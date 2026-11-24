"""Frozen two-head predictor and the experimental-record correction."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class ScPertEcho(nn.Module):
    """Compute two residuals for each perturbation and output gene."""
    def __init__(self, gene_count):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(514, 160), nn.GELU(), nn.LayerNorm(160))
        self.target = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128), nn.LayerNorm(128))
        self.gate = nn.Linear(160, 128)
        self.interaction = nn.Sequential(nn.Linear(288, 192), nn.GELU(), nn.Linear(192, 128))
        self.gene_id = nn.Embedding(gene_count, 32)
        self.genes = nn.Sequential(nn.Linear(160, 192), nn.GELU(), nn.Linear(192, 128), nn.LayerNorm(128))
        self.effect_view = nn.Parameter(torch.ones(2, 64))
        self.de_view = nn.Parameter(torch.ones(2, 64))

    def forward(self, target, anchors, view, genes, gene_index):
        code = nn.functional.one_hot(view, 2).to(target.dtype)
        state = self.context(torch.cat((anchors, code), dim=-1))
        gated = self.target(target) * (1 + 0.5 * torch.tanh(self.gate(state)))
        rna, de = self.interaction(torch.cat((gated, state), dim=-1)).chunk(2, dim=-1)
        gene_rna, gene_de = self.genes(torch.cat((genes, self.gene_id(gene_index)), dim=-1)).chunk(2, dim=-1)
        return (rna * self.effect_view[view]) @ gene_rna.T / 8, (de * self.de_view[view]) @ gene_de.T / 8


def read_arrays(value):
    if isinstance(value, (str, Path)):
        with np.load(value, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    return value


def load_model(weights):
    """Load numerical tensors from the published NPZ into an evaluation-mode CPU model."""
    data = read_arrays(weights)
    embed = data["gene_id.weight"]
    model = ScPertEcho(embed.shape[0])
    state = {}
    for key, value in data.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid model tensor: {key}")
        state[key] = torch.from_numpy(value)
    model.load_state_dict(state, strict=True)
    return model.eval()


def matrix(data, key, shape, allow_missing=False):
    value = np.asarray(data[key], dtype=np.float32)
    valid = np.logical_not(np.isinf(value)).all() if allow_missing else np.isfinite(value).all()
    if value.shape != shape or not valid:
        raise ValueError(f"{key} must have shape {shape} and valid numeric values")
    return value


def labels(data, key, unique=True):
    arr = np.asarray(data[key])
    if arr.ndim != 1 or not len(arr) or arr.dtype.kind not in "US" or not all(str(x).strip() for x in arr):
        raise ValueError(f"{key} must be a nonempty 1D array of text labels")
    arr = arr.astype(str)
    if unique and len(set(arr)) != len(arr):
        raise ValueError(f"{key} contains duplicate labels")
    return arr


def aligned(data, key, expected):
    if not np.array_equal(labels(data, key), expected):
        raise ValueError(f"{key} must match the predictor labels and order")


def integer_axis(data, key, size, low, high):
    arr = np.asarray(data[key])
    if arr.dtype.kind not in "iu" or arr.shape != (size,) or arr.min() < low or arr.max() >= high:
        raise ValueError(f"{key} must contain {size} integers between {low} and {high-1}")
    return torch.as_tensor(arr.astype(np.int64))


def apply_records(out, calibration, records):
    rows, genes = out["row_id"], out["gene_id"]
    aligned(records, "gene_id", genes)
    candidates = labels(records, "target_row", unique=False)
    if not set(candidates).issubset(set(rows)):
        raise ValueError("Record target_row contains an unknown forecast row")
    distance = matrix(records, "distance", (len(candidates),))
    if np.any(distance < 0):
        raise ValueError("Reference distances must be nonnegative")
    shape = (len(candidates), len(genes))
    for kind, source, dest in [("rna", "rna_z", "response"), ("de", "de_label", "de_score")]:
        eligible = np.asarray(records[kind + "_eligible"])
        if eligible.dtype != np.bool_ or eligible.shape != (len(candidates),):
            raise ValueError(f"{kind}_eligible must be a Boolean vector over records")
        values = matrix(records, source, shape, allow_missing=True)
        if kind == "de" and not np.isin(values[np.isfinite(values)], [0, 1]).all():
            raise ValueError("Measured DE labels must be 0, 1 or NaN")
        for i, row in enumerate(rows):
            selected = eligible & (candidates == row)
            if not np.any(selected):
                continue
            v = values[selected]
            valid = np.isfinite(v)
            ds = distance[selected]
            w = np.exp(-(ds - ds.min()) / 0.6)[:, None]
            if kind == "rna":
                change = np.clip(v, -8, 8) * calibration["response_sd"][i]
            else:
                change = v - out["de_score"][i]
            numerator = (w * np.where(valid, change, 0)).sum(axis=0)
            out[dest][i] += 0.6 * numerator / ((w * valid).sum(axis=0) + 2)


@torch.inference_mode()
def predict(model, inputs, calibration=None, records=None):
    """Predict from encoded inputs. Optional reference arrays calibrate responses and use measured records."""
    data = read_arrays(inputs)
    rows, genes = labels(data, "row_id"), labels(data, "gene_id")
    n, g = len(rows), len(genes)
    target = torch.from_numpy(matrix(data, "target_features", (n, 128)))
    anchors = torch.from_numpy(matrix(data, "control_anchors", (n, 512)))
    output = torch.from_numpy(matrix(data, "gene_features", (g, 128)))
    view = integer_axis(data, "view", n, 0, 2)
    ids = integer_axis(data, "gene_index", g, 0, model.gene_id.num_embeddings)
    if len(torch.unique(ids)) != g:
        raise ValueError("Output-gene indices must be unique")
    out = {"row_id": rows, "gene_id": genes, "rna_residual": np.empty((n, g), np.float32),
           "de_logit_residual": np.empty((n, g), np.float32)}
    for left in range(0, n, 64):
        group = slice(left, left + 64)
        for start in range(0, g, 2048):
            cols = slice(start, start + 2048)
            rna, de = model(target[group], anchors[group], view[group], output[cols], ids[cols])
            out["rna_residual"][group, cols] = rna.numpy()
            out["de_logit_residual"][group, cols] = de.numpy()
    if calibration is None:
        if records is not None:
            raise ValueError("Measured records require a calibration on the same gene scale")
        return out
    reference = read_arrays(calibration)
    aligned(reference, "row_id", rows)
    aligned(reference, "gene_id", genes)
    values = {k: matrix(reference, k, (n, g)) for k in ("response_mean", "response_sd", "prior_de_log_odds")}
    if np.any(values["response_sd"] <= 0):
        raise ValueError("All reference response scales must be positive")
    out["response"] = values["response_mean"] + values["response_sd"] * out["rna_residual"]
    out["de_score"] = torch.sigmoid(torch.from_numpy(values["prior_de_log_odds"] + out["de_logit_residual"])).numpy()
    if records is not None:
        apply_records(out, values, read_arrays(records))
    return out
