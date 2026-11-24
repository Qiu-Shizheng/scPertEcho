"""Map a human perturbation, control profile and output panel to frozen model coordinates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .tables import read_table, unique

EMBEDDINGS_SHA256 = "7ae69de3086f65a145ea571d4c34ad9fc52d9c60cd72b6fc02d3b213428c6cf7"
META_SHA256 = "edb1fa415cbc02c9d27101991407ac8a63b74f38f377a76c19a28b7ebddafcc1"


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode(encoding, annotations, controls, targets, genes):
    """Encode control averages, candidate symbols and a prespecified downstream gene panel."""
    with np.load(encoding, allow_pickle=False) as fit:
        union, symbols = fit["output_gene"], fit["symbol"]
        gene_index = {g: i for i, g in enumerate(union)}
        anchors = union[fit["anchor_index"]]
        head, table = read_table(controls, ["canonical_gene"])
        ctrl_ids = unique((r["canonical_gene"] for r in table), "untreated-control gene IDs")
        ctable = dict(zip(ctrl_ids, table))
        _, rows = read_table(targets, ["row_id", "target", "context"])
        row_ids = unique((r["row_id"] for r in rows), "target row IDs")
        context_set = {r["context"] for r in rows}
        if not context_set.issubset(set(head)-{"canonical_gene"}):
            raise ValueError("Each target context requires its own column of untreated-control averages")
        present = np.array([v in ctable for v in anchors])
        if present.mean() < .8:
            raise ValueError(f"Only {int(present.sum())}/512 control anchors supplied. At least 410 are required")
        ctx = {}
        for name in context_set:
            raw = np.array([float(ctable[v][name]) for v in anchors[present]], dtype=np.float32)
            if not np.isfinite(raw).all() or raw.min() < 0:
                raise ValueError("Use finite, nonnegative mean per-cell log1p(CP10k) controls")
            values = np.zeros(512, dtype=np.float32)
            values[present] = np.clip((raw-fit["anchor_center"][present])/fit["anchor_scale"][present], -6, 6)
            ctx[name] = values
        _, panel = read_table(genes, ["canonical_gene"])
        out = unique((r["canonical_gene"] for r in panel), "output genes")
        missing = [v for v in out if v not in gene_index]
        if missing:
            raise ValueError(f"Output genes absent from the trained gene axis: {missing[:5]}")
        index = np.array([gene_index[x] for x in out], dtype=np.int64)
        folder = Path(annotations)
        for name, expected in [("meta.json", META_SHA256), ("embeddings.npy", EMBEDDINGS_SHA256)]:
            if file_hash(folder/name) != expected:
                raise ValueError(f"Expected the pinned Arc PIE gene_text {name}")
        vocab = {key: i for i, key in enumerate(json.loads((folder/"meta.json").read_text())["keys"])}
        vec = np.load(folder/"embeddings.npy", mmap_mode="r", allow_pickle=False)
        basis, center, scale = fit["projection"], fit["annotation_center"], fit["annotation_scale"]
        def features(names):
            result = np.zeros((len(names), 128), dtype=np.float32)
            for start in range(0, len(names), 1024):
                indices = [i for i in range(start, min(start+1024, len(names))) if names[i] in vocab]
                if indices:
                    source = vec[[vocab[names[i]] for i in indices]]
                    result[indices] = np.clip(((source-center) @ basis.T)/scale, -8, 8)
            return result
        names = [r["target"] for r in rows]
        if any(not x or x != x.strip() for x in names):
            raise ValueError("Targets must be nonempty gene symbols")
        return dict(row_id=np.array(row_ids), gene_id=np.array(out), gene_index=index,
                    target_features=features(names), gene_features=features(symbols[index]),
                    control_anchors=np.stack([ctx[r["context"]] for r in rows]),
                    view=np.zeros(len(rows), dtype=np.int64),
                    target_annotation_found=np.array([n in vocab for n in names]),
                    gene_annotation_found=np.array([n in vocab for n in symbols[index]]))
