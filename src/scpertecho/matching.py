"""Identify which of two perturbations produced each of two anonymous response profiles."""
import numpy as np
from .tables import read_table, unique


def match_profiles(forecasts, observed, forecast_ids, observed_ids, genes, exclude):
    """Compare the two assignments on the prespecified common panel after target exclusion."""
    genes = unique(genes, "gene identifiers")
    forecast_ids = unique(forecast_ids, "candidate identifiers")
    observed_ids = unique(observed_ids, "sample identifiers")
    if len(forecast_ids) != 2 or len(observed_ids) != 2:
        raise ValueError("Provide exactly two forecasts and two anonymous measurements")
    f, o = (np.asarray(x, dtype=np.float64) for x in (forecasts, observed))
    if f.shape != (2, len(genes)) or o.shape != f.shape:
        raise ValueError("Both matrices must be two by the number of panel genes")
    excluded = set(exclude)
    keep = np.array([g not in excluded for g in genes])
    if not keep.any() or not np.isfinite(f[:, keep]).all() or not np.isfinite(o[:, keep]).all():
        raise ValueError("A common finite downstream panel with at least one gene is required")
    a, b = f[:, keep], o[:, keep]
    with np.errstate(over="raise", invalid="raise"):
        direct = float(np.square(a - b).sum())
        swapped = float(np.square(a - b[::-1]).sum())
        # This is one half of swapped minus direct, without subtracting two large cost sums.
        margin = float(np.dot(a[0] - a[1], b[0] - b[1]))
    tie = margin == 0
    order = observed_ids if margin > 0 else list(reversed(observed_ids))
    return {"genes_used": int(keep.sum()), "genes_excluded": int((~keep).sum()), "direct_cost": direct,
            "swapped_cost": swapped, "tie": tie, "assignment": None if tie else dict(zip(forecast_ids, order))}


def match_files(forecasts, observed, panel, exclude):
    _, p = read_table(panel, ["canonical_gene"])
    genes = unique((r["canonical_gene"] for r in p), "panel genes")
    # An experiment without direct target genes in the panel may use a header-only exclusion file.
    import csv
    with open(exclude, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["canonical_gene"]:
            raise ValueError("The screening-library exclusion file requires one canonical_gene column")
        excluded = [row["canonical_gene"] for row in reader]
    if any(not x or x != x.strip() for x in excluded) or len(excluded) != len(set(excluded)):
        raise ValueError("Excluded genes must be unique nonempty identifiers")
    keep = [g for g in genes if g not in set(excluded)]
    ids, values = [], []
    for path in (forecasts, observed):
        _, rows = read_table(path, ["sample_id", *keep])
        ids.append([r["sample_id"] for r in rows])
        values.append(np.array([[float(r[g]) if g in keep else 0.0 for g in genes] for r in rows]))
    return match_profiles(*values, *ids, genes, excluded)
