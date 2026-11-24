"""Read labelled user tables without altering column names or row order."""
import csv
from pathlib import Path


def read_table(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if len(set(header)) != len(header) or any(not k or k != k.strip() for k in header):
            raise ValueError(f"Invalid or duplicated column names in {path}")
        if not set(required).issubset(header):
            raise ValueError(f"{path} must contain the columns {required}")
        data = list(reader)
    if not data or any(None in row or None in row.values() for row in data):
        raise ValueError(f"Expected nonempty rectangular data in {path}")
    return header, data


def unique(values, name):
    values = list(values)
    if len(values) != len(set(values)) or not values or any(not v or v != v.strip() for v in values):
        raise ValueError(f"Expected unique nonempty {name}")
    return values
