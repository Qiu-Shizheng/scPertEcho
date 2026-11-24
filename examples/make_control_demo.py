"""Write a synthetic control profile in the measured gene coordinates of the public encoder."""
import argparse
import csv
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("encoding",type=Path)
p.add_argument("output",type=Path)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
with np.load(args.encoding,allow_pickle=False) as fit:
    anchors=fit["output_gene"][fit["anchor_index"]]
    levels=np.maximum(0,fit["anchor_center"]+.1*fit["anchor_scale"]*np.sin(np.arange(len(anchors))))
    symbols=fit["symbol"].tolist()
    panel=[fit["output_gene"][symbols.index(s)] for s in ("EPCAM","KRT8","KRT18","COL1A1","VIM")]
def write(name,header,rows):
    with (args.output/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
write("controls.csv",["canonical_gene","synthetic"],zip(anchors,levels))
write("targets.csv",["row_id","target","context"],[["target-1","GATA6","synthetic"],["target-2","SOX17","synthetic"]])
write("genes.csv",["canonical_gene"],[[x] for x in panel])
