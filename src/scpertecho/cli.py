"""Command-line tools for the scPertEcho model and two-profile assignment."""
import argparse
import json
from pathlib import Path
import numpy as np
from .encoding import encode
from .matching import match_files

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle:
        try:
            np.savez_compressed(handle, **data)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(prog="scpertecho",description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    def command(name, description, fields):
        part=commands.add_parser(name,help=description,description=description)
        for key,message in fields:
            part.add_argument("--"+key,required=True,type=Path,help=message)
        return part
    command("match","Assign two anonymous profiles to two candidate perturbations.",
            [("forecasts","Two predicted response profiles, CSV"),("observed","Two anonymous profiles, CSV"),
             ("panel","Prespecified panel of common genes, CSV"),("exclude","Genes targeted by the screening library, CSV")])
    command("demo","Run the frozen network and record correction on invented numerical inputs.",
            [("weights","Public fitted weight NPZ"),("output","New result NPZ")])
    command("encode","Encode target symbols and untreated expression using the fitted transforms.",
            [("encoding","Public fitted input transforms"),("annotations","Arc PIE gene_text directory"),
             ("controls","Untreated control averages, CSV"),("targets","Target and context table, CSV"),
             ("genes","Prespecified output gene IDs, CSV"),("output","New encoded input NPZ")])
    infer=command("predict","Forecast from prepared model inputs.",
                  [("weights","Public fitted weight NPZ"),("inputs","Encoded inputs from encode or the Python API"),("output","New result NPZ")])
    infer.add_argument("--calibration",type=Path,help="Aligned per-target reference means, scales and prior DE log odds")
    infer.add_argument("--records",type=Path,help="Aligned eligible measured records for the same targets")
    args=parser.parse_args(argv)
    if args.command=="match":
        print(json.dumps(match_files(args.forecasts,args.observed,args.panel,args.exclude),indent=2))
    elif args.command=="encode":
        out=encode(args.encoding,args.annotations,args.controls,args.targets,args.genes)
        save(args.output,out)
        print(f"{len(out['row_id'])} target rows and {len(out['gene_id'])} output genes")
    else:
        from .model import load_model, predict
        model=load_model(args.weights)
        if args.command=="demo":
            from .demo import demo_inputs
            out=predict(model,*demo_inputs())
        else:
            out=predict(model,args.inputs,args.calibration,args.records)
        save(args.output,out)
        report={"rows":len(out["row_id"]),"genes":len(out["gene_id"]),"outputs":list(out)}
        if args.command=="demo":
            report["first_response"]=round(float(out["response"][0,0]),6)
            report["first_de_score"]=round(float(out["de_score"][0,0]),6)
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
