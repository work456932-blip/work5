import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
from .config import Config
from .data import load_examples,load_labels,read_jsonl
from .backends import make_backend
from .pipeline import SceneRepair
from .baselines import infer_baseline
from .metrics import evaluate,localization_metrics

ABLATIONS = {"no_graph":{"use_graph":False},"no_necessity":{"use_necessity":False},
             "no_sufficiency":{"use_sufficiency":False},"no_stability":{"use_stability":False},
             "no_lcb":{"use_lcb":False},"earliest_root":{"earliest_root":True}}

def write_json(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

def main():
    p = argparse.ArgumentParser(description="Anonymous SceneRepair inference and delayed-label evaluation")
    sub = p.add_subparsers(dest="command",required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--config",required=True)
    infer.add_argument("--inputs",required=True)
    infer.add_argument("--image-root")
    infer.add_argument("--typed-trajectories",help="Optional JSONL of {id, trajectories}; annotations stay separate")
    infer.add_argument("--output",required=True)
    infer.add_argument("--seed",type=int)
    infer.add_argument("--method",choices=["scenerepair","direct","cot","self_consistency","typed_rerank","full_regeneration"],default="scenerepair")
    infer.add_argument("--ablation",choices=sorted(ABLATIONS))
    ev = sub.add_parser("evaluate")
    ev.add_argument("--predictions",required=True)
    ev.add_argument("--labels",required=True)
    ev.add_argument("--direct")
    ev.add_argument("--root-annotations")
    ev.add_argument("--localization-controls",action="store_true")
    ev.add_argument("--bootstrap-repeats",type=int,default=2000)
    ev.add_argument("--output",required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--reports",nargs="+",required=True)
    summary.add_argument("--output",required=True)
    args = p.parse_args()
    if args.command=="infer":
        c = Config.load(args.config)
        if args.seed is not None:
            c = replace(c,seed=args.seed)
        if args.ablation:
            c = replace(c,**ABLATIONS[args.ablation])
        backend = make_backend(c.backend_factory,c.backend_kwargs)
        supplied = None
        if args.typed_trajectories:
            if args.method != "scenerepair":
                raise ValueError("Supplied traces are only supported for SceneRepair diagnosis")
            supplied = {str(v["id"]):v["trajectories"] for v in read_jsonl(args.typed_trajectories)}
        runner = SceneRepair(backend,c)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("Use a fresh output path to preserve prediction artifacts")
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(str(output)+".config.json",asdict(c))
        # Inference has no --labels argument and this branch never calls load_labels.
        with output.open("w",encoding="utf-8") as f:
            for example in load_examples(args.inputs,args.image_root):
                row = runner.infer(example,supplied[example.id] if supplied is not None else None) if args.method=="scenerepair" else infer_baseline(example,backend,c,args.method)
                row["option_count"] = len(example.options)
                row["seed"] = c.seed
                f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                f.flush()
    elif args.command=="evaluate":
        predictions = read_jsonl(args.predictions)  # Predictions already persisted.
        labels = load_labels(args.labels)         # Labels enter here, after inference.
        report = evaluate(predictions,labels,read_jsonl(args.direct) if args.direct else None,args.bootstrap_repeats)
        if args.root_annotations:
            report["localization"] = localization_metrics(predictions,read_jsonl(args.root_annotations))
            if args.localization_controls:
                from .localization_baselines import localize_saved_records
                annotations = read_jsonl(args.root_annotations)
                report["localization_controls"] = {m:localization_metrics(localize_saved_records(predictions,m),annotations)
                    for m in ("uncertainty","anomaly","necessity","paired_no_graph")}
        write_json(args.output,report)
        print(json.dumps(report,indent=2,allow_nan=False))
    else:
        import numpy as np
        reports = [json.loads(Path(x).read_text()) for x in args.reports]
        if len(reports)!=3:
            raise ValueError("Paper protocol uses three independent seed reports")
        keys = set.intersection(*(set(r) for r in reports))
        values = {k:{"mean":float(np.mean([r[k] for r in reports])),"std":float(np.std([r[k] for r in reports],ddof=1))}
                  for k in sorted(keys) if all(isinstance(r[k],(int,float)) for r in reports)}
        write_json(args.output,{"seeds":3,"statistics":values,"source_reports":args.reports})

if __name__=="__main__":
    main()
