"""Development-only threshold selection; external benchmarks reuse the frozen YAML."""
from dataclasses import asdict,replace
import itertools
import numpy as np
from .diagnosis import select_root
from .graph import DAG
from .math_utils import lcb

def replay_trace(trace, config):
    """Replay diagnosis + stored repair draws. A changed root needs NEW candidates."""
    graph_raw = trace["diagnostic_graph"]
    graph = DAG(graph_raw["nodes"],{tuple(e) for e in graph_raw["edges"]})
    signals = {int(k):v for k,v in trace["signals"].items()}
    root,diagnosis = select_root(signals,graph,config)
    if root is None:
        return trace["original"]["answer_distribution"],"preserve"
    if root != trace["root"]:
        raise ValueError("Threshold replay changed the root; regenerate candidates on development inputs")
    candidates = [c for c in trace["acceptance"].get("candidates",[]) if c.get("valid")]
    if not candidates:
        return trace["original"]["answer_distribution"],"preserve"
    best = max(candidates,key=lambda c:lcb(c["gain_samples"],config.acceptance_confidence))
    bound = lcb(best["gain_samples"],config.acceptance_confidence)
    if not config.use_lcb or (bound>config.acceptance_margin and best["mean_consistency"]>=best["original_mean_consistency"]):
        return best["trajectory"]["answer_distribution"],"apply"
    return trace["original"]["answer_distribution"],"preserve"

def select_on_development(base, grid, run_development, objective):
    """Caller supplies dev-only inference and scoring functions; no test data used.

    run_development(config) must create/reuse label-free predictions BEFORE its
    separate objective(predictions) loads development labels. Persist returned
    selected config once; transfer tasks must not call this routine.
    """
    keys = sorted(grid)
    records,best = [],None
    for values in itertools.product(*(grid[k] for k in keys)):
        config = replace(base,**dict(zip(keys,values)))
        predictions = run_development(config)
        score = float(objective(predictions))
        row = {"parameters":dict(zip(keys,values)),"score":score,"config_hash":config.digest()}
        records.append(row)
        if best is None or score>best[0]:
            best = score,config
    return best[1],records
