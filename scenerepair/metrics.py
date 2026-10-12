"""Evaluation-only metrics. Percentages are on [0,100], distances in graph edges."""
import numpy as np
from scipy.stats import binomtest
from .graph import DAG

def iou(a,b):
    if a is None or b is None:
        return 0.
    width = max(0.,min(a[2],b[2])-max(a[0],b[0]))
    height = max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter = width*height
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area>0 else 0.

def macro(values, groups):
    return float(np.mean([np.mean(values[np.array(groups)==g]) for g in sorted(set(groups))]))

def paired_bootstrap(a,b,groups,repeats=2000,seed=0):
    a,b = np.asarray(a,float),np.asarray(b,float)
    rng = np.random.default_rng(seed)
    subsets = [np.flatnonzero(np.array(groups)==g) for g in sorted(set(groups))]
    deltas = []
    for _ in range(repeats):
        deltas.append(np.mean([np.mean((a-b)[rng.choice(s,len(s),replace=True)]) for s in subsets]))
    return [float(v*100) for v in np.quantile(deltas,[.025,.975])]

def mcnemar(a,b):
    a,b = np.asarray(a,bool),np.asarray(b,bool)
    improved,harmed = int(np.sum(a & ~b)),int(np.sum(~a & b))
    return {"wrong_to_correct_count":improved,"correct_to_wrong_count":harmed,
            "p_value":float(binomtest(improved,improved+harmed,.5).pvalue) if improved+harmed else 1.}

def ece(confidence,correct,bins=10):
    c,y = np.asarray(confidence),np.asarray(correct)
    out = 0.
    for i in range(bins):
        mask = (c>=i/bins)&((c<(i+1)/bins) if i<bins-1 else (c<=1))
        if mask.any():
            out += mask.mean()*abs(c[mask].mean()-y[mask].mean())
    return float(out)

def evaluate(predictions,labels,direct=None,bootstrap_repeats=2000):
    by_id = {str(v["id"]):v for v in predictions}
    if len(by_id)!=len(predictions) or set(by_id)!=set(labels):
        raise ValueError("Prediction IDs and evaluation-label IDs must match exactly")
    ids = sorted(by_id)
    groups = [by_id[i].get("group","") for i in ids]
    correct = np.array([by_id[i]["prediction"]["answer"]==labels[i]["answer"] for i in ids])
    confidence = [max(by_id[i]["prediction"]["answer_distribution"].values()) for i in ids]
    report = {"n":len(ids),"accuracy":100*macro(correct,groups),"micro_accuracy":100*float(correct.mean()),
              "ece":ece(confidence,correct),"group_accuracy":{g:100*float(correct[np.array(groups)==g].mean()) for g in sorted(set(groups))}}
    if all("option_count" in by_id[i] for i in ids):
        chance = np.array([1/by_id[i]["option_count"] for i in ids])
        report["chance_normalized_accuracy"] = 100*macro((correct-chance)/(1-chance),groups)
    boxed = [i for i in ids if "target_box" in labels[i]]
    if boxed:
        grounded = np.array([by_id[i]["prediction"]["answer"]==labels[i]["answer"] and
            by_id[i]["prediction"].get("target_view")==labels[i].get("target_view") and
            iou(by_id[i]["prediction"].get("target_box"),labels[i]["target_box"])>=.5 for i in boxed])
        report["grounded_answer_accuracy"] = 100*macro(grounded,[by_id[i].get("group","") for i in boxed])
    applied = np.array([by_id[i]["prediction"].get("decision")=="apply" for i in ids])
    report["apply_rate"] = 100*macro(applied,groups)
    if direct is not None:
        baseline = {str(v["id"]):v for v in direct}
        if set(baseline)!=set(by_id):
            raise ValueError("Direct comparison requires identical IDs")
        clean = np.array([baseline[i]["prediction"]["answer"]==labels[i]["answer"] for i in ids])
        fixed,broken = ~clean & correct,clean & ~correct
        report.update(wrong_to_correct=100*macro(fixed,groups),correct_to_wrong=100*macro(broken,groups),
            net_transition_gain=100*macro(fixed.astype(float)-broken.astype(float),groups),
            accuracy_gain=100*macro(correct.astype(float)-clean.astype(float),groups),
            gain_ci95=paired_bootstrap(correct,clean,groups,bootstrap_repeats),mcnemar=mcnemar(correct,clean))
        # Explicit denominators: avoid assigning the paper's undefined 'no-harm' convention.
        repaired_originally_correct = applied & clean
        report["no_harm_among_applied_originally_correct"] = (100*float(correct[repaired_originally_correct].mean()) if repaired_originally_correct.any() else None)
        report["harm_among_applied"] = 100*float(broken[applied].mean()) if applied.any() else None
    report["accepted_answer_error_risk"] = 100*float((~correct)[applied].mean()) if applied.any() else None
    report["risk_coverage"] = []
    for margin in [0.,.01,.02,.03,.05,.1]:
        selected, hypothetical_correct = [],[]
        for i in ids:
            row = by_id[i]
            votes, choices = {},[]
            for trace in row.get("traces",[]):
                if "original" not in trace:
                    continue
                gate = trace.get("acceptance",{})
                candidates = [c for c in gate.get("candidates",[]) if c.get("valid")]
                best = max(candidates,key=lambda c:c["lcb"]) if candidates else None
                apply = bool(best and best["lcb"]>margin and best["mean_consistency"]>=best["original_mean_consistency"])
                trajectory = best["trajectory"] if apply else trace["original"]
                weight = best["mean_consistency"] if apply else float(np.mean([v["consistency"] for v in trace["critic"].values()]))
                answer = max(trajectory["answer_distribution"],key=trajectory["answer_distribution"].get)
                votes[answer] = votes.get(answer,0.)+max(weight,1e-8)
                choices.append((answer,weight,apply))
            if votes:
                answer = max(votes,key=votes.get)
                representative = max((c for c in choices if c[0]==answer),key=lambda c:c[1])
                selected.append(representative[2])
                hypothetical_correct.append(answer==labels[i]["answer"])
            else:
                selected.append(False)
                hypothetical_correct.append(row["prediction"]["answer"]==labels[i]["answer"])
        mask = np.asarray(selected,bool)
        errors = ~np.asarray(hypothetical_correct,bool)
        report["risk_coverage"].append({"margin":margin,"coverage":float(mask.mean()),"accepted_answer_error_risk":float(errors[mask].mean()) if mask.any() else None})
    for key in ("calls","input_tokens","output_tokens","latency_seconds"):
        report["mean_"+key] = float(np.mean([by_id[i].get("cost",{}).get(key,0.) for i in ids]))
    return report

def localization_metrics(predictions, annotations):
    rows = {str(v["id"]):v for v in predictions}
    exact,within,distances,root_vs = [],[],[],[]
    excluded = 0
    for a in annotations:
        if a.get("ambiguous",False):
            excluded += 1
            continue
        row = rows[str(a["id"])]
        index = a.get("trajectory_index",row["prediction"].get("representative_trace"))
        trace = next((t for t in row.get("traces",[]) if t.get("index")==index),None)
        root = trace.get("root") if trace else None
        gold = a["root_idx"]
        exact.append(root==gold)
        graph_raw = trace.get("declared_graph",{}) if trace else {}
        graph = DAG(graph_raw.get("nodes",[]),{tuple(v) for v in graph_raw.get("edges",[])})
        d = graph.distance(root,gold)
        # Abstentions/disconnected diagnoses count as misses, with graph-size penalty.
        d = d if d is not None else max(1,len(graph.nodes))
        within.append(root is not None and d<=1)
        distances.append(d)
        # Independent pair discrimination, distinct from exact localization.
        # For each annotated root/symptom pair apply competitive ancestry to
        # those TWO candidates; a wrong third global root is not conflated here.
        if trace and "symptom_indices" in a:
            diagnostic_raw = trace.get("diagnostic_graph",graph_raw)
            diagnostic = DAG(diagnostic_raw.get("nodes",[]),{tuple(v) for v in diagnostic_raw.get("edges",[])})
            scores = {int(k):float(v) for k,v in trace.get("diagnosis",{}).get("scores",{}).items()}
            passing = set(trace.get("diagnosis",{}).get("passing",[]))
            margin = trace.get("diagnosis",{}).get("competitive_margin",.03)
            for symptom in a["symptom_indices"]:
                pair = [p for p in (gold,symptom) if p in passing]
                if not pair:
                    root_vs.append(False)
                    continue
                maximum = max(scores[p] for p in pair)
                pair = [p for p in pair if scores[p]>=maximum-margin]
                remaining = [p for p in pair if not (diagnostic.ancestors(p)&set(pair))]
                root_vs.append(len(remaining)==1 and remaining[0]==gold)
    if not exact:
        return {"n":0,"ambiguous_excluded":excluded}
    return {"n":len(exact),"ambiguous_excluded":excluded,"exact_accuracy":100*float(np.mean(exact)),
            "within_one_accuracy":100*float(np.mean(within)),"node_distance_mae":float(np.mean(distances)),
            "root_vs_symptom_accuracy":100*float(np.mean(root_vs)) if root_vs else None}
