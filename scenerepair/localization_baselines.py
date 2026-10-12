"""Root-localization controls on saved traces/probes; no new labels used."""
import copy
from .math_utils import root_score

def localize_saved_records(records, method):
    """Select by uncertainty, anomaly, necessity or graph-free paired effects.

    Each control operates on screened nodes from the full audit. The paired
    control uses A/N/U/Q without ancestry. Exact original baseline tuning is
    unspecified in the manuscript; these are explicit reference controls.
    """
    if method not in {"uncertainty","anomaly","necessity","paired_no_graph"}:
        raise ValueError("Unknown root-localization control")
    out = copy.deepcopy(records)
    for row in out:
        for trace in row.get("traces",[]):
            states = {s["step_idx"]:s for s in trace.get("original",{}).get("states",[])}
            signals = {int(k):v for k,v in trace.get("signals",{}).items()}
            scores = {}
            for i,s in signals.items():
                if method=="uncertainty":
                    value = 1-states[i]["confidence"]
                elif method=="anomaly":
                    value = s["A"]
                elif method=="necessity":
                    value = s["N"]
                else:
                    value = root_score([s[k] for k in ("A","N","U","Q")],[1]*4)
                scores[i] = value
            trace["root"] = max(scores,key=lambda i:(scores[i],-i)) if scores else None
            trace["diagnosis"] = {"scores":scores,"passing":list(scores),"status":method,"competitive_margin":0.}
            trace["diagnostic_graph"] = {"nodes":list(states),"edges":[]}
        representative = row["prediction"].get("representative_trace")
        trace = next((t for t in row.get("traces",[]) if t.get("index")==representative),None)
        if trace:
            row["prediction"]["root"] = trace["root"]
        row["localization_control"] = method
    return out
