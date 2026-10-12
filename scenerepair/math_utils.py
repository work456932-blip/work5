"""Paper Eqs. 3--8; JS uses natural logarithms and sample SD uses ddof=1."""
import math
from statistics import NormalDist
import numpy as np

def js_divergence(p, q):
    keys = sorted(p)
    if set(keys) != set(q):
        raise ValueError("Answer spaces differ")
    a,b = np.array([p[k] for k in keys]), np.array([q[k] for k in keys])
    m = (a+b)/2
    def kl(x):
        keep = x > 0
        return float(np.sum(x[keep] * np.log(x[keep]/m[keep])))
    return (kl(a)+kl(b))/2

def descendant_consistency(scores, node, graph):
    ids = {node} | graph.descendants(node)
    return float(np.mean([scores[k] for k in ids]))

def necessity(p, perturbed_p, before, after, weights):
    return weights[0]*js_divergence(p, perturbed_p) + weights[1]*max(before-after,0.)

def sufficiency(p, replaced_p, before_desc, after_desc, before_local, after_local, weights):
    return (weights[0]*max(after_desc-before_desc,0.)
            + weights[1]*max(after_local-before_local,0.)
            + weights[2]*max(max(replaced_p.values())-max(p.values()),0.))

def stability(answer_ids, decision_effects, consistency_effects, penalty=4.):
    # Appendix leaves Q unspecified: modal answer agreement times low-variance effects.
    if not answer_ids:
        return 0.
    agreement = max(answer_ids.count(v) for v in set(answer_ids))/len(answer_ids)
    return float(agreement * math.exp(-penalty*(np.var(decision_effects)+np.var(consistency_effects))))

def aggregate_families(records, epsilon=1e-8):
    # Effect-mass weights are computed from ALL valid realizations, never a best probe.
    groups = {}
    for r in records:
        if r["valid"]:
            groups.setdefault(r["family"], []).append(r)
    if not groups:
        return 0.,0.,0.
    masses = {k:sum(x["N"]+x["U"] for x in v) for k,v in groups.items()}
    denom = sum(masses.values())
    if denom <= epsilon:
        return 0.,0.,0.
    return tuple(float(sum(masses[k]/denom*np.mean([r[z] for r in v]) for k,v in groups.items())) for z in ("N","U","Q"))

def root_score(components, weights, epsilon=1e-8):
    return math.exp(sum(w*math.log(epsilon+v) for w,v in zip(weights,components))/sum(weights))

def lcb(deltas, confidence=.9):
    values = np.asarray(deltas,dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return float("-inf")
    return float(values.mean()-NormalDist().inv_cdf(confidence)*values.std(ddof=1)/math.sqrt(len(values)))
