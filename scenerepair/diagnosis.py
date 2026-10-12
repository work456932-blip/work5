from .math_utils import root_score

def screen(trajectory, components, config):
    eligible = [s.step_idx for s in trajectory.states if s.step_type != "answer_mapping"]
    # Conditional diagnosis: easy traces remain unchanged without interventional calls.
    if not any(components[i]["consistency"] < config.tau_c for i in eligible):
        return []
    ranked = sorted(eligible,key=lambda i:(components[i]["consistency"],i))
    return sorted(set(ranked[:config.candidate_nodes]) | {i for i in eligible if components[i]["consistency"] < config.tau_c})

def select_root(signals, diagnostic, config):
    scores,passing = {},[]
    for node,s in signals.items():
        components = [s["A"],s["N"],s["U"],s["Q"]]
        weights = list(config.root_weights)
        for flag,index in ((config.use_necessity,1),(config.use_sufficiency,2),(config.use_stability,3)):
            if not flag:
                weights[index] = 0.
        r = root_score(components,weights,config.epsilon)
        scores[node] = r
        valid_probes = any(v["valid"] for v in s["interventions"])
        if (valid_probes and s["A"] > 1-config.tau_c and
            (not config.use_necessity or s["N"] >= config.tau_n) and
            (not config.use_sufficiency or s["U"] >= config.tau_u) and
            (not config.use_stability or s["Q"] > config.stability_min) and r >= config.tau_r):
            passing.append(node)
    audit = {"scores":scores,"passing":passing,"competitive_margin":config.competitive_margin}
    if not passing:
        return None,{**audit,"status":"abstain_no_component_pass"}
    best = max(scores[i] for i in passing)
    competitive = passing if config.earliest_root else [i for i in passing if scores[i] >= best-config.competitive_margin]
    remaining = [i for i in competitive if not (diagnostic.ancestors(i) & set(competitive))] if config.use_graph else competitive
    audit.update(competitive=competitive,remaining=remaining,tie=len(remaining)>1)
    if len(remaining)>1 and config.abstain_nonancestral_ties and not config.earliest_root:
        return None,{**audit,"status":"abstain_ambiguous_roots"}
    root = min(remaining,key=lambda i:(diagnostic.depth(i),-scores[i],i))
    return root,{**audit,"status":"root_selected","high_confidence":signals[root]["critic_agreement"]}
