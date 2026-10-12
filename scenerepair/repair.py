from dataclasses import asdict
from difflib import SequenceMatcher
import numpy as np
from .schema import State, Trajectory, probabilities
from .math_utils import lcb
from .critic import consistencies

def editable_indices(root, declared, config):
    if config.full_regeneration or not config.use_graph:
        return set(declared.nodes)
    return {root} | declared.descendants(root)

def regenerate(session, example, original, editable, forced=None, key="repair", verified=None):
    fixed = [asdict(s) for s in original.states if s.step_idx not in editable]
    payload = {"trajectory":original.public(),"editable_indices":sorted(editable),"fixed_states":fixed,
               "verified_indices":sorted(verified or []),"forced_replacement":asdict(forced) if forced else None}
    raw = session.ask("repair",example,payload,key)
    replacements = [State.parse(s) for s in raw["states"]]
    updates = {s.step_idx:s for s in replacements}
    if len(updates) != len(replacements) or set(updates) != editable:
        raise ValueError("Repair must return exactly the editable closure")
    original_by_id = original.by_id()
    known_objects = {o for s in original.states for o in s.objects}
    for i,s in updates.items():
        if s.step_type != original_by_id[i].step_type or s.depends_on != original_by_id[i].depends_on:
            raise ValueError("Repair changed state type or graph structure")
        if not set(s.objects) <= known_objects or not set(s.evidence_views) <= set(example.images):
            raise ValueError("Repair introduced an unsupported object/view")
    if forced and asdict(updates[forced.step_idx]) != asdict(forced):
        raise ValueError("Intervention target was not held fixed")
    merged = original.copy()
    merged.states = [updates.get(s.step_idx,s) for s in merged.states]
    merged.answer_distribution = probabilities(raw["answer_distribution"],example.options)
    merged.target_box,merged.target_view = raw.get("target_box"),raw.get("target_view")
    merged = Trajectory.parse(merged.public(),example,len(original.states))
    for s in merged.states:
        if s.step_idx not in editable and asdict(s) != asdict(original_by_id[s.step_idx]):
            raise AssertionError("Immutable state changed")
    # A disconnected terminal state cannot authorize changing its answer or grounding.
    if original.states[-1].step_idx not in editable:
        if (merged.answer_distribution != original.answer_distribution or merged.target_box != original.target_box
            or merged.target_view != original.target_view):
            raise ValueError("Answer/grounding changed outside the authorized closure")
    return merged

def similarity(candidate, original):
    # Same-position structured-state string similarity; unchanged states contribute 1.
    return float(np.mean([SequenceMatcher(None,str(asdict(a)),str(asdict(b))).ratio()
                          for a,b in zip(candidate.states,original.states)]))

def objective(trajectory, original, components, config):
    c = float(np.mean(list(consistencies(components).values())))
    w = config.objective_weights
    return w[0]*c + w[1]*max(trajectory.answer_distribution.values()) + w[2]*similarity(trajectory,original)

def choose_repair(session, critic, example, original, root, declared, original_components, config):
    editable = editable_indices(root,declared,config)
    verified = {i for i,v in original_components.items() if v["consistency"] >= config.tau_c}
    candidates,details = [],[]
    # Candidate 0 is always the unchanged trace, not a re-generated copy.
    for k in range(config.repairs):
        try:
            candidate = regenerate(session,example,original,editable,key=f"candidate:{k}",verified=verified)
            candidates.append((k+1,candidate))
        except Exception as e:
            session.failure("repair_candidate",e)
            details.append({"candidate":k+1,"valid":False,"error_type":type(e).__name__})
    # Independent verifier draws; same candidate-0 draw is paired with each candidate.
    references = []
    try:
        for m in range(config.acceptance_repeats):
            reference = critic.score(example,original,f"acceptance:original:{m}")
            references.append((objective(original,original,reference,config),reference))
    except Exception as e:
        session.failure("acceptance_reference",e)
        return original,{"decision":"preserve","reason":"reference_verifier_failure","editable":sorted(editable),"candidates":details}
    scored = []
    for k,candidate in candidates:
        try:
            draws = [critic.score(example,candidate,f"acceptance:candidate:{k}:{m}") for m in range(config.acceptance_repeats)]
            deltas = [objective(candidate,original,v,config)-references[m][0] for m,v in enumerate(draws)]
            mean_c = float(np.mean([np.mean(list(consistencies(v).values())) for v in draws]))
            baseline_c = float(np.mean([np.mean(list(consistencies(v).values())) for _,v in references]))
            bound = lcb(deltas,config.acceptance_confidence)
            detail = {"candidate":k,"valid":True,"gain_samples":deltas,"lcb":bound,
                      "mean_consistency":mean_c,"original_mean_consistency":baseline_c,
                      "similarity":similarity(candidate,original),"trajectory":candidate.public(),
                      "verifier_draws":draws}
            details.append(detail)
            scored.append((bound,k,candidate,mean_c,baseline_c))
        except Exception as e:
            session.failure("acceptance_candidate",e)
            details.append({"candidate":k,"valid":False,"error_type":type(e).__name__})
    if not scored:
        return original,{"decision":"preserve","reason":"no_valid_repair","editable":sorted(editable),"candidates":details}
    # Exactly Eq. 8: select the maximum LCB first, then require both gate conditions.
    best = max(scored,key=lambda x:(x[0],-x[1]))
    bound,k,candidate,c,b = best
    apply = (bound > config.acceptance_margin and c >= b) if config.use_lcb else True
    return (candidate if apply else original), {"decision":"apply" if apply else "preserve",
        "selected_candidate":k if apply else 0,"max_lcb":bound,"editable":sorted(editable),
        "consistency_preserved":c>=b,"candidates":details,
        "original_verifier_draws":[v for _,v in references]}
