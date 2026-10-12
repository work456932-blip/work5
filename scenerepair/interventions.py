"""Paired admissible semantic probes, paper Eqs. 3--5."""
from dataclasses import asdict
from .schema import State
from .critic import consistencies
from .math_utils import aggregate_families, descendant_consistency, necessity, sufficiency, stability, js_divergence
from .repair import regenerate

FAMILIES = {
    "observation":["evidence"],
    "correspondence":["referent","evidence"],
    "reference_frame":["viewpoint","evidence"],
    "initial_state":["relation","referent"],
    "counterfactual_operation":["operation","scope"],
    "updated_relation":["relation","referent"],
    "visibility":["visibility","view_dropout"],
}
REALIZATIONS = ["canonical wording","meaning-equivalent lexical formulation","meaning-equivalent explicit referent formulation"]

def structural_admissibility(original, replacement, trajectory, example):
    if (original.step_idx,original.step_type,original.depends_on) != (replacement.step_idx,replacement.step_type,replacement.depends_on):
        return False
    entities = {o for s in trajectory.states for o in s.objects}
    if not set(replacement.objects) <= entities or not set(replacement.evidence_views) <= set(example.images):
        return False
    # Transformations may change direction/scope but not transformation class.
    if original.step_type == "counterfactual_operation":
        if (original.operation or {}).get("class") != (replacement.operation or {}).get("class"):
            return False
    return True

def probe_node(session, critic, example, trajectory, node, declared, diagnostic, components, config):
    target = trajectory.by_id()[node]
    before = consistencies(components)
    before_desc = descendant_consistency(before,node,diagnostic)
    editable = {node} | declared.descendants(node)
    # Interventions stay descendant-closed even in root-score ablations.
    records = []
    verified = {i for i,v in components.items() if v["consistency"] >= config.tau_c}
    for family in FAMILIES.get(target.step_type,[]):
        family_records,anchor = [],None
        for m in range(config.equivalent_variants):
            record = {"family":family,"realization":m,"valid":False}
            try:
                pair = session.ask("propose",example,{"trajectory":trajectory.public(),"target":asdict(target),
                    "family":family,"realization":REALIZATIONS[m % len(REALIZATIONS)],"anchor_pair":anchor,
                    "verified_ancestors":[asdict(trajectory.by_id()[i]) for i in sorted(declared.ancestors(node) & verified)]},f"probe:{node}:{family}:{m}")
                destroy,replace = State.parse(pair["destructive"]),State.parse(pair["corrective"])
                record["pair"] = pair
                if not all(structural_admissibility(target,s,trajectory,example) for s in (destroy,replace)):
                    record["reason"] = "structural_admissibility_failed"
                    records.append(record)
                    continue
                validity = session.ask("admissibility",example,{"trajectory":trajectory.public(),"target":node,
                    "pair":pair,"anchor_pair":anchor,"verified_ancestors":[asdict(trajectory.by_id()[i]) for i in sorted(declared.ancestors(node)&verified)]},f"validity:{node}:{family}:{m}")
                record["admissibility"] = validity
                if not all(validity.get(k) is True for k in ("destructive_valid","corrective_valid","equivalent_to_anchor")):
                    records.append(record)
                    continue
                if anchor is None:
                    anchor = pair
                td = regenerate(session,example,trajectory,editable,destroy,f"destructive:{node}:{family}:{m}",verified)
                tr = regenerate(session,example,trajectory,editable,replace,f"corrective:{node}:{family}:{m}",verified)
                cd = critic.score(example,td,f"destructive_verify:{node}:{family}:{m}")
                cr = critic.score(example,tr,f"corrective_verify:{node}:{family}:{m}")
                ds = descendant_consistency(consistencies(cd),node,diagnostic)
                rs = descendant_consistency(consistencies(cr),node,diagnostic)
                n = necessity(trajectory.answer_distribution,td.answer_distribution,before_desc,ds,config.necessity_weights)
                u = sufficiency(trajectory.answer_distribution,tr.answer_distribution,before_desc,rs,before[node],cr[node]["consistency"],config.sufficiency_weights)
                record.update(valid=True,N=n,U=u,destructive_answer=td.answer,corrective_answer=tr.answer,
                    destructive_answer_effect=js_divergence(trajectory.answer_distribution,td.answer_distribution),
                    corrective_answer_effect=js_divergence(trajectory.answer_distribution,tr.answer_distribution),
                    destructive_consistency_effect=before_desc-ds,corrective_consistency_effect=rs-before_desc,
                    destructive_trajectory=td.public(),corrective_trajectory=tr.public(),
                    destructive_critic=cd,corrective_critic=cr)
                family_records.append(record)
            except Exception as e:
                session.failure("intervention",e)
                record["reason"] = type(e).__name__
            records.append(record)
        q = 0.
        # A single surviving realization cannot establish linguistic stability.
        if len(family_records) >= 2:
            q = sum(stability([r[k+"_answer"] for r in family_records],
                    [r[k+"_answer_effect"] for r in family_records],
                    [r[k+"_consistency_effect"] for r in family_records],config.stability_variance_penalty)
                    for k in ("destructive","corrective"))/2
        for r in family_records:
            r["Q"] = q
    n,u,q = aggregate_families(records,config.epsilon)
    return {"A":1-before[node],"N":n,"U":u,"Q":q,"interventions":records,
            "critic_agreement":components[node]["agreement"]}
