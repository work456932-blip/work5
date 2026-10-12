"""Hybrid consistency: symbolic schema/geometry checks plus image-conditioned scores."""
from dataclasses import asdict
import json
import math
from .schema import State

INVERSE = {"left_of":"right_of","right_of":"left_of","above":"below","below":"above",
           "in_front_of":"behind","behind":"in_front_of","inside":"contains","contains":"inside",
           "near":"near","overlaps":"overlaps"}

def symbolic_score(state, trajectory, example):
    predecessors = [trajectory.by_id()[p] for p in state.depends_on]
    prior_objects = {v for s in predecessors for v in s.objects}
    checks = {"schema":True,
              "entity_continuity":not prior_objects or set(state.objects) <= prior_objects or state.step_type in {"observation","correspondence"},
              "view_references":set(state.evidence_views) <= set(example.images),
              "confidence":0 <= state.confidence <= 1}
    frames = {s.reference_frame for s in predecessors if s.reference_frame}
    checks["frame_compatibility"] = (not frames or state.reference_frame in frames or
        state.step_type in {"observation","correspondence","reference_frame"})
    op = state.operation or {}
    checks["operation_direction"] = op.get("direction") in {None,"clockwise","counterclockwise","left","right","up","down","forward","backward"}
    checks["operation_scope"] = op.get("scope") is None or isinstance(op["scope"], (str,list))
    relations = state.relations_after or state.relations_before
    triples = {(r.get("subject"),r.get("predicate"),r.get("object"),r.get("frame")) for r in relations}
    checks["relation_schema"] = all({"subject","predicate","object"} <= r.keys() for r in relations)
    checks["relation_entities"] = all(r.get("subject") in state.objects and r.get("object") in state.objects for r in relations)
    consistent = True
    for a,p,b,f in triples:
        # Only reject explicit contradictions; missing redundant inverse statements are allowed.
        if p in INVERSE and p != INVERSE[p]:
            if (a,INVERSE[p],b,f) in triples or (b,p,a,f) in triples:
                consistent = False
    checks["inverse_relations"] = consistent
    checks["visibility_entities"] = set(state.visibility) <= set(state.objects)
    return sum(checks.values())/len(checks), checks

class Critic:
    def __init__(self, session, config):
        self.session,self.config = session,config
        self.cache = {}

    def score(self, example, trajectory, repeat="initial"):
        raw = self.session.ask("verify",example,{"trajectory":trajectory.public()},repeat)
        records = {v["step_idx"]:v for v in raw["scores"]}
        if len(raw["scores"]) != len(records) or set(records) != set(trajectory.by_id()):
            raise ValueError("Verifier must score each state exactly once")
        result = {}
        for state in trajectory.states:
            key = json.dumps([asdict(state),[asdict(trajectory.by_id()[p]) for p in state.depends_on],example.public()],sort_keys=True)
            if key not in self.cache:
                self.cache[key] = symbolic_score(state,trajectory,example)
            sym,checks = self.cache[key]
            vis = float(records[state.step_idx]["consistency"])
            if not math.isfinite(vis) or not 0 <= vis <= 1:
                raise ValueError("Verifier consistency outside [0,1]")
            result[state.step_idx] = {"symbolic":sym,"vlm":vis,
                "consistency":self.config.symbolic_weight*sym+self.config.vlm_weight*vis,
                "agreement":abs(sym-vis)<=self.config.critic_disagreement_tolerance,
                "checks":checks,"supported_objects":records[state.step_idx].get("supported_objects",[]),
                "supported_views":records[state.step_idx].get("supported_views",[]),
                "reason":str(records[state.step_idx].get("reason",""))}
        return result

def consistencies(components):
    return {i:v["consistency"] for i,v in components.items()}
