"""Frozen label-free inference, Appendix B; audit data remain attached to each decision."""
import numpy as np
from .backends import ModelSession
from .critic import Critic, consistencies
from .schema import Trajectory, probabilities
from .graph import build_graphs
from .diagnosis import screen,select_root
from .interventions import probe_node
from .repair import choose_repair

def weighted_vote(trajectories):
    votes = {y:0. for y in trajectories[0][0].answer_distribution}
    for trajectory,weight,_ in trajectories:
        votes[trajectory.answer] += max(weight,1e-8)
    answer = max(votes,key=votes.get)
    representative = max((t for t in trajectories if t[0].answer==answer),key=lambda t:t[1])
    total = sum(votes.values())
    return answer,{y:v/total for y,v in votes.items()},representative

class SceneRepair:
    def __init__(self, backend, config):
        self.backend,self.config = backend,config

    def infer(self, example, provided_trajectories=None):
        session = ModelSession(self.backend,self.config,example.id)
        critic = Critic(session,self.config)
        traces,accepted = [],[]
        count = len(provided_trajectories) if provided_trajectories is not None else self.config.trajectories
        for i in range(count):
            try:
                # Synthetic-root evaluation override contains no root annotations.
                raw = provided_trajectories[i] if provided_trajectories is not None else session.ask("generate",example,{},f"trajectory:{i}")
                original = Trajectory.parse(raw,example,self.config.max_states)
                def confirm(a,b):
                    from dataclasses import asdict
                    return session.ask("confirm_edge",example,{"source":asdict(a),"target":asdict(b)},f"edge:{i}:{a.step_idx}:{b.step_idx}").get("confirmed") is True
                declared,diagnostic,edge_audit = build_graphs(original,confirm)
                components = critic.score(example,original,f"initial:{i}")
                record = {"index":i,"original":original.public(),"declared_graph":declared.public(),
                          "diagnostic_graph":diagnostic.public(),"edge_audit":edge_audit,"critic":components}
                signals = {node:probe_node(session,critic,example,original,node,declared,diagnostic,components,self.config)
                           for node in screen(original,components,self.config)}
                root,diagnosis = select_root(signals,diagnostic,self.config)
                record.update(signals=signals,root=root,diagnosis=diagnosis)
                if root is None:
                    selected,gate = original,{"decision":"preserve","reason":diagnosis["status"]}
                else:
                    selected,gate = choose_repair(session,critic,example,original,root,declared,components,self.config)
                record.update(selected=selected.public(),acceptance=gate)
                if gate["decision"] == "apply":
                    chosen = next(c for c in gate["candidates"] if c["candidate"]==gate["selected_candidate"])
                    consistency = chosen["mean_consistency"]
                else:
                    consistency = float(np.mean(list(consistencies(components).values())))
                accepted.append((selected,consistency,record))
                traces.append(record)
            except Exception as e:
                session.failure("typed_trajectory",e)
                traces.append({"index":i,"status":"fallback","error_type":type(e).__name__})
        if accepted:
            answer,distribution,(representative,weight,record) = weighted_vote(accepted)
            original_vote,_,_ = weighted_vote([(Trajectory.parse(t[2]["original"],example,self.config.max_states),t[1],t[2]) for t in accepted])
            prediction = {"answer":answer,"answer_distribution":distribution,"target_box":representative.target_box,
                "target_view":representative.target_view,"root":record["root"],
                "root_type":representative.by_id()[record["root"]].step_type if record["root"] is not None else None,
                "decision":record["acceptance"]["decision"],"root_score":record["diagnosis"]["scores"].get(record["root"],0.),
                "original_typed_vote":original_vote,"representative_trace":record["index"],"status":"ok"}
        else:
            # Deterministic preservation fallback: one direct request, no invented score.
            raw = session.ask("direct",example,{},"all_traces_failed")
            p = probabilities(raw["answer_distribution"],example.options)
            prediction = {"answer":max(p,key=p.get),"answer_distribution":p,"target_box":raw.get("target_box"),
                "target_view":raw.get("target_view"),"root":None,"root_type":None,"root_score":0.,
                "decision":"preserve","status":"direct_fallback"}
        return {"id":example.id,"group":example.group,"prediction":prediction,"traces":traces,
                "config_hash":self.config.digest(),"calls":session.records,"parse_events":session.parse_events,
                "cost":{"calls":len(session.records),"input_tokens":sum(c.get("input_tokens",0) for c in session.records),
                        "output_tokens":sum(c.get("output_tokens",0) for c in session.records),
                        "latency_seconds":sum(c["latency_seconds"] for c in session.records)}}
