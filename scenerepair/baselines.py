"""Shared-budget controls. Published third-party algorithms use external adapters."""
import numpy as np
from .backends import ModelSession
from .schema import Trajectory, probabilities
from .critic import Critic,consistencies
from .graph import DAG
from .repair import choose_repair

def infer_baseline(example, backend, config, method):
    session,records = ModelSession(backend,config,example.id),[]
    critic = Critic(session,config)
    if method in {"direct","cot","self_consistency"}:
        task = "direct" if method=="direct" else "cot"
        budget = config.trajectories if method=="self_consistency" else 1
        for i in range(budget):
            raw = session.ask(task,example,{},str(i))
            p = probabilities(raw["answer_distribution"],example.options)
            records.append({**raw,"answer_distribution":p,"answer":max(p,key=p.get),"weight":1.})
    elif method in {"typed_rerank","full_regeneration"}:
        budget = config.trajectories if method=="typed_rerank" else 1
        for i in range(budget):
            trajectory = Trajectory.parse(session.ask("generate",example,{},str(i)),example,config.max_states)
            graph = DAG([s.step_idx for s in trajectory.states])
            for s in trajectory.states:
                for p in s.depends_on:
                    graph.add(p,s.step_idx)
            for s in trajectory.states:
                s.depends_on = [p for p in s.depends_on if (p,s.step_idx) in graph.edges]
            comps = critic.score(example,trajectory,str(i))
            audit = None
            if method=="full_regeneration":
                # Same repair/scoring prompts, all states editable; no root localization.
                from dataclasses import replace
                full = replace(config,full_regeneration=True)
                trajectory,audit = choose_repair(session,critic,example,trajectory,graph.nodes[0],graph,comps,full)
            records.append({**trajectory.public(),"answer":trajectory.answer,
                            "weight":float(np.mean(list(consistencies(comps).values()))),"acceptance":audit})
    else:
        raise ValueError("Unknown implemented control; use a published-baseline adapter")
    if method=="self_consistency":
        votes = {y:sum(r["answer"]==y for r in records) for y in example.options}
        answer = max(votes,key=votes.get)
        selected = next(r for r in records if r["answer"]==answer)
        prediction = {**selected,"answer_distribution":{y:v/len(records) for y,v in votes.items()}}
    else:
        prediction = max(records,key=lambda r:r["weight"])
    return {"id":example.id,"group":example.group,"prediction":prediction,"method":method,
            "config_hash":config.digest(),"baseline_candidates":records,"calls":session.records,
            "parse_events":session.parse_events,"cost":{"calls":len(session.records),
                "input_tokens":sum(r.get("input_tokens",0) for r in session.records),
                "output_tokens":sum(r.get("output_tokens",0) for r in session.records),
                "latency_seconds":sum(r["latency_seconds"] for r in session.records)}}
