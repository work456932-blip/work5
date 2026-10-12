"""Declared graph authorizes edits; verified inferred edges only expand diagnosis."""
from dataclasses import dataclass, field

PRIORS = {("correspondence", "reference_frame"), ("reference_frame", "counterfactual_operation"),
          ("counterfactual_operation", "updated_relation")}

@dataclass
class DAG:
    nodes: list[int]
    edges: set[tuple[int, int]] = field(default_factory=set)

    def descendants(self, node):
        seen, todo = set(), [node]
        while todo:
            u = todo.pop()
            for a, b in self.edges:
                if a == u and b not in seen:
                    seen.add(b)
                    todo.append(b)
        return seen

    def ancestors(self, node):
        return {u for u in self.nodes if node in self.descendants(u)}

    def add(self, a, b):
        if a not in self.nodes or b not in self.nodes or a == b or a in self.descendants(b):
            return False
        self.edges.add((a, b))
        return True

    def depth(self, node):
        parents = [a for a,b in self.edges if b == node]
        return 0 if not parents else 1 + max(self.depth(p) for p in parents)

    def distance(self, a, b):
        if a not in self.nodes or b not in self.nodes:
            return None
        todo, seen = [(a, 0)], {a}
        while todo:
            u, d = todo.pop(0)
            if u == b:
                return d
            for v,w in self.edges:
                n = w if v == u else (v if w == u else None)
                if n is not None and n not in seen:
                    seen.add(n)
                    todo.append((n,d+1))
        return None

    def public(self):
        return {"nodes": self.nodes, "edges": sorted(self.edges)}

def build_graphs(trajectory, confirm):
    ids = [s.step_idx for s in trajectory.states]
    declared = DAG(ids)
    rejected = []
    for state in trajectory.states:
        for parent in state.depends_on:
            if not declared.add(parent, state.step_idx):
                rejected.append([parent, state.step_idx])
    # Sanitize rejected declarations in the actual trajectory before prompting/repairing.
    for s in trajectory.states:
        s.depends_on = [p for p in s.depends_on if (p,s.step_idx) in declared.edges]
    diagnostic = DAG(ids, set(declared.edges))
    inferred = []
    for a in trajectory.states:
        for b in trajectory.states:
            if a.step_idx < b.step_idx and (a.step_type,b.step_type) in PRIORS:
                edge = (a.step_idx,b.step_idx)
                if edge not in diagnostic.edges and confirm(a,b) and diagnostic.add(*edge):
                    inferred.append(list(edge))
    return declared, diagnostic, {"rejected_declared": rejected, "verified_inferred": inferred}
