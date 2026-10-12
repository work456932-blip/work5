from dataclasses import asdict, dataclass, field
import hashlib
import json
import yaml

@dataclass
class Config:
    # Appendix G / Table 12 defaults.
    trajectories: int = 3
    max_states: int = 8
    candidate_nodes: int = 4
    repairs: int = 3
    equivalent_variants: int = 3
    symbolic_weight: float = .35
    vlm_weight: float = .65
    tau_c: float = .58
    tau_n: float = .08
    tau_u: float = .04
    tau_r: float = .18
    competitive_margin: float = .03
    objective_weights: list[float] = field(default_factory=lambda: [.55, .25, .20])
    acceptance_repeats: int = 3
    acceptance_confidence: float = .90
    acceptance_margin: float = .02
    # Unspecified quantities: explicit reference conventions; see docs/IMPLEMENTATION.md.
    necessity_weights: list[float] = field(default_factory=lambda: [.5, .5])
    sufficiency_weights: list[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])
    root_weights: list[float] = field(default_factory=lambda: [1., 1., 1., 1.])
    stability_min: float = 0.
    stability_variance_penalty: float = 4.
    critic_disagreement_tolerance: float = .25
    epsilon: float = 1e-8
    abstain_nonancestral_ties: bool = True
    use_graph: bool = True
    use_necessity: bool = True
    use_sufficiency: bool = True
    use_stability: bool = True
    use_lcb: bool = True
    earliest_root: bool = False
    full_regeneration: bool = False
    temperature: float = .7
    max_output_tokens: int = 4096
    image_max_side: int = 1024
    seed: int = 42
    backend_factory: str = "scenerepair.backends:HTTPBackend"
    backend_kwargs: dict = field(default_factory=lambda: {"model_id": "Qwen3-VL-8B", "revision": "SET_LOCAL_FROZEN_REVISION"})

    def __post_init__(self):
        for key in ("trajectories", "max_states", "candidate_nodes", "repairs", "equivalent_variants", "acceptance_repeats"):
            if getattr(self, key) < 1:
                raise ValueError(key + " must be positive")
        if self.acceptance_repeats < 2 or not .5 < self.acceptance_confidence < 1:
            raise ValueError("LCB needs at least two repeats and confidence in (.5,1)")
        if abs(self.symbolic_weight + self.vlm_weight - 1) > 1e-6:
            raise ValueError("Critic weights must sum to one")
        for name, size in (("objective_weights",3),("necessity_weights",2),("sufficiency_weights",3),("root_weights",4)):
            values = getattr(self, name)
            if len(values) != size or any(v < 0 for v in values) or sum(values) <= 0:
                raise ValueError("Invalid weights: " + name)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**(yaml.safe_load(f) or {}))

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()
