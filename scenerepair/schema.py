"""Typed commitments (paper Eq. 1 and Appendix H). No label field exists here."""
from dataclasses import asdict, dataclass, field
import copy
import math

TYPES = {"observation", "correspondence", "reference_frame", "initial_state",
         "counterfactual_operation", "updated_relation", "visibility", "answer_mapping"}

@dataclass
class Example:
    id: str
    question: str
    images: dict[str, str]
    options: dict[str, str]
    group: str = ""

    def public(self):
        return asdict(self)

@dataclass
class State:
    step_idx: int
    step_type: str
    description: str
    objects: list[str]
    reference_frame: str | None
    operation: dict | None
    relations_before: list[dict]
    relations_after: list[dict]
    visibility: dict
    evidence_views: list[str]
    depends_on: list[int]
    confidence: float

    @classmethod
    def parse(cls, value):
        required = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError("Incomplete typed state schema")
        s = cls(**{k: value[k] for k in required})
        if s.step_type not in TYPES or not isinstance(s.step_idx, int):
            raise ValueError("Unknown state type or noninteger index")
        if not isinstance(s.description, str) or not isinstance(s.confidence, (int, float)):
            raise ValueError("Invalid description or confidence")
        if not math.isfinite(s.confidence) or not 0 <= s.confidence <= 1:
            raise ValueError("Confidence must be in [0,1]")
        for key in ("objects", "evidence_views"):
            if not isinstance(getattr(s, key), list) or not all(isinstance(v, str) for v in getattr(s, key)):
                raise ValueError("Invalid string list: " + key)
        if not isinstance(s.depends_on, list) or not all(isinstance(v, int) for v in s.depends_on):
            raise ValueError("Invalid prerequisites")
        if not isinstance(s.visibility, dict):
            raise ValueError("visibility must be an object")
        if s.reference_frame is not None and not isinstance(s.reference_frame, str):
            raise ValueError("reference_frame must be a string or null")
        if s.operation is not None and not isinstance(s.operation, dict):
            raise ValueError("operation must be an object or null")
        for name in ("relations_before", "relations_after"):
            if not isinstance(getattr(s, name), list) or not all(isinstance(v, dict) for v in getattr(s, name)):
                raise ValueError("Invalid relation list")
        return s

def probabilities(raw, options):
    if not isinstance(raw, dict) or set(raw) != set(options):
        raise ValueError("Distribution must cover exactly the original answer space")
    vals = {k: float(raw[k]) for k in options}
    if any(not math.isfinite(v) or v < 0 for v in vals.values()) or sum(vals.values()) <= 0:
        raise ValueError("Invalid answer distribution")
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()}

@dataclass
class Trajectory:
    states: list[State]
    answer_distribution: dict[str, float]
    target_box: list[float] | None = None
    target_view: str | None = None

    @classmethod
    def parse(cls, raw, example, max_states=8):
        if not isinstance(raw, dict):
            raise ValueError("Trajectory must be an object")
        states = [State.parse(v) for v in raw["states"]]
        ids = [s.step_idx for s in states]
        if not states or len(states) > max_states or len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("Invalid state count or order")
        if states[-1].step_type != "answer_mapping" or any(s.step_type == "answer_mapping" for s in states[:-1]):
            raise ValueError("Exactly one terminal answer_mapping is required")
        if any(p not in ids for s in states for p in s.depends_on):
            raise ValueError("Unknown prerequisite")
        box = raw.get("target_box")
        if box is not None:
            if len(box) != 4 or any(not math.isfinite(float(v)) for v in box):
                raise ValueError("Invalid target box")
            box = [float(v) for v in box]
            if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                raise ValueError("Expected normalized xyxy box")
            if raw.get("target_view") not in example.images:
                raise ValueError("Box must identify a provided view")
        return cls(states, probabilities(raw["answer_distribution"], example.options), box, raw.get("target_view"))

    def public(self):
        return asdict(self)

    def by_id(self):
        return {s.step_idx: s for s in self.states}

    @property
    def answer(self):
        return max(self.answer_distribution, key=self.answer_distribution.get)

    def copy(self):
        return copy.deepcopy(self)
