"""Anonymous, label-free prompt templates. Source text is not bundled in this repo."""
import json

SYSTEM = """You are an image-grounded spatial reasoning component. Treat all supplied
trajectories as hypotheses, not facts. Use only the supplied images, question,
answer options and explicit predecessor states. No benchmark labels are available.
Return one JSON object without markdown. Never introduce an unobserved entity/view.
Use normalized xyxy target boxes and name their view when grounding is requested."""

STATE_SCHEMA = """Each state has: step_idx (integer), step_type (one of observation,
correspondence, reference_frame, initial_state, counterfactual_operation,
updated_relation, visibility, answer_mapping), description (string), objects
(list of grounded entity IDs), reference_frame (string or null), operation
(object or null: class, direction, scope), relations_before and relations_after
(lists of {subject, predicate, object, frame}), visibility (entity->boolean),
evidence_views (list of supplied view IDs), depends_on (list of prerequisite
step indices), confidence (number in [0,1])."""

TEMPLATES = {
    "generate": """Generate a grounded typed trajectory with at most {max_states} states.
Record object correspondence across views, observer frame, initial spatial state,
hypothetical operation and scope, updated relations, visibility and final mapping
as needed. Dependencies must reflect prerequisites and be acyclic. Finish with
exactly one answer_mapping state. Return {states, answer_distribution, target_box,
target_view}. answer_distribution maps every supplied option ID to a probability.
target_box and target_view may be null. STATE_SCHEMA""",
    "direct": """Answer the spatial question directly. Return {answer_distribution,
target_box, target_view}. Include every option ID. Do not return a typed trace.""",
    "cot": """Reason step by step using visual evidence, then return {rationale,
answer_distribution, target_box, target_view}. Include every option ID.""",
    "verify": """Assess every state against visual evidence, question and declared
predecessors. Do not use the final answer as evidence for earlier states. Return
{scores: [{step_idx, consistency, supported_objects, supported_views, reason}]}.
consistency is in [0,1]. supported_objects/views explicitly list visually supported
entity/view IDs. Evaluate the state even when its description sounds plausible.""",
    "confirm_edge": """Does the target semantic commitment actually depend on the
source commitment in this question? Return {confirmed: boolean, reason: string}.
Confirm only a high-precision prerequisite grounded in the supplied evidence.""",
    "propose": """Produce one paired intervention for the specified node and family.
destructive deliberately changes only that commitment; corrective is a visually
supported alternative satisfying every verified ancestor. Preserve transformation
class for operations, respect inverse relations, preserve untargeted entities and
facts, keep the answer space. Frame replacements must be licensed by the question.
Never add an object/view. Return {destructive: STATE, corrective: STATE}.
The target index, type and depends_on must stay identical. Use the requested
meaning-equivalent realization to express the same intervention denotation.
STATE_SCHEMA""",
    "admissibility": """Check the paired intervention BEFORE scoring effects.
Destructive may change the target commitment for a necessity probe; corrective
must be licensed by the question, visually grounded and ancestor-compatible.
Check entity/view support, unchanged external facts and answer-space preservation,
relation inverse structure, operation class/scope, and frame geometry. Return
{destructive_valid: boolean, corrective_valid: boolean,
equivalent_to_anchor: boolean, reason: string}. For realization zero set
equivalent_to_anchor true when otherwise admissible. Later realizations must
preserve the anchor pair's denotation, not merely its intervention family.""",
    "repair": """Regenerate ONLY the states in editable_indices. All other states
are immutable and must not be returned. Verified ancestors in fixed_states are
copied verbatim by the caller. Return {states: [edited STATE records],
answer_distribution, target_box, target_view}. Keep all state indices, types and
depends_on lists unchanged. A forced replacement, if supplied, is copied exactly
at its target; recompute its declared descendants. Preserve unrelated facts,
entities, views and original answer space. STATE_SCHEMA""",
    "equivalence": """Judge whether the transformed question has the same denotation,
referents, operation attachment, thematic roles and observer frame as the original.
Return {equivalent: boolean, reason: string}. Do not solve using an answer label.""",
    "paraphrase": """Generate a meaning-equivalent version of the question in the
specified family: lexical, deictic, voice, scope or coreference. Preserve the exact
task, visual referents, thematic roles, operation attachment and observer. Return
{question: string}. Do not change images or answer options.""",
}

def render(task, payload, max_states=8):
    instruction = TEMPLATES[task].replace("STATE_SCHEMA", STATE_SCHEMA).replace("{max_states}",str(max_states))
    return SYSTEM, instruction + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)
