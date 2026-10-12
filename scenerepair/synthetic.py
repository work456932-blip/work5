"""Evaluation-only corruption helper; requires independently validated clean traces."""
from dataclasses import asdict
from .schema import State
from .repair import regenerate

def inject_verified_corruption(example, clean, root, corrupt_state, declared, session, validated_clean=False):
    """Inject a supplied first error and re-propagate its declared descendants.

    Validation of the clean trace AND unsupported nature of corrupt_state is the
    caller's annotation responsibility. A mutation by itself does not establish
    a real ground-truth causal root. No synthetic benchmark results are bundled.
    """
    if not validated_clean:
        raise ValueError("Synthetic roots require an independently validated clean trajectory")
    old = clean.by_id()[root]
    if old.step_type=="answer_mapping":
        raise ValueError("The terminal answer cannot be an explanatory root")
    replacement = State.parse(corrupt_state)
    if (replacement.step_idx,replacement.step_type,replacement.depends_on)!=(old.step_idx,old.step_type,old.depends_on):
        raise ValueError("Corruption must preserve typed structure")
    closure = {root} | declared.descendants(root)
    corrupted = regenerate(session,example,clean,closure,replacement,"synthetic_corruption")
    annotation = {"id":example.id,"root_idx":root,"root_type":old.step_type,
                  "descendant_indices":sorted(declared.descendants(root)),"ambiguous":False}
    # Store annotation separately from inference input.
    return corrupted,annotation
