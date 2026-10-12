"""Label-permutation audit compares decisions rather than timing/cost noise."""
def compare_predictions(first, second):
    a = {str(v["id"]):v["prediction"] for v in first}
    b = {str(v["id"]):v["prediction"] for v in second}
    if set(a)!=set(b):
        raise ValueError("Audit requires matching IDs")
    changed = [i for i in sorted(a) if a[i]!=b[i]]
    return {"examples":len(a),"changed":len(changed),"changed_ids":changed}
