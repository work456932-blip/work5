"""Meaning-preserving language and grounded-input controls (Sections 5.4, Appendix D)."""
from dataclasses import replace
from pathlib import Path
import random
from PIL import Image

LANGUAGE_FAMILIES = ("lexical","deictic","voice","scope","coreference")

def paraphrases(example, session):
    """Return label-free variants only after independent equivalence checking.

    The held-out human validation described in Appendix D remains a separate
    annotation step; this automatic check is not substituted for human judgment.
    """
    accepted,records = [],[]
    for family in LANGUAGE_FAMILIES:
        raw = session.ask("paraphrase",example,{"family":family},"language:"+family)
        question = raw["question"]
        if not isinstance(question,str) or not question.strip():
            raise ValueError("Invalid paraphrase")
        validity = session.ask("equivalence",example,{"original_question":example.question,
            "transformed_question":question},"language_validation:"+family)
        records.append({"family":family,"question":question,"validity":validity})
        if validity.get("equivalent") is True:
            accepted.append((family,replace(example,question=question)))
    return accepted,records

def visual_control(example, mode, output_dir, other_example=None, seed=0):
    """No downloading. Gray/low-resolution images are derived from local files."""
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    items = list(example.images.items())
    rng = random.Random(seed)
    if mode=="text_only":
        return replace(example,images={})
    if mode=="missing_view":
        if not items:
            return example
        removed = rng.choice(items)[0]
        return replace(example,images={k:v for k,v in items if k!=removed})
    if mode=="image_order":
        rng.shuffle(items)
        return replace(example,images=dict(items))
    if mode=="option_order":
        options = list(example.options.items())
        rng.shuffle(options)
        return replace(example,options=dict(options))
    if mode=="mismatched":
        if other_example is None or other_example.id==example.id or not other_example.images:
            raise ValueError("Mismatched control needs another example with local views")
        pool = list(other_example.images.values())
        return replace(example,images={k:pool[i%len(pool)] for i,(k,_) in enumerate(items)})
    if mode not in {"gray","low_resolution"}:
        raise ValueError("Unknown visual control")
    images = {}
    for i,(view,path) in enumerate(items):
        with Image.open(path) as im:
            im = im.convert("RGB")
            if mode=="gray":
                im = Image.new("RGB",im.size,(127,127,127))
            else:
                im.thumbnail((256,256))
            dest = output/f"view_{i}_{mode}.png"
            im.save(dest)
            images[view] = str(dest.resolve())
    return replace(example,images=images)

def language_metrics(clean, variants):
    """Compare matching IDs; task accuracy remains evaluation-only elsewhere."""
    by_id = {r["id"]:r["prediction"] for r in clean}
    result = {}
    for family,rows in variants.items():
        if {r["id"] for r in rows} != set(by_id):
            raise ValueError("Language consistency requires matching example IDs")
        answer = [r["prediction"]["answer"]==by_id[r["id"]]["answer"] for r in rows]
        root = [r["prediction"].get("root_type")==by_id[r["id"]].get("root_type") for r in rows
                if by_id[r["id"]].get("root_type") is not None and r["prediction"].get("root_type") is not None]
        result[family] = {"answer_consistency":sum(answer)/len(answer),
                          "root_type_consistency":sum(root)/len(root) if root else None,
                          "root_type_pair_count":len(root)}
    return result
