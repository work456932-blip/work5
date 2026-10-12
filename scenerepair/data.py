"""Dataset boundary: expose an explicit allowlist; labels use a different file/API."""
import json
from pathlib import Path
from .schema import Example

def load_examples(path, image_root=None):
    root = Path(image_root) if image_root else Path(path).parent
    seen = set()
    with open(path,encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            identifier = str(raw["id"])
            if identifier in seen:
                raise ValueError("Duplicate example ID")
            seen.add(identifier)
            images = raw["images"]
            if not isinstance(images,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in images.items()):
                raise ValueError("images must map view IDs to local paths")
            options = raw["options"]
            if not isinstance(options,dict) or len(options)<2 or not all(isinstance(k,str) and isinstance(v,str) for k,v in options.items()):
                raise ValueError("options must map IDs to answer text")
            # All other input fields (including answer, label, annotations, metadata) are ignored.
            yield Example(identifier,str(raw["question"]),{k:str((root/v).resolve()) for k,v in images.items()},options,str(raw.get("group","")))

def read_jsonl(path):
    with open(path,encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_labels(path):
    return {str(v["id"]):v for v in read_jsonl(path)}
