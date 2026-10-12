"""Frozen-model interfaces. HTTP uses an explicit project contract, not a vendor API."""
from dataclasses import dataclass
import base64
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time
from typing import Protocol
import urllib.request
from PIL import Image
from .prompts import render

@dataclass
class BackendReply:
    data: dict
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "unknown"
    revision: str = "unknown"

class Backend(Protocol):
    def complete(self, *, task: str, system: str, prompt: str, images: dict[str,str],
                 seed: int, temperature: float, max_output_tokens: int, image_max_side: int) -> BackendReply: ...

def decode_object(value):
    if isinstance(value, dict):
        return value
    value = value.strip()
    if value.startswith("```"):
        value = "\n".join(value.splitlines()[1:-1])
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("Model output must be a JSON object")
    return raw

class HTTPBackend:
    """Connect a local frozen Qwen/InternVL wrapper to docs/BACKEND.md contract."""
    def __init__(self, model_id, revision, endpoint_env="SCENEREPAIR_ENDPOINT", timeout=300):
        self.endpoint = os.environ.get(endpoint_env)
        if not self.endpoint:
            raise ValueError("Set " + endpoint_env + " or supply a Python backend factory")
        if not revision or revision.startswith("SET_"):
            raise ValueError("Set a concrete frozen backbone revision in the config")
        self.model_id, self.revision, self.timeout = model_id, revision, timeout

    def complete(self, **kwargs):
        encoded = []
        for view, path in kwargs.pop("images").items():
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((kwargs["image_max_side"],)*2)
                stream = io.BytesIO()
                im.save(stream, format="PNG")
                encoded.append({"view_id":view,"mime_type":"image/png", "base64":base64.b64encode(stream.getvalue()).decode()})
        body = {**kwargs,"images":encoded,"model_id":self.model_id,"revision":self.revision}
        headers = {"Content-Type":"application/json"}
        token = os.environ.get("SCENEREPAIR_API_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode(),headers=headers,method="POST")
        with urllib.request.urlopen(request,timeout=self.timeout) as response:
            raw = json.load(response)
        if raw.get("model_id") != self.model_id or raw.get("revision") != self.revision:
            raise ValueError("Server model/revision does not match the frozen configuration")
        usage = raw.get("usage",{})
        return BackendReply(decode_object(raw["data"]), int(usage.get("input_tokens",0)),
                            int(usage.get("output_tokens",0)), raw["model_id"], raw["revision"])

class ReplayBackend:
    """Replay previously captured real backend replies; never invent predictions."""
    def __init__(self, path):
        self.records = {}
        with open(path,encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.records[r["request_hash"]] = r["reply"]

    def complete(self, **kwargs):
        key = request_hash(kwargs)
        if key not in self.records:
            raise KeyError("No recorded model response for request " + key)
        return BackendReply(**self.records[key])

def request_hash(kwargs):
    return hashlib.sha256(json.dumps(kwargs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def make_backend(factory, kwargs):
    module,name = factory.split(":",1)
    return getattr(importlib.import_module(module),name)(**kwargs)

class ModelSession:
    def __init__(self, backend, config, example_id):
        self.backend, self.config, self.example_id = backend, config, example_id
        self.records, self.parse_events = [], []

    def ask(self, task, example, payload, independent_key=""):
        system,prompt = render(task,{"example":example.public(),**payload},self.config.max_states)
        purpose = f"{self.config.seed}:{self.example_id}:{task}:{independent_key}:{len(self.records)}"
        seed = int(hashlib.sha256(purpose.encode()).hexdigest()[:8],16)
        kwargs = dict(task=task,system=system,prompt=prompt,images=example.images,seed=seed,
                      temperature=self.config.temperature,max_output_tokens=self.config.max_output_tokens,
                      image_max_side=self.config.image_max_side)
        started = time.perf_counter()
        record = {"task":task,"seed":seed,"request_hash":request_hash(kwargs),
                  "prompt_hash":hashlib.sha256((system+prompt).encode()).hexdigest()}
        try:
            reply = self.backend.complete(**kwargs)
            data = decode_object(reply.data)
            record.update(input_tokens=reply.input_tokens,output_tokens=reply.output_tokens,
                          model_id=reply.model_id,revision=reply.revision,status="ok")
            return data
        except Exception as e:
            record.update(status="failed",error_type=type(e).__name__)
            raise
        finally:
            record["latency_seconds"] = time.perf_counter()-started
            self.records.append(record)

    def failure(self, stage, error):
        self.parse_events.append({"stage":stage,"status":"fallback","error_type":type(error).__name__})
