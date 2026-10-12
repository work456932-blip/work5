# Frozen multimodal backend contract

The code does not download model weights. The default adapter connects to an
already available local inference server through a project-specific HTTP JSON
contract. This is not advertised as an OpenAI/vLLM/vendor-compatible endpoint.

`Backend.complete` accepts task, system, prompt, view-ID/local-path mapping, seed,
temperature, output-token cap and image maximum side. It returns `BackendReply`
with a parsed JSON object, measured input/output token counts, model ID and the
exact frozen revision. Independent calls must actually honor their seeds.

## HTTP wrapper

Set `SCENEREPAIR_ENDPOINT` to the POST URL. An optional authentication token is
read only from `SCENEREPAIR_API_TOKEN`; it is never written into configs or logs.
Requests contain:

```json
{
  "task": "generate",
  "system": "label-free system instruction",
  "prompt": "stage instruction and input JSON",
  "images": [{"view_id":"view_0","mime_type":"image/png","base64":"..."}],
  "seed": 42,
  "temperature": 0.7,
  "max_output_tokens": 4096,
  "image_max_side": 1024,
  "model_id": "Qwen3-VL-8B",
  "revision": "your-frozen-local-revision"
}
```

Response envelope:

```json
{
  "data": {"states": [], "answer_distribution": {"A": 0.7, "B": 0.3}, "target_box": null, "target_view": null},
  "usage": {"input_tokens": 100, "output_tokens": 200},
  "model_id": "Qwen3-VL-8B",
  "revision": "your-frozen-local-revision"
}
```

The envelope above illustrates shape only; a `generate` response needs complete
typed states, including a terminal answer_mapping. Response data schemas for
each task are defined in `scenerepair/prompts.py`. The server may return data as
an object or as a JSON string. The client rejects mismatched model revisions.

## Distribution scoring

The manuscript requires answer distributions without specifying their extraction.
The reference prompt asks for a normalized option distribution. For a stronger
implementation, your wrapper can replace that field with normalized probabilities
obtained by scoring the original option IDs conditioned on the generated trace.
Use the SAME scoring definition, answer tokens, preprocessing and option ordering
for original, intervention and repair distributions. Do not substitute a single
selected answer with certainty 1, and never condition on evaluation labels.
Record that choice with the frozen deployment configuration.

## Local Python integration

Set `backend_factory` to `module_name:ClassName` and provide constructor kwargs.
Use `examples/backend_adapter.py` as the integration boundary. Qwen3-VL-32B and
InternVL3-8B require their own installed frozen inference wrappers, while the
SceneRepair graph/probe/gate implementation stays shared. Generator and verifier
calls can route to distinct frozen models based on task; expose both identities
and their revisions in audit metadata if using a cross-model verifier.

`ReplayBackend` reads saved REAL responses indexed by full request hash. Missing
requests raise an error. It is useful for inspection and never substitutes
synthetic answers for missing benchmark responses.
