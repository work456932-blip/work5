"""Example local frozen-model adapter. Wire your installed inference stack here.

This is the ONLY intentionally unimplemented external-model boundary. The
algorithm, graph, intervention, scoring, repair and evaluator are implemented.
"""
from scenerepair.backends import BackendReply,decode_object

class LocalFrozenBackend:
    def __init__(self, model_id, revision, **runtime_options):
        self.model_id,self.revision = model_id,revision
        self.runtime_options = runtime_options
        # Load your locally installed Qwen3-VL or InternVL wrapper here, freeze
        # all parameters, use evaluation mode, and retain the exact revision.

    def complete(self, *, task, system, prompt, images, seed, temperature,
                 max_output_tokens, image_max_side):
        # Set the model RNG to `seed`, preprocess supplied local views at the
        # common resolution, generate JSON, and return measured token usage.
        # Answer probabilities should be obtained with your configured model
        # scoring method (see docs/BACKEND.md), never from benchmark labels.
        raise NotImplementedError("Connect your frozen multimodal inference runtime here")
