from __future__ import annotations

import re
import sys

from accelerate import Accelerator
from PIL import Image

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from .generate import generate_response, load_generator


def boolean(value):
    if isinstance(value, str):
        if value.lower() not in {"true", "false"}:
            raise ValueError(f"Expected true or false, got {value!r}")
        return value.lower() == "true"
    return bool(value)


@register_model("vigil")
class VIGIL(lmms):
    is_simple = True

    def __init__(self, pretrained, device="auto", batch_size=1, dtype="bfloat16", revision=None,
                 trust_remote_code=False, max_length=16384, min_pixels=None, max_pixels=None,
                 max_num=None, use_thumbnail=None, seed=42, use_cache=True, **kwargs):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unknown model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("The VIGIL benchmark adapter uses batch_size=1")
        self.accelerator = Accelerator()
        self._rank = self.accelerator.process_index
        self._world_size = self.accelerator.num_processes
        if self._world_size > 1:
            device = str(self.accelerator.device)
        self._max_length = int(max_length)
        self.use_cache = boolean(use_cache)
        self._model, self.processor = load_generator(
            pretrained, device=device, dtype=dtype, seed=int(seed), revision=revision,
            trust_remote_code=boolean(trust_remote_code),
            min_pixels=int(min_pixels) if min_pixels is not None else None,
            max_pixels=int(max_pixels) if max_pixels is not None else None,
            max_num=int(max_num) if max_num is not None else None,
            use_thumbnail=boolean(use_thumbnail) if use_thumbnail is not None else None,
        )
        self._device = self._model.get_input_embeddings().weight.device

    @property
    def model(self):
        return self._model

    @property
    def config(self):
        return self._model.config

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return 1

    @property
    def eot_token_id(self):
        return getattr(self.processor, "end_token_id", self.tokenizer.eos_token_id)

    def loglikelihood(self, requests):
        raise NotImplementedError("Use the generative task variants with the VIGIL adapter")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Use single-round image/text tasks with the VIGIL adapter")

    def generate_until(self, requests):
        responses = []
        for request in requests:
            context, generation, doc_to_visual, doc_id, task, split = request.args
            options = dict(generation)
            until = options.pop("until", [])
            if isinstance(until, str):
                until = [until]
            if not isinstance(until, (list, tuple)) or any(not isinstance(stop, str) for stop in until):
                raise ValueError("until must be a string or a list of strings")
            maximum = int(options.pop("max_new_tokens", options.pop("max_gen_toks", 512)))
            temperature = options.pop("temperature", 0.0)
            temperature = float(temperature) if temperature is not None else 0.0
            sample = boolean(options.pop("do_sample", temperature > 0))
            if sample and temperature <= 0:
                temperature = 1.0
            if not sample:
                temperature = 0.0
            top_p = options.pop("top_p", 1.0)
            top_p = float(top_p) if top_p is not None else 1.0
            if not sample:
                top_p = 1.0
            if int(options.pop("num_beams", 1)) != 1:
                raise ValueError("VIGIL benchmarks support greedy or sampled decoding with num_beams=1")
            if float(options.pop("repetition_penalty", 1.0)) != 1.0:
                raise ValueError("VIGIL benchmark decoding uses repetition_penalty=1")
            options.setdefault("use_cache", self.use_cache)
            if options.get("top_k") is None:
                options.pop("top_k", None)
            extra = options.keys() - {"top_k", "use_cache", "eos_token_id", "pad_token_id"}
            if extra:
                raise ValueError(f"Unsupported generation options: {sorted(extra)}")
            document = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(document) if doc_to_visual is not None else []
            if visuals is None:
                visuals = []
            if isinstance(visuals, Image.Image):
                visuals = [visuals]
            if len(visuals) > 1 or any(not isinstance(image, Image.Image) for image in visuals):
                raise ValueError("VIGIL benchmark tasks must provide zero or one PIL image")
            prompt = re.sub(r"<image(?:\s+\d+)?>", "", context, flags=re.IGNORECASE).strip() if visuals else context
            response = generate_response(
                self.model, self.processor, {"id": str(doc_id), "prompt": prompt},
                max_new_tokens=maximum, max_length=self.max_length, temperature=temperature, top_p=top_p,
                image=visuals[0] if visuals else None, generation_kwargs=options,
            )
            for stop in until:
                if stop:
                    response = response.split(stop, 1)[0]
            responses.append(response)
            self.cache_hook.add_partial("generate_until", (context, generation), response)
        return responses


def register_lmms_model():
    import lmms_eval.models as models

    if hasattr(models, "MODEL_REGISTRY_V2"):
        from lmms_eval.models.registry_v2 import ModelManifest

        models.MODEL_REGISTRY_V2.register_manifest(
            ModelManifest(model_id="vigil", simple_class_path="vigil.lmms_adapter.VIGIL"), overwrite=True,
        )
    if hasattr(models, "AVAILABLE_SIMPLE_MODELS"):
        models.AVAILABLE_SIMPLE_MODELS["vigil"] = "vigil.lmms_adapter.VIGIL"
    if hasattr(models, "AVAILABLE_MODELS"):
        models.AVAILABLE_MODELS["vigil"] = "vigil.lmms_adapter.VIGIL"


def main():
    register_lmms_model()
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    sys.modules["vigil.lmms_adapter"] = sys.modules[__name__]
    main()
