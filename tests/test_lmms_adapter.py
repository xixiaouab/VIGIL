import copy
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image


@pytest.fixture
def adapter(monkeypatch):
    registrations, manifests, cli_calls = {}, [], []

    class Base:
        def __init__(self):
            self.task_dict = {}
            self.cached = []
            self.cache_hook = SimpleNamespace(add_partial=lambda *args: self.cached.append(args))

        @property
        def rank(self):
            return self._rank

        @property
        def world_size(self):
            return self._world_size

    def register(name):
        def decorate(cls):
            registrations[name] = cls
            return cls
        return decorate

    modules = {name: ModuleType(name) for name in ["lmms_eval", "lmms_eval.api", "lmms_eval.api.model",
               "lmms_eval.api.registry", "lmms_eval.models", "lmms_eval.models.registry_v2", "lmms_eval.__main__"]}
    modules["lmms_eval.api.model"].lmms = Base
    modules["lmms_eval.api.registry"].register_model = register
    modules["lmms_eval.models"].MODEL_REGISTRY_V2 = SimpleNamespace(
        register_manifest=lambda manifest, overwrite: manifests.append((manifest, overwrite)))
    modules["lmms_eval.models"].AVAILABLE_SIMPLE_MODELS = {}
    modules["lmms_eval.models"].AVAILABLE_MODELS = {}
    modules["lmms_eval.models.registry_v2"].ModelManifest = SimpleNamespace
    modules["lmms_eval.__main__"].cli_evaluate = lambda: cli_calls.append(True)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "vigil.lmms_adapter", raising=False)
    module = importlib.import_module("vigil.lmms_adapter")
    load_calls = []
    fake_model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)), config={})
    fake_processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=2))

    def load(*args, **kwargs):
        load_calls.append((args, kwargs))
        return fake_model, fake_processor

    monkeypatch.setattr(module, "load_generator", load)
    monkeypatch.setattr(module, "Accelerator", lambda: SimpleNamespace(process_index=0, num_processes=1, device="cpu"))
    return SimpleNamespace(module=module, registrations=registrations, manifests=manifests,
                           cli_calls=cli_calls, load_calls=load_calls, modules=modules)


def test_lmms_request_contract_pil_and_text_only(adapter, monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return "yes<END>extra"

    monkeypatch.setattr(adapter.module, "generate_response", generate)
    model = adapter.module.VIGIL("model", device="cpu", trust_remote_code="false")
    image = Image.new("RGB", (8, 8))
    model.task_dict = {"pope": {"test": [{"image": image}]}, "gsm8k": {"test": [{}]}}
    options = {"max_new_tokens": 17, "do_sample": False, "temperature": 0.8, "top_p": None,
               "num_beams": 1, "until": ["<END>"]}
    original = copy.deepcopy(options)
    requests = [SimpleNamespace(args=("<image> Is this visible?", options, lambda row: [row["image"]],
                                      0, "pope", "test")),
                SimpleNamespace(args=("What is 2 + 2?", {"until": "<END>"}, None, 0, "gsm8k", "test"))]
    assert model.generate_until(requests) == ["yes", "yes"]
    assert options == original
    assert calls[0][0][2]["prompt"] == "Is this visible?"
    assert calls[0][1]["image"] is image
    assert calls[0][1]["max_new_tokens"] == 17
    assert calls[0][1]["temperature"] == 0 and calls[0][1]["top_p"] == 1
    assert calls[1][1]["image"] is None
    assert model.cached[0] == ("generate_until", (requests[0].args[0], options), "yes")
    assert adapter.load_calls[0][1]["trust_remote_code"] is False


def test_lmms_distributed_rank_sampling_and_invalid_visuals(adapter, monkeypatch):
    monkeypatch.setattr(adapter.module, "Accelerator", lambda: SimpleNamespace(process_index=3, num_processes=4,
                                                                              device="cuda:1"))
    model = adapter.module.VIGIL("model", trust_remote_code="true")
    assert model.rank == 3 and model.world_size == 4
    assert adapter.load_calls[0][1]["device"] == "cuda:1"
    assert adapter.load_calls[0][1]["trust_remote_code"] is True
    model.task_dict = {"task": {"test": [{}]}}
    calls = []
    monkeypatch.setattr(adapter.module, "generate_response", lambda *args, **kwargs: calls.append(kwargs) or "answer")
    request = SimpleNamespace(args=("prompt", {"do_sample": True, "temperature": 0.6, "top_p": 0.8,
                                               "top_k": 5}, None, 0, "task", "test"))
    assert model.generate_until([request]) == ["answer"]
    assert calls[0]["temperature"] == 0.6 and calls[0]["top_p"] == 0.8
    assert calls[0]["generation_kwargs"]["top_k"] == 5
    image = Image.new("RGB", (4, 4))
    request = SimpleNamespace(args=("prompt", {}, lambda _: [image, image], 0, "task", "test"))
    with pytest.raises(ValueError, match="zero or one"):
        model.generate_until([request])


def test_lmms_registration_and_cli(adapter):
    assert adapter.registrations["vigil"] is adapter.module.VIGIL
    adapter.module.main()
    assert adapter.manifests[0][0].simple_class_path == "vigil.lmms_adapter.VIGIL"
    assert adapter.manifests[0][0].model_id == "vigil"
    assert adapter.manifests[0][1] is True
    assert adapter.cli_calls == [True]
