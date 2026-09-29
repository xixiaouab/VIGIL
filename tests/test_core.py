import copy
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import (LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration,
                          Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration)

from vigil.loss import vigil_loss
from vigil.masking import attention_masks
from vigil.model import VIGILModel, response_log_probs, sequence_log_probs

BACKBONES = ["qwen", "llava", pytest.param("internvl", marks=pytest.mark.skipif(
    not os.environ.get("VIGIL_INTERNVL_SOURCE"), reason="Set VIGIL_INTERNVL_SOURCE to the official remote-code directory"))]


def tiny_model(kind):
    torch.manual_seed(42)
    text = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                pad_token_id=0, attention_dropout=0.0)
    if kind == "qwen":
        text["rope_scaling"] = {"rope_type": "default", "mrope_section": [1, 1, 2]}
        config = Qwen2_5_VLConfig(
            text_config=text, image_token_id=120, video_token_id=123,
            vision_start_token_id=121, vision_end_token_id=122,
            vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                               patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                               window_size=8, fullatt_block_indexes=[1], out_hidden_size=32),
        )
        model = Qwen2_5_VLForConditionalGeneration(config).float()
        image_inputs = {"pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
        ids = [1, 121, 120, 120, 120, 120, 122, 7, 8, 9, 10]
    elif kind == "llava":
        text["model_type"] = "qwen2"
        config = LlavaOnevisionConfig(
            text_config=text, image_token_index=120, video_token_index=123,
            vision_config=dict(model_type="siglip_vision_model", hidden_size=16,
                               intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                               image_size=4, patch_size=2, vision_use_head=False),
            image_grid_pinpoints=[[4, 4]], vision_feature_layer=-1,
            vision_feature_select_strategy="full", vision_aspect_ratio="anyres_max_9",
        )
        model = LlavaOnevisionForConditionalGeneration(config).float()
        image_inputs = {"pixel_values": torch.randn(1, 2, 3, 4, 4), "image_sizes": torch.tensor([[4, 4]])}
        with torch.no_grad():
            size = torch.cat(model.model.get_image_features(**image_inputs)).shape[0]
        ids = [1] + [120] * size + [7, 8, 9, 10]
    else:
        source = Path(os.environ["VIGIL_INTERNVL_SOURCE"]).resolve()
        sys.path.insert(0, str(source.parent))
        import importlib
        InternConfig = importlib.import_module(source.name + ".configuration_internvl_chat").InternVLChatConfig
        InternModel = importlib.import_module(source.name + ".modeling_internvl_chat").InternVLChatModel
        text["architectures"] = ["InternLM2ForCausalLM"]
        config = InternConfig(
            llm_config=text, image_token_id=120,
            vision_config=dict(architectures=["InternVisionModel"], hidden_size=16,
                               intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                               image_size=8, patch_size=2, use_flash_attn=False, qk_normalization=False),
            force_image_size=8, downsample_ratio=0.5, ps_version="v2", template="internvl2_5",
        )
        model = InternModel(config, use_flash_attn=False).float()
        model.img_context_token_id = 120
        image_inputs = {"pixel_values": torch.randn(1, 3, 8, 8), "image_flags": torch.ones(1, 1, dtype=torch.long)}
        ids = [1, 121, 120, 120, 120, 120, 122, 7, 8, 9, 10]
    ids = torch.tensor([ids])
    labels = ids.clone()
    labels[:, :-3] = -100
    chosen = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, **image_inputs}
    rejected = {key: value.clone() for key, value in chosen.items()}
    rejected["input_ids"][0, -2:] = torch.tensor([11, 12])
    rejected["labels"][0, -2:] = torch.tensor([11, 12])
    return VIGILModel(model, logprob_chunk_size=2), {"chosen": chosen, "rejected": rejected}


def test_equations_and_matched_reference():
    cs = torch.tensor([-3.0, -5.0], requires_grad=True)
    rs = torch.tensor([-7.0, -6.0], requires_grad=True)
    cb = torch.tensor([-3.3, -8.0], requires_grad=True)
    rcs, rrs, rcb = torch.tensor([-4.0, -4.0]), torch.tensor([-6.0, -7.0]), torch.tensor([-5.0, -9.0])
    dpo = -F.logsigmoid(0.1 * ((cs - rcs) - (rs - rrs)))
    cvd = -F.logsigmoid(0.1 * ((cs - rcs) - (cb - rcb)))
    gate = 1 - torch.tanh((cs - cb).abs())
    for reduction in ("batch", "sample"):
        loss, metrics = vigil_loss(cs, rs, cb, rcs, rrs, rcb, gate_reduction=reduction)
        expected = dpo.mean() + (gate.mean() * cvd.mean() if reduction == "batch" else (gate * cvd).mean())
        torch.testing.assert_close(loss, expected)
        assert metrics["gate"].requires_grad is False
    loss.backward()
    assert cb.grad is not None and cb.grad.abs().sum() > 0
    shifted_ref, _ = vigil_loss(cs, rs, cb, rcs, rrs, rcb - 2)
    assert not torch.isclose(loss, shifted_ref)


def test_masks_preserve_causality_padding_and_visual_queries():
    padding = torch.tensor([[1, 1, 1, 1, 0]])
    visual = torch.tensor([[False, True, True, False, False]])
    see, blind = attention_masks(padding, visual, torch.float32)
    assert see.shape == (1, 1, 5, 5)
    assert see[0, 0, 3, 1] == 0 and torch.isneginf(blind[0, 0, 3, 1])
    assert blind[0, 0, 2, 1] == 0
    assert torch.isneginf(see[0, 0, 1, 2])
    assert torch.isneginf(blind[0, 0, :, 4]).all()
    sliding, _ = attention_masks(padding, visual, torch.float32, sliding_window=2)
    assert torch.isneginf(sliding[0, 0, 3, 1]) and sliding[0, 0, 3, 2] == 0


def test_chunked_response_logps_match_dense_and_gradients():
    torch.manual_seed(7)
    head = torch.nn.Linear(8, 16, bias=False)
    states = torch.randn(2, 5, 8, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, 5], [-100, 6, 7, -100, -100]])
    dense = sequence_log_probs(head(states), labels)
    chunked = response_log_probs(states, labels, head, chunk_size=2)
    torch.testing.assert_close(dense, chunked)
    dg = torch.autograd.grad(dense.sum(), (states, head.weight), retain_graph=True)
    cg = torch.autograd.grad(chunked.sum(), (states, head.weight))
    for a, b in zip(dg, cg):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("kind", BACKBONES)
def test_real_model_seeing_parity_and_blind_visual_invariance(kind):
    adapter, batch = tiny_model(kind)
    adapter.eval()
    clean = {key: value for key, value in batch["chosen"].items() if key != "labels"}
    with torch.no_grad():
        expected = sequence_log_probs(adapter.model(**clean).logits, batch["chosen"]["labels"])
        host, method = (adapter.model, "extract_feature") if kind == "internvl" else (adapter.model.model, "get_image_features")
        with patch.object(host, method, wraps=getattr(host, method)) as encoder:
            result = adapter(batch)
            assert encoder.call_count == 1
        changed = copy.deepcopy(batch)
        for key in ("chosen", "rejected"):
            changed[key]["pixel_values"] = changed[key]["pixel_values"] * -4 + 3
        changed_result = adapter(changed)
    torch.testing.assert_close(result["chosen_see"], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result["chosen_blind"], changed_result["chosen_blind"], atol=1e-6, rtol=0)
    assert not torch.allclose(result["chosen_see"], changed_result["chosen_see"], atol=1e-7, rtol=0)


@pytest.mark.parametrize("kind", BACKBONES)
def test_checkpoint_recompute_preserves_masks_and_gradient(kind):
    adapter, batch = tiny_model(kind)
    adapter.train()
    baseline = adapter(batch)
    (-baseline["chosen_blind"].sum()).backward()
    baseline_grads = {name: p.grad.clone() for name, p in adapter.named_parameters() if p.grad is not None}
    adapter.zero_grad(set_to_none=True)
    adapter.gradient_checkpointing_enable()
    checkpointed = adapter(batch)
    (-checkpointed["chosen_blind"].sum()).backward()
    for name, value in baseline.items():
        torch.testing.assert_close(value, checkpointed[name])
    for name, parameter in adapter.named_parameters():
        if name in baseline_grads:
            torch.testing.assert_close(parameter.grad, baseline_grads[name], atol=2e-6, rtol=2e-4)
    vision_terms = [p.grad.abs().sum().item() for name, p in adapter.named_parameters()
                    if any(part in name for part in ("visual.", "vision_tower.", "vision_model.")) and p.grad is not None]
    assert vision_terms and max(vision_terms) == 0


@pytest.mark.parametrize("kind", BACKBONES)
def test_training_objective_reaches_visual_encoder(kind):
    adapter, batch = tiny_model(kind)
    adapter.train()
    adapter.gradient_checkpointing_enable()
    scores = adapter(batch)
    references = {key: value.detach() for key, value in scores.items()}
    loss, _ = vigil_loss(**scores, **{"ref_" + key: value for key, value in references.items()})
    loss.backward()
    values = [p.grad.abs().sum().item() for name, p in adapter.named_parameters()
              if any(part in name for part in ("visual.", "vision_tower.", "vision_model.")) and p.grad is not None]
    assert values and max(values) > 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("kind", BACKBONES)
@pytest.mark.parametrize("side", ["left", "right"])
def test_padding_finite_and_response_invariance(kind, side):
    adapter, batch = tiny_model(kind)
    adapter.eval()
    with torch.no_grad():
        unpadded = adapter(batch)
    padded = copy.deepcopy(batch)
    for pair in padded.values():
        for key, value in list(pair.items()):
            if key in ("input_ids", "attention_mask", "labels"):
                padding = torch.full((1, 3), -100 if key == "labels" else 0, dtype=value.dtype)
                pair[key] = torch.cat((padding, value) if side == "left" else (value, padding), dim=1)
    results = adapter(padded)
    for key in unpadded:
        torch.testing.assert_close(unpadded[key], results[key], atol=1e-5, rtol=1e-5)
    (-sum(result.sum() for result in results.values())).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapter.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("kind", BACKBONES)
def test_generate_greedy_first_token_matches_full_forward(kind):
    adapter, batch = tiny_model(kind)
    adapter.eval()
    inputs = {key: value for key, value in batch["chosen"].items() if key != "labels"}
    with torch.no_grad():
        expected = adapter.model(**inputs).logits[:, -1].argmax(-1)
        generated = adapter.generate(**inputs, max_new_tokens=2, do_sample=False, eos_token_id=127, pad_token_id=0)
    torch.testing.assert_close(generated[:, inputs["input_ids"].shape[1]], expected)


@pytest.mark.skipif(not os.environ.get("VIGIL_INTERNVL_SOURCE"),
                    reason="Set VIGIL_INTERNVL_SOURCE to the official remote-code directory")
def test_internvl_remote_checkpoint_roundtrip(tmp_path):
    from tokenizers import Tokenizer, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    adapter, batch = tiny_model("internvl")
    adapter.eval()
    assert adapter.config.hidden_size == adapter.config.llm_config.hidden_size
    adapter.config.register_for_auto_class()
    adapter.model.register_for_auto_class("AutoModel")
    adapter.save_pretrained(tmp_path)
    vocab = {f"token{index}": index for index in range(128)}
    for token, index in (("<pad>", 0), ("<s>", 1), ("</s>", 2), ("<IMG_CONTEXT>", 120), ("<unk>", 127)):
        del vocab[f"token{index}"]
        vocab[token] = index
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")),
                                       unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(tmp_path)
    assert (tmp_path / "conversation.py").is_file()
    assert (tmp_path / "modeling_internlm2.py").is_file()
    loaded = VIGILModel.from_pretrained(tmp_path, dtype="float32", trust_remote_code=True, local_files_only=True)
    loaded.eval()
    assert loaded.config.hidden_size == adapter.config.hidden_size
    assert AutoTokenizer.from_pretrained(tmp_path, local_files_only=True).convert_tokens_to_ids("<IMG_CONTEXT>") == 120
    with torch.no_grad():
        expected, actual = adapter(batch), loaded(batch)
        for key in expected:
            torch.testing.assert_close(expected[key], actual[key])
