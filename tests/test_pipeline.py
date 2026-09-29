import json
import sys

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
    Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
)

from vigil import cache_reference, generate, train


def make_tiny_checkpoint(path):
    vocabulary = {word: index for index, word in enumerate([
        "<pad>", "<unk>", "<eos>", "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        "user", "assistant", "Yes", "No", ".", "Is", "there", "a", "dog", "?", "cat",
    ])}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
        additional_special_tokens=list(vocabulary)[3:9],
    )
    template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tokenizer.chat_template = template
    processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                              min_pixels=64, max_pixels=64),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                              size={"shortest_edge": 64, "longest_edge": 64}),
        chat_template=template,
    )
    config = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, pad_token_id=0, eos_token_id=2,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        image_token_id=7, video_token_id=8, vision_start_token_id=5, vision_end_token_id=6,
        vision_config=dict(depth=2, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=2,
                           temporal_patch_size=2, spatial_merge_size=2, window_size=8,
                           fullatt_block_indexes=[1], out_hidden_size=32),
    )
    torch.manual_seed(19)
    model = Qwen2_5_VLForConditionalGeneration(config).float()
    model.save_pretrained(path)
    processor.save_pretrained(path)
    return model.model.language_model.embed_tokens.weight.detach().clone()


def invoke(monkeypatch, module, arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, arguments)])
    module.main()


def test_real_processor_cache_train_and_generate_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    checkpoint = tmp_path / "base"
    original_weights = make_tiny_checkpoint(checkpoint)
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (50, 120, 200)).save(image)
    records = [
        {"id": "one", "image": "image.png", "prompt": "Is there a dog?", "chosen": "Yes.",
         "rejected": "No.", "category": "hallucination"},
        {"id": "two", "image": "image.png", "prompt": "Is there a cat?", "chosen": "No.",
         "rejected": "Yes.", "category": "vqa"},
    ]
    data = tmp_path / "preferences.jsonl"
    data.write_text("".join(json.dumps(record) + "\n" for record in records))
    cache = tmp_path / "reference.jsonl"
    invoke(monkeypatch, cache_reference, ["--model", checkpoint, "--data", data, "--output", cache,
                                          "--dtype", "float32", "--max-length", "128"])
    cached = [json.loads(line) for line in cache.read_text().splitlines()]
    assert [row["id"] for row in cached] == ["one", "two"]
    assert all(set(cache_reference.REFERENCE_FIELDS) <= row.keys() for row in cached)
    config = tmp_path / "train.json"
    output = tmp_path / "training"
    config.write_text(json.dumps({
        "model": str(checkpoint), "data": str(data), "reference_cache": str(cache),
        "output_dir": str(output), "dtype": "float32", "max_length": 128,
        "micro_batch_size": 1, "global_batch_size": 2,
        "training": {"use_cpu": True, "learning_rate": 1e-3, "warmup_ratio": 0,
                     "disable_tqdm": True},
    }))
    invoke(monkeypatch, train, ["--config", config])
    final = output / "final"
    assert (final / "config.json").is_file() and (final / "tokenizer.json").is_file()
    updated = Qwen2_5_VLForConditionalGeneration.from_pretrained(final, dtype=torch.float32)
    assert not torch.equal(original_weights, updated.model.language_model.embed_tokens.weight)
    predictions = tmp_path / "predictions.jsonl"
    invoke(monkeypatch, generate, ["--model", final, "--input", data, "--output", predictions,
                                   "--device", "cpu", "--dtype", "float32", "--max-length", "128",
                                   "--max-new-tokens", "2"])
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["one", "two"]
    assert all(isinstance(row["prediction"], str) for row in rows)
