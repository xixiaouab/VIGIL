import json

import pytest
import torch

from vigil.cache_reference import (
    REFERENCE_FIELDS, ReferenceCollator, ReferenceDataset, build_cache_metadata,
    dataset_fingerprint, file_fingerprint, load_reference_cache, metadata_path,
    model_fingerprint, write_reference_cache,
)


class Tokenizer:
    special_tokens_map = {"eos_token": "</s>"}
    padding_side = "right"
    chat_template = "{{ messages }}"

    def get_vocab(self):
        return {"</s>": 0, "yes": 1, "no": 2}


class Processor:
    tokenizer = Tokenizer()
    chat_template = "{{ messages }}"


@pytest.fixture
def records(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"test image contents")
    return [{"id": "a", "image": str(image), "prompt": "Is there a dog?",
             "chosen": "Yes.", "rejected": "No."},
            {"id": "b", "image": str(image), "prompt": "Describe the dog.",
             "chosen": "The dog is black.", "rejected": "The dog is white."}]


def metadata(records, **overrides):
    kwargs = {"model_name": "example/model", "revision": "immutable-commit",
              "processor": Processor(), "max_length": 4096,
              "model_config": {"model_type": "qwen2_5_vl"}}
    kwargs.update(overrides)
    return build_cache_metadata(records, **kwargs)


def scores(records):
    return {row["id"]: dict(zip(REFERENCE_FIELDS, [-2.0, -3.0, -4.0])) for row in records}


def test_cache_roundtrip_and_batch_attachment(tmp_path, records):
    cache = tmp_path / "reference.jsonl"
    expected = metadata(records)
    original = scores(records)
    write_reference_cache(cache, original, expected)
    loaded = load_reference_cache(cache, expected)
    assert loaded == original
    dataset = ReferenceDataset(records, loaded)
    batch = ReferenceCollator(lambda rows: {"chosen": len(rows)})([dataset[1], dataset[0]])
    assert batch["chosen"] == 2
    assert set(batch["reference"]) == set(REFERENCE_FIELDS)
    assert batch["reference"]["ref_chosen_see"].dtype == torch.float32
    assert batch["reference"]["ref_chosen_blind"].tolist() == [-4.0, -4.0]
    assert "reference" not in records[0]


def test_changed_image_or_negative_invalidates_cache(tmp_path, records):
    cache = tmp_path / "reference.jsonl"
    write_reference_cache(cache, scores(records), metadata(records))
    changed = [dict(row) for row in records]
    changed[0]["hard_negative"] = "Yes, and it has two tails."
    with pytest.raises(ValueError, match="dataset"):
        load_reference_cache(cache, metadata(changed))
    with open(records[0]["image"], "wb") as stream:
        stream.write(b"different image contents")
    with pytest.raises(ValueError, match="dataset"):
        load_reference_cache(cache, metadata(records))


@pytest.mark.parametrize("option,value", [("revision", "new-commit"), ("max_length", 2048),
                                          ("dtype", "float32"), ("attn_implementation", "eager")])
def test_changed_model_or_processing_invalidates_cache(tmp_path, records, option, value):
    cache = tmp_path / "reference.jsonl"
    write_reference_cache(cache, scores(records), metadata(records))
    with pytest.raises(ValueError, match="does not match"):
        load_reference_cache(cache, metadata(records, **{option: value}))


def test_corrupt_or_modified_scores_rejected(tmp_path, records):
    cache = tmp_path / "reference.jsonl"
    expected = metadata(records)
    write_reference_cache(cache, scores(records), expected)
    cache.write_text(cache.read_text().replace("-2.0", "-1.0"))
    with pytest.raises(ValueError, match="scores have changed"):
        load_reference_cache(cache, expected)


def test_missing_duplicate_and_nonfinite_rows_rejected(tmp_path, records):
    cache = tmp_path / "reference.jsonl"
    expected = metadata(records)
    with pytest.raises(ValueError, match="incomplete"):
        write_reference_cache(cache, {"a": scores(records)["a"]}, expected)
    write_reference_cache(cache, scores(records), expected)
    original = cache.read_text()
    for altered, message in [(original + original.splitlines()[0] + "\n", "Duplicate"),
                             (original.replace("-2.0", "NaN"), "Invalid"),
                             (original.splitlines()[0] + "\n", "IDs do not match")]:
        cache.write_text(altered)
        sidecar = json.loads(metadata_path(cache).read_text())
        sidecar["scores_sha256"] = file_fingerprint(cache)
        metadata_path(cache).write_text(json.dumps(sidecar))
        with pytest.raises(ValueError, match=message):
            load_reference_cache(cache, expected)


def test_duplicate_dataset_ids_and_missing_scores_fail(records):
    with pytest.raises(ValueError, match="unique"):
        dataset_fingerprint([records[0], records[0]])
    with pytest.raises(ValueError, match="Missing reference"):
        ReferenceDataset(records, {"a": scores(records)["a"]})


def test_local_checkpoint_content_changes_signature(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights one")
    before = model_fingerprint(str(tmp_path), model_config={"hidden_size": 32})
    weight.write_bytes(b"weights two")
    after = model_fingerprint(str(tmp_path), model_config={"hidden_size": 32})
    assert before != after


def test_resolved_model_commit_is_authoritative():
    before = model_fingerprint("example/model", "main", {"_commit_hash": "abc", "hidden_size": 32})
    after = model_fingerprint("example/model", None, {"_commit_hash": "abc", "hidden_size": 32})
    assert before == after
    with pytest.raises(ValueError, match="resolved model commit"):
        model_fingerprint("example/model", model_config={"hidden_size": 32})


def test_saved_qwen_checkpoint_metadata_matches_fresh_config(tmp_path, records):
    from transformers import AutoConfig
    from test_core import tiny_model
    from vigil.model import VIGILModel

    adapter, batch = tiny_model("qwen")
    checkpoint = tmp_path / "checkpoint"
    adapter.save_pretrained(checkpoint)
    original_config = AutoConfig.from_pretrained(checkpoint)
    reference = VIGILModel.from_pretrained(checkpoint, dtype="float32")
    reference.requires_grad_(False)
    reference.eval()
    with torch.inference_mode():
        values = reference(batch)
    subset = records[:1]
    cached_scores = {subset[0]["id"]: {"ref_" + key: value[0].item() for key, value in values.items()}}
    before = metadata(subset, model_name=str(checkpoint), revision=None,
                      model_config=original_config, dtype="float32")
    after = metadata(subset, model_name=str(checkpoint), revision=None,
                     model_config=reference.config, dtype="float32")
    assert before["signature"] == after["signature"]
    cache = tmp_path / "reference.jsonl"
    write_reference_cache(cache, cached_scores, before)
    assert load_reference_cache(cache, after) == cached_scores
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters())
