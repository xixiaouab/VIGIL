import copy
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vigil.data import PreferenceCollator, PreferenceDataset, negative_text, read_jsonl, validate_record, write_jsonl
from vigil.evaluate import align_records, bbox_iou, judge_vcc, parse_bbox, parse_vcc_label, pope_metrics, refcoco_metrics
from vigil.generate import generate_response
from vigil.mine import mine_record
from vigil.prepare import build_mixture, category_counts
from vigil.processing import InternVLProcessor, dynamic_tiles, load_processor


@pytest.fixture
def record(tmp_path):
    image = tmp_path / "test.png"
    Image.new("RGB", (8, 8), color=(40, 100, 220)).save(image)
    return {"id": "example", "image": str(image), "prompt": "How many?", "chosen": "Two.",
            "rejected": "Three.", "category": "hallucination"}


class FakeProcessor:
    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize
        prompt = messages[0]["content"][-1]["text"]
        text = "|vision|" + prompt + "|assistant|"
        if not add_generation_prompt:
            text += messages[1]["content"][0]["text"] + "|end|"
        return text

    def __call__(self, text, images=None, padding=False, truncation=False, add_special_tokens=False, return_tensors=None):
        assert truncation is False and add_special_tokens is False
        sequences = []
        for value in text:
            chunks = value.replace("|vision|", "ΩΩΩΩ").replace("|assistant|", "Σ").replace("|end|", "Ξ")
            sequences.append([2 if char == "Ξ" else ord(char) for char in chunks])
        width = max(map(len, sequences))
        padded, masks = [], []
        for sequence in sequences:
            pad = [0] * (width - len(sequence))
            mask = [1] * len(sequence)
            if self.padding_side == "left":
                padded.append(pad + sequence)
                masks.append([0] * len(pad) + mask)
            else:
                padded.append(sequence + pad)
                masks.append(mask + [0] * len(pad))
        result = {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}
        if images:
            assert all(isinstance(image, list) and len(image) == 1 for image in images)
            result["pixel_values"] = torch.tensor([list(image[0].getpixel((0, 0))) for image in images])
        return result

    def batch_decode(self, sequences, **kwargs):
        return ["".join(chr(token) for token in sequence.tolist()) for sequence in sequences]


def test_dataset_validates_without_mutating_source(record, tmp_path):
    original = copy.deepcopy(record)
    assert validate_record(record) == original
    record["image"] = "test.png"
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [record])
    data = PreferenceDataset(path)
    assert len(data) == 1
    assert data[0]["image"] == original["image"]
    assert read_jsonl(path)[0]["image"] == "test.png"
    with pytest.raises(FileExistsError):
        write_jsonl(path, [record])


def test_dataset_rejects_conflicting_and_identical_preferences(record, tmp_path):
    with pytest.raises(ValueError, match="identical"):
        validate_record({**record, "rejected": record["chosen"]})
    with pytest.raises(ValueError, match="category"):
        validate_record({**record, "category": "other"})
    with pytest.raises(ValueError, match="local"):
        validate_record({**record, "image": "https://example.com/image.png"})
    path = tmp_path / "duplicates.jsonl"
    write_jsonl(path, [record, record])
    with pytest.raises(ValueError, match="duplicate"):
        PreferenceDataset(path)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_collator_masks_prompt_and_padding_keeps_assistant_eos(record, padding_side):
    processor = FakeProcessor(padding_side)
    batch = [record, {**record, "id": "two", "prompt": "Count?", "chosen": "2", "hard_negative": "8"}]
    result = PreferenceCollator(processor, max_length=256)(batch)
    for branch in ("chosen", "rejected"):
        for index, row in enumerate(batch):
            labels = result[branch]["labels"][index]
            expected = row["chosen"] if branch == "chosen" else negative_text(row)
            assert labels[labels != -100].tolist() == [ord(char) for char in expected] + [2]
            assert torch.all(labels[result[branch]["attention_mask"][index] == 0] == -100)
    assert torch.equal(result["chosen"]["pixel_values"], result["rejected"]["pixel_values"])


def test_collator_refuses_visual_token_truncation(record):
    with pytest.raises(ValueError, match="max_length"):
        PreferenceCollator(FakeProcessor(), max_length=5)([record])


def test_collator_refuses_ambiguous_assistant_boundary(record):
    class BadProcessor(FakeProcessor):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return ("bad" if add_generation_prompt else "good") + "|end|"

    with pytest.raises(ValueError, match="assistant boundary"):
        PreferenceCollator(BadProcessor())([record])


def test_mixture_is_deduplicated_reproducible_and_stratified(record):
    counts = category_counts(120)
    assert counts == {"hallucination": 54, "vqa": 30, "ocr": 18, "math": 18}
    pool = [{**record, "id": f"{category}-{i:03}", "prompt": f"{category} {i}", "category": category}
            for category, count in counts.items() for i in range(count + 5)]
    first = build_mixture(pool + [pool[0]], size=120, fraction=0.25, seed=42)
    second = build_mixture(list(reversed(pool)), size=120, fraction=0.25, seed=42)
    full = build_mixture(pool, size=120, seed=42)
    assert first == second and len(first) == 30
    assert {row["id"] for row in first} <= {row["id"] for row in full}
    assert {category: sum(row["category"] == category for row in first) for category in counts} == category_counts(30)
    with pytest.raises(ValueError, match="need"):
        build_mixture(pool, size=1200)
    with pytest.raises(ValueError, match="Conflicting"):
        build_mixture(pool + [{**pool[0], "prompt": "different"}], size=120)


def test_mining_retains_source_pair(record, monkeypatch):
    answers = iter([record["chosen"], "Four objects."])
    monkeypatch.setattr("vigil.mine.generate_response", lambda *a, **kw: next(answers))
    args = SimpleNamespace(attempts=2, max_new_tokens=100, max_length=1000, temperature=0.8,
                           top_p=0.9, model="reference", seed=42)
    mined = mine_record(None, None, record, args)
    assert mined["hard_negative"] == "Four objects."
    assert all(mined[key] == value for key, value in record.items())
    assert "hard_negative" not in record


def test_generation_only_decodes_continuation(record):
    class FakeModel:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, **kwargs):
            assert not kwargs["do_sample"]
            assert "pixel_values" in kwargs
            return torch.cat([input_ids, torch.tensor([[ord("O"), ord("K")]])], dim=1)

    assert generate_response(FakeModel(), FakeProcessor(), record, max_new_tokens=2) == "OK"
    assert generate_response(FakeModel(), FakeProcessor(), {"prompt": "Count?"}, max_new_tokens=2,
                             image=Image.new("RGB", (8, 8))) == "OK"
    with pytest.raises(ValueError, match="max_length"):
        generate_response(FakeModel(), FakeProcessor(), record, max_length=3)


def test_pope_id_alignment_confusion_matrix_and_zero_division():
    reference = [{"id": str(i), "label": value} for i, value in enumerate(["yes", "yes", "no", "no"])]
    predicted = [{"id": str(i), "prediction": value} for i, value in enumerate([
        "Yes, a dog. Not relevant later.", "No, there is not.", "No.", "It is visible."])]
    result = pope_metrics(align_records(list(reversed(predicted)), reference))
    assert result["tp"] == result["tn"] == result["fp"] == result["fn"] == 1
    assert result["f1"] == result["accuracy"] == 0.5
    assert pope_metrics([({"prediction": "No."}, {"label": "no"})])["f1"] == 0
    with pytest.raises(ValueError, match="IDs do not match"):
        align_records(predicted[:-1], reference)


def test_refcoco_iou_formats_scales_and_invalid_predictions():
    assert parse_bbox([2, 3, 8, 7], "xywh") == (2, 3, 10, 10)
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    reference = {"bbox": [0, 0, 10, 10], "width": 20, "height": 20}
    metrics = refcoco_metrics([({"prediction": "Box: [0, 0, 0.5, 0.5]"}, reference),
                               ({"prediction": "No box"}, reference)], prediction_scale="unit")
    assert metrics["accuracy"] == 0.5 and metrics["invalid_predictions"] == 1


def test_vcc_uses_first_premise_and_local_image_only(record):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="INCONSISTENT: three objects"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    pred = {"id": "1", "prediction": "Long chain of thought", "first_visual_premise": "There are three objects."}
    reference = {"id": "1", "image": record["image"], "objects": ["apple", "cup"]}
    result = judge_vcc([(pred, reference)], "gpt-4o", client=client)
    assert not parse_vcc_label(result[0]["judgment"])
    content = calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(content[1]["text"])["first_visual_premise"] == pred["first_visual_premise"]
    assert pred["prediction"] not in content[1]["text"]
    with pytest.raises(ValueError):
        parse_vcc_label("Maybe")


class FakeConversation:
    system_message = "Inspect the image."
    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    sep = "<|im_end|>\n"

    def __init__(self):
        self.messages = []

    def append_message(self, role, content):
        self.messages.append((role, content))

    def get_prompt(self):
        result = "<|im_start|>system\n" + self.system_message + self.sep
        for role, content in self.messages:
            result += role + (content + self.sep if content is not None else "")
        return result


def fake_conversation_factory(_):
    return FakeConversation()


def tiny_internvl_processor():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "<s>", "</s>", "<IMG_CONTEXT>", "<img>", "</img>", "<|im_start|>", "<|im_end|>",
             "Two", "Three", ".", "user", "assistant", "system"]
    tokenizer = Tokenizer(models.WordLevel({token: index for index, token in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>",
                                       pad_token="</s>", unk_token="[UNK]", additional_special_tokens=words[3:8])
    config = SimpleNamespace(force_image_size=4, vision_config=SimpleNamespace(image_size=4, patch_size=2),
                             downsample_ratio=0.5, template="internvl2_5")
    return InternVLProcessor(tokenizer, config, fake_conversation_factory, max_num=2)


def test_internvl_dynamic_patches_visual_expansion_and_assistant_boundary(record, tmp_path):
    processor = tiny_internvl_processor()
    wide = Image.new("RGB", (8, 4), color="red")
    assert len(dynamic_tiles(wide, image_size=4, max_num=2)) == 3
    assert len(dynamic_tiles(wide, image_size=4, max_num=2, use_thumbnail=False)) == 2
    assert len(dynamic_tiles(Image.new("RGB", (4, 4)), image_size=4, max_num=2)) == 1
    wide.save(tmp_path / "wide.png")
    batch = PreferenceCollator(processor)([{**record, "image": str(tmp_path / "wide.png")}])
    chosen = batch["chosen"]
    assert chosen["pixel_values"].shape == (3, 3, 4, 4)
    assert chosen["image_flags"].shape == (3, 1)
    assert int((chosen["input_ids"] == processor.image_token_id).sum()) == 3
    labels = chosen["labels"][0]
    assert labels[labels != -100].tolist() == [8, 10, processor.end_token_id]
    assert torch.equal(chosen["pixel_values"], batch["rejected"]["pixel_values"])
    with pytest.raises(ValueError, match="truncated"):
        processor(["test"], truncation=True)


def test_internvl_requires_explicit_remote_code_and_preserves_saved_model_config(monkeypatch, tmp_path):
    from transformers import PretrainedConfig

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", lambda *args, **kwargs: ({"model_type": "internvl_chat"}, {}))
    with pytest.raises(ValueError, match="trust-remote-code"):
        load_processor("unused-model-name")
    processor = tiny_internvl_processor()
    existing = '{"trained_model_config": true}\n'
    (tmp_path / "config.json").write_text(existing)
    processor.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").read_text() == existing
    saved = json.loads((tmp_path / "vigil_processor_config.json").read_text())
    assert saved == {"max_num": 2, "use_thumbnail": True}
    assert (tmp_path / "conversation.py").is_file()
