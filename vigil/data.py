from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CATEGORY_WEIGHTS = {"hallucination": 0.45, "vqa": 0.25, "ocr": 0.15, "math": 0.15}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def negative_text(record: dict[str, Any]) -> str:
    value = record.get("hard_negative", record.get("rejected"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record.get('id', '?')}: provide rejected or hard_negative")
    return value


def image_path(value: str, image_root: str | Path | None = None) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError("image must be a local file path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(image_root or ".") / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_record(record: dict[str, Any], image_root: str | Path | None = None) -> dict[str, Any]:
    result = dict(record)
    for key in ("id", "prompt", "chosen", "category"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError(f"{result.get('id', '?')}: {key} must be a nonempty string")
    if result["category"] not in CATEGORY_WEIGHTS:
        raise ValueError(f"{result['id']}: category must be one of {list(CATEGORY_WEIGHTS)}")
    if result["chosen"].strip() == negative_text(result).strip():
        raise ValueError(f"{result['id']}: chosen and negative responses are identical")
    result["image"] = str(image_path(result.get("image"), image_root))
    return result


class PreferenceDataset(Sequence):
    def __init__(self, path: str | Path, image_root: str | Path | None = None):
        self.path = Path(path)
        root = image_root if image_root is not None else self.path.resolve().parent
        self.records = [validate_record(record, root) for record in read_jsonl(path)]
        identifiers = [record["id"] for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{path}: duplicate record IDs")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def conversation(prompt: str, response: str | None = None, with_image: bool = True) -> list[dict]:
    content = [{"type": "image"}] if with_image else []
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return messages


class PreferenceCollator:
    def __init__(self, processor, max_length: int = 4096, image_root: str | Path | None = None):
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        self.processor = processor
        self.max_length = max_length
        self.image_root = image_root

    def _encode(self, records: list[dict], images: list, response_key: str | None):
        texts = []
        for record in records:
            response = None
            if response_key == "chosen":
                response = record["chosen"]
            elif response_key == "rejected":
                response = negative_text(record)
            texts.append(self.processor.apply_chat_template(
                conversation(record["prompt"], response), tokenize=False,
                add_generation_prompt=response is None,
            ))
        return self.processor(
            text=texts, images=[[im] for im in images], padding=True,
            truncation=False, add_special_tokens=False, return_tensors="pt",
        )

    def __call__(self, records: list[dict]) -> dict:
        import torch
        from PIL import Image, ImageOps

        if not records:
            raise ValueError("Cannot collate an empty batch")
        images = []
        for record in records:
            with Image.open(image_path(record["image"], self.image_root)) as image:
                images.append(ImageOps.exif_transpose(image).convert("RGB"))
        prompt_inputs = self._encode(records, images, None)
        result = {}
        for response_key in ("chosen", "rejected"):
            batch = self._encode(records, images, response_key)
            input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
            labels = torch.full_like(input_ids, -100)
            for index, record in enumerate(records):
                positions = attention_mask[index].bool().nonzero(as_tuple=True)[0]
                full_ids = input_ids[index, positions]
                prompt_ids = prompt_inputs["input_ids"][index][prompt_inputs["attention_mask"][index].bool()]
                if len(full_ids) > self.max_length:
                    raise ValueError(f"{record['id']}: {response_key} has {len(full_ids)} tokens; "
                                     f"max_length={self.max_length}. Increase max_length or reduce image resolution.")
                prefix_length = len(prompt_ids)
                if len(full_ids) <= prefix_length or not torch.equal(full_ids[:prefix_length], prompt_ids):
                    raise ValueError(f"{record['id']}: assistant boundary does not match the chat template")
                labels[index, positions[prefix_length:]] = full_ids[prefix_length:]
            batch["labels"] = labels
            result[response_key] = dict(batch)
        return result
