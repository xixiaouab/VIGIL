from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import re
from collections import defaultdict
from pathlib import Path

from .data import image_path, read_jsonl

VCC_PROMPT = (
    "You are given an image, a list of ground-truth objects visible in the image, and the first visual premise "
    "extracted from a model’s reasoning. Decide whether the premise is visually consistent with the image. "
    "Answer with CONSISTENT or INCONSISTENT. If inconsistent, name the offending object, attribute, count, or "
    "spatial relation. Do not judge later reasoning steps or the final answer; evaluate only the first visual "
    "premise against the image evidence."
)


def align_records(predictions: list[dict], references: list[dict]) -> list[tuple[dict, dict]]:
    def indexed(rows, kind):
        result = {}
        for record in rows:
            value = record.get("id", record.get("question_id"))
            if value is None or str(value) in result:
                raise ValueError(f"{kind}: missing or duplicate ID {value!r}")
            result[str(value)] = record
        return result

    predicted, expected = indexed(predictions, "predictions"), indexed(references, "references")
    if not expected or predicted.keys() != expected.keys():
        raise ValueError(f"IDs do not match: {len(expected.keys() - predicted.keys())} missing, "
                         f"{len(predicted.keys() - expected.keys())} extra predictions")
    return [(predicted[key], reference) for key, reference in expected.items()]


def pope_answer(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("POPE prediction must be a nonempty string")
    words = text.split(".", 1)[0].replace(",", "").split(" ")
    return not any(word in {"No", "not", "no"} for word in words)


def pope_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    tp = tn = fp = fn = 0
    for prediction, reference in pairs:
        answer = prediction.get("prediction", prediction.get("answer"))
        pred = pope_answer(answer)
        label = reference.get("label")
        if label not in {"yes", "no"}:
            raise ValueError("POPE labels must be 'yes' or 'no'")
        truth = label == "yes"
        tp += int(pred and truth)
        tn += int(not pred and not truth)
        fp += int(pred and not truth)
        fn += int(not pred and truth)
    if not pairs:
        raise ValueError("No POPE examples")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"count": len(pairs), "accuracy": (tp + tn) / len(pairs), "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "yes_ratio": (tp + fp) / len(pairs), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def parse_bbox(value, box_format: str = "xyxy") -> tuple[float, float, float, float]:
    if isinstance(value, str):
        match = re.search(r"[\[(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
                          r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\])]", value)
        if not match:
            raise ValueError("Expected a four-coordinate bounding box")
        value = match.groups()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Expected a four-coordinate bounding box")
    x1, y1, x2, y2 = map(float, value)
    if not all(math.isfinite(coordinate) for coordinate in (x1, y1, x2, y2)):
        raise ValueError("Bounding box coordinates must be finite")
    if box_format == "xywh":
        x2, y2 = x1 + x2, y1 + y2
    elif box_format != "xyxy":
        raise ValueError("bbox_format must be xyxy or xywh")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have positive area")
    return x1, y1, x2, y2


def bbox_iou(first, second) -> float:
    a, b = parse_bbox(first), parse_bbox(second)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def refcoco_metrics(pairs: list[tuple[dict, dict]], threshold: float = 0.5,
                    prediction_scale: str = "pixels") -> dict:
    if not 0 < threshold <= 1 or not pairs:
        raise ValueError("Require examples and an IoU threshold in (0, 1]")
    ious, invalid = [], 0
    for prediction, reference in pairs:
        target = parse_bbox(reference["bbox"], reference.get("bbox_format", "xyxy"))
        try:
            predicted = parse_bbox(prediction.get("prediction", prediction.get("bbox")),
                                   prediction.get("prediction_bbox_format", "xyxy"))
            if prediction_scale != "pixels":
                if prediction_scale not in {"unit", "1000"}:
                    raise ValueError("Unknown prediction scale")
                divisor = 1.0 if prediction_scale == "unit" else 1000.0
                width, height = float(reference["width"]), float(reference["height"])
                if width <= 0 or height <= 0:
                    raise ValueError("Image dimensions must be positive")
                predicted = tuple(coord * dimension / divisor for coord, dimension in zip(
                    predicted, (width, height, width, height)))
            ious.append(bbox_iou(predicted, target))
        except (ValueError, TypeError):
            invalid += 1
            ious.append(0.0)
    return {"count": len(pairs), "accuracy": sum(value >= threshold for value in ious) / len(ious),
            "mean_iou": sum(ious) / len(ious), "threshold": threshold, "invalid_predictions": invalid}


def parse_vcc_label(value: str) -> bool:
    if not isinstance(value, str):
        raise ValueError("VCC needs a CONSISTENT or INCONSISTENT judgment")
    match = re.match(r"^\s*(CONSISTENT|INCONSISTENT)\b", value.upper())
    if not match:
        raise ValueError("VCC needs a CONSISTENT or INCONSISTENT judgment")
    return match.group(1) == "CONSISTENT"


def judge_vcc(pairs: list[tuple[dict, dict]], model_name: str, image_root=None, client=None) -> list[dict]:
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    results = []
    for prediction, reference in pairs:
        premise = prediction.get("first_visual_premise")
        objects = reference.get("objects")
        if not isinstance(premise, str) or not premise.strip() or not isinstance(objects, list):
            raise ValueError("VCC requires first_visual_premise in predictions and objects in references")
        path = image_path(reference["image"], image_root)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        response = client.chat.completions.create(
            model=model_name, temperature=0,
            messages=[{"role": "system", "content": VCC_PROMPT}, {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                {"type": "text", "text": json.dumps({"objects": objects, "first_visual_premise": premise},
                                                      ensure_ascii=False)},
            ]}],
        )
        judgment = response.choices[0].message.content
        parse_vcc_label(judgment)
        results.append({"id": reference.get("id", reference.get("question_id")), "judgment": judgment,
                        "judge_model": model_name})
    return results


def vcc_metrics(pairs: list[tuple[dict, dict]]) -> dict:
    values = [parse_vcc_label(prediction.get("judgment", prediction.get("vcc_label"))) for prediction, _ in pairs]
    if not values:
        raise ValueError("No VCC judgments")
    return {"count": len(values), "vcc": sum(values) / len(values)}


def main():
    parser = argparse.ArgumentParser(description="Score POPE, RefCOCO, or visual CoT consistency.")
    parser.add_argument("--task", choices=["pope", "refcoco", "vcc"], required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--prediction-scale", choices=["pixels", "unit", "1000"], default="pixels")
    parser.add_argument("--judge-model", help="Run the optional VCC API judge, e.g. gpt-4o")
    parser.add_argument("--image-root")
    args = parser.parse_args()
    if args.output and Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")
    if args.judge_model and args.task != "vcc":
        parser.error("--judge-model is only valid for VCC")
    references = read_jsonl(args.references)
    pairs = align_records(read_jsonl(args.predictions), references)
    judgments = None
    if args.judge_model:
        judgments = judge_vcc(pairs, args.judge_model, args.image_root or Path(args.references).resolve().parent)
        pairs = align_records(judgments, references)
    if args.task == "pope":
        scorer = pope_metrics
    elif args.task == "refcoco":
        scorer = lambda group: refcoco_metrics(group, args.iou_threshold, args.prediction_scale)
    else:
        scorer = vcc_metrics
    result = {"task": args.task, **scorer(pairs)}
    groups = defaultdict(list)
    for pair in pairs:
        if pair[1].get("split") is not None:
            groups[str(pair[1]["split"])].append(pair)
    if groups:
        result["splits"] = {name: scorer(group) for name, group in sorted(groups.items())}
    if judgments is not None:
        result["judgments"] = judgments
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
