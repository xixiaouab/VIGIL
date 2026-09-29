from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import PreferenceDataset
from .generate import add_model_arguments, generate_response, model_from_arguments

MINING_PROMPT = """Create one hard negative answer for a visual preference pair. Inspect the image and the grounded answer below. Write a fluent answer to the original question that introduces one plausible but absent object, incorrect attribute, count, or spatial relation. Keep the answer relevant and comparable in detail to the grounded answer. Return only the negative answer, without explanations or labels.

Original question: {prompt}
Grounded answer: {chosen}"""


def mine_record(model, processor, record: dict, args) -> dict:
    prompt_record = {**record, "prompt": MINING_PROMPT.format(prompt=record["prompt"], chosen=record["chosen"])}
    for _ in range(args.attempts):
        negative = generate_response(model, processor, prompt_record, max_new_tokens=args.max_new_tokens,
                                     max_length=args.max_length, temperature=args.temperature, top_p=args.top_p)
        if negative and negative.strip() != record["chosen"].strip():
            return {**record, "hard_negative": negative,
                    "hard_negative_source": {"model": args.model, "seed": args.seed}}
    raise ValueError(f"{record['id']}: no distinct hard negative after {args.attempts} attempts")


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives with the frozen reference checkpoint.")
    add_model_arguments(parser)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error("attempts must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    dataset = PreferenceDataset(args.input, args.image_root)
    model, processor = model_from_arguments(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in dataset:
            result = mine_record(model, processor, record, args)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
