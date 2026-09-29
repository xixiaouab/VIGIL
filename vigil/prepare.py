from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

from .data import CATEGORY_WEIGHTS, negative_text, read_jsonl, validate_record, write_jsonl


def category_counts(size: int) -> dict[str, int]:
    if size < 1:
        raise ValueError("size must be positive")
    exact = {category: size * weight for category, weight in CATEGORY_WEIGHTS.items()}
    counts = {category: math.floor(value) for category, value in exact.items()}
    remainder = size - sum(counts.values())
    ordering = sorted(exact, key=lambda key: (-(exact[key] - counts[key]), key))
    for category in ordering[:remainder]:
        counts[category] += 1
    return counts


def build_mixture(records: list[dict], size: int, fraction: float = 1.0, seed: int = 42) -> list[dict]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    counts = category_counts(size)
    unique, ids = {}, {}
    for record in records:
        fingerprint = json.dumps([record["image"], record["prompt"], record["chosen"], negative_text(record)])
        if record["id"] in ids and ids[record["id"]] != fingerprint:
            raise ValueError(f"Conflicting records use ID {record['id']}")
        ids[record["id"]] = fingerprint
        existing = unique.get(fingerprint)
        if existing and existing["category"] != record["category"]:
            raise ValueError(f"Duplicate preference pair has conflicting categories: {record['id']}")
        if existing is None or record["id"] < existing["id"]:
            unique[fingerprint] = record
    pools = {category: [] for category in CATEGORY_WEIGHTS}
    for record in unique.values():
        pools[record["category"]].append(record)
    rng = random.Random(seed)
    selected = []
    subset_counts = category_counts(max(1, math.floor(size * fraction)))
    for category, count in counts.items():
        pool = sorted(pools[category], key=lambda row: row["id"])
        if len(pool) < count:
            raise ValueError(f"{category}: need {count} distinct pairs, found {len(pool)}")
        rng.shuffle(pool)
        selected.extend(pool[:subset_counts[category]])
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Build the VIGIL preference mixture.")
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--size", type=int, default=120000)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")
    records = []
    for source in args.input:
        root = args.image_root if args.image_root is not None else Path(source).resolve().parent
        records.extend(validate_record(record, root) for record in read_jsonl(source))
    selected = build_mixture(records, args.size, args.fraction, args.seed)
    write_jsonl(args.output, selected)
    digest = hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
    print(json.dumps({"output": args.output, "pairs": len(selected), "seed": args.seed,
                      "categories": dict(Counter(row["category"] for row in selected)), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
