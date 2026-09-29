import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


REFERENCE_FIELDS = ("ref_chosen_see", "ref_rejected_see", "ref_chosen_blind")
CACHE_VERSION = 1


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fingerprint(value):
    raw = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(dataset):
    image_hashes, rows, ids = {}, [], []
    for record in dataset:
        row_id = str(record["id"])
        image_path = str(Path(record["image"]).resolve())
        if image_path not in image_hashes:
            image_hashes[image_path] = file_fingerprint(image_path)
        negative = record.get("hard_negative") or record.get("rejected")
        if not negative:
            raise ValueError(f"Record {row_id} has no rejected or hard_negative response")
        rows.append({"id": row_id, "image_sha256": image_hashes[image_path],
                     "prompt": record["prompt"], "chosen": record["chosen"], "rejected": negative})
        ids.append(row_id)
    if len(set(ids)) != len(ids):
        raise ValueError("Reference cache requires unique record IDs")
    if not ids:
        raise ValueError("Reference cache requires a nonempty dataset")
    return {"sha256": fingerprint(rows), "count": len(ids), "ids": ids}


def processor_fingerprint(processor):
    tokenizer = processor.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        backend = json.loads(backend.to_str())
        backend.pop("padding", None)
        backend.pop("truncation", None)
    image_processor = getattr(processor, "image_processor", None)
    description = {
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab": tokenizer.get_vocab(),
        "backend": backend,
        "special_tokens": tokenizer.special_tokens_map,
        "padding_side": tokenizer.padding_side,
        "processor_chat_template": getattr(processor, "chat_template", None),
        "tokenizer_chat_template": getattr(tokenizer, "chat_template", None),
        "image_processor": image_processor.to_dict() if image_processor is not None else None,
    }
    return fingerprint(description)


def model_fingerprint(model_name, revision=None, model_config=None):
    config = model_config.to_dict() if hasattr(model_config, "to_dict") else dict(model_config or {})
    resolved_revision = getattr(model_config, "_commit_hash", None) or config.get("_commit_hash") or revision

    def normalize(value):
        ignored = {"transformers_version", "torch_dtype", "dtype", "use_cache", "output_attentions",
                   "output_hidden_states", "return_dict"}
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()
                    if not str(key).startswith("_") and key not in ignored}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    config = normalize(config)
    model_path = Path(model_name)
    weights = {}
    if model_path.is_dir():
        for pattern in ("*.safetensors", "pytorch_model*.bin", "adapter_model*.bin"):
            for path in sorted(model_path.glob(pattern)):
                weights[path.name] = file_fingerprint(path)
        if not weights:
            raise ValueError(f"No checkpoint weights found in {model_path}")
    elif not resolved_revision:
        raise ValueError("A resolved model commit is required; pass the loaded AutoConfig")
    return {"name": str(model_name), "revision": resolved_revision,
            "config_sha256": fingerprint(config), "local_weights": weights}


def build_cache_metadata(dataset, *, model_name, revision=None, processor, max_length=4096,
                         model_config=None, dtype="bfloat16", attn_implementation="sdpa"):
    import transformers

    source_root = Path(__file__).parent
    sources = {name: file_fingerprint(source_root / name) for name in
               ("model.py", "masking.py", "data.py", "processing.py", "cache_reference.py")
               if (source_root / name).is_file()}
    payload = {
        "version": CACHE_VERSION,
        "dataset": dataset_fingerprint(dataset),
        "model": model_fingerprint(model_name, revision, model_config),
        "processor_sha256": processor_fingerprint(processor),
        "max_length": int(max_length),
        "dtype": str(dtype),
        "attn_implementation": attn_implementation,
        "score_reduction": "sum_response_token_logprobs",
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__},
        "source_sha256": sources,
    }
    return {**payload, "signature": fingerprint(payload)}


def _check_metadata(metadata):
    payload = {key: value for key, value in metadata.items() if key not in {"signature", "scores_sha256"}}
    if metadata.get("version") != CACHE_VERSION or metadata.get("signature") != fingerprint(payload):
        raise ValueError("Invalid reference-cache metadata signature")


def metadata_path(path):
    return Path(str(path) + ".meta.json")


def _load_rows(path):
    scores = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row["id"])
            if row_id in scores:
                raise ValueError(f"Duplicate reference-cache ID {row_id} at line {line_number}")
            values = {}
            for key in REFERENCE_FIELDS:
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"Invalid {key} for reference-cache ID {row_id}")
                values[key] = float(value)
            scores[row_id] = values
    return scores


def load_reference_cache(path, expected_metadata):
    _check_metadata(expected_metadata)
    with metadata_path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    _check_metadata(metadata)
    if metadata["signature"] != expected_metadata["signature"]:
        differences = [key for key in expected_metadata if key != "signature" and
                       expected_metadata[key] != metadata.get(key)]
        raise ValueError("Reference cache does not match: " + ", ".join(differences))
    if metadata.get("scores_sha256") != file_fingerprint(path):
        raise ValueError("Reference-cache scores have changed or are incomplete")
    scores = _load_rows(path)
    expected_ids = set(expected_metadata["dataset"]["ids"])
    if set(scores) != expected_ids:
        raise ValueError("Reference-cache IDs do not match the dataset")
    return scores


def _atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_reference_cache(path, scores, metadata):
    _check_metadata(metadata)
    if set(scores) != set(metadata["dataset"]["ids"]):
        raise ValueError("Cannot write an incomplete reference cache")
    lines = [json.dumps({"id": row_id, **scores[row_id]}, allow_nan=False)
             for row_id in metadata["dataset"]["ids"]]
    _atomic_write(path, "\n".join(lines) + "\n")
    metadata = {**metadata, "scores_sha256": file_fingerprint(path)}
    _atomic_write(metadata_path(path), json.dumps(metadata, indent=2) + "\n")
    load_reference_cache(path, metadata)


class ReferenceDataset(Dataset):
    def __init__(self, dataset, scores):
        self.dataset, self.scores = dataset, scores
        missing = [str(row["id"]) for row in dataset if str(row["id"]) not in scores]
        if missing:
            raise ValueError(f"Missing reference scores for ID {missing[0]}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = dict(self.dataset[index])
        row["reference"] = dict(self.scores[str(row["id"])])
        return row


class ReferenceCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, records):
        batch = self.collator(records)
        batch["reference"] = {key: torch.tensor([row["reference"][key] for row in records],
                                               dtype=torch.float32) for key in REFERENCE_FIELDS}
        return batch


def _to_device(batch, device):
    return {key: _to_device(value, device) if isinstance(value, dict) else
            value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="Cache frozen seeing/blind reference log-probabilities.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-num", type=int)
    parser.add_argument("--no-thumbnail", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device-map", choices=("auto",))
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if Path(args.output).exists() and not args.overwrite:
        parser.error("Output exists; choose another path or pass --overwrite")

    from transformers import AutoConfig
    from vigil.data import PreferenceCollator, PreferenceDataset
    from vigil.model import VIGILModel
    from vigil.processing import load_processor

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and args.device_map is not None:
        parser.error("Use --device-map auto in one process; torchrun assigns one GPU to each rank")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")

    dataset = PreferenceDataset(args.data, image_root=args.image_root)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision,
                                        trust_remote_code=args.trust_remote_code)
    resolved_revision = getattr(config, "_commit_hash", None) or args.revision
    processor_options = {key: value for key, value in
                         {"min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                          "max_num": args.max_num}.items()
                         if value is not None}
    if args.no_thumbnail:
        processor_options["use_thumbnail"] = False
    processor = load_processor(args.model, revision=resolved_revision,
                               trust_remote_code=args.trust_remote_code, **processor_options)
    metadata = build_cache_metadata(dataset, model_name=args.model, revision=args.revision,
                                   processor=processor, max_length=args.max_length, model_config=config,
                                   dtype=args.dtype, attn_implementation=args.attn_implementation) if rank == 0 else None
    if world > 1:
        shared = [metadata]
        torch.distributed.broadcast_object_list(shared, src=0)
        metadata = shared[0]
    collator = PreferenceCollator(processor, max_length=args.max_length)
    kwargs = {"revision": resolved_revision, "trust_remote_code": args.trust_remote_code}
    if args.device_map:
        kwargs["device_map"] = args.device_map
    reference = VIGILModel.from_pretrained(args.model, dtype=args.dtype,
                                          attn_implementation=args.attn_implementation, **kwargs)
    reference.requires_grad_(False)
    reference.eval()
    if args.device_map is None:
        reference.to(device)
    device = reference.get_input_embeddings().weight.device
    scores = {}
    indices = list(range(rank, len(dataset), world))
    with torch.inference_mode():
        for start in range(0, len(indices), args.batch_size):
            records = [dataset[index] for index in indices[start:start + args.batch_size]]
            values = reference(_to_device(collator(records), device))
            for offset, record in enumerate(records):
                scores[str(record["id"])] = {key: float(values[key.removeprefix("ref_")][offset].float().cpu())
                                              for key in REFERENCE_FIELDS}
            if rank == 0 and (start == 0 or start % (100 * args.batch_size) == 0 or
                              start + args.batch_size >= len(indices)):
                print(f"Cached {min(start + args.batch_size, len(indices))}/{len(indices)} records on rank 0", flush=True)
    if world == 1:
        write_reference_cache(args.output, scores, metadata)
    else:
        shard = Path(f"{args.output}.rank{rank:05d}-of-{world:05d}.jsonl")
        _atomic_write(shard, "\n".join(json.dumps({"id": key, **value}, allow_nan=False)
                                      for key, value in scores.items()) + "\n")
        torch.distributed.barrier()
        if rank == 0:
            merged = {}
            for shard_rank in range(world):
                shard_path = Path(f"{args.output}.rank{shard_rank:05d}-of-{world:05d}.jsonl")
                current = _load_rows(shard_path)
                if set(current) & set(merged):
                    raise ValueError("Duplicate IDs across reference-cache shards")
                merged.update(current)
            write_reference_cache(args.output, merged, metadata)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank == 0:
        print(f"Saved {len(dataset)} reference scores to {args.output}", flush=True)


if __name__ == "__main__":
    main()
