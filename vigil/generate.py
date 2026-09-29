from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import conversation, image_path, read_jsonl


def load_generator(model_name: str, device: str = "auto", dtype: str = "auto", seed: int = 42,
                   min_pixels: int | None = None, max_pixels: int | None = None,
                   revision: str | None = None, trust_remote_code: bool = False,
                   max_num: int | None = None, use_thumbnail: bool | None = None):
    from transformers import set_seed

    from .model import VIGILModel
    from .processing import load_processor

    set_seed(seed)
    processor_kwargs = {key: value for key, value in {"min_pixels": min_pixels, "max_pixels": max_pixels,
                        "max_num": max_num, "use_thumbnail": use_thumbnail}.items()
                        if value is not None}
    processor = load_processor(model_name, revision=revision, trust_remote_code=trust_remote_code, **processor_kwargs)
    kwargs = {"dtype": dtype, "attn_implementation": "sdpa", "revision": revision,
              "trust_remote_code": trust_remote_code}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = VIGILModel.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model.to(device)
    model.eval().requires_grad_(False)
    return model, processor


def generate_response(model, processor, record: dict, image_root=None, max_new_tokens: int = 512,
                      max_length: int = 8192, temperature: float = 0.0, top_p: float = 0.9,
                      image=None, generation_kwargs: dict | None = None) -> str:
    import torch
    from PIL import Image, ImageOps

    if max_new_tokens < 1 or max_length < 2 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid generation limits, temperature, or top_p")
    has_image = image is not None or bool(record.get("image"))
    messages = conversation(record["prompt"], with_image=has_image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [text], "return_tensors": "pt", "truncation": False, "add_special_tokens": False}
    if has_image:
        if image is None:
            with Image.open(image_path(record["image"], image_root)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        else:
            image = ImageOps.exif_transpose(image).convert("RGB")
        kwargs["images"] = [[image]]
    inputs = processor(**kwargs)
    prompt_length = inputs["input_ids"].shape[-1]
    if prompt_length + max_new_tokens > max_length:
        raise ValueError(f"{record.get('id', '?')}: prompt plus generation exceeds max_length={max_length}")
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if hasattr(processor, "end_token_id"):
        options["eos_token_id"] = processor.end_token_id
        options["pad_token_id"] = processor.tokenizer.pad_token_id
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p)
    extra_options = dict(generation_kwargs or {})
    if set(extra_options) & {"max_new_tokens", "do_sample", "temperature", "top_p"}:
        raise ValueError("Set generation length and sampling through their named arguments")
    options.update(extra_options)
    with torch.inference_mode():
        outputs = model.generate(**inputs, **options)
    return processor.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def add_model_arguments(parser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--max-num", type=int)
    parser.add_argument("--no-thumbnail", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=8192)


def model_from_arguments(args):
    return load_generator(args.model, args.device, args.dtype, args.seed, args.min_pixels, args.max_pixels,
                          args.revision, args.trust_remote_code, args.max_num, False if args.no_thumbnail else None)


def main():
    parser = argparse.ArgumentParser(description="Generate answers from a VIGIL checkpoint.")
    add_model_arguments(parser)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output already exists: {output}")
    records = read_jsonl(args.input)
    seen = set()
    root = args.image_root or Path(args.input).resolve().parent
    for record in records:
        if not isinstance(record.get("id"), (str, int)) or str(record["id"]) in seen:
            parser.error("Every input needs a unique id")
        seen.add(str(record["id"]))
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            parser.error(f"{record['id']}: prompt must be nonempty")
        if record.get("image"):
            record["image"] = str(image_path(record["image"], root))
    model, processor = model_from_arguments(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in records:
            prediction = generate_response(model, processor, record, root, args.max_new_tokens,
                                           args.max_length, args.temperature, args.top_p)
            handle.write(json.dumps({**record, "prediction": prediction}, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
