from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path


def dynamic_tiles(image, image_size=448, max_num=12, use_thumbnail=True):
    """InternVL tiling rule: https://huggingface.co/OpenGVLab/InternVL2_5-26B."""
    from PIL import Image

    if image_size < 1 or max_num < 1:
        raise ValueError("image_size and max_num must be positive")
    image = image.convert("RGB")
    width, height = image.size
    grids = {(columns, rows) for columns in range(1, max_num + 1)
             for rows in range(1, max_num + 1) if columns * rows <= max_num}
    best, gap = (1, 1), float("inf")
    for columns, rows in sorted(grids, key=lambda grid: grid[0] * grid[1]):
        distance = abs(width / height - columns / rows)
        enough_pixels = width * height > 0.5 * image_size**2 * columns * rows
        if distance < gap or (distance == gap and enough_pixels):
            best, gap = (columns, rows), distance
    columns, rows = best
    resized = image.resize((columns * image_size, rows * image_size), Image.Resampling.BICUBIC)
    tiles = [resized.crop((column * image_size, row * image_size,
                           (column + 1) * image_size, (row + 1) * image_size))
             for row in range(rows) for column in range(columns)]
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


class InternVLProcessor:
    def __init__(self, tokenizer, config, conversation_factory, max_num=12, use_thumbnail=True):
        if max_num < 1:
            raise ValueError("max_num must be positive")
        self.tokenizer, self.config = tokenizer, config
        self.conversation_factory = conversation_factory
        self.image_size = config.force_image_size or config.vision_config.image_size
        self.max_num, self.use_thumbnail = max_num, use_thumbnail
        self.num_image_token = int((self.image_size // config.vision_config.patch_size) ** 2
                                   * config.downsample_ratio**2)
        self.image_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if self.image_token_id is None or self.image_token_id == tokenizer.unk_token_id:
            raise ValueError("Tokenizer lacks <IMG_CONTEXT>")
        self.chat_template = config.template
        self.image_processor = self
        self.end_token_id = tokenizer.convert_tokens_to_ids(conversation_factory(config.template).sep.strip())

    def to_dict(self):
        template = self.conversation_factory(self.config.template)
        return {"processor_class": "InternVLProcessor", "image_size": self.image_size,
                "num_image_token": self.num_image_token, "max_num": self.max_num,
                "use_thumbnail": self.use_thumbnail, "template": self.config.template,
                "system_message": template.system_message, "separator": template.sep,
                "roles": list(template.roles), "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise ValueError("Render text first, then pass it with images to the processor")
        template = self.conversation_factory(self.config.template)
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                content = "".join("<image>\n" if item["type"] == "image" else item["text"] for item in content)
            if role == "system":
                template.system_message = content
            elif role in {"user", "assistant"}:
                template.append_message(template.roles[role == "assistant"], content)
            else:
                raise ValueError(f"Unsupported conversation role: {role}")
        if add_generation_prompt:
            template.append_message(template.roles[1], None)
        return (self.tokenizer.bos_token or "") + template.get_prompt()

    def __call__(self, text, images=None, padding=False, truncation=False,
                 add_special_tokens=False, return_tensors="pt", **kwargs):
        import numpy as np
        import torch

        if truncation:
            raise ValueError("InternVL visual tokens must not be truncated")
        if return_tensors != "pt":
            raise ValueError("InternVLProcessor returns PyTorch tensors")
        texts = [text] if isinstance(text, str) else list(text)
        groups = [[] for _ in texts] if images is None else images
        if not isinstance(groups, (list, tuple)):
            groups = [[groups]]
        elif groups and not isinstance(groups[0], (list, tuple)):
            groups = [[image] for image in groups]
        if len(groups) != len(texts):
            raise ValueError("One image group is required per text")
        tensors, counts, expanded = [], [], []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for prompt, group in zip(texts, groups):
            if prompt.count("<image>") != len(group):
                raise ValueError("Image placeholders and image count do not match")
            token_count = 0
            for image in group:
                tiles = dynamic_tiles(image, self.image_size, self.max_num, self.use_thumbnail)
                count = len(tiles) * self.num_image_token
                token_count += count
                prompt = prompt.replace("<image>", "<img>" + "<IMG_CONTEXT>" * count + "</img>", 1)
                for tile in tiles:
                    tensor = torch.from_numpy(np.asarray(tile, dtype=np.float32).copy()).permute(2, 0, 1) / 255
                    tensors.append((tensor - mean) / std)
            expanded.append(prompt)
            counts.append(token_count)
        batch = dict(self.tokenizer(expanded, padding=padding, truncation=False,
                                    add_special_tokens=add_special_tokens, return_tensors="pt", **kwargs))
        for index, count in enumerate(counts):
            if int((batch["input_ids"][index] == self.image_token_id).sum()) != count:
                raise ValueError("Tokenizer did not preserve the complete visual-token sequence")
        if tensors:
            batch["pixel_values"] = torch.stack(tensors)
            batch["image_flags"] = torch.ones(len(tensors), 1, dtype=torch.long)
        return batch

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        if not (path / "config.json").is_file():
            self.config.save_pretrained(path)
        settings = {"max_num": self.max_num, "use_thumbnail": self.use_thumbnail}
        (path / "vigil_processor_config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
        source = Path(inspect.getfile(self.conversation_factory))
        if source.resolve() != (path / "conversation.py").resolve():
            shutil.copy2(source, path / "conversation.py")


def load_processor(model, revision=None, trust_remote_code=False, **processor_options):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PretrainedConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_dict, _ = PretrainedConfig.get_config_dict(model, revision=revision)
    if config_dict.get("model_type") != "internvl_chat":
        return AutoProcessor.from_pretrained(model, revision=revision, trust_remote_code=trust_remote_code,
                                             **processor_options)
    if not trust_remote_code:
        raise ValueError("InternVL2.5 requires --trust-remote-code to load its official model code")
    settings_path = Path(model) / "vigil_processor_config.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    settings.update(processor_options)
    unknown = settings.keys() - {"max_num", "use_thumbnail"}
    if unknown:
        raise ValueError(f"Unsupported InternVL processor options: {sorted(unknown)}")
    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=True, use_fast=False)
    conversation_factory = get_class_from_dynamic_module("conversation.get_conv_template", model, revision=revision)
    return InternVLProcessor(tokenizer, config, conversation_factory, **settings)
