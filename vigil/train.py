import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, Trainer, TrainingArguments, set_seed

from .cache_reference import (
    ReferenceCollator, ReferenceDataset, build_cache_metadata, load_reference_cache,
)
from .data import PreferenceCollator, PreferenceDataset
from .loss import vigil_loss
from .model import VIGILModel
from .processing import load_processor


class VIGILTrainer(Trainer):
    def __init__(self, *args, beta=0.1, grounding_weight=1.0,
                 gate_reduction="batch", detach_gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.grounding_weight = grounding_weight
        self.gate_reduction = gate_reduction
        self.detach_gate = detach_gate
        self.model_accepts_loss_kwargs = False
        self.vigil_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        references = inputs.pop("reference")
        outputs = model(inputs)
        loss, metrics = vigil_loss(
            outputs["chosen_see"], outputs["rejected_see"], outputs["chosen_blind"],
            references["ref_chosen_see"], references["ref_rejected_see"],
            references["ref_chosen_blind"], beta=self.beta,
            grounding_weight=self.grounding_weight, gate_reduction=self.gate_reduction,
            detach_gate=self.detach_gate,
        )
        self.vigil_metrics = {k: float(v.detach().float().mean()) if torch.is_tensor(v)
                              else float(v) for k, v in metrics.items()}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        super().log({**logs, **self.vigil_metrics}, start_time)

    def _save(self, output_dir=None, state_dict=None):
        target = output_dir or self.args.output_dir
        Path(target).mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        weights = state_dict if state_dict is not None else model.state_dict()
        weights = {key.removeprefix("model."): value for key, value in weights.items()}
        model.model.save_pretrained(target, state_dict=weights, safe_serialization=True)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(target)
        torch.save(self.args, Path(target) / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        wrapper = self.accelerator.unwrap_model(model or self.model)
        return super()._load_from_checkpoint(resume_from_checkpoint, model=wrapper.model)


def make_training_arguments(config, world_size=1):
    if config.get("dtype", "bfloat16") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    batch = int(config.get("micro_batch_size", 1))
    global_batch = int(config.get("global_batch_size", 1024))
    divisor = batch * world_size
    if batch < 1 or global_batch < divisor or global_batch % divisor:
        raise ValueError("global_batch_size must be divisible by micro_batch_size × WORLD_SIZE")
    options = dict(
        output_dir=config["output_dir"], num_train_epochs=1,
        per_device_train_batch_size=batch, gradient_accumulation_steps=global_batch // divisor,
        learning_rate=5e-7, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.0, max_grad_norm=1.0, optim="adamw_torch",
        bf16=config.get("dtype", "bfloat16") == "bfloat16",
        fp16=config.get("dtype") == "float16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, label_names=[], report_to=[], seed=config.get("seed", 42),
        dataloader_num_workers=0, logging_steps=1, save_strategy="epoch",
        save_total_limit=2, ddp_find_unused_parameters=False,
    )
    options.update(config.get("training", {}))
    if config.get("fsdp"):
        options.update(fsdp="full_shard auto_wrap", fsdp_config=config["fsdp"])
    if config.get("deepspeed"):
        options["deepspeed"] = config["deepspeed"]
    if options.get("fsdp") and options.get("deepspeed"):
        raise ValueError("Choose FSDP or DeepSpeed for one training run")
    return TrainingArguments(**options)


def main():
    parser = argparse.ArgumentParser(description="Train VIGIL from paired preferences")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--reference-cache")
    parser.add_argument("--image-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("data", "reference_cache", "image_root", "output_dir"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in ("model", "data", "reference_cache", "output_dir"):
        if not config.get(key):
            parser.error(f"Missing {key} in config or CLI")
    set_seed(config.get("seed", 42))
    training_args = make_training_arguments(config, int(os.environ.get("WORLD_SIZE", 1)))
    model_name = config["model"]
    reference_name = config.get("reference_model", model_name)
    revision = config.get("revision")
    trust_remote_code = config.get("trust_remote_code", False)
    reference_config = AutoConfig.from_pretrained(reference_name, revision=revision,
                                                  trust_remote_code=trust_remote_code)
    resolved_revision = getattr(reference_config, "_commit_hash", None) or revision
    processor = load_processor(reference_name, revision=resolved_revision,
                               trust_remote_code=trust_remote_code,
                               **config.get("processor_options", {}))
    dataset = PreferenceDataset(config["data"], image_root=config.get("image_root"))
    maximum = config.get("max_length", 4096)
    attention = config.get("attn_implementation", "sdpa")
    metadata = build_cache_metadata(dataset, model_name=reference_name, revision=revision,
                                    processor=processor, max_length=maximum,
                                    model_config=reference_config,
                                    dtype=config.get("dtype", "bfloat16"),
                                    attn_implementation=attention)
    scores = load_reference_cache(config["reference_cache"], metadata)
    train_dataset = ReferenceDataset(dataset, scores)
    collator = ReferenceCollator(PreferenceCollator(processor, max_length=maximum,
                                                    image_root=config.get("image_root")))
    model = VIGILModel.from_pretrained(
        model_name, revision=resolved_revision if model_name == reference_name else revision,
        dtype=config.get("dtype", "bfloat16"), attn_implementation=attention,
        trust_remote_code=trust_remote_code,
    )
    model.requires_grad_(True)
    trainer = VIGILTrainer(
        model=model, args=training_args, train_dataset=train_dataset, data_collator=collator,
        processing_class=processor, beta=config.get("beta", 0.1),
        grounding_weight=config.get("grounding_weight", 1.0),
        gate_reduction=config.get("gate_reduction", "batch"),
        detach_gate=config.get("detach_gate", False),
    )
    if trainer.is_world_process_zero():
        target = Path(config["output_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=args.resume)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(Path(config["output_dir"]) / "final"))
    trainer.save_state()


if __name__ == "__main__":
    main()
