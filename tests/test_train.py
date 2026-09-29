import copy

import pytest
import torch
from transformers import TrainingArguments

from test_core import tiny_model
from vigil.model import VIGILModel
from vigil.train import VIGILTrainer, make_training_arguments


def test_effective_batch_is_preserved(tmp_path):
    args = make_training_arguments({"output_dir": str(tmp_path), "dtype": "float32",
                                    "global_batch_size": 1024, "micro_batch_size": 2}, 8)
    assert args.gradient_accumulation_steps == 64
    assert args.num_train_epochs == 1
    assert args.learning_rate == 5e-7
    with pytest.raises(ValueError, match="divisible"):
        make_training_arguments({"output_dir": str(tmp_path), "global_batch_size": 7}, 2)


def test_optimizer_save_reload_and_resume(tmp_path):
    model, batch = tiny_model("qwen")
    with torch.no_grad():
        refs = {"ref_" + key: value for key, value in model(batch).items()}
    data = [copy.deepcopy(batch) for _ in range(2)]

    def collate(rows):
        assert len(rows) == 1
        return {**copy.deepcopy(rows[0]), "reference": refs}

    def make_trainer(model, target, steps):
        return VIGILTrainer(
            model=model, train_dataset=data, data_collator=collate,
            args=TrainingArguments(output_dir=str(target), use_cpu=True, max_steps=steps,
                                   per_device_train_batch_size=1, learning_rate=1e-3,
                                   gradient_checkpointing=True,
                                   gradient_checkpointing_kwargs={"use_reentrant": False},
                                   remove_unused_columns=False, report_to=[], save_steps=1,
                                   logging_steps=1, label_names=[]),
        )

    before = model.model.lm_head.weight.detach().clone()
    trainer = make_trainer(model, tmp_path / "run", 1)
    trainer.train()
    assert not torch.equal(before, model.model.lm_head.weight)
    checkpoint = tmp_path / "run" / "checkpoint-1"
    reloaded = VIGILModel.from_pretrained(checkpoint, dtype="float32")
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        for key, values in model(batch).items():
            torch.testing.assert_close(values, reloaded(batch)[key])
    trainer = make_trainer(reloaded, tmp_path / "resume", 2)
    trainer.train(resume_from_checkpoint=str(checkpoint))
    assert trainer.state.global_step == 2
