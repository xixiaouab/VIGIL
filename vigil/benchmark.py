import argparse
import importlib.util
import shlex
import subprocess
import sys

TASKS = {
    "pope": "pope_adv,pope_pop,pope_random",
    "amber": "amber_g",
    "mathvista": "mathvista_testmini",
    "mmbench": "mmbench_en_dev",
    "seedbench": "seedbench",
    "mmlu": "mmlu_generative",
    "gsm8k": "gsm8k",
    "refcocog": "refcocog_bbox_rec_val",
}


def command(checkpoint, tasks, output, trust_remote_code=False):
    selected = []
    for task in tasks.split(","):
        if task not in TASKS:
            raise ValueError(f"Unknown benchmark {task}; choose from {', '.join(TASKS)}")
        selected.extend(TASKS[task].split(","))
    if "," in checkpoint:
        raise ValueError("Checkpoint paths cannot contain commas")
    model_args = f"pretrained={checkpoint}"
    if trust_remote_code:
        model_args += ",trust_remote_code=True"
    return [sys.executable, "-m", "vigil.lmms_adapter", "--model", "vigil",
            "--model_args", model_args,
            "--tasks", ",".join(dict.fromkeys(selected)), "--batch_size", "1",
            "--log_samples", "--output_path", output]


def main():
    parser = argparse.ArgumentParser(description="Launch benchmark evaluators with lmms-eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tasks", default="pope")
    parser.add_argument("--output", required=True)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    try:
        argv = command(args.checkpoint, args.tasks, args.output, args.trust_remote_code)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(argv), flush=True)
    if not args.print_command:
        if importlib.util.find_spec("lmms_eval") is None:
            parser.error("Install lmms-eval in the evaluation environment first")
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
