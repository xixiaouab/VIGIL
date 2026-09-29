# VIGIL

**Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs**

[Paper](https://arxiv.org/abs/2606.26387) · [Project](https://xixiaouab.github.io/VIGIL/)

## Install

Python 3.10+ and NVIDIA GPUs for training.

```bash
git clone https://github.com/xixiaouab/VIGIL.git
cd VIGIL
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

For DeepSpeed or InternVL:

```bash
pip install -e '.[deepspeed,internvl]'
```

## Prepare data

Place images in `data/images/` and preference pairs in `data/raw.jsonl`, one JSON object per line:

```json
{"id":"sample-001","image":"sample.jpg","prompt":"How many cups are visible?","chosen":"Two cups.","rejected":"Three cups.","category":"hallucination"}
```

`category` is `hallucination`, `vqa`, `ocr`, or `math`. `hard_negative`, when present, is used instead of `rejected`.

```bash
python -m vigil.prepare \
  --input data/raw.jsonl --image-root data/images \
  --output data/preferences.jsonl --size 120000 --seed 42
```

The mixture is 45% / 25% / 15% / 15%. Add `--fraction 0.25` for a stratified 30K subset of the same 120K pool. Use `--size` to select a different total.

Generate hard negatives with the base checkpoint:

```bash
python -m vigil.mine \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/preferences.jsonl --output data/train.jsonl
```

## Train

### 1. Cache the frozen reference scores

```bash
torchrun --standalone --nproc_per_node=8 -m vigil.cache_reference \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/train.jsonl --image-root data/images \
  --output data/reference.jsonl --max-length 4096
```

The cache contains chosen/negative seeing scores and chosen blind scores. Keep `reference.jsonl` and `reference.jsonl.meta.json` together. Rebuild the cache after changing the checkpoint, images, responses, or processor settings.

For a reference model spread across multiple GPUs, use one process with `--device-map auto`:

```bash
python -m vigil.cache_reference \
  --model Qwen/Qwen2.5-VL-72B-Instruct --device-map auto \
  --data data/train.jsonl --image-root data/images \
  --output data/reference-72b.jsonl
```

### 2. Start training

```bash
torchrun --standalone --nproc_per_node=8 -m vigil.train \
  --config configs/qwen2_5_vl_7b.json
```

| Configuration | Model | Training |
|---|---|---|
| `configs/qwen2_5_vl_7b.json` | Qwen2.5-VL-7B | FSDP, global batch 1024 |
| `configs/qwen2_5_vl_72b.json` | Qwen2.5-VL-72B | ZeRO-3, global batch 2048 |
| `configs/llava_onevision_7b.json` | LLaVA-OneVision-7B | FSDP, global batch 1024 |
| `configs/internvl2_5_26b.json` | InternVL2.5-26B | ZeRO-3 |
| `configs/qwen2_5_vl_7b_single.json` | Qwen2.5-VL-7B | Single GPU |

Use each model's own reference cache:

```bash
torchrun --standalone --nproc_per_node=8 -m vigil.train \
  --config configs/qwen2_5_vl_72b.json \
  --reference-cache data/reference-72b.jsonl
```

Training defaults: full parameter updates, one epoch, learning rate `5e-7`, cosine decay, warmup `0.03`, DPO `beta=0.1`, grounding weight `1.0`. Gradient accumulation is derived from `global_batch_size`, `micro_batch_size`, and GPU count.

The loss is `DPO + grounding_weight × gate × CVD`. The gate uses the seeing–blind log-probability gap. `gate_reduction="batch"` averages gates within each forward batch; `"sample"` applies them per example. `detach_gate` controls gate gradients. `processor_options` sets image resolution or InternVL tiling; use the same options when caching.

For InternVL reference caching:

```bash
python -m vigil.cache_reference \
  --model OpenGVLab/InternVL2_5-26B --trust-remote-code \
  --attn-implementation eager --device-map auto \
  --data data/train.jsonl --output data/reference-internvl.jsonl

torchrun --standalone --nproc_per_node=8 -m vigil.train \
  --config configs/internvl2_5_26b.json \
  --reference-cache data/reference-internvl.jsonl
```

Resume with `--resume outputs/qwen2_5_vl_7b/checkpoint-<step>`. The exported model and processor are saved under `<output_dir>/final/`.

## Run inference

Input JSONL:

```json
{"id":"sample-001","image":"sample.jpg","prompt":"Describe the image."}
```

```bash
python -m vigil.generate \
  --model outputs/qwen2_5_vl_7b/final \
  --input data/questions.jsonl --image-root data/images \
  --output outputs/predictions.jsonl --max-new-tokens 512
```

Add `--trust-remote-code` for InternVL. Omit `image` for text-only inputs. Each output preserves the input fields and adds `prediction`.

## Evaluate

POPE references contain `id`, `label` (`yes` / `no`), and optional `split`:

```bash
python -m vigil.evaluate --task pope \
  --predictions outputs/predictions.jsonl --references data/pope.jsonl \
  --output outputs/pope-metrics.json
```

RefCOCO references contain `id`, `bbox`, and `bbox_format` (`xyxy` / `xywh`). Predictions contain `[x1, y1, x2, y2]`:

```bash
python -m vigil.evaluate --task refcoco \
  --predictions outputs/predictions.jsonl --references data/refcocog.jsonl \
  --output outputs/refcocog-metrics.json
```

Use `--prediction-scale unit` or `--prediction-scale 1000` for normalized boxes; references then also need `width` and `height`.

For VCC, predictions contain `first_visual_premise`; references contain `image` and `objects`:

```bash
pip install -e '.[judge]'
python -m vigil.evaluate --task vcc --judge-model gpt-4o \
  --predictions outputs/premises.jsonl --references data/vcc.jsonl \
  --output outputs/vcc-metrics.json
```

Set `OPENAI_API_KEY` in your environment for the judge. To score existing `judgment` labels, omit `--judge-model`.

For the benchmark suite, install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) in an evaluation environment and run:

```bash
python -m vigil.benchmark \
  --checkpoint outputs/qwen2_5_vl_7b/final \
  --tasks pope,amber,mathvista,mmbench,seedbench \
  --output outputs/benchmarks
```

Also available: `mmlu`, `gsm8k`, and `refcocog`. Use `--print-command` to inspect the command. MMHal uses the [benchmark's evaluation script](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench/blob/main/eval_gpt4.py).

## Files

| File | Purpose |
|---|---|
| `vigil/prepare.py` | Validate, deduplicate, and mix preference data |
| `vigil/mine.py` | Generate hard negatives |
| `vigil/data.py` | Load pairs and build response-only labels |
| `vigil/processing.py` | Image processing and chat templates |
| `vigil/cache_reference.py` | Cache matched frozen-reference scores |
| `vigil/masking.py` | Construct seeing and blind attention masks |
| `vigil/model.py` | Shared visual encoding and model adapters |
| `vigil/loss.py` | DPO, CVD, dynamic gate, and VIGIL objective |
| `vigil/train.py` | Training, checkpoints, and resume |
| `vigil/generate.py` | Image and text inference |
| `vigil/evaluate.py` | POPE, RefCOCO, and VCC scoring |
| `vigil/benchmark.py` | Benchmark launch commands |
| `vigil/lmms_adapter.py` | Connect VIGIL checkpoints to lmms-eval |
| `configs/` | Model and distributed training settings |
| `tests/` | Loss, masks, data, checkpoints, and pipeline tests |
| `index.html`, `assets/` | Project website |

## Test

```bash
pytest -q
```

To include the InternVL adapter tests, download its Python model definitions:

```bash
pip install -e '.[test,internvl]'
hf download OpenGVLab/InternVL2_5-26B \
  --revision b537a9974b89cb621e9e6b9e7ebe2a904334fff0 \
  --include '*.py' --local-dir /tmp/vigil-internvl-source
touch /tmp/vigil-internvl-source/__init__.py
VIGIL_INTERNVL_SOURCE=/tmp/vigil-internvl-source pytest -q
```
