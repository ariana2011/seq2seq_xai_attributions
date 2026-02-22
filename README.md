# WMT-XAI: Attribution-Guided Neural Machine Translation

This repository provides code for computing source-attribution explanations from pretrained translation models and for fine-tuning translation models augmented with those attribution signals.

## Overview

The workflow has two main stages:

1. **Generate attributions** (`src/explain.py`) — run an XAI method (saliency, DeepLIFT, attention, etc.) over a translation corpus using [inseq](https://github.com/inseq-team/inseq) and save the resulting attribution matrices as JSONL files.
2. **Attribution-guided fine-tuning** (`src/hf_train.py`) — fine-tune a MarianMT or MBart-50 model using the HuggingFace `Seq2SeqTrainer`, optionally injecting the pre-computed attribution matrices into the model.

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `transformers`, `datasets`, `inseq`, `evaluate`, `torch`, `scikit-learn`, `sacrebleu`.

## Directory Structure

```
data/                        # Raw translation corpora (JSON/JSONL)
attributions/                # Generated attribution files (output of explain.py)
  <lang-pair>/
    <model>/
      <xi_method>/
        train/   validation/   test/
checkpoints/                 # Model checkpoints
src/
  explain.py                 # Attribution generation script
  hf_train.py                # Fine-tuning script
  model.py                   # Custom MarianMT model
  model_config.py            # Custom model configuration
  data_loader.py             # Dataset utilities
  attributor/                # Attribution injection modules
```

---

## 1. Generating Attributions (`explain.py`)

This script runs an XAI attribution method over a translation dataset and writes one JSONL file per `--max_num` samples.

### Arguments

| Argument | Short | Required | Description |
|---|---|---|---|
| `--xi_method` | | yes | Attribution method (see below) |
| `--data` | | yes | Path to the input JSON file under `data/` |
| `--data_name` | `-dn` | yes | Split name: `train`, `test`, or `validation` |
| `--data_folder` | `-df` | yes | Language-pair folder, e.g. `de-en`, `fr-en` |
| `--model` | | yes | `Marian` or `MBart` |
| `--input_lang` | | MBart only | Source language code, e.g. `de_DE` |
| `--output_lang` | | MBart only | Target language code, e.g. `en_XX` |
| `--max_num` | `-mxn` | | Number of samples to process (0 = all) |
| `--sample_num` | `-sn` | | Starting sample index (default: 0) |
| `--batch_size` | `-bs` | | Batch size (default: 20) |
| `--json_num` | `-jn` | | Output file index counter (default: 0) |
| `--matrix_op` | | | Aggregation over head dimension: `mean` or `l2_norm` (default: `mean`) |
| `--scaler` | | | Post-processing: `no-scaler`, `minmax`, `softmax`, `one_hot` (default: `no-scaler`) |
| `--generate_dest` | `-gd` | | Flag — generate target text instead of using gold references |
| `--resume` | | | Flag — skip already-written samples (uses `--resume_key` to match) |
| `--resume_key` | | | JSON key used for deduplication when resuming (default: `idx`) |



### Usage examples

**MarianMT — saliency on de-en training set:**
```bash
python src/explain.py \
  --model Marian \
  --xi_method saliency \
  --data de-en/train_filtered_full.json \
  --data_name train \
  --data_folder de-en \
  --max_num 200000
```



**MBart-50 — value zeroing on de-en:**
```bash
python src/explain.py \
  --model MBart \
  --xi_method value_zeroing \
  --data de-en/train_data.json \
  --data_name train \
  --data_folder de-en \
  --input_lang de_DE \
  --output_lang en_XX
```

**Resume an interrupted run:**
```bash
python src/explain.py \
  --model Marian \
  --xi_method saliency \
  --data de-en/train_filtered_full.json \
  --data_name train \
  --data_folder de-en \
  --matrix_op l2_norm \
  --max_num 200000 \
  --resume
```

### Output format

Each line in the output JSONL file is a JSON object:

```json
{
  "de": "<source sentence>",
  "en": "<target sentence>",
  "idx": 42,
  "shape_src": [15, 12],
  "attribution": [0.12, 0.03, "..."]
}
```

`attribution` is a flattened float array of shape `shape_src[0] x shape_src[1]` (target tokens x source tokens).
Output files are written to `attributions/<lang-pair>/<model>/<xi_method>/<data_name>/`.

---

## 2. Fine-tuning with Attribution Injection (`hf_train.py`)

This script wraps HuggingFace `Seq2SeqTrainer` to fine-tune a MarianMT or MBart-50 model, with optional attribution injection.

### Key arguments

All standard HuggingFace `Seq2SeqTrainingArguments` are supported. The most commonly used arguments are listed below.

**Data arguments:**

| Argument | Description |
|---|---|
| `--source_lang` | Source language code, e.g. `de` or `de_DE` for MBart |
| `--target_lang` | Target language code, e.g. `en` or `en_XX` for MBart |
| `--dataset_name` | Dataset variant (see below) |
| `--max_source_length` | Max tokenized source length (default: 1024) |
| `--max_target_length` | Max tokenized target length (default: 128) |
| `--metrics` | Comma-separated metrics: `sacrebleu,chrf,ter,rouge` (default: `sacrebleu`) |

**Dataset name options:**

| Value | Description |
|---|---|
| `custom` | Standard parallel data from `data/<src>-<tgt>/` |
| `custom100k` | Attribution-free data from `attributions/.../without_attr/` |
| `attribution_<method>` | Attribution-injected data, e.g. `attribution_saliency`, `attribution_deeplift` |

**Model arguments:**

| Argument | Description |
|---|---|
| `--model_name_or_path` | HuggingFace model ID or local path |
| `--num_heads_attr` | Number of attention heads to inject attribution into (default: 6) |
| `--operator` | How attribution is combined with attention: `addition`, `replace`, `multiply`, `average` |
| `--enc_att_op` | Flag — apply attribution injection to encoder self-attention |
| `--cross_att_op` | Flag — apply attribution injection to cross-attention |
| `--scaler` | Scale attributions before injection: `minmax`, `one_hot`, `softmax`, `random`, `one_diagonal` |
| `--approx_attributions_model` | Path to a model that predicts attributions on-the-fly instead of using dataset |
| `--generated` | Flag — use model-generated (rather than gold) target sequences in attribution data |

### Usage examples

**Baseline fine-tuning (no attribution) on de-en:**
```bash
python src/hf_train.py \
  --model_name_or_path Helsinki-NLP/opus-mt-de-en \
  --do_train \
  --do_eval \
  --source_lang de \
  --target_lang en \
  --dataset_name custom100k \
  --output_dir checkpoints/baseline_de_en \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --overwrite_output_dir \
  --predict_with_generate \
  --max_source_length 128 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --load_best_model_at_end True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --num_train_epochs 20
```

**Attribution-guided fine-tuning with saliency on de-en:**
```bash
python src/hf_train.py \
  --model_name_or_path Helsinki-NLP/opus-mt-de-en \
  --do_train \
  --do_eval \
  --source_lang de \
  --target_lang en \
  --dataset_name attribution_saliency \
  --output_dir checkpoints/saliency_de_en \
  --per_device_train_batch_size 40 \
  --per_device_eval_batch_size 40 \
  --overwrite_output_dir \
  --predict_with_generate \
  --max_source_length 128 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --load_best_model_at_end True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --num_train_epochs 20 \
  --num_heads_attr 8 \
  --operator addition \
  --enc_att_op
```


**Evaluation / prediction only:**
```bash
python src/hf_train.py \
  --model_name_or_path checkpoints/saliency_de_en \
  --do_predict \
  --source_lang de \
  --target_lang en \
  --dataset_name attribution_saliency \
  --output_dir checkpoints/saliency_de_en \
  --predict_with_generate \
  --max_source_length 128 \
  --max_target_length 128 \
  --per_device_eval_batch_size 40
```


### Supported language pairs

The custom model configuration supports: `de-en`, `ar-en`, `fr-en`, `en-zh`, `es-it`, `en-da`, and MBart-50 multilingual models.

---


# Attributor

A lightweight neural network that **learns to approximate XAI attribution matrices** produced by [inseq](https://inseq.org/). Once trained, it can replace expensive XAI computations (saliency, DeepLIFT, attention, integrated gradients, value zeroing, etc.) with a fast forward pass.

---

## Overview

`TargetSourceAttributor` is a small encoder–decoder transformer that takes tokenised source and target sentences and outputs a soft alignment matrix of shape `(target_len, source_len)`. It is trained by minimising KL divergence against attribution matrices previously computed by `explain.py`.

```
Source tokens ──► Transformer Encoder ──┐
                                         ├──► Cross-Attention ──► Head-Gate MLP ──► Attribution matrix
Target tokens ──► Transformer Decoder ──┘
```

---

## Prerequisites

Install the project dependencies from the root:

```bash
pip install -r requirements.txt
```

The attributor additionally requires `scipy` and `seaborn` for training diagnostics:

```bash
pip install scipy seaborn
```

---

## Directory layout

```
attributor/
├── NeuralBlocks.py            # Building blocks: TransformerBlock, CrossAttention, TAConfig, …
├── NeuralModule.py            # Base class with save_model / load_model helpers
├── TargetSourceAttributor.py  # Model definition
├── train.py                   # Training script
├── slurm_adel.py              # SLURM job generator (Marian tokeniser)
├── slurm_adel_mbart.py        # SLURM job generator (MBart tokeniser)
└── *.pt                       # Saved model checkpoints
```

---

## Step 1 – Generate attribution data

Before training the attributor you need JSONL attribution files produced by `explain.py` (see `src/README.md`). Each line must contain at minimum:

| Field | Description |
|---|---|
| `<src_lang>` | Source sentence string |
| `<tgt_lang>` | Target sentence string |
| `attribution` | Flat list of floats (the attribution matrix) |
| `shape_src` | `[source_len, target_len]` – shape used to reshape `attribution` |
| `idx` | Integer sample index |

---

## Step 2 – Train

Run `train.py` from inside the `attributor/` directory:

```bash
cd attributor/

python train.py \
  --model TargetSourceAttributor \
  --pair de-en \
  --tokenizer marian \
  --dataset_path /path/to/attributions/de-en/Marian/saliency/train/ \
  --run_name saliency \
  --n_batch 64 \
  --epochs 20
```

For MBart models swap `--tokenizer marian` for `--tokenizer mbart`.

### All training arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `TargetSourceAttributor` | Model class to use |
| `--pair` | *(required)* | Language pair, e.g. `de-en`, `fr-en`, `ar-en` |
| `--tokenizer` | `marian` | Tokeniser type: `marian` or `mbart` |
| `--dataset_path` | — | Path to a directory containing attribution JSONL files |
| `--run_name` | `""` | Tag appended to the saved checkpoint name |
| `--n_window` | `128` | Maximum sequence length (source and target) |
| `--n_embed` | `512` | Embedding / hidden dimension |
| `--n_heads` | `8` | Number of attention heads |
| `--n_vocabulary` | `59514` | Vocabulary size (overridden automatically from tokeniser) |
| `--n_batch` | `128` | Batch size |
| `--epochs` | `20` | Maximum training epochs |
| `--split_ratio` | `90/5/5` | Train / val / test split percentages |
| `--patience` | `3` | Early-stopping patience (epochs without val improvement) |
| `--from_pretrained` | flag | Resume from the checkpoint matching the run name |
| `--eval` | flag | Skip training and evaluate the existing checkpoint on the test set |
| `--generated` | `human` | Label for the data source (used in run naming) |

### Checkpoint naming

The saved checkpoint follows the pattern:

```
{model}_{run_name}_{pair}_{tokenizer}_OVERKILL.pt
```

For example: `TargetSourceAttributor_saliency_de-en_marian_OVERKILL.pt`

Checkpoints are saved to `../../models/` relative to `train.py` (i.e. `wmt-xai/models/`).

### Training outputs

| File | Description |
|---|---|
| `train_losses_<name>.txt` | Per-step training loss |
| `val_losses_<name>.txt` | Per-step validation loss |
| `val_loss_epochs_<name>.csv` | Per-epoch validation loss |
| `stats_<name>.json` | Final test-set metrics (KL divergence, top-k overlap, Kendall-τ, Frobenius norm) |

---

## Step 3 – Evaluate an existing checkpoint

```bash
python train.py \
  --model TargetSourceAttributor \
  --pair de-en \
  --tokenizer marian \
  --dataset_path /path/to/attributions/de-en/Marian/saliency/train/ \
  --run_name saliency \
  --from_pretrained \
  --eval
```

Reported metrics:

- **KL divergence** (mean, median, std) – primary training objective
- **Top-k overlap** (k = 3) – fraction of top-3 source tokens that match between prediction and ground truth
- **Kendall-τ** – rank correlation at the top-k positions
- **Frobenius norm** – element-wise difference between the two matrices

---

## Step 4 – Inference (Python API)

```python
from attributor.NeuralModule import NeuralModule

device = "cuda"
model = NeuralModule.load_model(
    name="TargetSourceAttributor_saliency_de-en_marian_OVERKILL",
    device=device,
    path="models/"          # root-relative path to the .pt file
)

# Inputs: tokenised tensors of shape (batch, seq_len), values are token ids
attribution_matrix = model(
    target,        # (B, T) – decoder token ids
    source,        # (B, S) – encoder token ids
    target_mask,   # (B, T) – 1 for real tokens, 0 for padding
    source_mask,   # (B, S) – 1 for real tokens, 0 for padding
)
# attribution_matrix: (B, T, S) – soft alignment, rows sum to 1
```

The returned tensor rows are probability distributions over source positions for each target token.

---


## License

Apache 2.0 — see the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for details.
