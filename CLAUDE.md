# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

FreeDave-RL applies GRPO (Group Relative Policy Optimization) to **Diffusion Language Models (DLMs)** — masked-token-prediction models like LLaDA, Dream, TraDo, and SDAR. It combines two upstream projects (see `README.md`): the [diffu-GRPO](https://github.com/dllm-reasoning/d1) trainer and [FreeDave](https://github.com/cychomatica/FreeDave) speculative decoding.

The core insight is adapting policy-gradient training (designed for autoregressive LMs) to the masked-diffusion paradigm by tracking **which diffusion step revealed each token** (the `trajectory_step_map`) and computing per-token log-probabilities against reconstructed partially-masked inputs at each step.

**Status: the RL training loss is a work in progress and not yet correct** (see `README.md` and recent commit messages `trainer loss debug; loss not ready yet`). The FreeDave decoding path and the chat demos are the mature parts of the repo. When touching `compute_loss` / `_get_per_token_logps_from_trajectory`, assume the math is still under active debugging.

## Installation

```bash
pip install -r requirements.txt
```

There is **no** `pyproject.toml`, `setup.py`, or `uv.lock`, and no installable package — modules are imported by path from the repo root, so run all commands from the repository root. `requirements.txt` pins a specific stack: `torch==2.6.0` (CUDA 12.4), `transformers==4.52.4`, `trl==0.19.1`, `peft==0.18.1`, `deepspeed==0.18.8`, and a **cp310 (Python 3.10)** `flash_attn` wheel.

## Running training

**Local (default `run.sh` runs TraDo-4B on 2 GPUs):**
```bash
bash run.sh
```
`run.sh` has two blocks — the active one launches `Gen-Verse/TraDo-4B-Instruct` with `slurm_scripts/accelerate_a5000x2.yaml` (2 processes, DeepSpeed ZeRO-2) and `slurm_scripts/train_trado.yaml`; the commented-out block is the `Dream-v0-Instruct-7B` variant.

**Direct accelerate launch (single GPU config is `accelerate.yaml`):**
```bash
accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port 12346 diffu_grpo_train.py \
    --config slurm_scripts/train_dream.yaml \
    --model_path Dream-org/Dream-v0-Instruct-7B \
    --num_iterations 12 \
    --dataset math \
    --run_name my_run \
    --output_dir checkpoints/my_run
```
Any YAML field can be overridden with a `--key value` flag (parsed by TRL's `TrlParser`).

**Slurm scripts** (`slurm_scripts/*.sbatch`) are legacy templates inherited from the upstream d1 repo — they hardcode a `storygen` SLURM account, an `LLaDA-8B-Instruct` path, and reference `./train.yaml` / `./accelerate_a100.yaml` (which do not exist here). Treat them as references, not runnable as-is.

## Chat demos (the mature, working code path)

```bash
python -m demos.chat_full_attn_dlm      # full-attention decoding
python -m demos.chat_block_attn_dlm     # block-attention + FreeDave speculative decoding
```
Both default to `Gen-Verse/TraDo-4B-Instruct`. Pass `--model_name <hf_id>` to switch models.

## Benchmarking FreeDave (the project's core metric)

`eval/freedave_bench.py` measures the quantities that matter for the fine-tuning goal — **acceptance/TPF (tokens-per-forward), TPS, and task accuracy** — across draft-step settings. `draft_steps=1` is one-token-per-step static decoding (the baseline); higher values are FreeDave. Run from the repo root:

```bash
python -m eval.freedave_bench \
    --model_name Gen-Verse/TraDo-4B-Instruct \
    --dataset math --split test --num_examples 100 \
    --block_length 4 --max_gen_length 256 --draft_steps 1 4 8 \
    --output eval/results/trado4b_math.json
```

Add `--eager` for FreeDave+ (Eager Mode, what the paper uses for TraDo). Note `--draft_mode tree_attention` (default) is incompatible with `flash_attention_2`, so do not force that attn implementation here. FreeDave is lossless w.r.t. static decode, so accuracy should be flat across `draft_steps`; the fine-tuning objective is to raise TPF/TPS at `draft_steps>1` while accuracy stays flat.

## Architecture

### Training pipeline (`diffu_grpo_train.py`)
Entry point. Selects dataset + reward functions from `--dataset`, loads the model (`AutoModelForCausalLM` when `"TraDo"` is in the model path, `AutoModel` otherwise), sets `use_cache=False` on the config, wraps with LoRA (targets all attention + MLP projections), and calls `DiffuGRPOTrainer.train()`. Sets `torch.cuda.set_device(LOCAL_RANK)` so each distributed rank uses its own GPU.

### `DiffuGRPOTrainer` (`diffu_grpo_trainer.py`)
Extends TRL's `GRPOTrainer`. Key overrides:
- **`generate` / `generate_with_trajectory`** — produces completions via the **FreeDave speculative (draft/verify) decode path**, branched by model: `block_decode_with_block_attention_FreeDave` for TraDo, `block_decode_with_full_attention_FreeDave` for Dream/LLaDA. Returns both `sequences` and a `trajectory_step_map` (the diffusion step at which each token was revealed).
- **`compute_loss`** — for the current inner-loop iteration (`_step % num_iterations`) it builds a `delta_mask` selecting only the completion tokens revealed at that trajectory step, computes their per-token log-probs via `_get_per_token_logps_from_trajectory`, and applies the GRPO clipped surrogate. Supports two-sided clipping (`epsilon_low`/`epsilon_high`, optional `delta` upper bound) and an optional KL term (`beta`).
- **`_get_per_token_logps_from_trajectory`** — reconstructs the partially-masked input for a given step (positions with `trajectory_step_map < step_idx` are visible, the rest masked) and reads log-probs at the revealed positions. This is the crux of adapting GRPO to diffusion and is the code most likely to be wrong while the loss is being debugged.
- **`forward_process`** — the masking/noising process used to build perturbed sequences.

### `DiffuGRPOConfig` (`diffu_grpo_config.py`)
Extends `transformers.TrainingArguments` (~500 lines, mostly upstream TRL GRPO fields). DLM-specific fields to know: `mask_id`, `eos_id`, `pad_id`, `block_length`, `diffusion_steps`, `remasking`, `random_masking`, `p_mask_prompt`, `cfg_scale`, `sdpa_additive_attention_mask`, `use_cache`, `dual_cache`. GRPO knobs: `loss_type` (`grpo` / `bnpo` (default) / `dr_grpo`), two-sided clipping (`delta`, `epsilon_high`), `scale_rewards`, `mask_truncated_completions`. `__post_init__` enforces the divisibility relationship between `generation_batch_size`, `per_device_train_batch_size × num_processes`, `steps_per_generation`, and `num_generations` — batch-size errors at startup come from here.

### `generation/` package (`generation/`) — core decoding library
Lives at the repo root (there is no `src/` directory). `generation/generation_core.py` (~1900 lines) holds the `DLMGeneration` class with four decode paths — `block_decode_with_full_attention[_FreeDave]` and `block_decode_with_block_attention[_FreeDave]` — where the `_FreeDave` variants add speculative draft/verify decoding.

- **AR-adapted DLM logit shift** — `is_ar_adapted_dlm_model` walks through PEFT/DeepSpeed/DDP wrappers to detect Dream-style backbones (class names in `AR_ADAPTED_DLM`). For those, `get_processed_model_outputs` shifts logits right by one (`cat([logits[:,:1], logits[:,:-1]])`) so next-token logits align to the current masked position; other DLMs use logits as-is.
- **`visibility_mask_to_sdpa_additive`** — converts 4D visibility masks (1/0) to SDPA additive bias (`0` / `finfo.min`). Used when `sdpa_additive_attention_mask=True`.
- **`cache_utils.py`** — `DynamicDualCache`: a two-stream KV cache for the draft+verify pattern in FreeDave speculative decoding.
- **`sampling_utils.py`** — top-k/top-p/min-p sampling for mask-token replacement.
- **`monitor_utils.py`** — `ForwardMonitor` counts model forward passes during generation (used in the demos to report speculative-decoding speedup).

### Model-specific configuration
Set these to match the backbone (values live in the `slurm_scripts/train_*.yaml` files):

| Model | `mask_id` | `sdpa_additive_attention_mask` | `use_cache` | Notes |
|---|---|---|---|---|
| LLaDA-8B-Instruct | 126336 | `true` | `false` | Standard masked DLM |
| Dream-v0-Instruct-7B | 151666 | `true` | `false` | AR-adapted; needs logit shift |
| TraDo-4B-Instruct | 151669 | `false` | `true` | AR-adapted; block attention; SDARAttention converts masks internally |

### Datasets and reward functions (`data_utils.py`, `reward_func.py`, `math500_utils.py`)

| `--dataset` | HF dataset | Reward functions |
|---|---|---|
| `gsm8k` | `openai/gsm8k` | xmlcount, soft/strict format, int check, correctness |
| `countdown` | `Jiayi-Pan/Countdown-Tasks-3to4` | countdown expression eval |
| `sudoku` | local CSV `../dataset/4x4_sudoku_unique_puzzles.csv` (relative to cwd; not in the repo — supply it) | sudoku correctness |
| `math` | `ankner/math-500` | correctness (LaTeX equiv), boxed + `<answer>`-tag format |
| `code` | `KodCode/KodCode-Light-RL-10K` | xmlcount, execute-and-test |

Each dataset function returns a `prompt` column (list of chat messages) plus task-specific answer columns passed as kwargs to the reward functions. `math500_utils.py` provides the MATH answer-equivalence machinery (`is_equiv`, `strip_string`, `remove_boxed`, `last_boxed_only_string`) used by the math reward. The `code` reward executes model-generated code against tests — it runs untrusted code in a subprocess, so be cautious when exercising that path locally.

## Config YAML structure

Training configs (`slurm_scripts/train_dream.yaml`, `train_llada.yaml`, `train_trado.yaml`) combine `ModelConfig` fields (LoRA settings, dtype, attn implementation) and `DiffuGRPOConfig`/`TrainingArguments` fields in one file, parsed together via `TrlParser((DiffuGRPOConfig, ModelConfig))`. Command-line `--key value` overrides any YAML field.
