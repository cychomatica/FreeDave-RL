#!/bin/bash -l
# ---------------------------------------------------------------------------
# Run on a Polaris LOGIN node (login nodes have direct internet).
# Pre-downloads the model + datasets into the eagle HF cache so the debug
# job runs reliably without depending on compute-node networking.
#   bash eval/polaris/prefetch.sh
# ---------------------------------------------------------------------------
set -e
# Login-node OpenBLAS tries to spawn one thread per core and trips RLIMIT_NPROC,
# which breaks even `import numpy`. Pin thread counts low for the import to work.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EAGLE=/eagle/lighthouse-purdue/michaelholm/freedave_bench
export HF_HOME=$EAGLE/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

module use /soft/modulefiles
module load conda
conda activate grpo-train

python - <<'PY'
from huggingface_hub import snapshot_download
from datasets import load_dataset
print("downloading TraDo-4B-Instruct ...")
snapshot_download("Gen-Verse/TraDo-4B-Instruct")
print("downloading math-500[test] ...")
load_dataset("ankner/math-500", split="test")
print("downloading gsm8k[test] (for later) ...")
load_dataset("openai/gsm8k", "main")
print("PREFETCH OK")
PY
