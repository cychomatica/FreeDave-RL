#!/bin/bash -l
# ---------------------------------------------------------------------------
# Build a Polaris-correct environment for the FreeDave benchmark, the ALCF way:
#   ALCF base conda (torch built for Polaris' CUDA/driver) + a venv on top that
#   layers ONLY the extra deps the repo needs. Never pip-install torch here --
#   ALCF advises against it, and a pip torch pulls a CUDA build the driver can't
#   run (that is what broke the grpo-train env: torch 2.11+cu130 on a 12.8 driver).
#
# Run on a LOGIN node:  bash eval/polaris/setup_env.sh
# ---------------------------------------------------------------------------
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

module use /soft/modulefiles
module load conda
conda activate base

VENV=/eagle/lighthouse-purdue/michaelholm/freedave_bench/venv
if [ ! -d "$VENV" ]; then
    echo "creating venv on top of base ($CONDA_PREFIX) ..."
    python -m venv "$VENV" --system-site-packages
fi
source "$VENV/bin/activate"

# Extra deps not in the ALCF base env (benchmark/inference only; no torch).
#  - Deprecated: needed by generation/generation_core.py
#  - transformers==4.52.4: TraDo's remote modeling_sdar.py imports LossKwargs,
#    which base transformers 4.53.3 no longer exposes. Pin to the repo version.
python -m pip install --no-input --quiet "Deprecated==1.3.1" "transformers==4.52.4"

echo "----------------------------------------------------------------------"
echo "python : $(which python)"
python -c "import torch; print('torch  :', torch.__version__, '| build CUDA', torch.version.cuda)"
echo "venv   : $VENV"
echo "----------------------------------------------------------------------"
echo "import test (repo benchmark chain):"
cd /home/michaelholm/FreeDave-RL
python - <<'PY'
import importlib
mods = ["torch","transformers","datasets","numpy","deprecated",
        "reward_func","math500_utils","data_utils",
        "generation.monitor_utils","generation.generation_core"]
bad = 0
for m in mods:
    try:
        importlib.import_module(m); print("  OK  ", m)
    except Exception as e:
        bad += 1; print("  FAIL", m, "::", type(e).__name__, e)
print("IMPORT_TEST", "PASS" if bad == 0 else f"FAIL ({bad})")
PY
