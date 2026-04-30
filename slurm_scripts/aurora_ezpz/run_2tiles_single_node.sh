#!/bin/bash -l
# ====================================================================
# WORKING reference: TraDo-4B GRPO on Aurora using ezpz launcher
# 1 node x 2 XPU tiles, ZeRO-3 + native PyTorch XCCL.
#
# This variant uses Sam Foreman's `ezpz` (https://github.com/saforem2/ezpz)
# for distributed setup and launching, instead of:
#   - the buggy `accelerate launch` on Aurora frameworks/2025.2.0
#     (`AttributeError: args.ipex`)
#   - manual PALS env-var marshalling via launch_per_rank.sh
#
# ezpz auto-detects the launcher (PALS, Slurm, mpiexec, local) and
# exports RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT
# correctly without per-rank wrappers.
# ====================================================================
set -e
unset -f __conda_hashr 2>/dev/null || true

WORK=${AURORA_WORK_DIR:-/lus/flare/projects/Aurora_deployment/$USER/freedave-rl}
REPO_DIR=${REPO_DIR:-$WORK/FreeDave-RL}
MODEL_DIR=${MODEL_DIR:-$WORK/models/TraDo-4B-Instruct}
RUN_DIR=$WORK/runs/ezpz_2tiles_$(date +%s)
mkdir -p "$RUN_DIR"

module use /soft/modulefiles 2>/dev/null || true
module load frameworks/2025.2.0 2>/dev/null || true
[ -n "$AURORA_VENV" ] && source "$AURORA_VENV/bin/activate"

# Install ezpz on first run (idempotent; harmless if already installed)
# Use `python -m pip` (bare `pip` shebang in Aurora ClearML agent venv is broken).
# DO NOT pass --user (rejected because dill is already in the venv site-packages).
# DO NOT pass --no-build-isolation (hatchling not pre-installed; isolation fetches it).
python -c "import ezpz" 2>/dev/null || python -m pip install 'ezpz @ git+https://github.com/saforem2/ezpz'

# libmpi.so.12 visibility (so CCL uses MPI transport, not silent OFI fallback)
LIBMPI_DIR=""
for d in "$I_MPI_ROOT/lib/release" "$I_MPI_ROOT/lib" \
         /opt/aurora/*/oneapi/mpi/latest/lib/release \
         /opt/aurora/*/oneapi/mpi/*/lib/release; do
    [ -e "$d/libmpi.so.12" ] && LIBMPI_DIR="$d" && break
done
if [ -z "$LIBMPI_DIR" ]; then
    F=$(find /opt -maxdepth 9 -name 'libmpi.so.12' 2>/dev/null | head -n 1)
    [ -n "$F" ] && LIBMPI_DIR=$(dirname "$F")
fi
[ -n "$LIBMPI_DIR" ] && export LD_LIBRARY_PATH="$LIBMPI_DIR:$LD_LIBRARY_PATH"

unset ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR

# Aurora oneCCL essentials per ALCF docs
export PALS_PMI=pmix
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=cxi

# ezpz reads NGPUS / NGPU_PER_HOST so it sets PPN automatically
export NGPU_PER_HOST=${PPN:-2}

export WANDB_MODE=offline
export WANDB_DIR=$RUN_DIR/wandb
export HF_HOME=$WORK/hf_cache
export TRANSFORMERS_CACHE=$WORK/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_USE_FSDP=false
export ACCELERATE_USE_IPEX=false

cd "$REPO_DIR"
git checkout -- diffu_grpo_train.py

# Inject Aurora compatibility patches (ezpz variant) at top of train script
python - <<'PY'
import pathlib
p = pathlib.Path("diffu_grpo_train.py")
src = p.read_text()
inject = (
    "try:\n"
    "    from slurm_scripts.aurora_ezpz.aurora_patch import apply_aurora_patches\n"
    "    apply_aurora_patches()\n"
    "except ImportError:\n"
    "    pass\n\n"
)
if "aurora_ezpz" not in src:
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i
            break
    p.write_text("".join(lines[:insert_at]) + inject + "".join(lines[insert_at:]))
PY

# Patch YAML for an Aurora-friendly smoke
NRANKS=${NRANKS:-2}
cp slurm_scripts/train_trado.yaml $RUN_DIR/train_trado.yaml.orig
python - <<PY
import pathlib, yaml
p = pathlib.Path("slurm_scripts/train_trado.yaml")
d = yaml.safe_load(p.read_text())
d["attn_implementation"] = "sdpa"
d["load_in_4bit"] = False
d["model_path"] = "$MODEL_DIR"
d["per_device_train_batch_size"] = 1
d["per_device_eval_batch_size"] = 1
d["num_generations"] = 2
d["gradient_accumulation_steps"] = 1
d["generation_batch_size"] = $NRANKS
d["num_train_epochs"] = 1
d["max_steps"] = 2
d["save_steps"] = 1000
d["save_total_limit"] = 1
d["dataset"] = "sudoku"
d["run_name"] = "trado_aurora_ezpz_2tiles"
d["output_dir"] = "$RUN_DIR/checkpoints"
d["bf16"] = True
d["use_vllm"] = False
d["max_completion_length"] = 128
d["max_prompt_length"] = 128
d["block_length"] = 32
d["diffusion_steps"] = 64
# Skip HF Trainer auto-init of clearml (no creds in user code, would crash run)
d["report_to"] = ["wandb"]
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY

cp $(dirname $0)/ds_config_zero3.json $RUN_DIR/ds_config_zero3.json

# === ezpz launcher ===
# `ezpz launch` handles MPI init, RANK/WORLD_SIZE detection, and exports
# torch.distributed env vars before the user script runs. No
# launch_per_rank.sh wrapper needed.
echo "=== ezpz launch (NGPU_PER_HOST=$NGPU_PER_HOST) ==="
ezpz launch \
    python diffu_grpo_train.py \
    --config slurm_scripts/train_trado.yaml \
    --model_path "$MODEL_DIR" \
    --num_iterations 2 --dataset sudoku \
    --run_name trado_aurora_ezpz_2tiles \
    --output_dir "$RUN_DIR/checkpoints" \
    --deepspeed "$RUN_DIR/ds_config_zero3.json" \
    2>&1 | tee "$RUN_DIR/train.log"

EXIT=${PIPESTATUS[0]}
echo "=== exit: $EXIT ==="
ls -la "$RUN_DIR" || true
exit $EXIT
