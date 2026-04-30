#!/bin/bash -l
# ====================================================================
# WORKING reference: TraDo-4B GRPO on Aurora, 1 node x 2 XPU tiles
#
# Confirmed: 2 GRPO steps in ~33s on Aurora compute, full ZeRO-3 +
# native PyTorch XCCL backend.  Comet ML + wandb both write offline runs.
# Checkpoints saved to OUTPUT_DIR.
#
# Usage (interactive on a compute node):
#   qsub -I -A <account> -q debug -l select=1 -l walltime=01:00:00 \
#        -l filesystems=home:flare
#   bash run_2tiles_single_node.sh
#
# Usage (under PBS as a batch job): see slurm_scripts/aurora/run_multinode.sh
# (which subsumes this single-node case via NNODES/PPN).
# ====================================================================
set -e
unset -f __conda_hashr 2>/dev/null || true

# ---- paths (edit for your project) ----
WORK=${AURORA_WORK_DIR:-/lus/flare/projects/Aurora_deployment/$USER/freedave-rl}
REPO_DIR=${REPO_DIR:-$WORK/FreeDave-RL}
MODEL_DIR=${MODEL_DIR:-$WORK/models/TraDo-4B-Instruct}
RUN_DIR=$WORK/runs/single_node_2tiles_$(date +%s)
mkdir -p "$RUN_DIR"

# ---- modules + venv ----
module use /soft/modulefiles 2>/dev/null || true
module load frameworks/2025.2.0 2>/dev/null || true
# Activate a venv that includes mpi4py, deepspeed 0.17.5, trl 0.19.1,
# transformers 4.56.1, peft 0.17.1.  The Aurora frameworks venv already
# has all of these.
[ -n "$AURORA_VENV" ] && source "$AURORA_VENV/bin/activate"

# ---- libmpi.so.12 visibility (so CCL uses MPI transport, not OFI fallback) ----
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

# ---- XPU tile visibility (these env vars hide xpu from PyTorch when set) ----
unset ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR

# ---- topology ----
NRANKS=${NRANKS:-2}
PPN=${PPN:-2}
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}
export GLOBAL_WORLD_SIZE=$NRANKS

# ---- Aurora oneCCL essentials (per https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/) ----
export PALS_PMI=pmix
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=cxi

# ---- HF + WANDB (offline) ----
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

# ---- Inject Aurora compatibility patches into diffu_grpo_train.py ----
git checkout -- diffu_grpo_train.py
python - <<'PY'
import pathlib
p = pathlib.Path("diffu_grpo_train.py")
src = p.read_text()
inject = (
    "try:\n"
    "    from slurm_scripts.aurora.aurora_patch import apply_aurora_patches\n"
    "    apply_aurora_patches()\n"
    "except ImportError:\n"
    "    pass\n\n"
)
if "aurora_patch" not in src:
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i
            break
    p.write_text("".join(lines[:insert_at]) + inject + "".join(lines[insert_at:]))
PY

# ---- Patch the YAML config for an Aurora-friendly smoke test ----
cp slurm_scripts/train_trado.yaml $RUN_DIR/train_trado.yaml.orig
python - <<PY
import pathlib, yaml
p = pathlib.Path("slurm_scripts/train_trado.yaml")
d = yaml.safe_load(p.read_text())
d["attn_implementation"] = "sdpa"   # flash_attn not on XPU
d["load_in_4bit"] = False           # bitsandbytes 8-bit not on XPU
d["model_path"] = "$MODEL_DIR"
d["per_device_train_batch_size"] = 1
d["per_device_eval_batch_size"] = 1
d["num_generations"] = 2
d["gradient_accumulation_steps"] = 1
d["generation_batch_size"] = $NRANKS  # TRL GRPO requires this multiple
d["num_train_epochs"] = 1
d["max_steps"] = 2
d["save_steps"] = 1000
d["save_total_limit"] = 1
d["dataset"] = "sudoku"
d["run_name"] = "trado_aurora_2tiles_smoke"
d["output_dir"] = "$RUN_DIR/checkpoints"
d["bf16"] = True
d["use_vllm"] = False
d["max_completion_length"] = 128
d["max_prompt_length"] = 128
d["block_length"] = 32
d["diffusion_steps"] = 64
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY

# ---- DeepSpeed ZeRO-3 config ----
cp $(dirname $0)/ds_config_zero3.json $RUN_DIR/ds_config_zero3.json

# ---- Launch ----
LAUNCH=$(dirname $0)/launch_per_rank.sh
echo "=== mpiexec $NRANKS ranks (single-node), MASTER=$MASTER_ADDR ==="
mpiexec -n $NRANKS --ppn $PPN $LAUNCH \
    python diffu_grpo_train.py \
    --config slurm_scripts/train_trado.yaml \
    --model_path "$MODEL_DIR" \
    --num_iterations 2 --dataset sudoku \
    --run_name trado_aurora_2tiles_smoke \
    --output_dir "$RUN_DIR/checkpoints" \
    --deepspeed "$RUN_DIR/ds_config_zero3.json" \
    2>&1 | tee "$RUN_DIR/train.log"

EXIT=${PIPESTATUS[0]}
echo "=== exit: $EXIT ==="
ls -la "$RUN_DIR" || true
exit $EXIT
