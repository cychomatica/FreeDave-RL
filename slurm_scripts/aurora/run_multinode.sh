#!/bin/bash -l
# ====================================================================
# Multi-node TraDo-4B GRPO on Aurora — DeepSpeed ZeRO-3 + XCCL
#
# STATUS: BLOCKED on a libfabric Cassini fabric bug (see README.md).
# Reaches the first cross-node DeepSpeed ZeRO-3 all_gather_into_tensor
# during checkpoint loading and crashes with:
#     cxil_map: write error  (many)
#     RuntimeError: oneCCL: allgatherv_ring.hpp:78
#       allgatherv_ring_blocking: ... fails with status: 1
# Reproducible across CCL_KVS_MODE=mpi/pmi, FI_MR_CACHE_MONITOR=
# userfaultfd/memhooks, with/without scale-tuning flags, and on both
# debug and gpu_hack PBS queues.
#
# Single-node up to 6 tiles works (use run_2tiles_single_node.sh).
# 12 tiles single-node OOMs at checkpoint load (12 * 9GB = 108GB host
# RAM materialized before ZeRO-3 partitioning).
#
# This script is the recipe-of-record for when the Aurora fabric issue
# is resolved.  Submit via PBS (qsub) on the debug or gpu_hack queue.
# ====================================================================
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
set -e
unset -f __conda_hashr 2>/dev/null || true

WORK=${AURORA_WORK_DIR:-/lus/flare/projects/Aurora_deployment/$USER/freedave-rl}
SENTINEL_DIR=$WORK/.multinode_sentinels
mkdir -p $SENTINEL_DIR
SENTINEL=$SENTINEL_DIR/done_${PBS_JOBID:-unknown}

# ---- head/non-head pattern (PBS pbsdsh launches run.sh on every node;
#      only head sees PBS_NODEFILE) ----
if [ -z "${PBS_NODEFILE:-}" ]; then
    echo "[non-head $(hostname)] waiting on sentinel"
    START=$(date +%s); MAX_WAIT=3300
    while [ ! -f $SENTINEL ]; do
        sleep 20
        ELAPSED=$(( $(date +%s) - START ))
        if [ $ELAPSED -gt $MAX_WAIT ]; then echo "[non-head] timeout"; exit 1; fi
    done
    echo "[non-head] done after $(( $(date +%s) - START ))s"
    exit 0
fi
trap 'touch $SENTINEL; echo "[head] wrote sentinel"' EXIT

REPO_DIR=${REPO_DIR:-$WORK/FreeDave-RL}
MODEL_DIR=${MODEL_DIR:-$WORK/models/TraDo-4B-Instruct}
RUN_DIR=$WORK/runs/multinode_${PBS_JOBID:-$(date +%s)}
mkdir -p "$RUN_DIR"

NNODES=$(wc -l < $PBS_NODEFILE)
PPN=${PPN:-2}
NRANKS=$((NNODES * PPN))

echo "============================================================"
echo "  multinode TraDo-4B GRPO: ${NNODES} nodes x ${PPN} tiles = ${NRANKS} ranks"
echo "  Head: $(hostname)   PBS: ${PBS_JOBID}"
cat $PBS_NODEFILE | sed 's/^/    /'
echo "============================================================"

module use /soft/modulefiles 2>/dev/null || true
module load frameworks/2025.2.0 2>/dev/null || true
[ -n "$AURORA_VENV" ] && source "$AURORA_VENV/bin/activate"

# libmpi.so.12 visibility
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

# Set MASTER_ADDR/PORT in parent shell so children inherit (PBS_NODEFILE
# does NOT propagate through mpiexec).
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | awk -F. '{print $1}')
export MASTER_PORT=${MASTER_PORT:-29500}
# Pass total world_size; PALS_RANKID is global rank, but PMI_SIZE may
# report per-node count -- launch_per_rank.sh prefers GLOBAL_WORLD_SIZE.
export GLOBAL_WORLD_SIZE=$NRANKS

# Aurora oneCCL essentials per ALCF docs
export PALS_PMI=pmix
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=cxi

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
d["run_name"] = "trado_aurora_multinode"
d["output_dir"] = "$RUN_DIR/checkpoints"
d["bf16"] = True
d["use_vllm"] = False
d["max_completion_length"] = 128
d["max_prompt_length"] = 128
d["block_length"] = 32
d["diffusion_steps"] = 64
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY

cp $(dirname $0)/ds_config_zero3.json $RUN_DIR/ds_config_zero3.json
LAUNCH=$(dirname $0)/launch_per_rank.sh

echo "=== mpiexec $NRANKS ranks (${NNODES}x${PPN}) MASTER=$MASTER_ADDR ==="
mpiexec -n $NRANKS --ppn $PPN $LAUNCH \
    python diffu_grpo_train.py \
    --config slurm_scripts/train_trado.yaml \
    --model_path "$MODEL_DIR" \
    --num_iterations 2 --dataset sudoku \
    --run_name trado_aurora_multinode \
    --output_dir "$RUN_DIR/checkpoints" \
    --deepspeed "$RUN_DIR/ds_config_zero3.json" \
    2>&1 | tee "$RUN_DIR/train.log"

EXIT=${PIPESTATUS[0]}
echo "=== exit: $EXIT ==="
ls -la "$RUN_DIR" || true
exit $EXIT
