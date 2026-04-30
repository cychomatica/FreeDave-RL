#!/bin/bash -l
# ====================================================================
# Multi-node TraDo-4B GRPO on Aurora using ezpz launcher
#
# Same as the slurm_scripts/aurora/ multinode recipe, but with `ezpz
# launch` replacing the manual mpiexec + launch_per_rank.sh + per-rank
# env-var marshalling.  Submit via PBS on the debug or gpu_hack queue.
#
# STATUS: same Aurora libfabric Cassini bug as the non-ezpz variant.
# See ../aurora/ISSUES.md for full bug record.
# ====================================================================
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q debug
set -e
unset -f __conda_hashr 2>/dev/null || true

WORK=${AURORA_WORK_DIR:-/lus/flare/projects/Aurora_deployment/$USER/freedave-rl}
SENTINEL_DIR=$WORK/.multinode_ezpz_sentinels
mkdir -p $SENTINEL_DIR
SENTINEL=$SENTINEL_DIR/done_${PBS_JOBID:-unknown}

# Head/non-head pattern (PBS pbsdsh launches script on every node;
# only head sees PBS_NODEFILE).  ezpz on the head will spawn the
# remote ranks via mpiexec under the hood, so non-head workers wait.
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
RUN_DIR=$WORK/runs/ezpz_multinode_${PBS_JOBID:-$(date +%s)}
mkdir -p "$RUN_DIR"

NNODES=$(wc -l < $PBS_NODEFILE)
PPN=${PPN:-2}
NRANKS=$((NNODES * PPN))

echo "============================================================"
echo "  ezpz multinode TraDo-4B GRPO: ${NNODES}x${PPN}=${NRANKS} ranks"
echo "  Head: $(hostname)   PBS: ${PBS_JOBID}"
cat $PBS_NODEFILE | sed 's/^/    /'
echo "============================================================"

module use /soft/modulefiles 2>/dev/null || true
module load frameworks/2025.2.0 2>/dev/null || true
[ -n "$AURORA_VENV" ] && source "$AURORA_VENV/bin/activate"

python -c "import ezpz" 2>/dev/null || python -m pip install --user 'ezpz @ git+https://github.com/saforem2/ezpz'

LIBMPI_DIR=""
for d in "$I_MPI_ROOT/lib/release" "$I_MPI_ROOT/lib" \
         /opt/aurora/*/oneapi/mpi/latest/lib/release \
         /opt/aurora/*/oneapi/mpi/*/lib/release; do
    [ -e "$d/libmpi.so.12" ] && LIBMPI_DIR="$d" && break
done
[ -z "$LIBMPI_DIR" ] && LIBMPI_DIR=$(dirname $(find /opt -maxdepth 9 -name 'libmpi.so.12' 2>/dev/null | head -n 1))
[ -n "$LIBMPI_DIR" ] && export LD_LIBRARY_PATH="$LIBMPI_DIR:$LD_LIBRARY_PATH"

unset ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR

# ezpz reads NGPU_PER_HOST and PBS_NODEFILE to determine topology
export NGPU_PER_HOST=$PPN

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
d["run_name"] = "trado_aurora_ezpz_multinode"
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

echo "=== ezpz launch (NGPU_PER_HOST=$NGPU_PER_HOST, NNODES=$NNODES, NRANKS=$NRANKS) ==="
ezpz launch \
    python diffu_grpo_train.py \
    --config slurm_scripts/train_trado.yaml \
    --model_path "$MODEL_DIR" \
    --num_iterations 2 --dataset sudoku \
    --run_name trado_aurora_ezpz_multinode \
    --output_dir "$RUN_DIR/checkpoints" \
    --deepspeed "$RUN_DIR/ds_config_zero3.json" \
    2>&1 | tee "$RUN_DIR/train.log"

EXIT=${PIPESTATUS[0]}
echo "=== exit: $EXIT ==="
ls -la "$RUN_DIR" || true
exit $EXIT
