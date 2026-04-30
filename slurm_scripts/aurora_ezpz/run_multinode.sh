#!/bin/bash -l
# ====================================================================
# WORKING multi-node TraDo-4B GRPO on Aurora (validated 2-node x 2-tile)
#
# This is the v8g recipe that finally trains end-to-end across nodes
# on Aurora gpu_hack.  Key fixes (vs the original ezpz launch path):
#
#   1. Use direct mpiexec (NOT `ezpz launch`).  ezpz works on a single
#      node but on Aurora gpu_hack its MPI_Init goes through PMIx and
#      hits PMIX_ERR_UNREACH cross-node.  We still use ezpz.setup_torch()
#      from user code for the Python-side torch.distributed bootstrap.
#
#   2. CCL_ATL_TRANSPORT=ofi (not mpi).  The MPI ATL path triggers the
#      Aurora libfabric "cxil_map: write error" bug at the first
#      cross-node oneCCL allgather.  OFI ATL bypasses it.
#
#   3. CCL_KVS_MODE=pmi (not mpi).  Required by oneCCL when ATL=ofi.
#
#   4. CXI provider workarounds:
#      - FI_CXI_DISABLE_HOST_REGISTER=1
#      - FI_CXI_OPTIMIZED_MRS=0
#      - FI_MR_CACHE_MAX_COUNT=0
#      These together make the Cassini provider stop trying the code
#      path that produces cxil_map errors.
#
#   5. report_to=["wandb"] in the train YAML.  Inside a ClearML agent
#      venv, HF Trainer auto-detects clearml and tries Task.init() with
#      no creds in the user code -> MissingConfigError.  Limit
#      report_to to wandb (or tensorboard) to skip that callback.
#
# Validated on Aurora 2026-04-30: 2 nodes x 2 tiles = 4 ranks,
# DeepSpeed ZeRO-3, max_steps=2 reached with real loss values.
#
# See ../aurora/ISSUES.md for the historical cxil_map debugging trail.
# ====================================================================
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -q gpu_hack
set -e
unset -f __conda_hashr 2>/dev/null || true

WORK=${AURORA_WORK_DIR:-/lus/flare/projects/Aurora_deployment/$USER/freedave-rl}
SENTINEL_DIR=$WORK/.multinode_ezpz_sentinels
mkdir -p $SENTINEL_DIR
SENTINEL=$SENTINEL_DIR/done_${PBS_JOBID:-unknown}

# Head/non-head pattern (PBS pbsdsh launches script on every node;
# only head sees PBS_NODEFILE).  Direct mpiexec on the head spawns the
# remote ranks; non-head workers wait on a sentinel file.
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
echo "  Aurora multinode TraDo-4B GRPO (working recipe v8g)"
echo "  ${NNODES}x${PPN}=${NRANKS} ranks   Head: $(hostname)"
echo "  PBS: ${PBS_JOBID}"
cat $PBS_NODEFILE | sed 's/^/    /'
echo "============================================================"

module use /soft/modulefiles 2>/dev/null || true
module load frameworks/2025.2.0 2>/dev/null || true
[ -n "$AURORA_VENV" ] && source "$AURORA_VENV/bin/activate"

# ezpz still used in user code for torch.distributed bootstrap, but NOT
# as the launcher.  Install if missing (NOT --user; --no-build-isolation
# fails because hatchling isn't pre-installed).
python -c "import ezpz" 2>/dev/null || python -m pip install 'ezpz @ git+https://github.com/saforem2/ezpz'

# libmpi.so.12 visibility for oneCCL OFI/MPI bootstrap.
LIBMPI_DIR=""
for d in "$I_MPI_ROOT/lib/release" "$I_MPI_ROOT/lib" \
         /opt/aurora/*/oneapi/mpi/latest/lib/release \
         /opt/aurora/*/oneapi/mpi/*/lib/release; do
    [ -e "$d/libmpi.so.12" ] && LIBMPI_DIR="$d" && break
done
[ -z "$LIBMPI_DIR" ] && LIBMPI_DIR=$(dirname $(find /opt -maxdepth 9 -name 'libmpi.so.12' 2>/dev/null | head -n 1))
[ -n "$LIBMPI_DIR" ] && export LD_LIBRARY_PATH="$LIBMPI_DIR:$LD_LIBRARY_PATH"

unset ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR

# ---- Working v8g multinode env ----
export NGPU_PER_HOST=$PPN
export PALS_PMI=pmix
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=ofi          # KEY: bypass MPI ATL (cxil_map path)
export CCL_KVS_MODE=pmi               # KEY: required when ATL=ofi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=cxi
# CXI provider workarounds for cxil_map: write error -- THE actual fix
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=0
export FI_MR_CACHE_MAX_COUNT=0
# torch.distributed master endpoint (head node)
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500

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
# v8g: skip HF Trainer auto-init of clearml (no creds in user code)
d["report_to"] = ["wandb"]
p.write_text(yaml.safe_dump(d, sort_keys=False))
PY

cp $(dirname $0)/ds_config_zero3.json $RUN_DIR/ds_config_zero3.json

echo "=== direct mpiexec (NRANKS=$NRANKS, MASTER_ADDR=$MASTER_ADDR, ATL=$CCL_ATL_TRANSPORT) ==="
mpiexec -n $NRANKS --ppn $PPN --hostfile $PBS_NODEFILE \
    --cpu-bind list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100 \
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
