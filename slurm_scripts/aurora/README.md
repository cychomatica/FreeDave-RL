# FreeDave-RL on Aurora (Intel XPU)

Port of TraDo-4B GRPO training from CUDA to Aurora's Intel XPU stack
(Intel Data Center GPU Max 1550, oneAPI/PyTorch 2.8/IPEX 2.8.10/native
XCCL backend) under `frameworks/2025.2.0`.

## Status

| Configuration | Result |
|---|---|
| 1 XPU tile | ✅ Trains end-to-end |
| 2 tiles, 1 node | ✅ 2 GRPO steps in ~33s, ZeRO-3 + XCCL |
| 6 tiles, 1 node | ✅ Full single-node training |
| 12 tiles, 1 node | ❌ Host-RAM OOM at checkpoint load (12 × ~9 GB shard materialized to RAM before ZeRO-3 partitioning) |
| ≥2 nodes | ❌ Blocked: `cxil_map: write error` + `oneCCL: allgatherv_ring` at first cross-node `dist.all_gather_into_tensor` during checkpoint load |

## Files

| File | Purpose |
|---|---|
| `aurora_patch.py` | The five XPU compatibility monkey-patches (mpi4py init, transformers `LossKwargs` stub, `flash_attn` stub, `peft.lora.inc` no-op, `torch.cuda → torch.xpu` aliases, per-rank `xpu.set_device`) |
| `launch_per_rank.sh` | PALS rank wrapper: maps `PALS_RANKID/GLOBAL_WORLD_SIZE/PALS_LOCAL_RANKID` → `RANK/WORLD_SIZE/LOCAL_RANK` |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 config (auto bucket sizes, no offload) |
| `run_2tiles_single_node.sh` | **Working** single-node 2-tile recipe |
| `run_multinode.sh` | Multi-node recipe (BLOCKED on Aurora fabric bug — see below) |

## How to use

The training entry-point `diffu_grpo_train.py` is unmodified in this
branch. The Aurora run scripts inject a 4-line conditional import of
`aurora_patch.apply_aurora_patches()` at the top of the file before
`mpiexec`:

```python
try:
    from slurm_scripts.aurora.aurora_patch import apply_aurora_patches
    apply_aurora_patches()
except ImportError:
    pass
```

This is a no-op on CUDA — the patches short-circuit when `torch.xpu`
isn't present.

### Quick start (single node, 2 tiles)

```bash
qsub -I -A <account> -q debug -l select=1 -l walltime=01:00:00 \
     -l filesystems=home:flare
# on the compute node:
export AURORA_WORK_DIR=/lus/flare/projects/<project>/$USER/freedave-rl
export REPO_DIR=$AURORA_WORK_DIR/FreeDave-RL
export MODEL_DIR=$AURORA_WORK_DIR/models/TraDo-4B-Instruct
bash slurm_scripts/aurora/run_2tiles_single_node.sh
```

### Multi-node (currently broken)

```bash
qsub slurm_scripts/aurora/run_multinode.sh -A <account> \
     -l select=2 -l walltime=01:00:00
```

Crashes at first cross-node DeepSpeed ZeRO-3 weight all-gather. See
[Open issue](#open-issue-multi-node-cassini-fabric) below.

## Aurora oneCCL env recipe

Per the [ALCF oneCCL docs](https://docs.alcf.anl.gov/aurora/data-science/frameworks/oneCCL/):

```bash
export PALS_PMI=pmix
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export FI_MR_CACHE_MONITOR=userfaultfd
export CCL_ZE_IPC_EXCHANGE=sockets
export FI_PROVIDER=cxi
# Make Intel MPI libmpi.so.12 visible so CCL uses MPI transport
# (not silent OFI fallback):
export LD_LIBRARY_PATH=/opt/aurora/<ver>/oneapi/mpi/<ver>/lib:$LD_LIBRARY_PATH
```

### Why each patch

1. **mpi4py init first** — `CCL_KVS_MODE=mpi` requires `MPI_Init()` to
   have been called before any CCL collective. Otherwise: `Attempting
   to use an MPI routine before initializing or after finalizing MPICH`
2. **flash_attn stub** — flash-attn is CUDA-only. Newer transformers
   import `flash_attn.ops.triton.layer_norm.rms_norm_fn` at import time
   even when `attn_implementation="sdpa"`. The stub provides an XPU
   fallback `rms_norm_fn`.
3. **peft.lora.inc no-op** — peft 0.17 unconditionally imports the
   Habana Intel Neural Compressor `dispatch_inc` path on XPU; we no-op
   it and stub `neural_compressor.torch` submodules.
4. **torch.cuda → torch.xpu aliases** — TRL/DeepSpeed call
   `torch.cuda.{empty_cache,synchronize,...}` and `torch.cuda.amp.autocast`
   unconditionally; we redirect to the XPU equivalents.
5. **`torch.xpu.set_device(LOCAL_RANK)` per rank** — required before
   any other XPU allocation so each rank binds to its own tile.

### Why explicit `RANK/WORLD_SIZE` mapping

On Aurora's PALS launcher, `PMI_RANK` / `PMI_SIZE` are EMPTY for
multi-node `mpiexec`. Only `PALS_RANKID` (global rank) and
`PALS_LOCAL_RANKID` are populated. We export
`GLOBAL_WORLD_SIZE=$NRANKS` from the parent shell so it propagates
to all child processes.

## Open issue: multi-node Cassini fabric

Symptom (from `run_multinode.sh`):

```
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
cxil_map: write error
cxil_map: write error
... (many)
[rank0]: RuntimeError: oneCCL: allgatherv_ring.hpp:78
  allgatherv_ring_blocking: EXCEPTION:
  atl_comm->recv(ep_idx, recv_ptr, recv_block_size, left, tag, recv_req)
  fails with status: 1
```

Reproducible at:
- 2 nodes × 1 tile (smallest possible inter-node test)
- 2 nodes × 2 tiles
- 2 nodes × 6 tiles
- both `debug` and `gpu_hack` PBS queues
- both `FI_MR_CACHE_MONITOR=userfaultfd` and `=memhooks`
- with and without ALCF scale-tuning flags
  (`CCL_ALLREDUCE_SCALEOUT`, `CCL_BCAST=double_tree`, `CCL_KVS_USE_MPI_RANKS=1`)

Always crashes at the very first cross-node `dist.all_gather_into_tensor`
call inside DeepSpeed ZeRO-3's `_load_state_dict_into_zero3_model →
GatheredParameters → _allgather_params`. The single-node case (any
tile count up to 6) works because no cross-node collectives happen.

A standalone reproducer that strips DeepSpeed/transformers/FreeDave-RL
and just calls `dist.all_gather_into_tensor` after `mpi4py.MPI` init +
`init_process_group(backend="xccl")` hangs at the very first 1KB
allgather (no `cxil_map` printed) — suggesting the bug is in the
XCCL/libfabric path, not in DeepSpeed's pattern.

## Tested versions

```
frameworks/2025.2.0
  PyTorch 2.8.0a0+gitba56102 (XPU build)
  IPEX 2.8.10+git09505bb
  oneCCL via torch.distributed native XCCL backend (no oneccl_bindings_for_pytorch)
  transformers 4.56.1
  trl 0.19.1
  deepspeed 0.17.5
  peft 0.17.1
  mpi4py 4.1.0
  numpy 2.0.2
```
