# Open Issues — FreeDave-RL on Aurora

Tracking issues encountered while porting TraDo-4B GRPO training to
Intel Aurora XPU. File a corresponding GitHub Issue (and an ALCF
helpdesk ticket) for each one.

---

## #1 — `cxil_map: write error` + `oneCCL: allgatherv_ring` failure on first cross-node `dist.all_gather_into_tensor` (multi-node)

**Severity:** Blocks all multi-node training.
**Status:** Open. Reproducible across every env permutation tried.
**Layer:** Aurora libfabric Cassini provider / oneCCL.

### Symptom

In `run_multinode.sh` (ZeRO-3 + XCCL, 2 nodes × 2 tiles = 4 ranks),
training reaches the first cross-node DeepSpeed all-gather during
checkpoint shard load and crashes:

```
[2026-04-30 09:14:35,415] [INFO] [TorchCheckpointEngine] Initialized
[r0] xpu device 0, tile_count=12, WORLD_SIZE=4
[r0] init_process_group(xccl) OK
[r0] first barrier OK
...
finished initializing model - num_params = 399, num_elems = 4.41B
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
cxil_map: write error
cxil_map: write error
... (many)
Loading checkpoint shards:   0%|          | 0/2 [00:03<?, ?it/s]
[rank2]: Traceback (most recent call last):
[rank2]:   File ".../diffu_grpo_train.py", line 264, in <module>
[rank2]:     main(...)
[rank2]:   File ".../transformers/integrations/deepspeed.py", line 341, in load
[rank2]:     with deepspeed.zero.GatheredParameters(...):
[rank2]:   File ".../deepspeed/runtime/zero/partition_parameters.py", line 2012,
[rank2]:     dist.all_gather_into_tensor(flat_tensor, ...)
[rank2]:   File ".../torch/distributed/distributed_c10d.py", line 3986,
[rank2]:     work = group._allgather_base(output_tensor, input_tensor, opts)
[rank2]: RuntimeError: oneCCL: allgatherv_ring.hpp:78 allgatherv_ring_blocking:
[rank2]:  EXCEPTION: atl_comm->recv(ep_idx, recv_ptr, recv_block_size, left,
[rank2]:                            tag, recv_req)  fails with status: 1
```

### What works (single-node)

- 1 XPU tile: trains
- 2 / 6 tiles, 1 node: trains (2 GRPO steps in 33s for 2 tiles)
- 12 tiles, 1 node: separate problem — host RAM OOM at checkpoint load
  (12 × ~9 GB shard materialized to RAM before ZeRO-3 partitioning)

### Tested permutations (all fail identically)

| Variable | Values tried |
|---|---|
| Topology | 2×1, 2×2, 2×6 (ranks per node) |
| `FI_MR_CACHE_MONITOR` | `userfaultfd`, `memhooks` |
| `CCL_ATL_TRANSPORT` | `mpi` (with libmpi.so.12 visible), default |
| `CCL_KVS_MODE` | `mpi` (with `mpi4py.MPI` init), default |
| Scale-tuning flags | with / without `CCL_ALLREDUCE_SCALEOUT`, `CCL_BCAST=double_tree`, `CCL_KVS_USE_MPI_RANKS=1`, `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`, `CCL_KVS_CONNECTION_TIMEOUT`, `CCL_CONFIGURATION` |
| PBS queue | `debug`, `gpu_hack` |

The crash signature is invariant.

### Standalone reproducer

A minimal reproducer (no DeepSpeed, no transformers, no FreeDave-RL)
that does:

```python
from mpi4py import MPI                      # initializes MPI
import torch
torch.xpu.set_device(LOCAL_RANK)
import torch.distributed as dist
dist.init_process_group(backend="xccl",
                        rank=RANK, world_size=WORLD_SIZE)
dist.barrier()                              # OK
shard = torch.full((1024,), float(RANK), dtype=torch.bfloat16,
                   device=f"xpu:{LOCAL_RANK}")
out = torch.empty((1024 * WORLD_SIZE,), dtype=torch.bfloat16,
                  device=f"xpu:{LOCAL_RANK}")
dist.all_gather_into_tensor(out, shard)    # HANGS forever (no error printed)
```

…hangs indefinitely on the first 1KB allgather (no `cxil_map` error
printed, no traceback). This isolates the bug to the bare XCCL +
libfabric Cassini path; DeepSpeed's pattern is not the trigger.

### Environment

```
frameworks/2025.2.0
  PyTorch 2.8.0a0+gitba56102 (XPU)
  IPEX 2.8.10+git09505bb
  torch.distributed XCCL backend (native, PyTorch 2.8)
  mpi4py 4.1.0 (Intel MPI 2021.16, /opt/aurora/25.190.0/oneapi/mpi/2021.16)
  Cray PALS 1.8 (mpiexec at /opt/cray/pals/1.8/bin/mpiexec)
```

### What the env block looks like at crash time

```bash
PALS_PMI=pmix
CCL_PROCESS_LAUNCHER=pmix
CCL_ATL_TRANSPORT=mpi
CCL_KVS_MODE=mpi
FI_MR_CACHE_MONITOR=userfaultfd
CCL_ZE_IPC_EXCHANGE=sockets
FI_PROVIDER=cxi
LD_LIBRARY_PATH=/opt/aurora/25.190.0/oneapi/mpi/2021.16/lib:$LD_LIBRARY_PATH
```

Plus PyTorch distributed env from PALS (`PALS_RANKID` → `RANK`,
`GLOBAL_WORLD_SIZE` → `WORLD_SIZE`, `PALS_LOCAL_RANKID` → `LOCAL_RANK`).

### Workaround attempts that did NOT help

- Switch `FI_MR_CACHE_MONITOR=userfaultfd` → `memhooks`
- Drop all scale-tuning flags (essentials only)
- Switch from `CCL_ATL_TRANSPORT=mpi` to `ofi`
- Try gpu_hack PBS queue instead of debug

### Possible next steps

1. ALCF helpdesk ticket with the standalone reproducer
2. Pre-shard the checkpoint to avoid the cross-node param all-gather
   during loading
3. Try a different `FI_PROVIDER` (`tcp` would be slow but might work)
4. Try setting `FI_MR_CACHE_MAX_COUNT=0` to disable the libfabric MR
   cache entirely

---

## #2 — 12 tiles single-node host-RAM OOM at checkpoint load

**Severity:** Blocks full single-node training.
**Status:** Open. Workaround: use ≤6 tiles per node.
**Layer:** transformers/DeepSpeed checkpoint loading, not Aurora-specific.

### Symptom

`run_multinode.sh` with `NNODES=1`, `PPN=12`: ZeRO-3 partition
succeeds (`finished initializing model — num_params = 399, num_elems
= 4.41B`), then rank 0 SIGKILL at `Loading checkpoint shards: 0%`.

### Root cause

Each rank loads the full ~9 GB checkpoint shard set to host RAM
before ZeRO-3 partitions parameters to XPU. With 12 ranks per node
and 12 × 9 = 108 GB needed, the node's ~512 GB physical RAM is
exhausted by the time the OS account for kernel + agent overhead.

### Possible fixes

- `low_cpu_mem_usage=True` on `from_pretrained` (tried — did not help)
- Use `HfDeepSpeedConfig` BEFORE `from_pretrained` so ZeRO-3-aware
  loading skips rank-0 materialization
- Pre-shard the checkpoint and load each shard only on the ranks
  that need its parameters

---

## #3 — `accelerate launch` fails with `AttributeError: args.ipex`

**Severity:** Cosmetic — workaround in place.
**Status:** Closed by workaround.
**Layer:** Bundled `accelerate==1.10.1` in frameworks/2025.2.0.

### Symptom

`accelerate launch ...` raises `AttributeError: 'Namespace' object
has no attribute 'ipex'` immediately.

### Workaround

Skip `accelerate launch`. Use `mpiexec -n $NRANKS python <script>`
directly, with `launch_per_rank.sh` mapping PALS env vars to
`torch.distributed`'s expected env vars.
