# FreeDave-RL on Aurora — multi-node validated

Working multi-node TraDo-4B GRPO recipe for Aurora XPU. The single-node
2-tile recipe uses `ezpz launch`; the multi-node recipe uses direct
`mpiexec` (with `ezpz.setup_torch()` still called from user code for the
torch.distributed bootstrap) plus the CXI workarounds that dodge the
"cxil_map: write error" Cassini bug.

## Status (2026-04-30)

| Scale | Recipe | Status |
|---|---|---|
| 1 node × 2 tiles | `ezpz launch` + `ezpz.setup_torch()` | ✅ trains end-to-end (max_steps=2 reached) |
| **2 nodes × 2 tiles** | **direct mpiexec + OFI ATL + CXI workarounds + report_to=wandb** | ✅ **trains end-to-end** (max_steps=2, train_loss=0.0004) |

## The cxil_map saga (short version)

The `mpiexec` + `CCL_ATL_TRANSPORT=mpi` path that works for the single-node
2-tile run dies cross-node with `cxil_map: write error` followed by
`oneCCL allgatherv_ring EXCEPTION: atl_comm->recv fails with status: 1`
at the first cross-node ZeRO-3 `all_gather_into_tensor`. We tried 7+
variants (PMIx vs PMI-1, ezpz launch vs direct mpiexec, security plugin
toggles) — all hit the same bug as long as ATL=mpi.

The workaround is **OFI ATL + CXI knobs**:

```bash
export CCL_ATL_TRANSPORT=ofi          # KEY: bypass MPI ATL (cxil_map path)
export CCL_KVS_MODE=pmi               # KEY: required by oneCCL when ATL=ofi
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_HOST_REGISTER=1 # KEY: dodge Cassini host-register path
export FI_CXI_OPTIMIZED_MRS=0         # KEY
export FI_MR_CACHE_MAX_COUNT=0        # KEY
```

With these, all 4 ranks complete the cross-node ZeRO-3 model load and the
trainer reaches `step 2/2` with real loss values.

## Files

| File | Purpose |
|---|---|
| `aurora_patch.py` | XPU-compat shims (transformers `LossKwargs`, flash_attn stub, peft.lora.inc no-op, torch.cuda→torch.xpu aliases) plus `ezpz.setup_torch()` |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 config |
| `run_2tiles_single_node.sh` | Single-node 2-tile recipe via `ezpz launch` |
| `run_multinode.sh` | **Multi-node v8g recipe via direct mpiexec + OFI/CXI workarounds** |

## How to use

### Single node (2 tiles)

```bash
qsub -I -A <account> -q gpu_hack -l select=1 -l walltime=01:00:00 \
     -l filesystems=home:flare
export AURORA_WORK_DIR=/lus/flare/projects/<project>/$USER/freedave-rl
export REPO_DIR=$AURORA_WORK_DIR/FreeDave-RL
export MODEL_DIR=$AURORA_WORK_DIR/models/TraDo-4B-Instruct
bash slurm_scripts/aurora_ezpz/run_2tiles_single_node.sh
```

### Multi node (2 nodes × 2 tiles, 4 ranks)

```bash
qsub slurm_scripts/aurora_ezpz/run_multinode.sh -A <account> \
     -l select=2 -l walltime=01:00:00
```

The scripts auto-install `ezpz` from GitHub on first run if it isn't
already importable.

## Multi-node recipe details (v8g)

Why direct mpiexec instead of `ezpz launch`:

- `ezpz launch` works on a single node but on Aurora gpu_hack its
  `mpiexec` invocation goes through PMIx and hits `PMIX_Init returned -25`
  (PMIX_ERR_UNREACH) cross-node.
- Direct `mpiexec --hostfile $PBS_NODEFILE` with `PALS_PMI=pmix` works.
- `ezpz.setup_torch()` is still called from user code — it handles the
  Python-side `torch.distributed.init_process_group` with the XCCL
  backend on XPU. So we keep ezpz's nice rank discovery and just bypass
  its launcher.

Why `report_to=["wandb"]` instead of the trainer default:

- Inside the Aurora ClearML agent venv, HF Trainer auto-detects the
  `clearml` package and tries to spawn a Trainer-side `Task.init()` from
  the user code. The user code has no ClearML credentials → trainer
  callback raises `MissingConfigError` and crashes the run after the
  model has loaded. Setting `report_to=["wandb"]` (or `["tensorboard"]`)
  in the train YAML skips the clearml callback entirely.

## Install gotchas

```bash
# Right way to install ezpz on Aurora ClearML agent venv:
python -m pip install 'ezpz @ git+https://github.com/saforem2/ezpz'
```

- Use `python -m pip` — the bare `pip` shebang in the agent venv is broken.
- Do NOT pass `--user` — `dill` is already in the venv site-packages and
  pip refuses `--user` when a transitive dep would not take precedence.
- Do NOT pass `--no-build-isolation` — `hatchling` (ezpz's PEP 517
  backend) is not pre-installed; isolation lets pip fetch it.

## Historical context

See `../aurora/ISSUES.md` for the full debug trail of the cxil_map bug
across 7+ launcher / transport / PMI variants. The TL;DR: it's an Aurora
libfabric Cassini provider bug in the MPI-ATL code path; the OFI-ATL
path with the CXI knobs above sidesteps it.
