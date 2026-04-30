# FreeDave-RL on Aurora — ezpz launcher variant

Same FreeDave-RL Aurora XPU port as
[`slurm_scripts/aurora/`](../aurora/), but using
[ezpz](https://github.com/saforem2/ezpz) (Sam Foreman's ALCF distributed
launcher) instead of:

- bare `mpiexec` + `launch_per_rank.sh` env-var marshalling
- the buggy `accelerate launch` on Aurora frameworks/2025.2.0
  (`AttributeError: args.ipex`)

ezpz auto-detects the launcher (PALS, Slurm, mpiexec, local, etc.),
inits MPI, and exports the correct
`RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT` before
the user code runs — no per-rank wrapper needed.

## Status

Single-node 2 tiles trains end-to-end (max_steps=2 reached, real loss
values logged), **but** ezpz.setup_torch() on Aurora returns
`world_size=1` per rank — each rank thinks it is the sole rank and runs
standalone training (loads full 4.41B model, no XCCL gradient sync).
True multi-rank distributed via this path is not validated yet.

For a validated multi-tile distributed run, use the non-ezpz variant
(`../aurora/`) which uses `mpiexec` + a per-rank wrapper + explicit
`mpi4py.MPI` import; that recipe trains on up to 6 tiles end-to-end.

Multi-node still hits the Aurora libfabric Cassini bug independent of
launcher choice — see `../aurora/ISSUES.md`.

## Files

| File | Purpose |
|---|---|
| `aurora_patch.py` | Same XPU compat patches as the non-ezpz variant, MINUS the explicit `mpi4py.MPI` import (ezpz handles that) |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 config |
| `run_2tiles_single_node.sh` | Single-node 2-tile recipe via `ezpz launch` |
| `run_multinode.sh` | Multi-node recipe via `ezpz launch` |

## How to use

```bash
# Single-node, interactive
qsub -I -A <account> -q debug -l select=1 -l walltime=01:00:00 \
     -l filesystems=home:flare
export AURORA_WORK_DIR=/lus/flare/projects/<project>/$USER/freedave-rl
export REPO_DIR=$AURORA_WORK_DIR/FreeDave-RL
export MODEL_DIR=$AURORA_WORK_DIR/models/TraDo-4B-Instruct
bash slurm_scripts/aurora_ezpz/run_2tiles_single_node.sh

# Multi-node
qsub slurm_scripts/aurora_ezpz/run_multinode.sh -A <account> \
     -l select=2 -l walltime=01:00:00
```

The script auto-installs `ezpz` from GitHub on first run if it isn't
already importable.

## Why ezpz over `mpiexec` + custom wrapper

| Concern | mpiexec + launch_per_rank.sh | ezpz |
|---|---|---|
| RANK / WORLD_SIZE detection | Manual: read `PALS_RANKID`, `GLOBAL_WORLD_SIZE`, `PALS_LOCAL_RANKID` in a wrapper | `ezpz.setup_torch()` in user code (Aurora: see status caveat) |
| MPI init | Manual: import `mpi4py.MPI` first in user code | `ezpz.setup_torch()` handles it |
| MASTER_ADDR | Manual: `head -n 1 $PBS_NODEFILE` in parent shell | Auto |
| Per-rank wrapper script | Required (`launch_per_rank.sh`) | Not needed |
| Cross-system portability | Aurora-specific PALS env vars | Same script works on Polaris / Frontier / Perlmutter / local |

The Aurora oneCCL env recipe (CCL_*, FI_*, libmpi.so.12 visibility)
remains the same — ezpz doesn't replace those, just the
launcher-and-rank-detection layer.

## Recipe (v7, 2026-04-30)

User script needs only:

```python
import ezpz
ezpz.setup_torch()           # MPI init + torch.distributed (XCCL on XPU)
device = ezpz.get_torch_device()
```

(The full `aurora_patch.py` wraps this plus the unrelated XPU-compat
shims for transformers / flash_attn / peft / torch.cuda aliases.)

Install ezpz once into the venv (NOT `--user`, NOT
`--no-build-isolation`):

```bash
python -m pip install 'ezpz @ git+https://github.com/saforem2/ezpz'
```

(Use `python -m pip` because the Aurora ClearML agent venv has a broken
`pip` shebang; `--user` is rejected because `dill` is already in the
venv's site-packages; `--no-build-isolation` fails because `hatchling`
is not pre-installed.)
