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

Same as the non-ezpz variant — see `../aurora/README.md` and
`../aurora/ISSUES.md`. Single-node up to 6 tiles works; multi-node
blocked on the same Aurora libfabric Cassini bug regardless of
launcher choice.

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
| RANK / WORLD_SIZE detection | Manual: read `PALS_RANKID`, `GLOBAL_WORLD_SIZE`, `PALS_LOCAL_RANKID` in a wrapper | Auto: ezpz detects PALS / Slurm / mpiexec |
| MPI init | Manual: import `mpi4py.MPI` first in user code | Auto: `ezpz launch` calls `setup_torch_distributed` before user code |
| MASTER_ADDR | Manual: `head -n 1 $PBS_NODEFILE` in parent shell | Auto |
| Per-rank wrapper script | Required (`launch_per_rank.sh`) | Not needed |
| Cross-system portability | Aurora-specific PALS env vars | Same script works on Polaris / Frontier / Perlmutter / local |

The Aurora oneCCL env recipe (CCL_*, FI_*, libmpi.so.12 visibility)
remains the same — ezpz doesn't replace those, just the
launcher-and-rank-detection layer.
