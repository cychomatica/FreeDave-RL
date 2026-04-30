#!/bin/bash
# Per-rank wrapper for Aurora PALS launcher.
#
# PALS_RANKID is the GLOBAL rank (0..WORLD_SIZE-1) on Aurora — PMI_RANK
# is empty under PALS multinode mpiexec.  We export torch.distributed's
# expected env vars from PALS_*, falling back to PMI_* for older envs.
#
# WORLD_SIZE comes from GLOBAL_WORLD_SIZE which the parent shell exports
# before mpiexec — PMI_SIZE may report a per-node count under PALS.
export RANK=${PALS_RANKID:-${PMI_RANK:-0}}
export WORLD_SIZE=${GLOBAL_WORLD_SIZE:-${PMI_SIZE:-1}}
export LOCAL_RANK=${PALS_LOCAL_RANKID:-${PMI_LOCAL_RANK:-0}}
exec "$@"
