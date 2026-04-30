"""Aurora XPU compatibility patches for FreeDave-RL.

Apply five small monkey-patches that let the CUDA-only training stack
(transformers + peft + flash-attn + DeepSpeed) run on Intel Aurora XPU
(Intel Data Center GPU Max 1550) under the Aurora frameworks/2025.2.0
software stack (PyTorch 2.8 + IPEX 2.8.10 + native XCCL backend).

Usage in an entry-point script (e.g. diffu_grpo_train.py):

    try:
        from slurm_scripts.aurora.aurora_patch import apply_aurora_patches
        apply_aurora_patches()
    except ImportError:
        pass

The patches are no-ops on CUDA (they short-circuit when torch.xpu is
absent or when the symbols they would inject already exist).

What each patch does:

1. _aurora_init_mpi
   Imports mpi4py.MPI to trigger MPI_Init().  Required because Aurora
   oneCCL is configured with CCL_KVS_MODE=mpi + CCL_KVS_USE_MPI_RANKS=1
   per ALCF docs; without an explicit MPI_Init() before the first CCL
   collective, CCL aborts with "Attempting to use an MPI routine before
   initializing or after finalizing MPICH".

2. _aurora_patch_transformers_compat
   Stubs `transformers.utils.LossKwargs` for the case where TRL 0.19
   imports it from a transformers version that doesn't define it.

3. _aurora_patch_flash_attn_stub
   Installs a fake `flash_attn` package providing a CPU/XPU rms_norm_fn
   fallback.  flash-attn is CUDA-only and unavailable on XPU; without
   the stub, importing newer transformers/peft modules fails at import
   time even when attn_implementation="sdpa" is selected.

4. _aurora_patch_peft_inc
   Disables peft 0.17's optional `dispatch_inc` (Habana Intel Neural
   Compressor) path which is unconditionally imported on XPU and pulls
   in HPU-only modules.  Also stubs the empty `neural_compressor.torch`
   submodules it expects.

5. _aurora_patch_torch_cuda_aliases
   Aliases `torch.cuda.{empty_cache, synchronize, set_device, current_device,
   device_count}` -> `torch.xpu.*`, and overrides `torch.cuda.amp.autocast`
   to construct an XPU autocast context.  Required because TRL/DeepSpeed
   ZeRO-3 still call torch.cuda.* unconditionally even when device.type
   is "xpu".

6. _aurora_bind_xpu_tile
   Calls torch.xpu.set_device(LOCAL_RANK) so each rank is bound to its
   own XPU tile before any other XPU allocation happens.

Tested with frameworks/2025.2.0 (PyTorch 2.8.0a0+gitba56102, IPEX
2.8.10+git09505bb, transformers 4.56.1, trl 0.19.1, deepspeed 0.17.5,
peft 0.17.1) on Aurora compute nodes.
"""

from __future__ import annotations

import os


def _aurora_init_mpi() -> None:
    try:
        from mpi4py import MPI  # noqa: F401  -- triggers MPI_Init()
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        print(f"[aurora-patch] MPI initialized: rank={rank} size={size}", flush=True)
    except Exception as e:
        print(f"[aurora-patch] MPI init failed (skipping): {e}", flush=True)


def _aurora_patch_transformers_compat() -> None:
    try:
        import transformers.utils as _u
        from typing import TypedDict

        if not hasattr(_u, "LossKwargs"):
            class LossKwargs(TypedDict, total=False):
                pass

            _u.LossKwargs = LossKwargs
    except Exception as e:
        print(f"[aurora-patch] LossKwargs stub failed: {e}")


def _aurora_patch_flash_attn_stub() -> None:
    import importlib.machinery
    import sys
    import types

    if "flash_attn" in sys.modules:
        return

    import torch

    def _make_pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        m.__path__ = []  # type: ignore[attr-defined]
        return m

    flash_attn = _make_pkg("flash_attn")
    flash_attn.__version__ = "0.0.0-aurora-stub"
    fa_ops = _make_pkg("flash_attn.ops")
    fa_triton = _make_pkg("flash_attn.ops.triton")
    fa_layer_norm = _make_pkg("flash_attn.ops.triton.layer_norm")

    def rms_norm_fn(
        x,
        weight,
        bias=None,
        residual=None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        **kwargs,
    ):
        if residual is not None:
            if residual_in_fp32:
                x = (x.float() + residual.float()).to(x.dtype)
            else:
                x = x + residual.to(x.dtype)
            residual_out = x
        else:
            residual_out = x
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x.float() * torch.rsqrt(var + eps)).to(weight.dtype)
        out = x_norm * weight
        if bias is not None:
            out = out + bias
        return (out, residual_out) if prenorm else out

    fa_layer_norm.rms_norm_fn = rms_norm_fn

    def _not_impl(*a, **k):
        raise NotImplementedError("flash_attn not available on Aurora XPU; use sdpa")

    flash_attn.flash_attn_func = _not_impl
    flash_attn.flash_attn_varlen_func = _not_impl
    flash_attn.flash_attn_qkvpacked_func = _not_impl
    flash_attn.ops = fa_ops
    fa_ops.triton = fa_triton
    fa_triton.layer_norm = fa_layer_norm

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.ops"] = fa_ops
    sys.modules["flash_attn.ops.triton"] = fa_triton
    sys.modules["flash_attn.ops.triton.layer_norm"] = fa_layer_norm

    for modname in ("transformers.utils.import_utils", "transformers.utils"):
        try:
            import importlib

            mod = importlib.import_module(modname)
            for fn in (
                "is_flash_attn_2_available",
                "is_flash_attn_3_available",
                "is_flash_attn_greater_or_equal_2_10",
            ):
                if hasattr(mod, fn):
                    setattr(mod, fn, lambda *a, **k: False)
        except Exception:
            pass


def _aurora_patch_peft_inc() -> None:
    try:
        import importlib

        def _noop_dispatch_inc(target, adapter_name, lora_config=None, **kwargs):
            return None

        for modname in (
            "peft.tuners.lora.inc",
            "peft.tuners.lora.model",
            "peft.tuners.lora.layer",
        ):
            try:
                m = importlib.import_module(modname)
                if hasattr(m, "dispatch_inc"):
                    m.dispatch_inc = _noop_dispatch_inc
            except Exception:
                pass

        import importlib.machinery
        import sys
        import types

        for name in (
            "neural_compressor.torch",
            "neural_compressor.torch.algorithms",
            "neural_compressor.torch.algorithms.fp8_quant",
            "neural_compressor.torch.algorithms.fp8_quant._quant_common",
            "neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules",
        ):
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                m.__path__ = []  # type: ignore[attr-defined]
                sys.modules[name] = m

        helper = sys.modules[
            "neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules"
        ]

        class _Sentinel:
            pass

        helper.PatchedLinear = _Sentinel
        helper.PatchedLinearAllReduce = _Sentinel
        helper.PatchedLmHeadLinearAllreduce = _Sentinel
    except Exception as e:
        print(f"[aurora-patch] peft.lora.inc patch skipped: {e}")


def _aurora_patch_torch_cuda_aliases() -> None:
    import torch

    if not hasattr(torch, "xpu"):
        return

    cuda = torch.cuda
    xpu = torch.xpu
    for name in ("empty_cache", "synchronize", "set_device", "current_device", "device_count"):
        if hasattr(xpu, name):
            setattr(cuda, name, getattr(xpu, name))

    try:
        from torch.amp import autocast as _amp_autocast

        def _xpu_autocast(*args, **kwargs):
            kwargs.pop("device_type", None)
            return _amp_autocast("xpu", *args, **kwargs)

        cuda.amp.autocast = _xpu_autocast
    except Exception:
        pass


def _aurora_bind_xpu_tile() -> None:
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(local_rank)
        print(
            f"[aurora-patch] RANK={os.environ.get('RANK')} "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
            f"LOCAL_RANK={local_rank} bound to xpu:{local_rank}",
            flush=True,
        )


def apply_aurora_patches() -> None:
    """Apply all Aurora XPU compatibility patches.

    Safe to call on CUDA — torch.cuda aliases short-circuit, mpi4py is
    a no-op import if MPI is already initialized, and the flash_attn
    stub only installs if `flash_attn` isn't already imported.
    """
    _aurora_init_mpi()
    _aurora_patch_transformers_compat()
    _aurora_patch_flash_attn_stub()
    _aurora_patch_peft_inc()
    _aurora_patch_torch_cuda_aliases()
    _aurora_bind_xpu_tile()


if __name__ == "__main__":
    apply_aurora_patches()
