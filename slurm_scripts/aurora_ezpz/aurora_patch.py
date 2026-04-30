"""Aurora XPU compatibility patches for FreeDave-RL (ezpz variant).

Same five XPU compat patches as `slurm_scripts/aurora/aurora_patch.py`,
EXCEPT the MPI init step is delegated to ezpz: when the launcher is
`ezpz launch ...`, ezpz already calls `setup_torch_distributed` which
imports mpi4py.MPI and configures torch.distributed.

What this version still does:
1. transformers `LossKwargs` stub
2. flash_attn stub package (CPU/XPU rms_norm_fn fallback)
3. peft.lora.inc no-op + neural_compressor stubs
4. torch.cuda -> torch.xpu aliases + amp.autocast override
5. Per-rank torch.xpu.set_device(LOCAL_RANK)

Usage in diffu_grpo_train.py:

    try:
        from slurm_scripts.aurora_ezpz.aurora_patch import apply_aurora_patches
        apply_aurora_patches()
    except ImportError:
        pass

Tested with frameworks/2025.2.0 + ezpz HEAD on Aurora.
"""

from __future__ import annotations

import os


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

    def rms_norm_fn(x, weight, bias=None, residual=None, eps=1e-6,
                    prenorm=False, residual_in_fp32=False, **kwargs):
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

    def _ni(*a, **k):
        raise NotImplementedError("flash_attn not available on Aurora XPU; use sdpa")

    flash_attn.flash_attn_func = _ni
    flash_attn.flash_attn_varlen_func = _ni
    flash_attn.flash_attn_qkvpacked_func = _ni
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

        def _noop(target, adapter_name, lora_config=None, **kwargs):
            return None

        for modname in (
            "peft.tuners.lora.inc",
            "peft.tuners.lora.model",
            "peft.tuners.lora.layer",
        ):
            try:
                m = importlib.import_module(modname)
                if hasattr(m, "dispatch_inc"):
                    m.dispatch_inc = _noop
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


def apply_aurora_patches() -> None:
    """Apply Aurora XPU compatibility patches.

    Skip the mpi4py init step — `ezpz launch` (or `ezpz.setup_torch_distributed()`)
    handles MPI init and torch.distributed setup before the user code runs.
    """
    _aurora_patch_transformers_compat()
    _aurora_patch_flash_attn_stub()
    _aurora_patch_peft_inc()
    _aurora_patch_torch_cuda_aliases()
    _aurora_bind_xpu_tile()


if __name__ == "__main__":
    apply_aurora_patches()
