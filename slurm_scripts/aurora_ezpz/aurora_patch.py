"""Aurora XPU compatibility patches for FreeDave-RL (ezpz variant).

This is the ezpz launcher variant of the patch shim. The single change vs the
non-ezpz `slurm_scripts/aurora/aurora_patch.py` is at the top: we call
`ezpz.setup_torch()` to discover RANK / WORLD_SIZE / LOCAL_RANK / device and
initialize torch.distributed (XCCL backend on XPU). No manual `mpi4py.MPI`
import or PALS env-var bridging.

What this version still does (same as non-ezpz variant):
1. transformers `LossKwargs` stub          (transformers 4.56 + recent torch)
2. flash_attn stub package                  (CPU/XPU rms_norm_fn fallback)
3. peft.lora.inc no-op + neural_compressor  (skip Habana FP8 path on XPU)
4. torch.cuda -> torch.xpu aliases          (let HF/transformers use XPU)
5. Per-rank torch.xpu.set_device(LOCAL_RANK)

Usage in diffu_grpo_train.py:

    try:
        from slurm_scripts.aurora_ezpz.aurora_patch import apply_aurora_patches
        apply_aurora_patches()
    except ImportError:
        pass

KNOWN ISSUE (as of 2026-04-30 + ezpz 0.12.5 + frameworks/2025.2.0):
`ezpz.setup_torch()` on Aurora compute nodes via `ezpz launch` returns
`world_size=1` per rank (each rank thinks it is the sole rank). Training
runs end-to-end on a single tile but is NOT a true distributed group --
each rank loads the full model and does standalone training. Multi-tile /
multi-node distributed comm via this path is not validated yet.

Tested with frameworks/2025.2.0 + ezpz 0.12.5 on Aurora gpu_hack.
"""

from __future__ import annotations


def _ezpz_setup_torch():
    """Call ezpz.setup_torch() -- handles MPI init + torch.distributed init."""
    import ezpz
    rank = ezpz.setup_torch()
    device = ezpz.get_torch_device()
    print(f"[aurora-patch] ezpz.setup_torch() -> rank={rank}, device={device}", flush=True)


def _patch_transformers_losskwargs():
    """transformers 4.56 imports LossKwargs lazily; stub it on older paths."""
    try:
        import transformers.utils as _u
        from typing import TypedDict
        if not hasattr(_u, "LossKwargs"):
            class LossKwargs(TypedDict, total=False):
                pass
            _u.LossKwargs = LossKwargs
    except Exception as e:
        print(f"[aurora-patch] LossKwargs stub failed: {e}")


def _patch_flash_attn_stub():
    """Stub flash_attn (no XPU build) + provide a pure-torch rms_norm_fn."""
    import sys
    import types
    import importlib.machinery
    import torch

    if "flash_attn" in sys.modules:
        return

    def _make_pkg(name):
        m = types.ModuleType(name)
        m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        m.__path__ = []
        return m

    flash_attn = _make_pkg("flash_attn")
    flash_attn.__version__ = "0.0.0-aurora-stub"
    fa_ops = _make_pkg("flash_attn.ops")
    fa_triton = _make_pkg("flash_attn.ops.triton")
    fa_layer_norm = _make_pkg("flash_attn.ops.triton.layer_norm")

    def rms_norm_fn(x, weight, bias=None, residual=None, eps=1e-6,
                    prenorm=False, residual_in_fp32=False, **kwargs):
        if residual is not None:
            x = (x.float() + residual.float()).to(x.dtype) if residual_in_fp32 else x + residual.to(x.dtype)
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
        raise NotImplementedError("flash_attn not on XPU")

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
            for fn in ("is_flash_attn_2_available",
                       "is_flash_attn_3_available",
                       "is_flash_attn_greater_or_equal_2_10"):
                if hasattr(mod, fn):
                    setattr(mod, fn, lambda *a, **k: False)
        except Exception:
            pass


def _patch_peft_lora_inc():
    """peft 0.17 dispatch_inc imports neural_compressor's Habana FP8 path."""
    try:
        import importlib

        def _noop(target, adapter_name, lora_config=None, **kwargs):
            return None

        for modname in ("peft.tuners.lora.inc",
                        "peft.tuners.lora.model",
                        "peft.tuners.lora.layer"):
            try:
                m = importlib.import_module(modname)
                if hasattr(m, "dispatch_inc"):
                    m.dispatch_inc = _noop
            except Exception:
                pass

        import sys
        import types
        import importlib.machinery
        for name in ("neural_compressor.torch",
                     "neural_compressor.torch.algorithms",
                     "neural_compressor.torch.algorithms.fp8_quant",
                     "neural_compressor.torch.algorithms.fp8_quant._quant_common",
                     "neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules"):
            if name not in sys.modules:
                m = types.ModuleType(name)
                m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
                m.__path__ = []
                sys.modules[name] = m
        helper = sys.modules["neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules"]

        class _S:
            pass

        helper.PatchedLinear = _S
        helper.PatchedLinearAllReduce = _S
        helper.PatchedLmHeadLinearAllreduce = _S
    except Exception as e:
        print(f"[aurora-patch] peft.lora.inc patch skipped: {e}")


def _patch_torch_cuda_aliases():
    """Make torch.cuda.* fall through to torch.xpu.* so HF/transformers work."""
    import torch
    if not hasattr(torch, "xpu"):
        return
    cuda = torch.cuda
    xpu = torch.xpu
    for name in ("empty_cache", "synchronize", "set_device",
                 "current_device", "device_count"):
        if hasattr(xpu, name):
            setattr(cuda, name, getattr(xpu, name))
    try:
        from torch.amp import autocast as _ac

        def _xa(*a, **k):
            k.pop("device_type", None)
            return _ac("xpu", *a, **k)

        cuda.amp.autocast = _xa
    except Exception:
        pass


def _bind_xpu_tile():
    """Bind this rank's torch.xpu device to LOCAL_RANK."""
    import os
    import torch
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.set_device(lr)


def apply_aurora_patches() -> None:
    """Apply all patches in the right order.  Call once, early in user code."""
    _ezpz_setup_torch()           # MUST be first -- initializes distributed
    _patch_transformers_losskwargs()
    _patch_flash_attn_stub()
    _patch_peft_lora_inc()
    _patch_torch_cuda_aliases()
    _bind_xpu_tile()
