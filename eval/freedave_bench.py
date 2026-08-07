"""
FreeDave efficiency + accuracy benchmark.

Baseline measurement harness for "fine-tune a DLM so FreeDave decodes faster
without losing accuracy". It runs the FreeDave block-attention decode at several
draft-step settings on a reasoning benchmark and reports, for each setting:

  - acc      : task accuracy (%), scored with the repo's own reward logic
  - tok/ex   : mean generated tokens per example
  - NFE      : total model forward passes (from ForwardMonitor)
  - TPF      : tokens-per-forward = sum(tokens) / sum(NFE)   <- FreeDave headline metric
  - TPS      : tokens-per-second  = sum(tokens) / sum(time)
  - steps    : mean diffusion transfer-steps per example
  - vs d=1   : TPS / TPF speedup relative to draft_steps=1 (which IS static decoding)

`draft_steps=1` reproduces one-token-per-step static decoding, so a single run
over `--draft_steps 1 4 8` gives the static baseline AND the FreeDave points on
one apples-to-apples code path (mirrors Fig. 3 of arXiv:2510.00294).

Because FreeDave is lossless *relative to static decode*, accuracy should be flat
across draft_steps for a given model; TPF/TPS should rise. After fine-tuning for
higher acceptance, TPF/TPS should rise further while accuracy stays flat -- that
is the whole objective, and this script is how you measure it.

Usage (run from the repo root, on a GPU box):

    python -m eval.freedave_bench \
        --model_name Gen-Verse/TraDo-4B-Instruct \
        --dataset math --split test --num_examples 100 \
        --block_length 4 --max_gen_length 256 \
        --draft_steps 1 4 8 \
        --output eval/results/trado4b_math.json

Add --eager to reproduce FreeDave+ (Eager Mode), which the paper uses for TraDo.
"""

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

from data_utils import get_math_questions, get_gsm8k_questions, set_random_seed
from reward_func import extract_xml_answer
from math500_utils import remove_boxed, last_boxed_only_string, is_equiv
from generation.generation_core import DLMGeneration
from generation.monitor_utils import ForwardMonitor


# ----------------------------------------------------------------------------- data
def build_examples(dataset, split, num_examples, seed):
    """Return a list of (messages, gold_answer) for the chosen benchmark."""
    if dataset == "math":
        ds = get_math_questions(split)
    elif dataset == "gsm8k":
        ds = get_gsm8k_questions(split)
    else:
        raise ValueError(f"Unsupported dataset for this harness: {dataset}")
    ds = ds.shuffle(seed=seed)
    if num_examples is not None and num_examples < len(ds):
        ds = ds.select(range(num_examples))
    return [(ex["prompt"], ex["answer"]) for ex in ds]


# ------------------------------------------------------------------------- scoring
def _extract_boxed(text):
    try:
        return remove_boxed(last_boxed_only_string(text))
    except Exception:
        return None


def _norm_num(s):
    if s is None:
        return None
    return s.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")


def _gsm8k_pred(text):
    """Final numeric answer: prefer a \\boxed{} value, else the last number in the text.
    (TraDo answers GSM8K with a boxed/plain number, not the <answer> XML the prompt asks
    for, so tag-based extraction fails -- last-number is the standard GSM8K heuristic.)"""
    boxed = _extract_boxed(text)
    src = boxed if boxed else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", src)
    return nums[-1].replace(",", "") if nums else None


def score(dataset, completion_text, gold):
    """True if the completion is correct, using the same logic as reward_func."""
    if dataset == "math":
        return bool(is_equiv(_extract_boxed(completion_text), _extract_boxed(gold)))
    # gsm8k: numeric match on boxed-or-last-number
    pred = _gsm8k_pred(completion_text)
    g = _norm_num(gold)
    if pred is None or g is None:
        return False
    try:
        return abs(float(pred) - float(g)) < 1e-6
    except ValueError:
        return pred == g


# --------------------------------------------------------------------------- model
def load_model_and_tokenizer(model_name, dtype):
    model_cls = AutoModelForCausalLM if "TraDo" in model_name else AutoModel
    model = model_cls.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def resolve_special_ids(tokenizer, args):
    """Prefer explicit CLI ids; else derive from the tokenizer like the demos do."""
    def lookup(tok_key):
        name = tokenizer.special_tokens_map.get(tok_key)
        if name is None:
            return None
        return tokenizer.added_tokens_encoder.get(name, tokenizer.convert_tokens_to_ids(name))

    mask_id = args.mask_id if args.mask_id is not None else lookup("mask_token")
    eos_id = args.eos_id if args.eos_id is not None else lookup("eos_token")
    pad_id = args.pad_id if args.pad_id is not None else (lookup("pad_token") or eos_id)
    if mask_id is None or eos_id is None:
        raise ValueError(
            "Could not resolve mask/eos ids from the tokenizer; pass "
            "--mask_id/--eos_id/--pad_id explicitly (TraDo: 151669/151643/151643)."
        )
    return int(mask_id), int(eos_id), int(pad_id)


# --------------------------------------------------------------------------- decode
def count_generated_tokens(completion_ids, eos_id):
    """Meaningful tokens = up to and including the first EOS (rest is padding)."""
    eos_pos = (completion_ids == eos_id).nonzero(as_tuple=True)[0]
    if eos_pos.numel() > 0:
        return int(eos_pos[0].item()) + 1
    return int(completion_ids.numel())


@torch.no_grad()
def decode_one(dlm_gen, model, input_ids, attention_mask, ids, draft_steps, args):
    """Run one generation and return (sequences, nfe, elapsed_seconds)."""
    mask_id, eos_id, pad_id = ids
    common = dict(
        model=model,
        input_ids=input_ids,
        temperature=args.temperature,
        top_p=None,
        top_k=None,
        alg_temp=None,
        block_length=args.block_length,
        max_gen_length=args.max_gen_length,
        decoding_steps=args.max_gen_length,
        draft_steps=draft_steps,
        eager_acceptance_mode=args.eager,
        draft_mode=args.draft_mode,
        use_cache=args.use_cache,
        mask_token_id=mask_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        early_exit=args.early_exit,
    )

    monitor = ForwardMonitor(model)
    monitor.start()
    torch.cuda.synchronize()
    t0 = time.time()
    if args.decode == "block":
        out = dlm_gen.block_decode_with_block_attention_FreeDave(
            attention_mask=attention_mask, **common
        )
    else:  # full-attention DLMs (Dream / LLaDA)
        out = dlm_gen.block_decode_with_full_attention_FreeDave(
            dual_cache=args.dual_cache, **common
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    monitor.stop()
    return out, monitor.get_nfe(), elapsed


# ----------------------------------------------------------------------------- main
def run(args):
    set_random_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = load_model_and_tokenizer(args.model_name, dtype)
    ids = resolve_special_ids(tokenizer, args)
    dlm_gen = DLMGeneration(sdpa_additive_attention_mask=args.sdpa_additive_attention_mask)
    examples = build_examples(args.dataset, args.split, args.num_examples, args.seed)
    # Data-parallel sharding: each shard (GPU) takes a strided slice of the same
    # shuffled subset, so shards are disjoint and their union is the full set.
    if args.num_shards > 1:
        examples = examples[args.shard_id::args.num_shards]
    print(f"Loaded {len(examples)} {args.dataset}/{args.split} examples "
          f"(shard {args.shard_id}/{args.num_shards}) | model={args.model_name} | "
          f"decode={args.decode} eager={args.eager} early_exit={args.early_exit}\n")

    results = {}
    for d in args.draft_steps:
        n_correct = 0
        sum_tokens = sum_nfe = 0
        sum_time = 0.0
        sum_steps = 0
        # warmup (kernel compilation / cache) -- not recorded
        if examples:
            msgs, _ = examples[0]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_prompt_length)
            decode_one(dlm_gen, model,
                       enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda"),
                       ids, d, args)

        for i, (msgs, gold) in enumerate(examples):
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_prompt_length)
            input_ids = enc["input_ids"].to("cuda")
            attn = enc["attention_mask"].to("cuda")
            prompt_len = input_ids.shape[1]

            out, nfe, elapsed = decode_one(dlm_gen, model, input_ids, attn, ids, d, args)

            completion_ids = out.sequences[0, prompt_len:].cpu()
            n_tok = count_generated_tokens(completion_ids, ids[1])
            steps = int(out.trajectory_step_map.max().item()) + 1 if out.trajectory_step_map.numel() else 0
            content = tokenizer.decode(completion_ids[:n_tok], skip_special_tokens=True)
            correct = score(args.dataset, content, gold)

            n_correct += int(correct)
            sum_tokens += n_tok
            sum_nfe += nfe
            sum_time += elapsed
            sum_steps += steps
            torch.cuda.empty_cache()

            if args.verbose:
                print(f"  d={d} [{i+1}/{len(examples)}] {'OK ' if correct else 'XX '}"
                      f"tok={n_tok} nfe={nfe} steps={steps} {elapsed:.2f}s")

        n = len(examples)
        results[d] = {
            "acc": 100.0 * n_correct / n if n else 0.0,
            "n_correct": n_correct,
            "n": n,
            "mean_tokens": sum_tokens / n if n else 0.0,
            "total_nfe": sum_nfe,
            "tpf": sum_tokens / sum_nfe if sum_nfe else 0.0,
            "tps": sum_tokens / sum_time if sum_time else 0.0,
            "mean_steps": sum_steps / n if n else 0.0,
            # raw sums kept so multi-GPU shards can be merged exactly (see aggregate_shards.py)
            "sum_tokens": sum_tokens,
            "sum_time": sum_time,
            "sum_steps": sum_steps,
        }
        # Save after each draft-step setting so a walltime cutoff keeps finished points.
        _save(args, results)

    _report(args, results)


def _save(args, results):
    if not args.output:
        return
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"[saved {len(results)} setting(s) -> {args.output}]")


def _report(args, results):
    base = results.get(1)
    print("\n" + "=" * 92)
    print(f"FreeDave benchmark  |  {args.model_name}  |  {args.dataset}/{args.split}")
    print("=" * 92)
    hdr = f"{'d':>4} {'acc%':>7} {'correct':>9} {'tok/ex':>8} {'NFE':>8} {'TPF':>7} {'TPS':>8} {'steps':>7} {'TPSx':>6} {'TPFx':>6}"
    print(hdr)
    print("-" * 92)
    for d in args.draft_steps:
        r = results[d]
        tpsx = r["tps"] / base["tps"] if base and base["tps"] else 1.0
        tpfx = r["tpf"] / base["tpf"] if base and base["tpf"] else 1.0
        print(f"{d:>4} {r['acc']:>7.2f} {r['n_correct']:>4}/{r['n']:<4} "
              f"{r['mean_tokens']:>8.1f} {r['total_nfe']:>8} {r['tpf']:>7.3f} "
              f"{r['tps']:>8.2f} {r['mean_steps']:>7.1f} {tpsx:>5.2f}x {tpfx:>5.2f}x")
    print("=" * 92)
    print("d=1 is static (one-token-per-step) decoding. Goal of fine-tuning: push TPF/TPS "
          "up at d>1 while acc% stays flat.")


def parse_args():
    p = argparse.ArgumentParser(description="FreeDave efficiency + accuracy benchmark")
    p.add_argument("--model_name", type=str, default="Gen-Verse/TraDo-4B-Instruct")
    p.add_argument("--dataset", type=str, default="math", choices=["math", "gsm8k"])
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--num_examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draft_steps", type=int, nargs="+", default=[1, 4, 8],
                   help="List of draft-step settings; 1 == static decoding baseline.")
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--max_gen_length", type=int, default=256)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--draft_mode", type=str, default="tree_attention",
                   choices=["tree_attention", "batch_expanding"])
    p.add_argument("--eager", action="store_true",
                   help="Eager Mode (FreeDave+); the paper uses this for TraDo.")
    p.add_argument("--early_exit", action="store_true", default=True,
                   help="Stop decoding once a block is all-EOS (realistic TPF/TPS). On by default.")
    p.add_argument("--no_early_exit", dest="early_exit", action="store_false")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Data-parallel shards (one per GPU). Each process handles a strided slice.")
    p.add_argument("--shard_id", type=int, default=0, help="This process's shard index [0, num_shards).")
    p.add_argument("--decode", type=str, default="block", choices=["block", "full"],
                   help="block: TraDo/SDAR block-attention; full: Dream/LLaDA full-attention.")
    p.add_argument("--use_cache", action="store_true", default=True)
    p.add_argument("--no_use_cache", dest="use_cache", action="store_false")
    p.add_argument("--dual_cache", action="store_true", default=False)
    p.add_argument("--sdpa_additive_attention_mask", action="store_true", default=False,
                   help="Set for full-attention Dream/LLaDA; leave off for TraDo.")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--mask_id", type=int, default=None)
    p.add_argument("--eos_id", type=int, default=None)
    p.add_argument("--pad_id", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
