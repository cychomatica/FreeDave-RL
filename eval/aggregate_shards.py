"""
Merge per-shard FreeDave benchmark JSONs (one per GPU) into a single result.

Each shard ran a disjoint strided slice of the same shuffled subset, so summing
the raw counters (n, n_correct, sum_tokens, total_nfe, sum_time, sum_steps) across
shards reproduces exactly what a single-GPU run over all examples would report.

Usage:
    python -m eval.aggregate_shards <shard0.json> <shard1.json> ... --output merged.json
"""

import argparse
import json


def merge(shard_paths):
    shards = [json.load(open(p)) for p in shard_paths]
    config = dict(shards[0]["config"])
    config.pop("shard_id", None)
    config["num_shards_merged"] = len(shards)

    # union of draft-step keys present in every shard
    d_keys = set(shards[0]["results"])
    for s in shards[1:]:
        d_keys &= set(s["results"])

    merged = {}
    for d in sorted(d_keys, key=int):
        acc = dict(n=0, n_correct=0, sum_tokens=0, total_nfe=0, sum_time=0.0, sum_steps=0)
        for s in shards:
            r = s["results"][d]
            acc["n"] += r["n"]
            acc["n_correct"] += r["n_correct"]
            acc["sum_tokens"] += r.get("sum_tokens", r["mean_tokens"] * r["n"])
            acc["total_nfe"] += r["total_nfe"]
            acc["sum_time"] += r.get("sum_time", (r["mean_tokens"] * r["n"] / r["tps"]) if r["tps"] else 0.0)
            acc["sum_steps"] += r.get("sum_steps", r["mean_steps"] * r["n"])
        n = acc["n"]
        merged[d] = {
            "acc": 100.0 * acc["n_correct"] / n if n else 0.0,
            "n_correct": acc["n_correct"],
            "n": n,
            "mean_tokens": acc["sum_tokens"] / n if n else 0.0,
            "total_nfe": acc["total_nfe"],
            "tpf": acc["sum_tokens"] / acc["total_nfe"] if acc["total_nfe"] else 0.0,
            "tps": acc["sum_tokens"] / acc["sum_time"] if acc["sum_time"] else 0.0,
            "mean_steps": acc["sum_steps"] / n if n else 0.0,
        }
    return config, merged


def report(config, results):
    ds = config.get("dataset"); split = config.get("split"); model = config.get("model_name")
    draft_steps = sorted(results, key=int)
    base = results.get("1")
    print("\n" + "=" * 92)
    print(f"FreeDave benchmark (merged {config.get('num_shards_merged')} shards)  |  {model}  |  {ds}/{split}")
    print("=" * 92)
    print(f"{'d':>4} {'acc%':>7} {'correct':>9} {'tok/ex':>8} {'NFE':>8} {'TPF':>7} {'TPS':>8} {'steps':>7} {'TPSx':>6} {'TPFx':>6}")
    print("-" * 92)
    for d in draft_steps:
        r = results[d]
        tpsx = r["tps"] / base["tps"] if base and base["tps"] else 1.0
        tpfx = r["tpf"] / base["tpf"] if base and base["tpf"] else 1.0
        print(f"{d:>4} {r['acc']:>7.2f} {r['n_correct']:>4}/{r['n']:<4} "
              f"{r['mean_tokens']:>8.1f} {r['total_nfe']:>8} {r['tpf']:>7.3f} "
              f"{r['tps']:>8.2f} {r['mean_steps']:>7.1f} {tpsx:>5.2f}x {tpfx:>5.2f}x")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="per-shard JSON files")
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()
    config, results = merge(args.shards)
    report(config, results)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"config": config, "results": results}, f, indent=2)
        print(f"\nSaved merged -> {args.output}")


if __name__ == "__main__":
    main()
