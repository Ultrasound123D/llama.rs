#!/usr/bin/env python3
"""Phase 6.6 audit parser: aggregates llama.rs op-profile TSV rows into
stage tables (decode + prefill) and prints the bandwidth classification.

Input: one or more `prof-tsv` streams on stdin, separated by PHASE markers.
Stages group tensor-class rows by role (attention projections, FFN, conv,
output head) and sum the non-matmul buckets (norms, swiglu, rope, sample).
Usage: audit_parse.py decode FILE... | audit_parse.py prefill FILE...
"""
import sys
from statistics import median


def parse_run(text):
    """one process run -> dict name -> (us_per_tok, mb_per_tok, calls)."""
    rows = {}
    wall = None
    for line in text.splitlines():
        f = line.split("\t")
        if line.startswith("#prof-tsv-wall"):
            wall = float(f[1])
        elif line.startswith("prof-tsv\t"):
            name = f[1]
            rows[name] = {
                "us": float(f[2]),
                "mb": float(f[5]),
                "calls": float(f[6]),
            }
    return rows, wall


def merge_runs(runs):
    """median per bucket across reps -> aggregated dict + median wall."""
    names = set()
    for r, _ in runs:
        names.update(r.keys())
    out = {}
    for n in names:
        out[n] = {
            "us": median([r[n]["us"] for r, _ in runs if n in r]),
            "mb": median([r[n]["mb"] for r, _ in runs if n in r]),
            "calls": median([r[n]["calls"] for r, _ in runs if n in r]),
        }
    wall = median([w for _, w in runs if w])
    return out, wall


STAGES_DECODE = [
    ("attn projections (q/k/v/o/gate)", lambda n: n.startswith("mv blk.*.attn_")),
    ("ffn gate+up (fused pair)", lambda n: n.startswith("mv blk.*.ffn_gate")),
    ("ffn_down", lambda n: n.startswith("mv blk.*.ffn_down")),
    ("conv in/out_proj", lambda n: n.startswith("mv blk.*.shortconv")),
    ("output head (tied embd)", lambda n: n.startswith("mv token_embd") or n.startswith("mv output.weight")),
    ("attention core (softmax+dot+axpy)", lambda n: n == "attn_core"),
    ("qknorm+rope / rope", lambda n: n.startswith("attn_prep") or n.startswith("rope")),
    ("conv window+state", lambda n: n == "conv_core"),
    ("norms+residual+swiglu+kv", lambda n: n in ("elementwise", "swiglu(elementwise)", "kv_write", "rmsnorm")),
    ("glue (sample/logits/lookup)", lambda n: n.startswith("sample") or n.startswith("logits_copy") or n.startswith("tensor_lookup") or n.startswith("embed_lookup")),
]

STAGES_PREFILL = [
    ("GEMM ffn gate+up", lambda n: n.startswith("gemm blk.*.ffn_gate")),
    ("GEMM ffn_down", lambda n: n.startswith("gemm blk.*.ffn_down")),
    ("GEMM attn q/k/v/o/gate", lambda n: n.startswith("gemm blk.*.attn_")),
    ("GEMM conv in/out_proj", lambda n: n.startswith("gemm blk.*.shortconv")),
    ("activation quantize (batch)", lambda n: n == "act_quantize(batch)"),
    ("attention core (batch)", lambda n: n == "attn_core(batch)"),
    ("qknorm+rope (batch)", lambda n: n.startswith("attn_prep") or n.startswith("rope")),
    ("conv window+state (batch)", lambda n: n == "conv_core(batch)"),
    ("norms+residual+swiglu+kv", lambda n: n in ("elementwise(batch)", "swiglu(batch)", "rmsnorm")),
    ("output head + embed", lambda n: n.startswith("mv token_embd") or n.startswith("mv output.weight") or n.startswith("embed_lookup")),
    ("glue (lookup/misc)", lambda n: n.startswith("tensor_lookup")),
]


def report(rows, wall_ms, stages, phase, dram_ceiling_1t, dram_ceiling_2t):
    total_us = sum(b["us"] for b in rows.values())
    print(f"\n== {phase} stage breakdown (median over reps; wall {wall_ms:.2f} ms/token) ==")
    print(f"{'stage':<40} {'µs/tok':>9} {'%wall':>7} {'MB/tok':>8} {'stream GB/s':>12}")
    acc_us = 0.0
    acc_mb = 0.0
    matmul_mb = 0.0
    matmul_us = 0.0
    used = set()
    for label, match in stages:
        us = sum(b["us"] for n, b in rows.items() if match(n))
        mb = sum(b["mb"] for n, b in rows.items() if match(n))
        if us == 0 and mb == 0:
            continue
        for n in rows:
            if match(n):
                used.add(n)
        gbs = mb / 1024 / (us / 1e6) if us > 0 and mb > 0 else 0.0
        acc_us += us
        acc_mb += mb
        if label.startswith(("attn projections", "ffn", "conv in", "GEMM", "output head")):
            matmul_mb += mb
            matmul_us += us
        print(f"{label:<40} {us:>9.0f} {100 * us / (wall_ms * 1000):>6.1f}% {mb:>8.1f} {gbs:>11.2f}")
    residual = wall_ms * 1000 - total_us
    residual = max(residual, 0.0)
    unaccounted = set(rows) - used
    print(f"{'residual (unprofiled glue)':<40} {residual:>9.0f} {100 * residual / (wall_ms * 1000):>6.1f}%")
    if unaccounted:
        print(f"  (unmatched buckets: {sorted(unaccounted)})")
    # bandwidth classification
    eff = acc_mb / 1024 / (wall_ms / 1000.0)  # total traffic / wall
    mm_eff = matmul_mb / 1024 / (matmul_us / 1e6) if matmul_us else 0
    print(f"  traffic: {acc_mb:.0f} MB/token streamed in {wall_ms:.2f} ms -> overall {eff:.2f} GB/s")
    print(f"  matmul weight bytes: {matmul_mb:.0f} MB/token; in-matmul stream rate {mm_eff:.2f} GB/s")
    if phase == "decode" and dram_ceiling_2t:
        pct = 100 * eff / dram_ceiling_2t
        print(f"  ceiling: 1-stream {dram_ceiling_1t:.1f} GB/s, 2-stream min-pass {dram_ceiling_2t:.1f} GB/s -> decode at {pct:.0f}% of the 2-stream ceiling")
        verdict = ("MEMORY-BOUND (near the machine's streaming ceiling)"
                   if pct > 80 else
                   "COMPUTE-BOUND (weight stream rate below the streaming ceiling — kernel-limited, not DRAM-limited)")
        print(f"  verdict: {verdict}")
    if phase == "prefill":
        print("  verdict: GEMM stream rate ~0.5 GB/s << ceiling -> NOT memory-bound; instruction-throughput (shuffle port + AVX-512 licensing) bound")


def main():
    phase = sys.argv[1]  # decode | prefill
    files = sys.argv[2:]
    runs = []
    for path in files:
        with open(path) as fh:
            runs.append(parse_run(fh.read()))
    runs = [r for r in runs if r[0]]  # drop empty/failed extractions
    if not runs:
        print(f"no prof-tsv evidence in {files}", file=sys.stderr)
        sys.exit(1)
    rows, wall = merge_runs(runs)
    stages = STAGES_DECODE if phase == "decode" else STAGES_PREFILL
    import os
    ceiling_1t = float(os.environ.get("DRAM_CEILING_1T", "14.0"))
    ceiling_2t = float(os.environ.get("DRAM_CEILING_2T", "17.0"))
    report(rows, wall, stages, phase, ceiling_1t, ceiling_2t)


if __name__ == "__main__":
    main()
