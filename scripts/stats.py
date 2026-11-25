#!/usr/bin/env python3
"""median / p10 / p90 over space- or newline-separated numbers (stdin)."""
import sys

vals = sorted(float(x) for x in sys.stdin.read().replace("\n", " ").split() if x)
if not vals:
    print("0 0 0")
    sys.exit(0)


def pct(p):
    k = (len(vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


print(f"{pct(0.5):.2f} {pct(0.10):.2f} {pct(0.90):.2f}")
