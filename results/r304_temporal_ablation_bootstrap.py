#!/usr/bin/env python3
"""
R3-04 temporal-stage ablation: reproduces the headline accuracy figures and
computes the run-clustered bootstrap confidence intervals for the gated-frame
raw-vs-fused heading-error improvement reported in Section 4.4 (Results).

Input : cabbage_perframe.csv  (one row per frame; columns:
        video, frame, t_sec, veg, raw_deg, fused_deg, gated,
        ct_px, fused_ct_px, gt_deg, gt_ct_px)
        gt_deg / gt_ct_px are present only on the 150-frame labelled subset.

Run   : python3 r304_temporal_ablation_bootstrap.py   (from any working directory)
Seed  : 12345  (fixed for reproducibility)  |  B = 10000 resamples
"""
import csv, statistics as st, random
from pathlib import Path

CSV = str(Path(__file__).resolve().parent / "cabbage_perframe.csv")
SEED, B = 12345, 10000
RUN = {"rec_20251130_030636.avi": "run1", "rec_20251130_124123.avi": "run2"}


def f(x):
    x = (x or "").strip()
    return float(x) if x not in ("", "nan", "NaN") else None


def gated(r):
    return r["gated"].strip() in ("1", "1.0", "True", "true")


def pct(a, p):
    a = sorted(a)
    if not a:
        return float("nan")
    k = (len(a) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


def err(r, col):
    return abs(f(r[col]) - f(r["gt_deg"]))


def boot_ci(frames_by_run, fn, rng):
    out = []
    for _ in range(B):
        samp = []
        for fr in frames_by_run.values():
            n = len(fr)
            samp += [fr[rng.randrange(n)] for _ in range(n)]
        out.append(fn(samp))
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def main():
    rng = random.Random(SEED)
    rows = list(csv.DictReader(open(CSV)))
    lab = [r for r in rows if f(r["gt_deg"]) is not None]

    # --- headline reproduction (all 150 labelled frames) ---
    hraw = [abs(f(r["raw_deg"]) - f(r["gt_deg"])) for r in lab]
    hfus = [abs(f(r["fused_deg"]) - f(r["gt_deg"])) for r in lab]
    print(f"Labelled frames: {len(lab)}  (expect 150)")
    print(f"Heading raw   med {st.median(hraw):.1f} mean {st.mean(hraw):.1f} p90 {pct(hraw,90):.1f}  [6.3/9.0/19.4]")
    print(f"Heading fused med {st.median(hfus):.1f} mean {st.mean(hfus):.1f} p90 {pct(hfus,90):.1f}  [5.0/6.3/13.2]")

    # --- gated-frame ablation + selectivity ---
    g = [r for r in lab if gated(r)]
    ng = [r for r in lab if not gated(r)]
    er = [err(r, "raw_deg") for r in g]
    ef = [err(r, "fused_deg") for r in g]
    print(f"\nGated frames: {len(g)} (expect 62)")
    print(f"Ablation: raw med {st.median(er):.1f} -> fused med {st.median(ef):.1f}  [11.8 -> 8.1]")
    print(f"Selectivity: gated mean raw {st.mean(er):.1f} vs non-gated {st.mean([err(r,'raw_deg') for r in ng]):.1f}  [14.4 vs 5.1]")

    # --- run-clustered bootstrap CI for the paired improvement ---
    g_by = {rn: [r for r in g if RUN[r["video"]] == rn] for rn in ("run1", "run2")}
    med_pair = lambda S: st.median([err(r, "raw_deg") - err(r, "fused_deg") for r in S])
    diff_med = lambda S: st.median([err(r, "raw_deg") for r in S]) - st.median([err(r, "fused_deg") for r in S])
    lo1, hi1 = boot_ci(g_by, med_pair, rng)
    lo2, hi2 = boot_ci(g_by, diff_med, rng)
    print(f"\nRun-clustered bootstrap (B={B}, seed={SEED}):")
    print(f"  Median paired improvement  {med_pair(g):+.2f} deg  95% CI [{lo1:+.2f}, {hi1:+.2f}]")
    print(f"  Difference of medians      {diff_med(g):+.2f} deg  95% CI [{lo2:+.2f}, {hi2:+.2f}]")
    for rn in ("run1", "run2"):
        lo, hi = boot_ci({rn: g_by[rn]}, med_pair, rng)
        print(f"  {rn} paired improvement   {med_pair(g_by[rn]):+.2f} deg  95% CI [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
