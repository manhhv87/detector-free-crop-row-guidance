#!/usr/bin/env python
"""
eval_stats.py -- confirmatory statistics for the paper, computed from the per-frame CSVs.

Adds the rigour the headline comparisons need, all from the existing per-frame logs:
  - the reconstructed YOLOv5 front end put through the SAME temporal stage as the detector-free pipeline
    (a fair raw-vs-fused, both-front-ends comparison);
  - paired per-frame differences with bootstrap confidence intervals for the load-bearing comparisons;
  - accuracy stratified by how far off-centre the ground-truth row sits (|cross-track| terciles), since the
    runs were logged under closed-loop centring;
  - per-run replication of the temporal-stage effect;
  - a calibration sensitivity band on the centimetre cross-track.

Reads results/cabbage_perframe.csv (ExG raw/fused + GT) and results/yolo_perframe.csv. Pure numpy.
"""
from __future__ import annotations
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import guidance_curve as gc
import calib
from temporal_guidance import TemporalGuidance

RNG = np.random.default_rng(0)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(v):
    a = np.asarray([x for x in v if x is not None], float)
    return (np.median(a), a.mean(), np.percentile(a, 90)) if len(a) else (float("nan"),) * 3


def boot_ci(diffs, B=10000):
    """Bootstrap 95% CI of the median of paired per-frame differences (frame-level i.i.d. resampling)."""
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(d), size=(B, len(d)))
    meds = np.median(d[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def boot_ci_clustered(diffs, runs, B=10000):
    """Run-clustered 95% CI: frames are resampled WITHIN each run, preserving per-run counts, so the
    within-run correlation of 1 fps frames is not mistaken for extra independent evidence. More conservative
    (wider) than the frame-level CI, which is the point the reviewers raised."""
    d = np.asarray(diffs, float)
    runs = np.asarray(runs)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    groups = [np.where(runs == r)[0] for r in np.unique(runs)]
    meds = np.empty(B)
    for b in range(B):
        pick = np.concatenate([g[RNG.integers(0, len(g), size=len(g))] for g in groups])
        meds[b] = np.median(d[pick])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def frame_index(name):
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 0


def main():
    exg = {r["frame"]: r for r in csv.DictReader(open("results/cabbage_perframe.csv", newline=""))}
    yolo = {r["frame"]: r for r in csv.DictReader(open("results/yolo_perframe.csv", newline=""))}

    # ---- YOLO fused: run the reconstructed detector stream through the SAME temporal stage ----
    groups = defaultdict(list)
    for fr, r in yolo.items():
        groups[r["video"]].append(fr)
    yfus = {}
    for v, frs in groups.items():
        frs.sort(key=frame_index)
        tg = TemporalGuidance()
        for fr in frs:
            d, c = f(yolo[fr]["yolo_deg"]), f(yolo[fr]["yolo_ct_px"])
            fh = tg.update_heading(d, present=d is not None)["heading"]
            fc = tg.update_crosstrack(c, present=c is not None)["crosstrack"]
            yfus[fr] = (fh, fc)

    lab = [fr for fr, e in exg.items() if e.get("gt_deg")]
    runs = sorted({exg[fr]["video"] for fr in lab})

    def err(fr, key):
        """absolute error vs GT for a given series on frame fr; returns None if missing."""
        gtd, gtct = f(exg[fr]["gt_deg"]), f(exg[fr]["gt_ct_px"])
        if key == "exg_raw_h":  return None if f(exg[fr]["raw_deg"]) is None else abs(f(exg[fr]["raw_deg"]) - gtd)
        if key == "exg_fus_h":  return None if f(exg[fr]["fused_deg"]) is None else abs(f(exg[fr]["fused_deg"]) - gtd)
        if key == "yolo_raw_h": return None if (fr not in yolo or f(yolo[fr]["yolo_deg"]) is None) else abs(f(yolo[fr]["yolo_deg"]) - gtd)
        if key == "yolo_fus_h": return None if (fr not in yfus or yfus[fr][0] is None) else abs(yfus[fr][0] - gtd)
        if key == "exg_raw_c":  return None if f(exg[fr]["ct_px"]) is None else abs(f(exg[fr]["ct_px"]) - gtct)
        if key == "exg_fus_c":  return None if f(exg[fr]["fused_ct_px"]) is None else abs(f(exg[fr]["fused_ct_px"]) - gtct)
        if key == "yolo_raw_c": return None if (fr not in yolo or f(yolo[fr]["yolo_ct_px"]) is None) else abs(f(yolo[fr]["yolo_ct_px"]) - gtct)
        if key == "yolo_fus_c": return None if (fr not in yfus or yfus[fr][1] is None) else abs(yfus[fr][1] - gtct)

    print(f"# ACCURACY vs GT (n={len(lab)}), raw and fused for both front ends")
    print(f"{'series':24s} heading med/mean/p90 | cross-track med/mean/p90 (px)")
    for name, kh, kc in [("ExG raw", "exg_raw_h", "exg_raw_c"), ("ExG + temporal", "exg_fus_h", "exg_fus_c"),
                         ("YOLOv5 raw", "yolo_raw_h", "yolo_raw_c"), ("YOLOv5 + temporal", "yolo_fus_h", "yolo_fus_c")]:
        h = [err(fr, kh) for fr in lab]; c = [err(fr, kc) for fr in lab]
        print("%-24s %5.2f / %5.2f / %5.2f | %5.1f / %5.1f / %5.1f" % (name, *pct(h), *pct(c)))

    # ---- paired bootstrap CIs for the load-bearing comparisons ----
    def paired(ka, kb):
        keep = [fr for fr in lab if err(fr, ka) is not None and err(fr, kb) is not None]
        d = [err(fr, ka) - err(fr, kb) for fr in keep]
        runs = [exg[fr]["video"] for fr in keep]
        return np.median(d), boot_ci(d), boot_ci_clustered(d, runs), len(d)
    print("\n# PAIRED DIFFERENCES (median; 95% CI frame-level | run-clustered)")
    for label, ka, kb in [("PRIMARY ExG+temporal - YOLO raw (heading)", "exg_fus_h", "yolo_raw_h"),
                          ("PRIMARY ExG+temporal - YOLO raw (cross-track px)", "exg_fus_c", "yolo_raw_c"),
                          ("ablation ExG fused - YOLO fused (heading)", "exg_fus_h", "yolo_fus_h"),
                          ("ExG raw - ExG fused (heading)", "exg_raw_h", "exg_fus_h"),
                          ("ablation ExG fused - YOLO fused (cross-track px)", "exg_fus_c", "yolo_fus_c")]:
        m, (lo, hi), (clo, chi), n = paired(ka, kb)
        sig = "" if clo <= 0 <= chi else "  (clustered CI excludes 0)"
        print("  %-42s med=%+.2f  frame[%+.2f,%+.2f]  run[%+.2f,%+.2f]  n=%d%s"
              % (label, m, lo, hi, clo, chi, n, sig))

    # ---- stratify by how far off-centre the GT row sits (|GT cross-track|) ----
    labc = [(fr, abs(f(exg[fr]["gt_ct_px"]))) for fr in lab if f(exg[fr]["gt_ct_px"]) is not None]
    vals = np.array([v for _, v in labc])
    t1, t2 = np.percentile(vals, [33, 66])
    print(f"\n# STRATIFIED by |GT cross-track| (terciles at {t1:.0f}, {t2:.0f} px)")
    print(f"{'stratum':18s} n | ExG+temporal head | YOLO+temporal head (median deg)")
    for nm, sel in [("near-centre", vals <= t1), ("mid", (vals > t1) & (vals <= t2)), ("off-centre", vals > t2)]:
        frs = [fr for (fr, _), s in zip(labc, sel) if s]
        eh = [err(fr, "exg_fus_h") for fr in frs]
        yh = [err(fr, "yolo_fus_h") for fr in frs]
        print("  %-16s %2d | %5.2f | %5.2f" % (nm, len(frs), np.median([x for x in eh if x is not None]),
                                               np.median([x for x in yh if x is not None])))

    # ---- per-run replication of the temporal-stage effect (gated frames raw vs fused) ----
    print("\n# PER-RUN temporal-stage effect (gate-suppressed labelled frames, heading err vs GT)")
    for run in runs:
        g = [fr for fr in lab if exg[fr]["video"] == run and exg[fr].get("gated") == "1"]
        if not g:
            print(f"  {run}: no gated+labelled frames"); continue
        raw = np.median([abs(f(exg[fr]["raw_deg"]) - f(exg[fr]["gt_deg"])) for fr in g])
        fus = np.median([abs(f(exg[fr]["fused_deg"]) - f(exg[fr]["gt_deg"])) for fr in g if f(exg[fr]["fused_deg"]) is not None])
        print(f"  {run}: n_gated={len(g)}  raw={raw:.1f}  fused={fus:.1f} deg")

    # ---- calibration sensitivity: cm per pixel at the near row under pose perturbation ----
    def cmperpx(h_mm, pitch_deg):
        p = np.radians(pitch_deg)
        Rwc = np.array([[1, 0, 0], [0, -np.sin(p), -np.cos(p)], [0, np.cos(p), -np.sin(p)]])
        t = -Rwc @ np.array([0, 0, h_mm])
        P = calib.K @ np.column_stack([Rwc[:, 0], Rwc[:, 1], t])
        Hinv = np.linalg.inv(P)
        yn = gc.Y_CROSSTRACK * 480
        g0 = Hinv @ np.array([320.0, yn, 1.0]); g1 = Hinv @ np.array([321.0, yn, 1.0])
        return abs((g1[0] / g1[2] - g0[0] / g0[2]) / 10.0)
    base = cmperpx(650, 35.0)
    grid = [cmperpx(650 + dh, 35.0 + dp) for dh in (-50, 0, 50) for dp in (-5, 0, 5)]
    print(f"\n# CALIBRATION SENSITIVITY (near-row cm/px)")
    print(f"  base={base:.4f}  under +/-5cm height, +/-5deg tilt: [{min(grid):.4f}, {max(grid):.4f}] "
          f"(median 1.1 cm -> [{1.1*min(grid)/base:.1f}, {1.1*max(grid)/base:.1f}] cm)")


if __name__ == "__main__":
    main()
