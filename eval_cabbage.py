#!/usr/bin/env python
"""
eval_cabbage.py -- honest, auditable in-field evaluation of the detector-free cabbage-row guidance.

Runs the cabbage_row estimator over the EXTRACTED FRAMES in temporal order per video (so every
number is reproducible AND labelable), applies temporal fusion, writes a per-frame CSV, and reports:

  RELIABILITY : row-found rate overall + by vegetation tertile (manifest 'veg' column).
  ACCURACY    : heading + cross-track error vs human GT (annotate_rows labels.json), raw AND fused.
  TEMPORAL    : (ANTI-TAUTOLOGY) the fusion gate (--max-jump) is a PHYSICAL heading-rate limit per
                frame-interval, set INDEPENDENTLY of any reporting threshold. We do NOT headline the
                count of jumps the gate rejects. We report instead:
                  (a) suppression as a RATE at several thresholds {10,15,20} deg (descriptive only),
                  (b) where GT exists, heading ERROR-vs-GT on the raw blow-up frames, raw vs fused
                      (the only non-circular robustness evidence),
                  (c) a false-reject proxy: GT error on gated vs non-gated frames (a gate that fires
                      on GOOD frames is lag, not safety).

CSV -> results/cabbage_perframe.csv  (video, frame, t_sec, veg, raw_deg, fused_deg, gated, ct_px, gt_deg, gt_ct_px)

Run:
  python eval_cabbage.py --frames datasets/CabbageNav/frames \
      --labels datasets/CabbageNav/frames/labels.json --out results/cabbage_perframe.csv
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import cabbage_row as cr
import guidance_curve as gc
import calib


def gt_of(coeffs, H, W):
    cf = np.array(coeffs, float)
    return gc.heading_lookahead(cf, H), gc.crosstrack_px(cf, H, W)


def frame_index(name):
    try:
        return int(name.split("__f")[-1].split(".")[0])
    except Exception:
        return 0


def pct(x):
    x = np.asarray(x, float)
    return (np.median(x), x.mean(), np.percentile(x, 90)) if x.size else (np.nan, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--labels", default=None, help="labels.json from annotate_rows (enables ACCURACY + anti-tautology)")
    ap.add_argument("--out", default="results/cabbage_perframe.csv")
    ap.add_argument("--max-jump", type=float, default=8.0,
                    help="fusion gate = max plausible heading change per frame-interval (deg). At 1 fps this is "
                         "the robot's physical heading-rate limit in deg/s; set from the platform, NOT from a metric.")
    ap.add_argument("--win", type=int, default=15, help="temporal median/EMA window")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[10, 15, 20],
                    help="reporting thresholds for the (descriptive) jump-rate table")
    args = ap.parse_args()
    fdir = Path(args.frames)

    # ---- manifest: per-frame veg + video grouping + time ----
    man = {}
    mpath = fdir / "manifest.csv"
    if mpath.exists():
        for r in csv.DictReader(open(mpath, newline="")):
            man[r["frame"]] = {"video": r["video"], "t_sec": float(r.get("t_sec", 0) or 0),
                               "veg": float(r.get("veg", 0) or 0)}
    labels = json.load(open(args.labels)) if (args.labels and Path(args.labels).is_file()) else {}

    # group frames by video, ordered in time
    groups = defaultdict(list)
    if man:
        for nm, m in man.items():
            groups[m["video"]].append(nm)
    else:  # no manifest -> one anonymous group, alphabetical
        groups["all"] = [p.name for p in fdir.glob("*.jpg")]
    for v in groups:
        groups[v].sort(key=frame_index)

    from temporal_guidance import TemporalGuidance
    rows = []  # the per-frame CSV
    for video, names in groups.items():
        tg = TemporalGuidance(win=args.win, max_jump_deg=args.max_jump)
        prev_fused = None
        for nm in names:
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            H, W = img.shape[:2]
            g = cr.guidance(img)
            raw = g["heading"] if g else None
            ct = g["crosstrack_px"] if g else None
            if raw is not None:
                gated = prev_fused is not None and abs(raw - prev_fused) > args.max_jump  # gate would fire
                fused = tg.update_heading(raw)["heading"]
                prev_fused = fused
            else:
                gated = False
                tg.update_heading(None, present=False)
                fused = None
            fused_ct = tg.update_crosstrack(ct, present=ct is not None)["crosstrack"]  # stabilise e too
            gt_deg = gt_ct = None
            if nm in labels:
                gt_deg, gt_ct = gt_of(labels[nm]["coeffs"], labels[nm].get("H", H), labels[nm].get("W", W))
            mm = man.get(nm, {})
            rows.append({"video": video, "frame": nm, "t_sec": mm.get("t_sec", 0), "veg": mm.get("veg", 0),
                         "raw_deg": raw, "fused_deg": fused, "gated": int(gated), "ct_px": ct,
                         "fused_ct_px": fused_ct, "gt_deg": gt_deg, "gt_ct_px": gt_ct})

    # ---- write auditable CSV ----
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "t_sec", "veg", "raw_deg", "fused_deg",
                                          "gated", "ct_px", "fused_ct_px", "gt_deg", "gt_ct_px"])
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {len(rows)} frames -> {args.out}\n")

    found = [r for r in rows if r["raw_deg"] is not None]
    n = len(rows)
    print(f"# RELIABILITY  (n={n} frames, {len(groups)} videos)")
    print(f"  row found: {len(found)}/{n} ({100*len(found)/max(1,n):.1f}%)")
    if found:  # by vegetation tertile
        veg = np.array([r["veg"] for r in rows], float)
        lo, hi = np.percentile(veg, [33, 66])
        for lab, sel in [("low-veg", veg <= lo), ("mid-veg", (veg > lo) & (veg <= hi)), ("high-veg", veg > hi)]:
            idx = np.where(sel)[0]
            fnd = sum(1 for i in idx if rows[i]["raw_deg"] is not None)
            print(f"    {lab:9s}: {fnd}/{len(idx)} found ({100*fnd/max(1,len(idx)):.0f}%)")

    # ---- CROSS-TRACK MAGNITUDE (cm via shared-robot homography; no GT needed) ----
    cmag = [abs(calib.px_cross_to_cm(r["fused_ct_px"], 480, 640)) for r in found if r["fused_ct_px"] is not None]
    if cmag:
        print(f"\n# CROSS-TRACK MAGNITUDE (|offset of the row from centre|, cm via calib homography)")
        print("  |cross-track| cm: median={:.1f} mean={:.1f} p90={:.1f}".format(*pct(cmag)))

    # ---- ACCURACY vs GT ----
    lab = [r for r in rows if r["gt_deg"] is not None and r["raw_deg"] is not None]
    print(f"\n# ACCURACY vs human GT  ({len(lab)} labelled frames)")
    if lab:
        eh_raw = [abs(r["raw_deg"] - r["gt_deg"]) for r in lab]
        eh_fus = [abs(r["fused_deg"] - r["gt_deg"]) for r in lab if r["fused_deg"] is not None]
        ect_raw = [abs(r["ct_px"] - r["gt_ct_px"]) for r in lab if r["ct_px"] is not None and r["gt_ct_px"] is not None]
        ect_fus = [abs(r["fused_ct_px"] - r["gt_ct_px"]) for r in lab if r["fused_ct_px"] is not None and r["gt_ct_px"] is not None]
        ecm_fus = [abs(calib.px_cross_to_cm(r["fused_ct_px"], 480, 640) - calib.px_cross_to_cm(r["gt_ct_px"], 480, 640))
                   for r in lab if r["fused_ct_px"] is not None and r["gt_ct_px"] is not None]
        print("  heading err deg     raw  : median={:.2f} mean={:.2f} p90={:.2f}".format(*pct(eh_raw)))
        print("  heading err deg     fused: median={:.2f} mean={:.2f} p90={:.2f}".format(*pct(eh_fus)))
        print("  cross-track err px  raw  : median={:.1f} mean={:.1f} p90={:.1f}".format(*pct(ect_raw)))
        print("  cross-track err px  fused: median={:.1f} mean={:.1f} p90={:.1f}".format(*pct(ect_fus)))
        print("  cross-track err cm  fused: median={:.2f} mean={:.2f} p90={:.2f}".format(*pct(ecm_fus)))
    else:
        print("  (none yet -> label ~120-150 frames with annotate_rows.py; this is the GO/NO-GO number)")

    # ---- TEMPORAL (anti-tautology) ----
    print(f"\n# TEMPORAL  (fusion gate = {args.max_jump:.0f} deg/frame, a PHYSICAL limit, not a metric)")
    print("  (a) jump-rate, DESCRIPTIVE only (gate is on by construction -- not the headline result):")
    for video, names in groups.items():
        seq_raw = [r["raw_deg"] for r in rows if r["video"] == video and r["raw_deg"] is not None]
        seq_fus = [r["fused_deg"] for r in rows if r["video"] == video and r["fused_deg"] is not None]
        dr = np.abs(np.diff(seq_raw)) if len(seq_raw) > 1 else np.array([])
        df = np.abs(np.diff(seq_fus)) if len(seq_fus) > 1 else np.array([])
        rates = " ".join(f">{int(t)}d raw={int((dr>t).sum())}/{dr.size} fused={int((df>t).sum())}/{df.size}"
                         for t in args.thresholds)
        print(f"    {video}: {rates}")
    blow = [r for r in rows if r["gt_deg"] is not None and r["gated"]]
    print("  (b) error-vs-GT on the gated/blow-up frames (the NON-circular evidence):")
    if blow:
        er = [abs(r["raw_deg"] - r["gt_deg"]) for r in blow]
        ef = [abs(r["fused_deg"] - r["gt_deg"]) for r in blow if r["fused_deg"] is not None]
        print(f"    on {len(blow)} gated+labelled frames: |raw-GT| median={np.median(er):.1f}  "
              f"|fused-GT| median={np.median(ef):.1f} deg")
    else:
        print("    (needs GT labels ON gated frames -> label some blow-up frames)")
    # (c) false-reject proxy
    gated_lab = [r for r in rows if r["gt_deg"] is not None and r["raw_deg"] is not None]
    if gated_lab:
        g_err = [abs(r["raw_deg"] - r["gt_deg"]) for r in gated_lab if r["gated"]]
        ng_err = [abs(r["raw_deg"] - r["gt_deg"]) for r in gated_lab if not r["gated"]]
        if g_err and ng_err:
            print("  (c) false-reject proxy: mean |raw-GT| on gated={:.1f} vs non-gated={:.1f} deg "
                  "(gated should be HIGHER -> gate fires on bad frames)".format(np.mean(g_err), np.mean(ng_err)))


if __name__ == "__main__":
    main()
