#!/usr/bin/env python
"""
exp_wrongrow.py -- seed-corridor sweep for the wrong-row tail, on the same frames/GT as eval_cabbage.

Runs the full per-video stream in time order (so the temporal stage has its history) for a few seed
corridor half-widths and reports, for each:
  - all-150-GT heading error (raw/fused) and fused cross-track error;
  - the wrong-row count (raw |ct - gt| > 100 px) and the fused cross-track error on the 220 mm baseline's
    wrong-row frames (the same set across widths, so improvements are measured on the same frames);
  - C3 (gated frames raw->fused heading) and gate selectivity, to catch regressions.

This is how the +/-180 mm default in cabbage_row.row_anchors was chosen: 180 mm cuts the wrong-row count and
the cross-track tail without regressing heading or C3, while 150 mm over-tightens and worsens the tail.
Stateful/feedback front-end variants (carrying the previous fused cross-track back as a predicted seed) were
also tried and DISCARDED -- they destabilise into a self-reinforcing wrong-row lock. Read-only on results/.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import cabbage_row as cr
import guidance_curve as gc
from temporal_guidance import TemporalGuidance

CORRIDORS = [220.0, 180.0, 150.0]


def frame_index(name):
    try:
        return int(name.split("__f")[-1].split(".")[0])
    except Exception:
        return 0


def pct(x):
    x = np.asarray([v for v in x if v is not None], float)
    return (np.median(x), x.mean(), np.percentile(x, 90)) if x.size else (np.nan, np.nan, np.nan)


def run(safe_mm, groups, fdir, labels):
    rows = []
    for video, names in groups.items():
        tg = TemporalGuidance(win=15, max_jump_deg=8.0)
        prev_fused = None
        for nm in names:
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            H, W = img.shape[:2]
            g = cr.guidance_coeffs(img, safe_mm=safe_mm)
            raw = ct = None
            if g is not None:
                raw = gc.heading_lookahead(g, H); ct = gc.crosstrack_px(g, H, W)
            if raw is not None:
                gated = prev_fused is not None and abs(raw - prev_fused) > 8.0
                fused = tg.update_heading(raw)["heading"]; prev_fused = fused
            else:
                gated = False; tg.update_heading(None, present=False); fused = None
            fused_ct = tg.update_crosstrack(ct, present=ct is not None)["crosstrack"]
            gt_deg = gt_ct = None
            if nm in labels:
                cf = np.array(labels[nm]["coeffs"], float)
                hh = labels[nm].get("H", H); ww = labels[nm].get("W", W)
                gt_deg = gc.heading_lookahead(cf, hh); gt_ct = gc.crosstrack_px(cf, hh, ww)
            rows.append(dict(frame=nm, raw=raw, fused=fused, gated=gated, ct=ct,
                             fused_ct=fused_ct, gt_deg=gt_deg, gt_ct=gt_ct))
    return rows


def report(name, rows, base_wrong):
    lab = [r for r in rows if r["gt_deg"] is not None and r["raw"] is not None]
    eh_raw = [abs(r["raw"] - r["gt_deg"]) for r in lab]
    eh_fus = [abs(r["fused"] - r["gt_deg"]) for r in lab if r["fused"] is not None]
    ect = [abs(r["fused_ct"] - r["gt_ct"]) for r in lab if r["fused_ct"] is not None and r["gt_ct"] is not None]
    wrong = {r["frame"] for r in lab if r["ct"] is not None and r["gt_ct"] is not None and abs(r["ct"] - r["gt_ct"]) > 100}
    onbase = [abs(r["fused_ct"] - r["gt_ct"]) for r in lab
              if r["frame"] in base_wrong and r["fused_ct"] is not None and r["gt_ct"] is not None]
    g = [r for r in lab if r["gated"]]
    c3r = np.median([abs(r["raw"] - r["gt_deg"]) for r in g]) if g else float("nan")
    c3f = np.median([abs(r["fused"] - r["gt_deg"]) for r in g if r["fused"] is not None]) if g else float("nan")
    print(f"\n=== corridor {name} mm ===")
    print("  heading raw   med/mean/p90: %.2f / %.2f / %.2f" % pct(eh_raw))
    print("  heading fused med/mean/p90: %.2f / %.2f / %.2f" % pct(eh_fus))
    print("  cross-track fused med/p90 : %.1f / %.1f px" % (pct(ect)[0], pct(ect)[2]))
    print("  wrong-row count (raw|ct-gt|>100): %d" % len(wrong))
    if onbase:
        print("  fused cross-track err on 220mm wrong-row frames: med %.0f / p90 %.0f px"
              % (np.median(onbase), np.percentile(onbase, 90)))
    print("  C3 gated heading raw->fused: %.1f -> %.1f (n_gated=%d)" % (c3r, c3f, len(g)))
    return wrong


def _default_labels():
    """The repo ships labels1.json (primary annotator) and labels2.json (second annotator);
    a merged labels.json is optional, so fall back to the primary rather than failing."""
    base = Path("datasets/CabbageNav/frames")
    merged = base / "labels.json"
    return str(merged if merged.exists() else base / "labels1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--labels", default=_default_labels(),
                    help="GT labels; defaults to the merged labels.json when present, "
                         "else the primary annotator's labels1.json")
    args = ap.parse_args()
    fdir = Path(args.frames)
    man = {r["frame"]: r["video"] for r in csv.DictReader(open(fdir / "manifest.csv", newline=""))}
    labels = json.load(open(args.labels)) if Path(args.labels).is_file() else {}
    groups = defaultdict(list)
    for nm, v in man.items():
        groups[v].append(nm)
    for v in groups:
        groups[v].sort(key=frame_index)
    print(f"frames={sum(len(v) for v in groups.values())}  labelled={len(labels)}  videos={len(groups)}")

    base = run(220.0, groups, fdir, labels)
    base_wrong = {r["frame"] for r in base if r["gt_deg"] is not None and r["ct"] is not None
                  and r["gt_ct"] is not None and abs(r["ct"] - r["gt_ct"]) > 100}
    report("220 (baseline)", base, set())
    print(f"\n(220 mm wrong-row set = {len(base_wrong)} frames; same set scored for every width)")
    for sm in CORRIDORS:
        if sm == 220.0:
            continue
        report(str(int(sm)), run(sm, groups, fdir, labels), base_wrong)


if __name__ == "__main__":
    main()
