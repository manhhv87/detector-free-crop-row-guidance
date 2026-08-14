#!/usr/bin/env python
"""
iaa_floor.py -- inter-annotator agreement (IAA) floor for the GT guidance line.
Two annotators independently labelled the same frames (convention A); the floor is
the median absolute difference between their look-ahead heading (deg) and cross-track
(px) -- the smallest meaningful error for the accuracy comparison.

Run:  python results/iaa_floor.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import guidance_curve as gc
FR = os.path.join(ROOT, "datasets", "CabbageNav", "frames")


def hd(lbl):
    cf = np.array(lbl["coeffs"], float)
    H = lbl.get("H", 480); W = lbl.get("W", 640)
    return gc.heading_lookahead(cf, H), gc.crosstrack_px(cf, H, W)


def stat(a):
    return dict(median=float(np.median(a)), mean=float(np.mean(a)),
                p90=float(np.percentile(a, 90)), max=float(np.max(a)))


def main():
    L1 = json.load(open(os.path.join(FR, "labels1.json")))
    L2 = json.load(open(os.path.join(FR, "labels2.json")))
    common = sorted(set(L1) & set(L2))
    dh, dc = [], []
    for k in common:
        h1, c1 = hd(L1[k]); h2, c2 = hd(L2[k])
        dh.append(abs(h1 - h2)); dc.append(abs(c1 - c2))
    dh, dc = np.array(dh), np.array(dc)
    print(f"IAA over n={len(common)} common frames (annotator 1 vs 2)")
    print("  heading |dtheta| deg:", {k: round(v, 2) for k, v in stat(dh).items()})
    print("  cross-track |de| px :", {k: round(v, 2) for k, v in stat(dc).items()})
    try:
        import calib
        dccm = np.array([abs(calib.px_cross_to_cm(c, 480, 640)) for c in dc])
        print("  cross-track |de| cm :", {k: round(v, 2) for k, v in stat(dccm).items()})
    except Exception as e:
        print("  (cm skipped:", repr(e)[:60], ")")


if __name__ == "__main__":
    main()
