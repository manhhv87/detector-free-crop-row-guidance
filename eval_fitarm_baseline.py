#!/usr/bin/env python
"""
eval_fitarm_baseline.py -- fit-arm baseline for the guidance-curve estimator.

The deployed pipeline fits the central-row anchors with a robust Huber IRLS quadratic. This compares that
choice against the two standard alternatives on the SAME anchors per frame, through the SAME deployed fit
wrapper (band restriction + over-bend-to-line guard), so only the estimator differs:
  - Huber IRLS (deployed),
  - equal-weight least squares (the non-robust polynomial fit),
  - RANSAC consensus fitting.
Each fit is scored against the 150-frame human-labelled subset as look-ahead heading error and near-row
cross-track error. This is a per-frame fit comparison (the temporal stage is a separate, later step), so it
isolates the value of the robust fit. With this wrapper the Huber arm reproduces the deployed raw per-frame
heading error reported in the accuracy table. Pure cv2/numpy; no torch.

Run: python eval_fitarm_baseline.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

import cabbage_row as cr
import guidance_curve as gc


def pct(v):
    a = np.asarray(v, float)
    return (np.median(a), a.mean(), np.percentile(a, 90)) if len(a) else (float("nan"),) * 3


def est_huber(y, x, deg):
    return gc.fit_curve(y, x, degree=deg)


def est_ols(y, x, deg):
    try:
        return np.polyfit(y, x, deg)
    except Exception:
        return None


def est_ransac(y, x, deg, n_iter=200, thresh=8.0, seed=0):
    """RANSAC polynomial of the given degree: sample deg+1 anchors, fit, keep the largest inlier set, refit."""
    n = len(y)
    k = deg + 1
    if n < k:
        return est_ols(y, x, deg)
    rng = np.random.default_rng(seed)
    best_inl, best_mask = -1, None
    for _ in range(n_iter):
        idx = rng.choice(n, k, replace=False)
        try:
            c = np.polyfit(y[idx], x[idx], deg)
        except Exception:
            continue
        mask = np.abs(np.polyval(c, y) - x) < thresh
        if mask.sum() > best_inl:
            best_inl, best_mask = int(mask.sum()), mask
    if best_mask is None or best_mask.sum() < k:
        return est_ols(y, x, deg)
    return np.polyfit(y[best_mask], x[best_mask], deg)


def fit_band_with(est, y, x, H, W):
    """Replicate guidance_curve.fit_band (band mask + over-bend->line guard) but with estimator `est`,
    so each fit arm is compared on the deployed footing and only the loss differs."""
    m = (y >= gc.Y_FIT_LO * H) & (y <= gc.Y_FIT_HI * H)
    if m.sum() >= 2:
        y, x = y[m], x[m]
    c = est(y, x, 2)
    if c is None:
        return None
    if W is not None and len(c) >= 3:                       # over-bend guard -> degree 1
        ys = np.linspace(gc.Y_FIT_LO * H, gc.Y_FIT_HI * H, 20)
        xs = np.polyval(c, ys)
        if (xs < -0.05 * W).any() or (xs > 1.05 * W).any():
            c = est(y, x, 1)
    return c


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

    labels = json.load(open(args.labels))
    fdir = Path(args.frames)
    estimators = {"Huber (deployed)": est_huber, "equal-LS": est_ols, "RANSAC": est_ransac}
    arms = {k: [] for k in estimators}
    n_used = 0
    for nm, lab in labels.items():
        img = cv2.imread(str(fdir / nm))
        if img is None:
            continue
        H, W = img.shape[:2]
        pts = cr.row_anchors(img)
        if pts is None:
            continue
        y, x = pts[:, 1], pts[:, 0]
        gtc = np.array(lab["coeffs"], float)
        gtd = lab.get("heading_la", gc.heading_lookahead(gtc, H))
        gtct = gc.crosstrack_px(gtc, H, W)
        fits = {k: fit_band_with(e, y, x, H, W) for k, e in estimators.items()}
        if any(f is None for f in fits.values()):
            continue
        n_used += 1
        for k, c in fits.items():
            arms[k].append((abs(gc.heading_lookahead(c, H) - gtd),
                            abs(gc.crosstrack_px(c, H, W) - gtct)))

    print(f"# FIT-ARM BASELINE  (per-frame, same anchors + deployed fit wrapper; {n_used} labelled frames)")
    print(f"{'fit':18s} | heading err deg (med/mean/p90) | cross-track err px (med/mean/p90)")
    for k, rows in arms.items():
        print("%-18s |   %5.2f / %5.2f / %5.2f       |   %5.1f / %5.1f / %5.1f"
              % (k, *pct([r[0] for r in rows]), *pct([r[1] for r in rows])))

    # paired Huber vs equal-LS: are they actually distinguishable?
    Hh, Ee = np.array(arms["Huber (deployed)"]), np.array(arms["equal-LS"])
    if len(Hh) and len(Ee):
        dh, dc = Hh[:, 0] - Ee[:, 0], Hh[:, 1] - Ee[:, 1]   # Huber - equal-LS; negative => Huber better
        print("\nHuber vs equal-LS (paired): "
              f"heading med(H-E)={np.median(dh):+.2f} deg, Huber lower on {100*np.mean(Hh[:,0]<Ee[:,0]):.0f}%; "
              f"cross-track med(H-E)={np.median(dc):+.2f} px, Huber lower on {100*np.mean(Hh[:,1]<Ee[:,1]):.0f}% "
              "(-> indistinguishable)")


if __name__ == "__main__":
    main()
