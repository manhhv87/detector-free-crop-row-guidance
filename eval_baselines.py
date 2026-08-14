#!/usr/bin/env python
"""
eval_baselines.py -- front-end, vegetation-index and temporal baselines for the detector-free pipeline.

Tests the paper's "the front end / fit is secondary, the temporal stage is the lever" thesis by swapping the
parts that the fit-arm comparison did not. Every method feeds the SAME downstream (robust fit where it
applies, then the deployed temporal stage), so only the named component differs, and each is scored against
the 150-frame human-labelled subset.

  Front ends (anchor/line source): ExG column-peak (deployed), centroid-per-band, Hough lines, periodic_row.
  Vegetation indices (same peak tracker, swapped map): ExG (deployed), ExGR, CIVE, NDI.
  Temporal stage (ExG front end): raw, median+EMA+gate (deployed), Kalman.

Pure cv2/numpy; no torch. Run: python eval_baselines.py
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import cabbage_row as cr
import guidance_curve as gc
import calib
from periodic_row import exg as pr_exg, guidance_line_periodic
from temporal_guidance import TemporalGuidance


# ---------- vegetation indices -> veg map (higher = more vegetation), normalised like ExG ----------
def _norm(v):
    v = np.clip(v, 0.0, None)
    m = v.max()
    return v / m if m > 0 else v


def _chan(img):
    b, g, r = (img[:, :, i].astype(np.float32) for i in range(3))
    s = b + g + r + 1e-6
    return r / s, g / s, b / s


def idx_exg(img):
    return pr_exg(img)


def idx_exgr(img):
    r, g, b = _chan(img)
    return _norm((2 * g - r - b) - (1.4 * r - g))      # Excess Green minus Excess Red


def idx_cive(img):
    r, g, b = _chan(img)
    return _norm(-(0.441 * r - 0.811 * g + 0.385 * b))  # CIVE (negated: low CIVE = vegetation)


def idx_ndi(img):
    r, g, b = _chan(img)
    return _norm((g - r) / (g + r + 1e-6))             # normalised difference index


# ---------- front ends: image -> guidance coeffs (np.polyval order) or None ----------
def fe_exg_peak(img):
    pts = cr.row_anchors(img)
    if pts is None:
        return None
    H, W = img.shape[:2]
    return gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)


def fe_index_peak(img, idx):
    pts = cr.row_anchors(img, V=idx(img))
    if pts is None:
        return None
    H, W = img.shape[:2]
    return gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)


def fe_centroid(img):
    H, W = img.shape[:2]
    V = idx_exg(img)
    pts = []
    for y in np.linspace(gc.Y_FIT_HI * H, gc.Y_FIT_LO * H, 9):
        y0, y1 = int(y - 0.04 * H), int(y + 0.04 * H)
        prof = V[max(0, y0):y1].sum(axis=0)
        cb = calib.corridor_px(0.5 * (y0 + y1), W)          # vegetation centroid within the safety corridor
        lo, hi = (int(max(0, cb[0])), int(min(W, cb[1]))) if cb else (0, W)
        seg = prof[lo:hi]
        if seg.sum() < 1e-6:
            continue
        cx = lo + float(np.sum(np.arange(len(seg)) * seg) / seg.sum())
        pts.append((cx, 0.5 * (y0 + y1)))
    if len(pts) < 3:
        return None
    pts = np.array(pts, float)
    return gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)


def fe_hough(img):
    H, W = img.shape[:2]
    V = idx_exg(img)
    mask = (V > 0.2).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40,
                            minLineLength=int(0.15 * H), maxLineGap=25)
    if lines is None:
        return None
    pts = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if abs(y2 - y1) > abs(x2 - x1) and abs(0.5 * (x1 + x2) - W / 2) < 0.30 * W:  # near-vertical, central
            pts += [(x1, y1), (x2, y2)]
    if len(pts) < 2:
        return None
    pts = np.array(pts, float)
    return gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2 if len(pts) >= 3 else 1)


def fe_periodic(img):
    gl = guidance_line_periodic(img)
    return np.array([gl["a"], gl["b"]], float) if gl else None


# ---------- a minimal constant-velocity Kalman filter on the heading stream ----------
class KF1D:
    def __init__(self, q=0.5, r=9.0):
        self.q, self.r = q, r
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.Hm = np.array([[1.0, 0.0]])
        self.x = None
        self.P = None

    def update(self, z):
        if self.x is None:
            if z is None or not np.isfinite(z):
                return None
            self.x = np.array([z, 0.0]); self.P = np.eye(2) * 10.0
            return z
        self.x = self.F @ self.x                              # predict
        self.P = self.F @ self.P @ self.F.T + self.q * np.eye(2)
        if z is not None and np.isfinite(z):                 # update
            y = z - (self.Hm @ self.x)[0]
            S = (self.Hm @ self.P @ self.Hm.T)[0, 0] + self.r
            K = (self.P @ self.Hm.T / S)[:, 0]
            self.x = self.x + K * y
            self.P = (np.eye(2) - np.outer(K, self.Hm)) @ self.P
        return float(self.x[0])


def pct(v):
    a = np.asarray([x for x in v if x is not None], float)
    return (np.median(a), a.mean(), np.percentile(a, 90)) if len(a) else (float("nan"),) * 3


def frame_index(name):
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 0


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
    labels = json.load(open(args.labels))
    gt = {}
    for nm, lab in labels.items():
        c = np.array(lab["coeffs"], float)
        gt[nm] = (lab.get("heading_la", gc.heading_lookahead(c, 480)), c)

    groups = defaultdict(list)
    mpath = fdir / "manifest.csv"
    if mpath.exists():
        for r in csv.DictReader(open(mpath, newline="")):
            groups[r["video"]].append(r["frame"])
    else:
        groups["all"] = [p.name for p in fdir.glob("*.jpg")]
    for v in groups:
        groups[v].sort(key=frame_index)

    methods = {
        "ExG-peak (deployed)": fe_exg_peak,
        "centroid-band": fe_centroid,
        "Hough lines": fe_hough,
        "periodic_row": fe_periodic,
        "ExGR-peak": lambda im: fe_index_peak(im, idx_exgr),
        "CIVE-peak": lambda im: fe_index_peak(im, idx_cive),
        "NDI-peak": lambda im: fe_index_peak(im, idx_ndi),
    }
    # per method: errors vs GT
    raw_err = {k: [] for k in methods}      # (heading_err, ct_err)
    fus_err = {k: [] for k in methods}
    emit = {k: 0 for k in methods}
    nframes = 0
    # temporal-method comparison on the ExG front end
    temp_err = {"raw": [], "median+EMA+gate": [], "Kalman": []}

    for video, names in groups.items():
        tg = {k: TemporalGuidance() for k in methods}
        prevf = {k: None for k in methods}
        kf = KF1D()
        for nm in names:
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            nframes += 1
            H, W = img.shape[:2]
            g = gt.get(nm)
            for k, fn in methods.items():
                try:
                    cf = fn(img)
                except Exception:
                    cf = None
                if cf is None:
                    tg[k].update_heading(None, present=False)
                    continue
                emit[k] += 1
                rh = gc.heading_lookahead(cf, H)
                rct = gc.crosstrack_px(cf, H, W)
                fh = tg[k].update_heading(rh)["heading"]
                fct = tg[k].update_crosstrack(rct)["crosstrack"]
                if g is not None:
                    gtd = g[0]; gtct = gc.crosstrack_px(g[1], H, W)
                    raw_err[k].append((abs(rh - gtd), abs(rct - gtct)))
                    if fh is not None:
                        fus_err[k].append((abs(fh - gtd), abs(fct - gtct) if fct is not None else None))
                if k == "ExG-peak (deployed)":
                    kh = kf.update(rh)
                    if g is not None:
                        temp_err["raw"].append(abs(rh - g[0]))
                        if fh is not None:
                            temp_err["median+EMA+gate"].append(abs(fh - g[0]))
                        if kh is not None:
                            temp_err["Kalman"].append(abs(kh - g[0]))

    print(f"# FRONT-END + INDEX BASELINES   (n={nframes} frames; errors vs {len(gt)} GT frames)")
    print(f"{'method':20s} | emit% | raw heading (med/mean/p90) | fused heading (med/mean/p90)")
    for k in methods:
        rh = pct([e[0] for e in raw_err[k]])
        fh = pct([e[0] for e in fus_err[k]])
        print("%-20s | %4.0f%% |   %5.2f / %5.2f / %5.2f      |   %5.2f / %5.2f / %5.2f"
              % (k, 100 * emit[k] / max(1, nframes), *rh, *fh))

    print(f"\n# TEMPORAL STAGE   (ExG-peak front end, heading err vs GT, deg)")
    for k in ("raw", "median+EMA+gate", "Kalman"):
        print("%-18s : med/mean/p90 = %5.2f / %5.2f / %5.2f" % (k, *pct(temp_err[k])))


if __name__ == "__main__":
    main()
