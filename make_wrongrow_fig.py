#!/usr/bin/env python
"""
make_wrongrow_fig.py -- wrong-row failure-case overlay (R6-3): the ExG tracker locks
onto a NEIGHBOURING row; the human-GT central row is overlaid for contrast.
Reuses the live estimator (cabbage_row.track) and the standard overlay style.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import cv2

from make_cabbage_figs import track   # same ExG tracker + Huber fit as the method figure
import guidance_curve as gc

FR = Path("datasets/CabbageNav/frames")
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
_LF = FR / "labels.json"
if not _LF.exists():
    _LF = FR / "labels1.json"   # primary annotator (convention A) when a merged labels.json is absent
LABELS = json.load(open(_LF))


def dashed_curve(vis, coeffs, H, color, thick=3, dash=14, gap=10):
    ys = np.arange(int(gc.Y_FIT_LO * H), int(0.99 * H))
    on = True; run = 0; prev = None
    for y in ys:
        p = (int(np.polyval(coeffs, y)), int(y))
        run += 1
        if prev and on:
            cv2.line(vis, prev, p, color, thick)
        prev = p
        if run >= (dash if on else gap):
            on = not on; run = 0


def render(nm):
    img = cv2.imread(str(FR / nm))
    H, W = img.shape[:2]
    pts, cf = track(img)
    vis = img.copy()
    cv2.line(vis, (W // 2, 0), (W // 2, H), (200, 200, 200), 1)
    # pipeline fitted curve (green)
    prev = None
    for y in np.linspace(gc.Y_FIT_LO * H, 0.99 * H, 80):
        p = (int(gc.x_at(cf, y)), int(y))
        if prev:
            cv2.line(vis, prev, p, (0, 230, 0), 3)
        prev = p
    for (x, y) in pts:                                   # ExG peaks (orange)
        cv2.circle(vis, (int(x), int(y)), 5, (0, 165, 255), -1)
    yc = gc.Y_CROSSTRACK * H; xc = gc.x_at(cf, yc)
    cv2.line(vis, (W // 2, int(yc)), (int(xc), int(yc)), (255, 90, 0), 3)   # pipeline cross-track (blue)
    # GT central row (magenta dashed) + its cross-track
    gt = LABELS[nm]["coeffs"]
    dashed_curve(vis, gt, H, (255, 0, 255), thick=3)
    gx = int(np.polyval(gt, yc))
    cv2.circle(vis, (gx, int(yc)), 6, (255, 0, 255), -1)
    e_pred = gc.crosstrack_px(cf, H, W)
    e_gt = float(np.polyval(gt, yc)) - W / 2.0
    cv2.rectangle(vis, (0, 0), (W, 56), (0, 0, 0), -1)
    cv2.putText(vis, f"pipeline e={e_pred:+.1f}px (neighbour row)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 0), 2, cv2.LINE_AA)
    # all three printed offsets carry one decimal, so the difference is exact rather than
    # reconciled against rounded values
    cv2.putText(vis, f"GT e={e_gt:+.1f}px (central row)  ->  wrong-row error {abs(e_pred - e_gt):.1f}px",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 255), 2, cv2.LINE_AA)
    print(f"  e_pred={e_pred:+.4f}  e_gt={e_gt:+.4f}  |diff|={abs(e_pred - e_gt):.4f}")
    return img, vis


def main():
    cands = sys.argv[1:] or ["videos__rec_20251130_030636__f013560.jpg"]
    for i, nm in enumerate(cands):
        img, vis = render(nm)
        panel = np.hstack([img, np.full((img.shape[0], 12, 3), 255, np.uint8), vis])
        out = OUT / (f"_cand_{i}.png" if len(cands) > 1 else "fig_wrongrow.png")
        cv2.imwrite(str(out), panel)
        print("wrote", out.name, "from", nm)


if __name__ == "__main__":
    main()
