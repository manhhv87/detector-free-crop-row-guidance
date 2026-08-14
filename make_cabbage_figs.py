#!/usr/bin/env python
"""
make_cabbage_figs.py -- generate the paper's method/qualitative figures from the real robot frames.

Produces (in paper/figures/):
  fig_cabbage_method.png   : raw forward-POV frame | overlay (ExG column-peak points, robust quadratic
                             guidance curve x=P(y), look-ahead tangent = heading, near-row cross-track).
  fig_cabbage_gallery.png  : 2x2 overlays across vegetation-density levels (qualitative robustness).
No ground truth needed; pure illustration of the live estimator on real frames.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import cv2

from periodic_row import exg
import guidance_curve as gc

FR = Path("datasets/CabbageNav/frames")
OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)


def track(img, n_bands=9, corridor=0.18, smooth=21):
    H, W = img.shape[:2]; V = exg(img); pts = []; cur = None
    for y in np.linspace(gc.Y_FIT_HI * H, gc.Y_FIT_LO * H, n_bands):
        y0, y1 = int(y - 0.04 * H), int(y + 0.04 * H)
        prof = V[max(0, y0):y1].sum(axis=0)
        if smooth > 1:
            prof = np.convolve(prof, np.ones(smooth) / smooth, "same")
        if prof.max() < 1e-6:
            continue
        thr = prof.mean() + 0.3 * prof.std()
        pk = [i for i in range(3, len(prof) - 3) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > thr]
        if not pk:
            continue
        cur = min(pk, key=lambda x: abs(x - W / 2)) if cur is None else \
            min([x for x in pk if abs(x - cur) < corridor * W] or [cur], key=lambda x: abs(x - cur))
        pts.append((cur, 0.5 * (y0 + y1)))
    if len(pts) < 3:
        return None, None
    pts = np.array(pts, float)
    return pts, gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)


def draw(img, pts, coeffs):
    H, W = img.shape[:2]; vis = img.copy()
    cv2.line(vis, (W // 2, 0), (W // 2, H), (200, 200, 200), 1)            # image centre
    prev = None                                                            # fitted curve (green)
    for y in np.linspace(gc.Y_FIT_LO * H, 0.99 * H, 80):
        p = (int(gc.x_at(coeffs, y)), int(y))
        if prev:
            cv2.line(vis, prev, p, (0, 230, 0), 3)
        prev = p
    for (x, y) in pts:                                                     # tracked ExG peaks (orange)
        cv2.circle(vis, (int(x), int(y)), 5, (0, 165, 255), -1)
    yl = gc.Y_LOOKAHEAD * H; xl = gc.x_at(coeffs, yl); m = gc.slope_at(coeffs, yl); L = 0.30 * H
    cv2.line(vis, (int(xl - m * L), int(yl - L)), (int(xl + m * 0.12 * L), int(yl + 0.12 * L)), (0, 0, 255), 3)
    cv2.circle(vis, (int(xl), int(yl)), 7, (0, 0, 255), 2)                 # look-ahead tangent (red)
    yc = gc.Y_CROSSTRACK * H; xc = gc.x_at(coeffs, yc)
    cv2.line(vis, (W // 2, int(yc)), (int(xc), int(yc)), (255, 90, 0), 3)  # cross-track (blue)
    th = gc.heading_lookahead(coeffs, H); e = gc.crosstrack_px(coeffs, H, W)
    cv2.rectangle(vis, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(vis, f"theta={th:+.1f} deg   e={e:+.0f} px", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def pick(rows, video, q):
    sub = [r for r in rows if r["video"] == video]
    sub.sort(key=lambda r: float(r["veg"]))
    return sub[min(len(sub) - 1, int(q * len(sub)))]["frame"]


def main():
    rows = list(csv.DictReader(open(FR / "manifest.csv", newline="")))
    vids = sorted({r["video"] for r in rows})
    # method figure: a clear, mid-vegetation frame
    nm = pick(rows, vids[-1], 0.6)
    img = cv2.imread(str(FR / nm)); pts, cf = track(img)
    vis = draw(img, pts, cf)
    panel = np.hstack([img, np.full((img.shape[0], 12, 3), 255, np.uint8), vis])
    cv2.imwrite(str(OUT / "fig_cabbage_method.png"), panel)
    print("wrote fig_cabbage_method.png from", nm)
    # gallery: 4 frames across vegetation-density levels, all from run 2.
    # Hand-picked CLEAR mid-row frames (verified frame-by-frame): people gather at the
    # headlands where each pass starts, so only the mid-row stretches are free of legs
    # and feet; these four are people-free with a clearly trackable central row.
    picks = ["videos__rec_20251130_124123__f003960.jpg",
             "videos__rec_20251130_124123__f012160.jpg",
             "videos__rec_20251130_124123__f013260.jpg",
             "videos__rec_20251130_124123__f014060.jpg"]
    tiles = []
    for nm in picks:
        im = cv2.imread(str(FR / nm)); p, c = track(im)
        tiles.append(draw(im, p, c) if c is not None else im)
    row1 = np.hstack([tiles[0], tiles[1]]); row2 = np.hstack([tiles[2], tiles[3]])
    cv2.imwrite(str(OUT / "fig_cabbage_gallery.png"), np.vstack([row1, row2]))
    print("wrote fig_cabbage_gallery.png from", picks)


if __name__ == "__main__":
    main()
