#!/usr/bin/env python
"""
cabbage_row.py -- detector-free central crop-row guidance for the cabbage robot (forward POV).

The robot's onboard camera (640x480) sees a cabbage bed receding ahead. Cabbage is dense/high-contrast,
so a simple Excess-Green column-profile tracker locates the central row reliably (verified: row found in
100% of sampled robot-run frames) -- no trained detector needed. Returns a curved guidance fit
(guidance_curve) so heading = tangent at the look-ahead row. Consumed by eval_cabbage.

guidance_coeffs(img) -> polynomial coeffs (np.polyfit order) of x=P(y), or None.
guidance(img)        -> {coeffs, heading, crosstrack_px, cx, cy} or None.
"""
from __future__ import annotations
import numpy as np

from periodic_row import exg
import guidance_curve as gc

try:
    import calib as _calib                     # for the ground-corridor gate; optional
except Exception:
    _calib = None


def _peaks(p, thr):
    return [i for i in range(3, len(p) - 3) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > thr]


def row_anchors(img, n_bands=9, corridor=0.18, smooth=21, safe_mm=180.0, gate="seed", V=None):
    """Track the central cabbage row bottom->top via Excess-Green column peaks; return the anchor points.
    V: optional precomputed vegetation map (higher = more vegetation); defaults to the Excess-Green index.
    Passing a different index here runs the identical tracker on a different front end (for baselines).

    safe_mm/gate: the ground 'virtual safety corridor' (+/-safe_mm around the robot centreline) that drops
    side-row peaks. gate='seed' applies it only when choosing the FIRST (near-row) anchor -- this stops the
    track from starting on a neighbour row without clipping the far look-ahead anchors (a narrow far-field
    corridor otherwise hurts heading). gate='all' applies it to every band; gate=None or safe_mm=None
    disables it. Needs calib.

    safe_mm defaults to 180 mm, inside the chassis half-width (220 mm = half the Wsafe=440 mm straddle
    corridor). The neighbour rows sit near the chassis edge, so when the central row is momentarily gapped a
    +/-220 mm seed band still admits the neighbour and the (memoryless) tracker re-seeds onto it every frame
    of the gap. Tightening the SEED band to +/-180 mm excludes that neighbour while still admitting the
    central row's lateral wander; this cuts the wrong-row frames without regressing heading or the temporal
    gate (validated on the labelled subset)."""
    H, W = img.shape[:2]
    if V is None:
        V = exg(img)
    pts, cur = [], None
    for y in np.linspace(gc.Y_FIT_HI * H, gc.Y_FIT_LO * H, n_bands):
        y0, y1 = int(y - 0.04 * H), int(y + 0.04 * H)
        prof = V[max(0, y0):y1].sum(axis=0)
        if smooth > 1:
            prof = np.convolve(prof, np.ones(smooth) / smooth, "same")
        if prof.max() < 1e-6:
            continue
        thr = prof.mean() + 0.3 * prof.std()
        pk = _peaks(prof, thr)
        if safe_mm and gate and _calib is not None and (gate == "all" or cur is None):
            cb = _calib.corridor_px(0.5 * (y0 + y1), W, half_mm=safe_mm)   # drop out-of-corridor side rows
            if cb is not None:
                pk = [x for x in pk if cb[0] <= x <= cb[1]]
        if not pk:
            continue
        if cur is None:
            cur = min(pk, key=lambda x: abs(x - W / 2))          # bottom: nearest image centre
        else:
            cand = [x for x in pk if abs(x - cur) < corridor * W]
            cur = min(cand or [cur], key=lambda x: abs(x - cur))  # track without jumping rows
        pts.append((cur, 0.5 * (y0 + y1)))
    return np.array(pts, float) if len(pts) >= 3 else None


def guidance_coeffs(img, **kw):
    """Robust Huber quadratic fit x=P(y) to the central-row anchors of row_anchors()."""
    pts = row_anchors(img, **kw)
    if pts is None:
        return None
    H, W = img.shape[:2]
    return gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)


def guidance(img):
    cf = guidance_coeffs(img)
    if cf is None:
        return None
    H, W = img.shape[:2]
    return {"coeffs": cf, "heading": gc.heading_lookahead(cf, H), "crosstrack_px": gc.crosstrack_px(cf, H, W)}


if __name__ == "__main__":
    import sys, cv2, math
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is None:
        print("usage: cabbage_row.py <frame.jpg>"); raise SystemExit
    g = guidance(img)
    print("None" if g is None else {k: (round(v, 2) if isinstance(v, float) else v)
                                    for k, v in g.items() if k in ("heading", "crosstrack_px")})
