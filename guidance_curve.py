#!/usr/bin/env python
"""
guidance_curve.py -- shared curved-guidance model for crop rows.

Crop rows bow in the far field (lens distortion, terrain, contour planting), so a single straight
line x=a*y+b is biased exactly where steering is sensitive. We model the central row as a low-order
POLYNOMIAL x = P(y) fit ROBUSTLY on the reliable near-to-mid band, and -- crucially for control --
take the steering HEADING as the TANGENT of the curve at a fixed LOOK-AHEAD row (not the global
slope), with cross-track measured at a near row. This matches pure-pursuit/Stanley control and is
standard practice in lane detection.

All coords are image pixels (y down, x right). Coeffs follow numpy.polyfit (highest power first);
degree 1 reduces to the old straight line, so this is backward compatible.
"""
from __future__ import annotations
import math
import numpy as np

# fractions of image height H
Y_FIT_LO = 0.45      # fit only the reliable near-to-mid band (drop the curvy/uncertain far field)
Y_FIT_HI = 0.98
Y_LOOKAHEAD = 0.65   # heading = tangent of the curve here (the steering reference row)
Y_CROSSTRACK = 0.92  # cross-track = offset here (near the robot)


def fit_curve(cy, cx, degree=2, irls_iters=5):
    """Robust (Huber-IRLS) polynomial fit x = P(y). Returns coeffs (np.polyfit order).
    Falls back to a lower degree if too few points. degree=1 => straight line."""
    cy = np.asarray(cy, float); cx = np.asarray(cx, float)
    n = len(cy)
    deg = min(degree, max(1, n - 1))
    if n < 2:
        return np.array([0.0, float(cx[0]) if n else 0.0])
    if cy.max() - cy.min() < 1e-6:
        return np.array([0.0, float(np.median(cx))])
    w = np.ones(n)
    coeffs = np.polyfit(cy, cx, deg, w=w)
    for _ in range(int(irls_iters)):
        r = np.abs(cx - np.polyval(coeffs, cy))
        s = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 1.345 * s, 1.0, 1.345 * s / (r + 1e-9))
        coeffs = np.polyfit(cy, cx, deg, w=w)
    return coeffs


def x_at(coeffs, y):
    return float(np.polyval(coeffs, y))


def slope_at(coeffs, y):
    """dx/dy at row y (the local tangent slope)."""
    return float(np.polyval(np.polyder(np.asarray(coeffs, float)), y))


def heading_at(coeffs, y):
    """Steering heading (deg from vertical) = atan of the tangent slope at look-ahead row y."""
    return math.degrees(math.atan(slope_at(coeffs, y)))


def heading_lookahead(coeffs, H):
    return heading_at(coeffs, Y_LOOKAHEAD * H)


def crosstrack_px(coeffs, H, W):
    """Lateral offset of the row from image centre at the near cross-track row (px)."""
    return x_at(coeffs, Y_CROSSTRACK * H) - W / 2.0


def fit_band(cy, cx, H, W=None, degree=2):
    """Fit points inside the reliable [Y_FIT_LO, Y_FIT_HI]*H band. If W is given and the quadratic
    leaves the image [0,W] within the band (over-bending on a few frames), fall back to a line."""
    cy = np.asarray(cy, float); cx = np.asarray(cx, float)
    m = (cy >= Y_FIT_LO * H) & (cy <= Y_FIT_HI * H)
    if m.sum() >= 2:
        cy, cx = cy[m], cx[m]
    coeffs = fit_curve(cy, cx, degree=degree)
    if W is not None and len(coeffs) >= 3:                 # over-bend guard -> degree 1
        ys = np.linspace(Y_FIT_LO * H, Y_FIT_HI * H, 20)
        xs = np.polyval(coeffs, ys)
        if (xs < -0.05 * W).any() or (xs > 1.05 * W).any():
            coeffs = fit_curve(cy, cx, degree=1)
    return coeffs


if __name__ == "__main__":
    # smoke: a curved row (quadratic) -- straight fit biases heading; curve+tangent recovers it.
    H, W = 1080, 1920
    ys = np.linspace(0.45 * H, 0.98 * H, 12)
    true = np.array([0.00035, -0.20, 1000.0])          # x = 0.00035 y^2 - 0.20 y + 1000 (bows)
    xs = np.polyval(true, ys) + np.random.default_rng(0).normal(0, 3, len(ys))
    cf2 = fit_curve(ys, xs, degree=2)
    cf1 = fit_curve(ys, xs, degree=1)
    yla = Y_LOOKAHEAD * H
    print(f"true tangent@LA = {math.degrees(math.atan(np.polyval(np.polyder(true), yla))):+.2f} deg")
    print(f"quadratic  tangent@LA = {heading_at(cf2, yla):+.2f} deg")
    print(f"straight-line heading = {heading_at(cf1, yla):+.2f} deg  (biased on a curved row)")
