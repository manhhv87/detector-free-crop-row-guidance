#!/usr/bin/env python
"""
periodic_row.py -- detector-free crop-row guidance from the PERIODIC structure of the rows.

Why this exists: detect-then-fit (RANSAC/IRLS over per-plant boxes) is structurally helpless when
few plants are detected -- early growth, gaps, occlusion -- because there are <3 points to fit a
line (we observed exactly this on real field video: missed near rows -> garbage heading). Crop rows,
however, form a near-PERIODIC pattern across the whole image. This module fits a periodic comb to the
vegetation column-profile in several horizontal bands, which POOLS weak evidence across ALL rows, so
the central row is located by the grid even when its own plants are sparse/absent. The guidance line
is the central comb line. Pure NumPy/OpenCV (CPU, no detector, NPU-irrelevant).

Public API:
  exg(img)                          -> Excess-Green vegetation map (float >=0)
  guidance_line_periodic(img, ...)  -> {a, b, score, pts} for x = a*y + b (px), or None
"""
from __future__ import annotations
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


def exg(img_bgr) -> np.ndarray:
    """Excess Green = 2g - r - b on chromatic-normalized channels (robust to brightness).
    Highlights crop foliage; small/young plants still register, which is what we need."""
    b, g, r = (img_bgr[:, :, i].astype(np.float32) for i in range(3))
    s = b + g + r + 1e-6
    rn, gn, bn = r / s, g / s, b / s
    v = 2.0 * gn - rn - bn
    v = np.clip(v, 0.0, None)
    if v.max() > 0:
        v /= v.max()
    return v


def _dominant_period(profile, pmin, pmax):
    """Dominant spatial period of a 1-D profile via autocorrelation (returns float or None)."""
    p = profile - profile.mean()
    if np.allclose(p, 0):
        return None
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(max(pmin, 2)), int(min(pmax, len(ac) - 1))
    if hi <= lo + 1:
        return None
    seg = ac[lo:hi]
    k = int(np.argmax(seg)) + lo
    if ac[k] < 0.15:               # too weak -> no clear periodicity
        return None
    return float(k)


def _comb_phase(profile, T):
    """Best comb phase phi in [0,T): maximizes sum of profile at {phi + k*T}. Returns (phi, strength)."""
    W = len(profile)
    best_phi, best_val = 0.0, -1.0
    for phi in np.arange(0, T, max(1.0, T / 24.0)):
        xs = np.arange(phi, W, T)
        xi = np.clip(np.round(xs).astype(int), 0, W - 1)
        val = float(profile[xi].sum()) / (len(xi) + 1e-6)
        if val > best_val:
            best_val, best_phi = val, float(phi)
    return best_phi, best_val


def guidance_line_periodic(img_bgr, y_top=0.45, y_bot=0.97, n_bands=7,
                           min_period_frac=0.05, max_period_frac=0.34, smooth=9):
    """Estimate the central crop-row guidance line x = a*y + b from row PERIODICITY.

    For each horizontal band we build the vegetation column-profile, find its dominant period T,
    fit a periodic comb (phase) that pools ALL rows, and take the comb line nearest the image centre
    as the central-row crossing at that band. A robust line is fit through the per-band crossings.
    The score is the mean comb strength (periodicity confidence). Returns None if no band shows
    usable periodicity (then the caller should fall back to the detector readout)."""
    if cv2 is None:
        raise RuntimeError("needs opencv")
    H, W = img_bgr.shape[:2]
    V = exg(img_bgr)
    cxc = W / 2.0
    pmin, pmax = min_period_frac * W, max_period_frac * W
    ys = np.linspace(y_top, y_bot, n_bands + 1)
    bands = []   # (yc, combs[], T, strength), collected per band
    for i in range(n_bands):
        y0, y1 = int(ys[i] * H), int(ys[i + 1] * H)
        if y1 <= y0 + 1:
            continue
        prof = V[y0:y1].sum(axis=0)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            prof = np.convolve(prof, k, mode="same")
        if prof.max() < 1e-6:
            continue
        T = _dominant_period(prof, pmin, pmax)
        if T is None:
            continue
        phi, strength = _comb_phase(prof, T)
        combs = np.arange(phi, W, T)
        if combs.size:
            bands.append((0.5 * (y0 + y1), combs, T, strength))
    if len(bands) < 3:
        return None

    # Select ONE coherent central row by RANSAC over the comb crossings (not greedy per-band, which
    # zig-zags between adjacent rows). The comb supplies clean candidate row points in EVERY band --
    # including sparse ones where plant detection has nothing -- so a near-vertical, near-centre line
    # can always be fit. We prefer lines that (a) stay near vertical, (b) pass near image-bottom-centre,
    # and (c) hit a crossing in as many bands as possible.
    Tmed = float(np.median([T for _, _, T, _ in bands]))
    thr = 0.30 * Tmed
    y_bottom = y_bot * H
    # Grid-search the central row as a NEAR-VERTICAL line anchored near image-bottom-centre, scored
    # by Gaussian-weighted alignment to the comb crossings (vegetation), with a mild slope penalty.
    # This disambiguates the central row from the dense crossings (every row x every band): a slanted
    # line that merely clips one crossing per band scores lower than the true near-vertical central row.
    x0_grid = np.linspace(cxc - 0.12 * W, cxc + 0.12 * W, 25)   # central row crosses bottom near centre
    a_grid = np.linspace(-0.35, 0.35, 29)                       # near-vertical
    best, best_score = None, -1e9
    for a in a_grid:
        for x0 in x0_grid:
            b = x0 - a * y_bottom
            sc = 0.0
            for yc, combs, T, strength in bands:
                pred = a * yc + b
                d = np.min(np.abs(combs - pred))
                sc += strength * np.exp(-(d / thr) ** 2)        # reward crossings ON the line
            sc -= 1.5 * abs(a)                                  # prefer vertical (robot ~aligned)
            if sc > best_score:
                best_score, best = sc, (a, b)
    a0, b0 = best
    # refine: one crossing per band nearest the chosen line, then robust (Huber-IRLS) fit
    pts, strengths = [], []
    for yc, combs, T, strength in bands:
        pred = a0 * yc + b0
        x = float(combs[np.argmin(np.abs(combs - pred))])
        if abs(x - pred) <= thr:
            pts.append((x, yc)); strengths.append(strength)
    if len(pts) < 3:
        return None
    pts = np.array(pts, float)
    pts = np.array(pts, float)
    xs, yy = pts[:, 0], pts[:, 1]
    # robust (Huber-IRLS) fit x = a*y + b through the central-comb crossings
    a, b = np.polyfit(yy, xs, 1)
    for _ in range(5):
        r = np.abs(xs - (a * yy + b))
        s = 1.4826 * np.median(r) + 1e-6
        w = np.where(r <= 1.345 * s, 1.0, 1.345 * s / (r + 1e-9))
        A = np.vstack([yy * w, w]).T
        sol, *_ = np.linalg.lstsq(A, xs * w, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    return {"a": a, "b": b, "score": float(np.mean(strengths)), "pts": pts}


if __name__ == "__main__":
    import sys
    if cv2 is None:
        raise SystemExit("opencv required")
    path = sys.argv[1] if len(sys.argv) > 1 else None
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    gl = guidance_line_periodic(img)
    print("periodic guidance:", None if gl is None else {k: gl[k] for k in ("a", "b", "score")})
