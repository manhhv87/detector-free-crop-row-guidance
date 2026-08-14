#!/usr/bin/env python
"""
temporal_guidance.py -- temporally-robust crop-row guidance: the honest improvement to detect-then-fit.

A real robot heading cannot jump 30 deg in one frame. Per-frame detect-then-fit occasionally blows up
(few/clustered detections -> a wildly wrong line, e.g. the +33 deg frame we observed). This module fuses
the per-frame guidance line over the video stream and REJECTS/HOLDS such single-frame outliers, turning
a jittery per-frame estimate into a stable heading stream -- and flags low-trust frames for the safety
gate. Pure NumPy; works on any per-frame (a, b, trust) stream from any readout.

  TemporalGuidance.update(a, b, present, trust) -> {a, b, heading, gated}
  smooth_stream(records, ...)                    -> list of fused records (offline / batch)
"""
from __future__ import annotations
import math
from collections import deque

import numpy as np


class TemporalGuidance:
    """Robust online filter on the guidance line. Maintains a sliding window of ACCEPTED headings;
    a new frame is accepted only if it is present, trusted, and within `max_jump_deg` of the window
    median (the physical-plausibility gate). Otherwise the last good estimate is HELD and the frame is
    marked gated (the controller should slow/abstain). Accepted headings are EMA-smoothed."""

    def __init__(self, win=15, max_jump_deg=8.0, ema=0.4, min_trust=0.0, warmup=3, max_jump_px=130.0):
        self.win = win; self.max_jump = max_jump_deg; self.ema = ema
        self.min_trust = min_trust; self.warmup = warmup
        self.hist = deque(maxlen=win)        # accepted headings (deg)
        self.a = None; self.b = None; self.heading = None; self.n = 0
        self.max_jump_px = max_jump_px       # cross-track plausibility gate (px/frame)
        self.ct_hist = deque(maxlen=win)     # accepted cross-track offsets (px)
        self.ct = None

    def update_crosstrack(self, ct, present=True):
        """Same physical-plausibility gate + EMA as update_heading, applied to the cross-track offset (px),
        so a single-frame wrong-row jump in e is rejected/held instead of passed straight through. The
        cross-track had no temporal stage before, which is why its error tail was dominated by wrong-row
        frames. Returns {crosstrack, gated}."""
        ref = float(np.median(self.ct_hist)) if self.ct_hist else None
        accept = present and ct is not None and math.isfinite(ct)
        if accept and ref is not None and len(self.ct_hist) >= self.warmup and abs(ct - ref) > self.max_jump_px:
            accept = False
        if accept:
            self.ct_hist.append(ct)
            self.ct = ct if self.ct is None else (1 - self.ema) * self.ct + self.ema * ct
        return {"crosstrack": self.ct, "gated": not accept}

    def update_heading(self, heading, present=True, trust=1.0):
        """Fuse a precomputed steering heading (deg, e.g. the look-ahead tangent of a curved fit).
        Same physical-plausibility gate as update(); returns {heading, gated}."""
        self.n += 1
        ref = float(np.median(self.hist)) if self.hist else None
        accept = present and heading is not None and math.isfinite(heading) and trust >= self.min_trust
        if accept and ref is not None and len(self.hist) >= self.warmup and abs(heading - ref) > self.max_jump:
            accept = False
        gated = not accept
        if accept:
            self.hist.append(heading)
            self.heading = heading if self.heading is None else (1 - self.ema) * self.heading + self.ema * heading
        return {"heading": self.heading, "gated": gated, "raw_heading": heading, "n": self.n}

    def update(self, a, b, present=True, trust=1.0):
        self.n += 1
        h = math.degrees(math.atan(a)) if (present and a is not None and math.isfinite(a)) else None
        ref = float(np.median(self.hist)) if self.hist else None
        gated = False
        accept = present and h is not None and trust >= self.min_trust
        if accept and ref is not None and len(self.hist) >= self.warmup and abs(h - ref) > self.max_jump:
            accept = False                    # physically implausible single-frame jump -> reject
        if accept:
            self.hist.append(h)
            if self.heading is None:
                self.heading, self.a, self.b = h, a, b
            else:                              # EMA on heading + line params
                self.heading = (1 - self.ema) * self.heading + self.ema * h
                self.a = (1 - self.ema) * self.a + self.ema * a
                self.b = (1 - self.ema) * self.b + self.ema * b
        else:
            gated = True                       # hold last good (a,b,heading unchanged)
        return {"a": self.a, "b": self.b, "heading": self.heading, "gated": gated,
                "raw_heading": h, "n": self.n}


def smooth_stream(records, win=15, max_jump_deg=8.0, ema=0.4, min_trust=0.0):
    """records: list of dicts with keys a,b (per-frame line; None if no line), optional 'trust'.
    Returns the same list with fused fields a_t,b_t,heading_t,gated added."""
    tg = TemporalGuidance(win, max_jump_deg, ema, min_trust)
    out = []
    for r in records:
        a = r.get("a"); present = a is not None and (isinstance(a, float) and math.isfinite(a))
        o = tg.update(a if present else 0.0, r.get("b", 0.0), present=present, trust=r.get("trust", 1.0))
        rr = dict(r); rr.update(a_t=o["a"], b_t=o["b"], heading_t=o["heading"], gated=o["gated"])
        out.append(rr)
    return out


if __name__ == "__main__":
    # smoke: a clean heading track with injected single-frame spikes (the +33 deg failures).
    rng = np.random.default_rng(0)
    true = 2.0 + 0.5 * np.sin(np.linspace(0, 6, 120))          # gentle real heading drift (deg)
    raw = true + rng.normal(0, 0.8, 120)
    spikes = rng.choice(120, 10, replace=False)
    raw[spikes] += rng.choice([-1, 1], 10) * rng.uniform(20, 40, 10)   # blow-ups
    recs = [{"a": math.tan(math.radians(h)), "b": 300.0} for h in raw]
    fused = smooth_stream(recs, win=15, max_jump_deg=8.0, ema=0.4)
    ht = np.array([f["heading_t"] for f in fused])
    err_raw = np.abs(raw - true); err_fused = np.abs(ht - true)
    print(f"raw   heading err: mean={err_raw.mean():.2f} max={err_raw.max():.2f} deg (spikes present)")
    print(f"fused heading err: mean={err_fused.mean():.2f} max={err_fused.max():.2f} deg")
    print(f"gated {sum(f['gated'] for f in fused)}/120 frames (the rejected blow-ups + low-trust)")
    print(f"-> temporal fusion removes the single-frame blow-ups: {err_fused.max() < 10 < err_raw.max()}")
