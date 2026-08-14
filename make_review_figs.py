#!/usr/bin/env python
"""
make_review_figs.py -- data plots requested in review (R6-3) + the gate-threshold
sensitivity sweep (R5-3, Route B). All panels are built from the released per-frame
CSVs (results/cabbage_perframe.csv, results/yolo_perframe.csv) plus the actual
TemporalGuidance class, so every figure is reproducible from the released artefacts.

Outputs -> paper/figures/:
  fig_error_cdf.pdf      heading + cross-track error CDF (raw / fused / YOLOv5), 150 labelled frames
  fig_heading_trace.pdf  per-run heading vs frame, raw + fused, gated frames marked
  fig_gate_sweep.pdf     fused heading error vs jump-gate threshold (sensitivity)
Also prints the heading- and cross-track-gate sweep tables to stdout.

Run:  python make_review_figs.py
"""
from __future__ import annotations
import csv, math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from temporal_guidance import TemporalGuidance

CAB = "results/cabbage_perframe.csv"
YOLO = "results/yolo_perframe.csv"
OUT = "paper/figures"
RUN = {"rec_20251130_030636.avi": "run 1", "rec_20251130_124123.avi": "run 2"}


def f(x):
    x = (x or "").strip()
    return float(x) if x not in ("", "nan", "NaN") else None


def fidx(nm):
    try:
        return int(nm.split("__f")[-1].split(".")[0])
    except Exception:
        return 0


def load():
    cab = list(csv.DictReader(open(CAB)))
    yolo = {(r["video"], r["frame"]): r for r in csv.DictReader(open(YOLO))}
    for r in cab:
        y = yolo.get((r["video"], r["frame"]))
        r["yolo_deg"] = y["yolo_deg"] if y else None
        r["yolo_ct_px"] = y["yolo_ct_px"] if y else None
    return cab


def cdf(ax, data, label, prec=1, **kw):
    d = np.sort(np.asarray(data, float))
    ax.plot(d, np.linspace(0, 1, len(d)), label=f"{label} (med {np.median(d):.{prec}f})", **kw)


def fig_error_cdf(cab):
    lab = [r for r in cab if f(r["gt_deg"]) is not None]
    h_raw = [abs(f(r["raw_deg"]) - f(r["gt_deg"])) for r in lab if f(r["raw_deg"]) is not None]
    h_fus = [abs(f(r["fused_deg"]) - f(r["gt_deg"])) for r in lab if f(r["fused_deg"]) is not None]
    h_yolo = [abs(f(r["yolo_deg"]) - f(r["gt_deg"])) for r in lab if f(r["yolo_deg"]) is not None]
    c_raw = [abs(f(r["ct_px"]) - f(r["gt_ct_px"])) for r in lab if f(r["ct_px"]) is not None and f(r["gt_ct_px"]) is not None]
    c_fus = [abs(f(r["fused_ct_px"]) - f(r["gt_ct_px"])) for r in lab if f(r["fused_ct_px"]) is not None and f(r["gt_ct_px"]) is not None]
    c_yolo = [abs(f(r["yolo_ct_px"]) - f(r["gt_ct_px"])) for r in lab if f(r["yolo_ct_px"]) is not None and f(r["gt_ct_px"]) is not None]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    cdf(ax[0], h_raw, "ExG raw", color="0.6", lw=1.6)
    cdf(ax[0], h_fus, "ExG + temporal", color="C0", lw=2.0)
    cdf(ax[0], h_yolo, "YOLOv5", color="C3", lw=2.0, ls="--")
    ax[0].set_xlabel("heading error vs GT (deg)"); ax[0].set_ylabel("cumulative fraction")
    ax[0].set_xlim(0, 30); ax[0].set_title("(a) heading error"); ax[0].legend(fontsize=8, loc="lower right")
    cdf(ax[1], c_raw, "ExG raw", prec=0, color="0.6", lw=1.6)
    cdf(ax[1], c_fus, "ExG + temporal", prec=0, color="C0", lw=2.0)
    cdf(ax[1], c_yolo, "YOLOv5", prec=0, color="C3", lw=2.0, ls="--")
    ax[1].set_xlabel("cross-track error vs GT (px)"); ax[1].set_ylabel("cumulative fraction")
    ax[1].set_xlim(0, 100); ax[1].set_title("(b) cross-track error"); ax[1].legend(fontsize=8, loc="lower right")
    for a in ax:
        a.grid(alpha=0.3); a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_error_cdf.pdf"); plt.close(fig)
    print("[fig] fig_error_cdf.pdf")


def fig_heading_trace(cab):
    by = defaultdict(list)
    for r in cab:
        by[r["video"]].append(r)
    for v in by:
        by[v].sort(key=lambda r: fidx(r["frame"]))
    vids = [v for v in RUN if v in by]
    fig, ax = plt.subplots(len(vids), 1, figsize=(9.2, 4.6), sharex=False)
    if len(vids) == 1:
        ax = [ax]
    for a, v in zip(ax, vids):
        rs = by[v]
        t = [f(r["t_sec"]) for r in rs]
        raw = [f(r["raw_deg"]) for r in rs]
        fus = [f(r["fused_deg"]) for r in rs]
        a.plot(t, raw, color="0.7", lw=0.8, label="raw per-frame")
        a.plot(t, fus, color="C0", lw=1.6, label="fused (temporal)")
        gx = [t[i] for i, r in enumerate(rs) if r["gated"].strip() in ("1", "1.0", "True", "true") and raw[i] is not None]
        gy = [raw[i] for i, r in enumerate(rs) if r["gated"].strip() in ("1", "1.0", "True", "true") and raw[i] is not None]
        a.scatter(gx, gy, s=10, color="C3", zorder=3, label=f"gated ({len(gx)})")
        a.set_ylabel("heading (deg)"); a.set_title(RUN[v], fontsize=10, loc="left")
        a.grid(alpha=0.3); a.set_ylim(-35, 35)
    ax[0].legend(fontsize=8, ncol=3, loc="upper right")
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_heading_trace.pdf"); plt.close(fig)
    print("[fig] fig_heading_trace.pdf")


def run_temporal(cab, max_jump_deg=8.0, max_jump_px=130.0):
    """Re-run the actual temporal stage on raw_deg/ct_px at given gate thresholds; returns fused dicts keyed by (video,frame)."""
    by = defaultdict(list)
    for r in cab:
        by[r["video"]].append(r)
    for v in by:
        by[v].sort(key=lambda r: fidx(r["frame"]))
    out = {}
    for v, rs in by.items():
        tg = TemporalGuidance(win=15, max_jump_deg=max_jump_deg, max_jump_px=max_jump_px)
        for r in rs:
            raw = f(r["raw_deg"]); ct = f(r["ct_px"])
            fh = tg.update_heading(raw if raw is not None else None, present=raw is not None)["heading"]
            fc = tg.update_crosstrack(ct, present=ct is not None)["crosstrack"]
            out[(r["video"], r["frame"])] = (fh, fc)
    return out


def sweep(cab):
    lab = [r for r in cab if f(r["gt_deg"]) is not None]
    def herr(fused):
        e = [abs(fused[(r["video"], r["frame"])][0] - f(r["gt_deg"])) for r in lab
             if fused[(r["video"], r["frame"])][0] is not None]
        return np.median(e), np.mean(e), np.percentile(e, 90)
    def cerr(fused):
        e = [abs(fused[(r["video"], r["frame"])][1] - f(r["gt_ct_px"])) for r in lab
             if fused[(r["video"], r["frame"])][1] is not None and f(r["gt_ct_px"]) is not None]
        return np.median(e), np.mean(e), np.percentile(e, 90)
    h_thr = [4, 6, 8, 10, 12, 16, 1e9]
    print("\n# R5-3 heading-gate sweep (median/mean/p90 fused heading err vs GT, 150 labelled)")
    hm = []
    for t in h_thr:
        med, mean, p90 = herr(run_temporal(cab, max_jump_deg=t))
        hm.append((t, med, mean, p90))
        tag = "none" if t > 1e8 else f"{int(t)}"
        print(f"  dtheta_max={tag:>4} deg : median={med:.2f}  mean={mean:.2f}  p90={p90:.2f}")
    px_thr = [80, 110, 130, 160, 200, 1e9]
    print("\n# R5-2 cross-track-gate sweep (median/mean/p90 fused cross-track err px vs GT)")
    for t in px_thr:
        med, mean, p90 = cerr(run_temporal(cab, max_jump_px=t))
        tag = "none" if t > 1e8 else f"{int(t)}"
        print(f"  de_max={tag:>4} px : median={med:.2f}  mean={mean:.2f}  p90={p90:.2f}")
    # figure: heading sweep curve
    xs = [t for t, *_ in hm[:-1]]
    med = [m for _, m, _, _ in hm[:-1]]; p90 = [p for *_, p in hm[:-1]]
    none_med = hm[-1][1]; none_p90 = hm[-1][3]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(xs, med, "o-", color="C0", label="median")
    ax.plot(xs, p90, "s--", color="C1", label="90th percentile")
    ax.axhline(none_med, color="C0", ls=":", lw=1, alpha=0.7, label="no gate (median)")
    ax.axhline(none_p90, color="C1", ls=":", lw=1, alpha=0.7, label="no gate (p90)")
    ax.axvline(8, color="0.5", lw=1)
    ax.annotate("adopted 8 deg", (8, ax.get_ylim()[1]), fontsize=8, ha="center", va="top")
    ax.set_xlabel(r"heading jump-gate threshold $\Delta\theta_{\max}$ (deg)")
    ax.set_ylabel("fused heading error vs GT (deg)")
    ax.set_title("Gate-threshold sensitivity (150 labelled frames)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_gate_sweep.pdf"); plt.close(fig)
    print("\n[fig] fig_gate_sweep.pdf")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    cab = load()
    fig_error_cdf(cab)
    fig_heading_trace(cab)
    sweep(cab)


if __name__ == "__main__":
    main()
