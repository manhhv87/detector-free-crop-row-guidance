#!/usr/bin/env python
"""
bench_runtime.py -- per-frame wall-clock cost of the detector-free pipeline on the target board.

The manuscript claims the pipeline needs no GPU and runs on the robot's CPU-only computer, but says
nothing about how long a frame takes. That leaves a reviewer unable to tell whether the 1 fps
operating regime is a design choice or a throughput ceiling. This measures it.

What is timed is exactly the deployed path, one frame at a time, the way the robot would run it:

    decode -> Excess-Green + Otsu + column-peak track   (cabbage_row.row_anchors)
           -> robust Huber IRLS quadratic fit           (guidance_curve.fit_band)
           -> look-ahead heading + cross-track          (guidance_curve.heading_lookahead / crosstrack_px)
           -> temporal stage (median, EMA, jump gate)   (temporal_guidance.TemporalGuidance)

Frame decode is timed separately and reported apart, since on the robot the frame arrives from the
camera rather than from disk and should not be charged to the pipeline.

Run on the board itself (not on a laptop -- the number is meaningless off the target hardware):

    python bench_runtime.py --frames datasets/CabbageNav/frames --limit 300

Report the median and the 90th percentile, not the mean: a few slow frames from OS scheduling should
not be hidden, and neither should they set the headline figure.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import cabbage_row as cr
import guidance_curve as gc
from temporal_guidance import TemporalGuidance


def board_info():
    """Best-effort identification of the machine, so the number in the paper is attributable."""
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "cv2_threads": cv2.getNumThreads()}
    for path, key in [("/proc/device-tree/model", "board"),
                      ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", "cpu0_khz"),
                      ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "governor")]:
        try:
            info[key] = Path(path).read_text().strip("\x00\n ")
        except OSError:
            pass
    try:
        info["cores"] = len([l for l in Path("/proc/cpuinfo").read_text().splitlines()
                             if l.startswith("processor")])
    except OSError:
        pass
    return info


def summarise(name, samples_ms):
    s = sorted(samples_ms)
    med = statistics.median(s)
    p90 = s[max(0, int(round(0.9 * len(s))) - 1)]
    print(f"  {name:34s} median {med:7.1f} ms   p90 {p90:7.1f} ms   "
          f"min {s[0]:6.1f}   max {s[-1]:7.1f}")
    return med, p90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--limit", type=int, default=300, help="frames to time after warm-up")
    ap.add_argument("--warmup", type=int, default=20, help="untimed frames first (page cache, JIT, clocks)")
    ap.add_argument("--threads", type=int, default=None,
                    help="pin cv2 thread count; use 1 to report single-core cost")
    args = ap.parse_args()

    if args.threads is not None:
        cv2.setNumThreads(args.threads)

    print("# BOARD")
    for k, v in board_info().items():
        print(f"  {k:14s} {v}")

    files = sorted(Path(args.frames).glob("*.jpg"))[: args.warmup + args.limit]
    if len(files) < args.warmup + 10:
        raise SystemExit(f"need at least {args.warmup + 10} frames in {args.frames}, found {len(files)}")

    # Warm-up: first frames pay for page cache, library init and CPU frequency ramp.
    tg = TemporalGuidance()
    for f in files[: args.warmup]:
        img = cv2.imread(str(f))
        g = cr.guidance(img)
        if g:
            tg.update_heading(g["heading"]); tg.update_crosstrack(g["crosstrack_px"])

    decode, front, fit, geom, temporal, total = [], [], [], [], [], []
    tg = TemporalGuidance()
    emitted = 0

    for f in files[args.warmup:]:
        t0 = time.perf_counter()
        img = cv2.imread(str(f))
        t1 = time.perf_counter()
        if img is None:
            continue
        H, W = img.shape[:2]

        pts = cr.row_anchors(img)
        t2 = time.perf_counter()

        cf = None if pts is None else gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)
        t3 = time.perf_counter()

        if cf is None:
            heading = ct = None
        else:
            heading = gc.heading_lookahead(cf, H)
            ct = gc.crosstrack_px(cf, H, W)
            emitted += 1
        t4 = time.perf_counter()

        tg.update_heading(heading, present=heading is not None)
        tg.update_crosstrack(ct, present=ct is not None)
        t5 = time.perf_counter()

        decode.append((t1 - t0) * 1e3)
        front.append((t2 - t1) * 1e3)
        fit.append((t3 - t2) * 1e3)
        geom.append((t4 - t3) * 1e3)
        temporal.append((t5 - t4) * 1e3)
        total.append((t5 - t1) * 1e3)          # decode excluded: the camera supplies the frame

    n = len(total)
    print(f"\n# PER-FRAME COST  (n={n} frames, {emitted} emitted a guidance line)")
    summarise("ExG + Otsu + column-peak track", front)
    summarise("Huber IRLS quadratic fit", fit)
    summarise("heading + cross-track", geom)
    summarise("temporal stage", temporal)
    print("  " + "-" * 74)
    med, p90 = summarise("PIPELINE TOTAL (no decode)", total)
    summarise("(JPEG decode, off-robot only)", decode)

    print(f"\n# HEADLINE")
    print(f"  {med:.1f} ms per frame at the median ({1000.0 / med:.1f} fps sustained), "
          f"{p90:.1f} ms at the 90th percentile")
    margin = 1000.0 / med
    print(f"  The runs were sampled at 1 fps; the pipeline is {margin:.0f}x that rate, so the sampling "
          f"rate is\n  a property of the recording, not a throughput limit."
          if margin >= 2 else
          f"  The runs were sampled at 1 fps and the pipeline sustains {margin:.1f} fps, so throughput "
          f"is a real\n  constraint and should be reported as such.")


if __name__ == "__main__":
    main()
