#!/usr/bin/env python
"""
eval_yolo_baseline.py -- the DEPLOYED-DETECTOR baseline for the detector-free pipeline.

The cabbage robot's own controller follows the row with a trained YOLOv5 cabbage detector feeding a PID
loop (platform thesis). This script reconstructs that detector's guidance signal on the SAME frames the
detector-free pipeline was evaluated on, so the two front ends can be compared head to head.

Crucially, the ONLY thing that differs from the detector-free pipeline is the front end: the central-row
boxes are the YOLOv5 cabbage detections inside the +/-220 mm ground safety corridor (the deployed
controller's neighbour-row filter) instead of the Excess-Green column peaks within the same corridor.
Everything downstream is shared and unchanged -- the robust quadratic fit x=P(y) (guidance_curve.fit_band),
the look-ahead tangent heading, the near-row cross-track, and the metric calibration (calib). This keeps the
comparison apples-to-apples: trained detector vs training-free vegetation index, same guidance geometry.

It writes results/yolo_perframe.csv (yolo_deg, yolo_ct_px, yolo_ct_cm per frame) and, if the detector-free
results/cabbage_perframe.csv is present, prints the agreement between the two (how closely the training-free
pipeline reproduces the deployed detector's guidance).

NOTE: this is a BASELINE/EVALUATION tool, not part of the detector-free pipeline. It needs torch + the
YOLOv5 weights (weights/bestinworld.pt) and is the only place YOLO is used in the repo; the deployed
pipeline remains pure cv2/numpy. torch is imported lazily so the geometry can be unit-tested without it.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import guidance_curve as gc
try:
    import calib
except Exception:
    calib = None


def guidance_from_centers(centers, H, W):
    """Keep the cabbage detections inside the +/-Wsafe/2 ground safety corridor about the robot centreline --
    the exact rule the deployed controller uses to discard neighbour-row plants -- then robustly fit x=P(y)
    to the central-row boxes through the SAME fit/look-ahead/calibration as the ExG pipeline. The corridor is
    essential: without it, a nearest-running-column tracker wanders onto side rows whenever many cabbages are
    detected, which inflates the error precisely on the densely-detected frames."""
    if len(centers) == 0:
        return None
    cen = np.asarray(centers, float)
    if calib is not None:                                   # corridor filter (deployed controller's rule)
        kept = [(cx, cy) for cx, cy in cen
                for cb in [calib.corridor_px(cy, W)]
                if cb is None or cb[0] <= cx <= cb[1]]
        if len(kept) >= 2:
            cen = np.asarray(kept, float)
    if len(cen) < 2:
        return None
    coeffs = gc.fit_band(cen[:, 1], cen[:, 0], H, W=W, degree=2 if len(cen) >= 3 else 1)
    if coeffs is None:
        return None
    ct_cm = calib.crosstrack_cm(coeffs, H, W) if calib is not None else ""
    return {"coeffs": coeffs, "deg": gc.heading_lookahead(coeffs, H),
            "ct_px": gc.crosstrack_px(coeffs, H, W), "ct_cm": ct_cm, "n_anchor": len(cen)}


def _load_model(weights):
    import torch
    # classic YOLOv5 (models.yolo.DetectionModel) -> load via torch.hub 'custom'
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights)
    return model


def detect_centers(model, img_bgr, conf=0.25):
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    model.conf = conf
    res = model(rgb)
    det = res.xyxy[0].cpu().numpy() if hasattr(res, "xyxy") else np.empty((0, 6))
    if len(det) == 0:
        return np.empty((0, 2))
    cx = (det[:, 0] + det[:, 2]) / 2.0
    cy = (det[:, 1] + det[:, 3]) / 2.0
    return np.column_stack([cx, cy])


def frame_index(name):
    import re
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--weights", default="weights/bestinworld.pt")
    ap.add_argument("--out", default="results/yolo_perframe.csv")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--exg-csv", default="results/cabbage_perframe.csv",
                    help="detector-free per-frame CSV to compare against")
    ap.add_argument("--selftest", action="store_true", help="test the geometry without torch/YOLO")
    args = ap.parse_args()

    if args.selftest:
        H, W = 480, 640
        # two synthetic cabbage rows; the central one sits near x=W/2 and bends slightly right
        cen = []
        for y in range(460, 200, -30):
            cen.append((W / 2 + 0.0008 * (y - 460) ** 2, y))        # central row
            cen.append((W / 2 - 300, y)); cen.append((W / 2 + 310, y))  # neighbour rows (outside the corridor)
        g = guidance_from_centers(cen, H, W)
        print("[selftest] corridor-filtered fit:", None if g is None else
              {"deg": round(g["deg"], 2), "ct_px": round(g["ct_px"], 1), "n": g["n_anchor"]})
        print("[selftest] side rows dropped, central kept (ct_px should be small, near 0):",
              g is not None and abs(g["ct_px"]) < 40)
        return

    fdir = Path(args.frames)
    man = {}
    mp = fdir / "manifest.csv"
    if mp.exists():
        for r in csv.DictReader(open(mp, newline="")):
            man[r["frame"]] = {"video": r["video"], "t_sec": float(r.get("t_sec", 0) or 0)}
    groups = defaultdict(list)
    if man:
        for nm, m in man.items():
            groups[m["video"]].append(nm)
    else:
        groups["all"] = [p.name for p in fdir.glob("*.jpg")]
    for v in groups:
        groups[v].sort(key=frame_index)

    import cv2
    model = _load_model(args.weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for video, names in groups.items():
        for nm in names:
            img = cv2.imread(str(fdir / nm))
            if img is None:
                continue
            H, W = img.shape[:2]
            cen = detect_centers(model, img, conf=args.conf)
            g = guidance_from_centers(cen, H, W)
            t = man.get(nm, {}).get("t_sec", 0)
            rows.append({"video": video, "frame": nm, "t_sec": t, "n_det": len(cen),
                         "yolo_deg": "" if g is None else f"{g['deg']:.4f}",
                         "yolo_ct_px": "" if g is None else f"{g['ct_px']:.2f}",
                         "yolo_ct_cm": "" if (g is None or g['ct_cm'] == "") else f"{g['ct_cm']:.2f}"})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame", "t_sec", "n_det", "yolo_deg", "yolo_ct_px", "yolo_ct_cm"])
        w.writeheader(); w.writerows(rows)
    emit = sum(1 for r in rows if r["yolo_deg"] != "")
    print(f"[yolo] {len(rows)} frames, detector emitted a line on {emit} ({100*emit/max(1,len(rows)):.1f}%) -> {args.out}")

    # ---- agreement with the detector-free pipeline ----
    exg_path = Path(args.exg_csv)
    if not exg_path.is_file():
        print(f"[yolo] {exg_path} not found -> skip agreement (run eval_cabbage.py first)")
        return
    exg = {r["frame"]: r for r in csv.DictReader(open(exg_path, newline=""))}
    dh, dpx = [], []
    yd, ed = [], []
    for r in rows:
        e = exg.get(r["frame"])
        if not e or r["yolo_deg"] == "" or not e.get("raw_deg"):
            continue
        yv, ev = float(r["yolo_deg"]), float(e["raw_deg"])   # per-frame front-end vs front-end (both unfiltered)
        dh.append(abs(yv - ev)); yd.append(yv); ed.append(ev)
        if r["yolo_ct_px"] != "" and e.get("ct_px"):
            dpx.append(abs(float(r["yolo_ct_px"]) - float(e["ct_px"])))
    if dh:
        dh = np.array(dh)
        corr = float(np.corrcoef(yd, ed)[0, 1]) if len(yd) > 2 else float("nan")
        print(f"\n# AGREEMENT  detector-free (ExG) vs deployed YOLOv5, both emitting ({len(dh)} frames)")
        print(f"  |heading_ExG - heading_YOLO|  median={np.median(dh):.2f} mean={dh.mean():.2f} p90={np.percentile(dh,90):.2f} deg")
        print(f"  heading correlation r={corr:.3f}")
        if dpx:
            dpx = np.array(dpx)
            print(f"  |cross-track px| diff          median={np.median(dpx):.1f} mean={dpx.mean():.1f} px")


if __name__ == "__main__":
    main()
