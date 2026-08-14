#!/usr/bin/env python
"""
select_cabbage.py -- auto-pick the cabbage frames from the robot runs (user reviews the result).

Cabbage leaves are glaucous blue-green and grow as spaced rosettes on bare soil (moderate green cover,
bluer hue, lots of brown); the adjacent leafy-green beds (mustard/lettuce) are vivid yellow-green and
dense (very high green cover, yellower hue). We score each frame on a near-field central ROI by green
coverage + the hue/blueness of the vegetation, then classify cabbage vs other.

Modes:
  --mode features : print features for a sample (to calibrate thresholds), write features.csv for all.
  --mode select   : classify all frames, write selected.txt + review contact sheets (cabbage / other).
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np, cv2

FR = Path("datasets/CabbageNav/frames")


def feats(img):
    H, W = img.shape[:2]
    roi = img[int(0.55 * H):int(0.97 * H), int(0.20 * W):int(0.80 * W)]
    b, g, r = (roi[:, :, i].astype(np.float32) for i in range(3))
    exg = 2 * g - r - b
    veg = exg > 25                                   # vegetation mask
    vf = float(veg.mean())
    if veg.sum() < 200:
        return vf, np.nan, np.nan
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)[veg]       # OpenCV hue 0..179
    # blue-greenness: cabbage glaucous -> b closer to g (grey/blue-green); lettuce -> g >> b
    bg = (g - b)[veg]
    return vf, float(np.median(hue)), float(np.median(bg))


def sample_frames(per=10):
    rows = list(csv.DictReader(open(FR / "manifest.csv", newline="")))
    vids = sorted({r["video"] for r in rows})
    out = []
    for vi, v in enumerate(vids):
        sub = sorted([r["frame"] for r in rows if r["video"] == v])
        for k in np.linspace(0, len(sub) - 1, per).astype(int):
            out.append((f"R{vi+1}", sub[k]))
    return out, rows, vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["features", "select"], default="features")
    ap.add_argument("--hue", type=float, nargs=2, default=[68, 100], help="hue [lo,hi] for cabbage (blue-green)")
    ap.add_argument("--bg", type=float, default=24.0, help="max median (G-B): cabbage glaucous is low, vivid greens high")
    args = ap.parse_args()

    if args.mode == "features":
        samp, _, _ = sample_frames(10)
        print(f"{'tag':10s} {'frame':40s} {'veg_frac':>8s} {'hue':>6s} {'g-b':>6s}")
        for tag, nm in samp:
            im = cv2.imread(str(FR / nm))
            vf, hue, bg = feats(im)
            print(f"{tag:10s} {nm:40s} {vf:8.3f} {hue:6.1f} {bg:6.1f}")
        return

    # ---- select all ----
    rows = list(csv.DictReader(open(FR / "manifest.csv", newline="")))
    sel, rej = [], []
    for r in rows:
        nm = r["frame"]; im = cv2.imread(str(FR / nm))
        if im is None:
            continue
        vf, hue, bg = feats(im)
        ok = (not np.isnan(hue)) and (args.hue[0] <= hue <= args.hue[1]) and (bg <= args.bg)
        (sel if ok else rej).append(nm)
    Path("results").mkdir(exist_ok=True)
    open("results/selected_cabbage.txt", "w").write("\n".join(sel))
    print(f"selected {len(sel)}/{len(sel)+len(rej)} as cabbage; rejected {len(rej)}")

    def sheet(names, path, n=40, cols=8):
        names = names[:: max(1, len(names) // n)][:n]
        cells = []
        for nm in names:
            im = cv2.imread(str(FR / nm))
            if im is None:
                continue
            im = cv2.resize(im, (200, 150))
            cells.append(im)
        rowsimg = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols) if len(cells[i:i + cols]) == cols]
        if rowsimg:
            cv2.imwrite(path, np.vstack(rowsimg)); print("wrote", path, len(cells), "cells")

    sheet(sel, "results/review_cabbage.png")
    sheet(rej, "results/review_other.png")


if __name__ == "__main__":
    main()
