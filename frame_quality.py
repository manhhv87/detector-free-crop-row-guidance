#!/usr/bin/env python
"""
frame_quality.py -- deep per-frame quality assessment for the cabbage guidance dataset.

It scores every frame on two families of criteria and flags poor frames so the dataset can be pruned to
the most reliable subset. No labels needed.

  IMAGE quality
    sharpness     variance of the Laplacian (motion blur from the moving robot -> low)
    brightness    mean luma; over_exp / under_exp = fraction of clipped pixels
    contrast      luma standard deviation

  CROP-ROW SIGNAL quality (what the guidance estimator actually depends on)
    veg_frac      Excess-Green vegetation fraction in the central corridor (is there a crop signal?)
    n_anchors     how many of the scanned bands yield a central-row peak (row coverage; max = n_bands)
    reach_la      does the tracked row reach the look-ahead row (~0.65H), where the heading is read?
    fit_resid     median |x_i - P(y_i)| of the robust quadratic fit (px); high = scattered/ambiguous row
    peak_ratio    central-row peak height vs the next-strongest peak in the band (low = competing rows)
    over_bend     the quadratic left the image and had to fall back to a line (unstable curvature)
    heading,e     reported heading (deg) and cross-track (% half-width); extreme values are suspect

  REDUNDANCY
    dup_diff      mean abs luma difference to the previous frame of the same run (low = near-duplicate)

A composite quality in [0,100] and a list of failed checks are written per frame. Critical checks
(weak crop signal, sparse/short track, poor fit, ambiguous row, severe blur/exposure, over-bend) mark a
frame as a drop candidate.

Usage:
  python frame_quality.py --frames datasets/CabbageNav/frames            # report + worst contact sheet
  python frame_quality.py --frames datasets/CabbageNav/frames --move     # quarantine the drop candidates
"""
from __future__ import annotations
import argparse, csv, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2

from periodic_row import exg
import guidance_curve as gc

CORRIDOR = 0.18      # central corridor half-width as a fraction of W for peak tracking
N_BANDS = 9


def _peaks(p, thr):
    return [i for i in range(3, len(p) - 3) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > thr]


def analyse(img):
    H, W = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = {}
    # ---- image quality ----
    m["sharpness"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
    m["brightness"] = float(g.mean())
    m["contrast"] = float(g.std())
    m["over_exp"] = float((g > 250).mean())
    m["under_exp"] = float((g < 15).mean())
    # ---- crop-row signal ----
    V = exg(img)                                       # normalized ExG, for peak tracking (matches pipeline)
    cx0, cx1 = int(0.30 * W), int(0.70 * W)
    bb, gg, rr = (img[:, :, i].astype(np.float32) for i in range(3))
    ss = bb + gg + rr + 1e-6
    exr = np.clip(2 * gg / ss - rr / ss - bb / ss, 0.0, None)   # absolute chromatic ExG, for crop-presence
    m["veg_frac"] = float((exr[int(0.45 * H):, cx0:cx1] > 0.1).mean())
    pts, cur, ratios = [], None, []
    for y in np.linspace(gc.Y_FIT_HI * H, gc.Y_FIT_LO * H, N_BANDS):
        y0, y1 = int(y - 0.04 * H), int(y + 0.04 * H)
        prof = V[max(0, y0):y1].sum(axis=0)
        prof = np.convolve(prof, np.ones(21) / 21, "same")
        if prof.max() < 1e-6:
            continue
        thr = prof.mean() + 0.3 * prof.std()
        pk = _peaks(prof, thr)
        if not pk:
            continue
        heights = sorted((prof[i] for i in pk), reverse=True)
        ratios.append(heights[0] / (heights[1] + 1e-6) if len(heights) > 1 else 3.0)
        cur = min(pk, key=lambda x: abs(x - W / 2)) if cur is None else \
            min([x for x in pk if abs(x - cur) < CORRIDOR * W] or [cur], key=lambda x: abs(x - cur))
        pts.append((cur, 0.5 * (y0 + y1)))
    m["n_anchors"] = len(pts)
    m["peak_ratio"] = float(np.median(ratios)) if ratios else 0.0
    if len(pts) >= 3:
        pts = np.array(pts, float)
        coeffs = gc.fit_band(pts[:, 1], pts[:, 0], H, W=W, degree=2)
        resid = np.abs(pts[:, 0] - np.polyval(coeffs, pts[:, 1]))
        m["fit_resid"] = float(np.median(resid))
        m["reach_la"] = int(pts[:, 1].min() <= gc.Y_LOOKAHEAD * H + 0.04 * H)
        m["over_bend"] = int(len(coeffs) < 3)         # fell back to a line
        m["heading"] = float(gc.heading_lookahead(coeffs, H))
        m["e_pct"] = float(gc.crosstrack_px(coeffs, H, W) / (W / 2) * 100)
    else:
        m.update(fit_resid=np.nan, reach_la=0, over_bend=1, heading=np.nan, e_pct=np.nan)
    return m


def flags_for(m, T):
    f = []
    if m["sharpness"] < T["sharp"]:                         f.append("blurry")
    if m["brightness"] < 45 or m["under_exp"] > 0.55:       f.append("dark")
    if m["brightness"] > 215 or m["over_exp"] > 0.08:       f.append("bright")
    if m["contrast"] < 22:                                  f.append("low_contrast")
    if m["veg_frac"] < 0.015:                               f.append("weak_veg")
    if m["n_anchors"] < 4:                                  f.append("sparse_track")
    if not m["reach_la"]:                                   f.append("no_lookahead")
    if not np.isnan(m["fit_resid"]) and m["fit_resid"] > 45:f.append("poor_fit")
    if m["peak_ratio"] and m["peak_ratio"] < 1.03:          f.append("ambiguous_row")
    if m["over_bend"]:                                      f.append("over_bend")
    if not np.isnan(m["heading"]) and abs(m["heading"]) > 35: f.append("extreme_heading")
    return f


# poor_fit is NOT critical: a moderate robust-fit residual is normal for discrete, staggered cabbage
# plants with parallel rows, so it over-flags good frames. It is kept as an informational metric only.
CRITICAL = {"weak_veg", "sparse_track", "no_lookahead", "over_bend", "blurry", "dark", "bright"}


def quality(m, T):
    def clip01(x): return max(0.0, min(1.0, x))
    sharp = clip01((m["sharpness"] - T["sharp_lo"]) / (T["sharp_hi"] - T["sharp_lo"] + 1e-6))
    expo = 1.0 - clip01(m["over_exp"] / 0.08) * 0.5 - clip01(m["under_exp"] / 0.55) * 0.5
    contr = clip01((m["contrast"] - 18) / 40)
    track = clip01(m["n_anchors"] / N_BANDS)
    fit = 0.0 if np.isnan(m["fit_resid"]) else clip01(1 - m["fit_resid"] / 40)
    unamb = clip01((m["peak_ratio"] - 1.0) / 1.5)
    veg = clip01(m["veg_frac"] / 0.15)
    reach = 1.0 if m["reach_la"] else 0.4
    q = (0.16 * sharp + 0.08 * expo + 0.08 * contr + 0.18 * track * reach +
         0.30 * fit + 0.05 * unamb + 0.15 * veg)
    return round(100 * clip01(q), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="datasets/CabbageNav/frames")
    ap.add_argument("--out", default="results/frame_quality.csv")
    ap.add_argument("--worst", type=int, default=48, help="render this many lowest-quality frames to a contact sheet")
    ap.add_argument("--q-drop", type=float, default=45.0, help="quality below this is a drop candidate")
    ap.add_argument("--dup-diff", type=float, default=2.0, help="mean luma diff below this = near-duplicate")
    ap.add_argument("--move", action="store_true", help="quarantine drop candidates (reversible)")
    ap.add_argument("--quarantine", default="datasets/CabbageNav/_lowquality",
                    help="where --move puts drop candidates")
    args = ap.parse_args()
    fdir = Path(args.frames)
    man = {r["frame"]: r for r in csv.DictReader(open(fdir / "manifest.csv", newline=""))} \
        if (fdir / "manifest.csv").exists() else {}

    def fidx(n):
        try: return int(n.split("__f")[-1].split(".")[0])
        except Exception: return 0
    names = sorted([p.name for p in fdir.glob("*.jpg")], key=fidx)
    if not names:
        raise SystemExit(f"no frames in {fdir}")

    rows, prev_small, prev_vid = [], {}, None
    print(f"analysing {len(names)} frames ...")
    for k, nm in enumerate(names):
        img = cv2.imread(str(fdir / nm))
        if img is None:
            continue
        m = analyse(img)
        vid = man.get(nm, {}).get("video", "all")
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)
        m["dup_diff"] = float(np.abs(small - prev_small[vid]).mean()) if vid in prev_small else 99.0
        prev_small[vid] = small
        m["frame"] = nm; m["video"] = vid
        rows.append(m)
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(names)}")

    sharps = np.array([r["sharpness"] for r in rows])
    T = {"sharp": 0.5 * float(np.median(sharps)),          # blur flag: clearly soft relative to the dataset
         "sharp_lo": float(np.percentile(sharps, 10)),     # composite normalisation band
         "sharp_hi": float(np.percentile(sharps, 90))}
    for r in rows:
        r["flags"] = flags_for(r, T)
        r["quality"] = quality(r, T)
        r["drop"] = int(r["quality"] < args.q_drop or bool(set(r["flags"]) & CRITICAL))
        r["near_dup"] = int(r["dup_diff"] < args.dup_diff)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["frame", "video", "quality", "drop", "near_dup", "flags", "sharpness", "brightness", "contrast",
            "over_exp", "under_exp", "veg_frac", "n_anchors", "reach_la", "fit_resid", "peak_ratio",
            "over_bend", "heading", "e_pct", "dup_diff"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c) if c != "flags" else "|".join(r["flags"]) for c in cols])

    # ---- summary ----
    q = np.array([r["quality"] for r in rows])
    print(f"\n[csv] {len(rows)} frames -> {args.out}")
    print(f"quality: median={np.median(q):.1f} mean={q.mean():.1f} p10={np.percentile(q,10):.1f} min={q.min():.1f}")
    fc = defaultdict(int)
    for r in rows:
        for fl in r["flags"]:
            fc[fl] += 1
    print("flag counts:", dict(sorted(fc.items(), key=lambda x: -x[1])))
    drops = [r for r in rows if r["drop"]]
    dups = [r for r in rows if r["near_dup"]]
    print(f"drop candidates: {len(drops)}/{len(rows)} ({100*len(drops)/len(rows):.0f}%) "
          f"| near-duplicates: {len(dups)}")
    open(Path(args.out).with_name("frame_quality_drop.txt"), "w").write(
        "\n".join(r["frame"] for r in drops))

    # ---- contact sheet of the worst frames ----
    worst = sorted(rows, key=lambda r: r["quality"])[:args.worst]
    cells, cols_n = [], 6
    for r in worst:
        im = cv2.imread(str(fdir / r["frame"]))
        if im is None:
            continue
        im = cv2.resize(im, (200, 150))
        cv2.rectangle(im, (0, 0), (200, 26), (0, 0, 0), -1)
        tag = f"{r['quality']:.0f} {('|'.join(r['flags']))[:22]}"
        cv2.putText(im, tag, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(im, r["frame"].split("__f")[-1][:10], (3, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(im)
    if cells:
        grid = [np.hstack(cells[i:i + cols_n]) for i in range(0, len(cells), cols_n) if len(cells[i:i + cols_n]) == cols_n]
        sheet = Path(args.out).with_name("frame_quality_worst.png")
        cv2.imwrite(str(sheet), np.vstack(grid)); print(f"[worst] {sheet}")

    # ---- optional quarantine ----
    if args.move and drops:
        quar = Path(args.quarantine)
        quar.mkdir(parents=True, exist_ok=True)
        keep = set(names) - {r["frame"] for r in drops}
        for r in drops:
            src = fdir / r["frame"]
            if src.exists():
                shutil.move(str(src), str(quar / r["frame"]))
        if man:                                          # filter manifest to kept frames
            mp = fdir / "manifest.csv"
            allrows = list(csv.DictReader(open(mp, newline="")))
            flds = list(allrows[0].keys())
            shutil.copy2(str(mp), str(quar / "manifest_before_quality.csv"))
            with open(mp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=flds); w.writeheader()
                w.writerows([rr for rr in allrows if rr["frame"] in keep])
        print(f"[move] quarantined {len(drops)} low-quality frames -> {quar}")
    elif drops:
        print(f"\nReview frame_quality_worst.png + frame_quality_drop.txt. "
              f"Re-run with --move to quarantine the {len(drops)} drop candidates.")


if __name__ == "__main__":
    main()
