#!/usr/bin/env python
"""
select_label_set.py -- pick a stratified subset of frames for human ground-truth
labelling of the crop-row guidance curve.

Accuracy vs human GT is ESTIMATED from a representative sample, not from every
frame (1 fps frames are highly correlated, so the marginal information per extra
label is small).

By default it draws a single REPRESENTATIVE sample, stratified by
video x vegetation-tertile in proportion to stratum size and spaced evenly in
time within each stratum, so the set spans both runs and all growth stages with
little redundancy and the headline accuracy estimate is unbiased. The natural
blow-up rate is high (~40% of frames have |raw_deg - fused_deg| >= 5 deg), so a
representative 150 already contains ~60 hard frames -- enough to measure the
temporal-robustness split (error-vs-GT on the gated frames, the C3 evidence)
without skewing the headline. Each pick's hardness is recorded so that split can
be made afterwards. An optional --hard-frac (off by default) deliberately boosts
the hardest frames if an even larger hard sample is wanted, at the cost of a
pessimistic headline bias.

It reads the per-frame CSV from eval_cabbage.py, copies the chosen frames into an
output folder (no manifest, so annotate_rows.py just globs them), and writes
_selection.csv documenting every pick. Selection is fully deterministic.

Label the result with:
    python annotate_rows.py --frames <out-dir> --out datasets/CabbageNav/frames/labels.json
"""
import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path


def tertile_thresholds(values):
    s = sorted(values)
    n = len(s)
    return s[n // 3], s[2 * n // 3]


def even_spaced(items, k):
    """Pick k items evenly spaced across a list (k <= len(items))."""
    n = len(items)
    if k <= 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cabbage_perframe.csv",
                    help="per-frame CSV from eval_cabbage.py")
    ap.add_argument("--frames", default="datasets/CabbageNav/frames",
                    help="source dir holding the jpg frames")
    ap.add_argument("--out-dir", default="datasets/CabbageNav/label_subset",
                    help="destination folder for the selected frames")
    ap.add_argument("--n", type=int, default=150, help="total frames to select")
    ap.add_argument("--hard-frac", type=float, default=0.0,
                    help="OPTIONAL fraction of N to force to the hardest blow-up frames "
                         "(default 0 = unbiased representative sample; the natural ~40%% "
                         "hard rate already gives enough hard frames for the C3 split)")
    ap.add_argument("--hard-deg", type=float, default=5.0,
                    help="|raw-fused| threshold (deg) that counts a frame as 'hard'")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    if not rows:
        raise SystemExit(f"no rows in {args.csv} -- run eval_cabbage.py first")

    for r in rows:
        r["veg"] = float(r["veg"] or 0)
        r["t_sec"] = float(r["t_sec"] or 0)
        raw, fused = r.get("raw_deg") or "", r.get("fused_deg") or ""
        r["hardness"] = abs(float(raw) - float(fused)) if raw and fused else 0.0

    lo, hi = tertile_thresholds([r["veg"] for r in rows])
    for r in rows:
        r["tert"] = "low" if r["veg"] <= lo else ("high" if r["veg"] > hi else "mid")

    n_hard = round(args.n * args.hard_frac)
    n_core = args.n - n_hard

    # ---- representative core: video x tertile, proportional, temporally spaced ----
    strata = defaultdict(list)
    for r in rows:
        strata[(r["video"], r["tert"])].append(r)
    for k in strata:
        strata[k].sort(key=lambda r: r["t_sec"])

    total = len(rows)
    raw_alloc = {k: n_core * len(v) / total for k, v in strata.items()}
    alloc = {k: int(math.floor(x)) for k, x in raw_alloc.items()}
    short = n_core - sum(alloc.values())  # distribute leftovers by largest remainder
    for k in sorted(raw_alloc, key=lambda k: raw_alloc[k] - alloc[k], reverse=True)[:short]:
        alloc[k] += 1

    selected = {}  # frame -> row (dict preserves insertion order, dedups by name)
    for k, v in strata.items():
        for r in even_spaced(v, alloc[k]):
            selected[r["frame"]] = r

    # ---- guaranteed hard quota: top |raw-fused| frames not already in the core ----
    for r in sorted(rows, key=lambda r: r["hardness"], reverse=True):
        if len(selected) >= args.n:
            break
        selected.setdefault(r["frame"], r)

    # ---- top up if rounding left us short ----
    for r in sorted(rows, key=lambda r: r["t_sec"]):
        if len(selected) >= args.n:
            break
        selected.setdefault(r["frame"], r)

    picks = list(selected.values())

    # ---- copy frames + write the selection manifest ----
    src, out = Path(args.frames), Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    missing = 0
    for r in picks:
        sp = src / r["frame"]
        if sp.is_file():
            shutil.copy2(sp, out / r["frame"])
        else:
            missing += 1
    with open(out / "_selection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "video", "tert", "t_sec", "veg", "hardness"])
        for r in sorted(picks, key=lambda r: (r["video"], r["t_sec"])):
            w.writerow([r["frame"], r["video"], r["tert"], f'{r["t_sec"]:.0f}',
                        f'{r["veg"]:.4f}', f'{r["hardness"]:.2f}'])

    # ---- summary ----
    print(f"selected {len(picks)} / {args.n} frames -> {out}  (missing source: {missing})")
    by_v, by_t = defaultdict(int), defaultdict(int)
    for r in picks:
        by_v[r["video"]] += 1
        by_t[(r["video"], r["tert"])] += 1
    for v in sorted(by_v):
        print(f"  {v}: {by_v[v]}")
        for t in ("low", "mid", "high"):
            print(f"      {t:4s}: {by_t[(v, t)]}")
    hard_in = sum(1 for r in picks if r["hardness"] >= args.hard_deg)
    print(f"hard frames in set (|raw-fused| >= {args.hard_deg:g} deg): {hard_in}  "
          f"(reserved quota >= {n_hard})")
    print(f"\nNext, label them:\n  python annotate_rows.py --frames {out} "
          f"--out {src}/labels.json")


if __name__ == "__main__":
    main()
