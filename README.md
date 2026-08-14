# Detector-free cabbage crop-row guidance

A deliberately simple, **training-free** vision pipeline that produces a
crop-row guidance line for a low-cost cabbage field robot, evaluated in the
field. It uses only classical computer vision (NumPy + OpenCV): there is no
neural detector, no learned weights, and no train/test split to leak. The same
code runs unchanged across crops and growth stages.

The pipeline turns a forward-view camera frame into the two signals a steering
controller consumes:

- **heading** (degrees) -- the look-ahead tangent to the fitted row, and
- **cross-track offset** -- the lateral distance to the row, in pixels and, via
  a measured camera-to-ground homography, in centimetres.

## How it works

1. **Excess-Green central-row estimator** -- a per-frame column-peak tracker
   localises the central crop row from an Excess-Green vegetation index,
   seeded inside a ±180 mm ground safety corridor.
2. **Robust curve fit** -- a quadratic `x = P(y)` is fitted to the row anchors
   with a robust (Huber) loss to tolerate gaps and outliers.
3. **Guidance extraction** -- the heading is the look-ahead tangent to `P(y)`;
   the cross-track is its lateral offset at the near row.
4. **Temporal stabilisation** -- a sliding-window median, an exponential moving
   average, and a physical max-jump gate stabilise both signals over time.

## Repository layout

### Pipeline

| File | Role |
|------|------|
| `cabbage_row.py` | Excess-Green central-row estimator (column-peak tracker) |
| `guidance_curve.py` | Robust quadratic row fit -> look-ahead heading + cross-track |
| `temporal_guidance.py` | Temporal stage: sliding median + EMA + jump gate |
| `periodic_row.py` | Shared Excess-Green helper (`exg`) / periodic-row prototype |
| `calib.py` | Camera-to-ground homography -> cross-track in centimetres |

### Data preparation

| File | Role |
|------|------|
| `extract_frames.py` | Sample frames from the robot's forward-view videos |
| `select_cabbage.py` | Excess-Green classifier to screen cabbage vs other greens |
| `select_label_set.py` | Pick the stratified subset sent for manual labelling |
| `frame_quality.py` | Advisory frame-quality report (sharpness, exposure, row signal) |
| `annotate_rows.py` | Manual ground-truth labelling of the row centreline |

### Evaluation

| File | Role |
|------|------|
| `eval_cabbage.py` | Main evaluation: accuracy vs ground truth + temporal stability |
| `eval_stats.py` | Paired bootstrap CIs, off-centre stratification, per-run replication |
| `eval_baselines.py` | Front-end, vegetation-index and temporal-filter baselines |
| `eval_fitarm_baseline.py` | Fit-arm comparison (Huber vs equal-weight LS vs RANSAC) |
| `eval_yolo_baseline.py` | Trained YOLOv5 detector baseline through the same corridor and fit |
| `exp_wrongrow.py` | Seed-corridor sweep against the wrong-row tail |
| `verify_paper.py` | Re-derives every number in the manuscript from `results/*.csv` |
| `verify_refs.py` | Resolves every bibliography entry against Crossref and DataCite |
| `bench_runtime.py` | Per-frame wall-clock cost of the pipeline on the target board |

### Figures and manuscript

| File | Role |
|------|------|
| `make_cabbage_figs.py` | Method-overlay and qualitative-gallery figures |
| `make_review_figs.py` | Error CDFs, heading traces, gate-threshold sweep |
| `make_wrongrow_fig.py` | Wrong-row failure-case overlay |
| `run_all.sh` | End-to-end driver (extract -> [label] -> evaluate) |

## What is and is not in this repository

Committed, so that every number in the manuscript can be recomputed **without
the imagery**:

- `results/cabbage_perframe.csv` -- per-frame raw and fused heading and
  cross-track, the gate flag, and the ground-truth values on the labelled subset
- `results/yolo_perframe.csv` -- the trained-detector baseline, per frame
- `results/frame_quality.csv` -- per-frame image- and row-signal quality
- `datasets/CabbageNav/frames/manifest.csv` -- frame index, run, timestamp,
  resolution, Excess-Green coverage
- `datasets/CabbageNav/frames/labels1.json`, `labels2.json` -- both annotators'
  guidance lines (clicked points, fitted coefficients, look-ahead heading)

Not committed: the frames and the source videos. They are available from the
corresponding author on reasonable request; see the manuscript's *Data and Code
Availability* section. The trained detector weights are not redistributed here
either.

## Quickstart

```bash
pip install -r requirements.txt
```

The manuscript's numbers can be checked straight from the committed tables, with
no imagery and no video:

```bash
python verify_paper.py               # re-derives the reported figures
python eval_stats.py                  # paired bootstrap CIs and stratifications
python results/r304_temporal_ablation_bootstrap.py
```

With the image data in place, the full pipeline runs end to end:

```bash
bash run_all.sh
```

Labelling a subset for ground-truth accuracy is a manual step. The evaluation
scripts read `labels.json` if present and otherwise fall back to `labels1.json`,
the primary annotator's labels:

```bash
python annotate_rows.py --frames datasets/CabbageNav/frames \
    --out datasets/CabbageNav/frames/labels1.json --limit 150
```

## Paper

This repository accompanies the manuscript *In-Field Accuracy Evaluation of a
Detector-Free Vision Pipeline for Cabbage Crop-Row Guidance*. The manuscript
itself is not distributed here; please refer to the published version.

`verify_paper.py` re-derives the reported figures from the committed tables. Its
first two gates check the LaTeX build and cross-references and are skipped in this
repository, since the manuscript sources are not included; the numeric gates run
as normal.
