# Detector-free cabbage crop-row guidance

A training-free vision pipeline that turns a forward-view camera frame into the
two signals a steering controller consumes: a **heading** (the look-ahead tangent
to the fitted row) and a **cross-track offset**. Classical computer vision only —
NumPy and OpenCV, no neural detector, no learned weights, no GPU.

An Excess-Green column-peak tracker locates the central row inside a ±180 mm
ground safety corridor, a robust Huber IRLS quadratic `x = P(y)` is fitted to the
row anchors, and a temporal stage of median, EMA and a kinematic jump gate
stabilises both signals over time.

## Reproducing the reported numbers

The per-frame results and both annotators' guidance lines are committed, so every
figure in the manuscript can be recomputed **without the imagery**:

```bash
pip install -r requirements.txt
python verify_paper.py     # re-derives the reported numbers from results/*.csv
python eval_stats.py       # paired bootstrap CIs and stratifications
python verify_refs.py      # resolves the bibliography against Crossref/DataCite
```

With the image data in place the full pipeline runs end to end with
`bash run_all.sh`, and `bench_runtime.py` times it on the target board.

## Layout

| Path | Role |
|---|---|
| `cabbage_row.py`, `guidance_curve.py`, `temporal_guidance.py`, `calib.py` | the pipeline |
| `extract_frames.py`, `select_cabbage.py`, `select_label_set.py`, `annotate_rows.py` | data preparation |
| `eval_*.py`, `exp_wrongrow.py` | evaluation, baselines and ablations |
| `make_*_figs.py` | figures |
| `results/` | per-frame results and the frame-quality report |
| `datasets/CabbageNav/frames/` | frame manifest and both annotators' labels |

## Data

Committed: the per-frame CSVs, the frame manifest, and `labels1.json` /
`labels2.json`. Not committed: the frames, the source videos and the detector
weights — available from the corresponding author on reasonable request.

## Paper

This repository accompanies *In-Field Accuracy Evaluation of a Detector-Free
Vision Pipeline for Cabbage Crop-Row Guidance*. The manuscript is not distributed
here, so the LaTeX gates of `verify_paper.py` are skipped; the numeric gates run
as normal.
