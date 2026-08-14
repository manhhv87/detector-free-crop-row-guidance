#!/usr/bin/env bash
# =====================================================================
# Cabbage robot crop-row guidance -- detector-free, real-field evaluation.
#   extract frames -> (label a subset) -> evaluate estimator vs GT + temporal stability
# Run from the repo root:  bash run_all.sh
#
# Pure cv2/numpy: NO torch, NO detector training, NO CRDLD/CRBD. The guidance is the
# Excess-Green central-row estimator (cabbage_row.py) + curved fit (guidance_curve.py)
# + temporal fusion (temporal_guidance.py), evaluated on the 2 real forward-POV robot
# runs in datasets/CabbageNav/videos/.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

VIDEOS="${VIDEOS:-datasets/CabbageNav/videos}"   # robot forward-POV videos (rec_*.avi)
FRAMES="${FRAMES:-datasets/CabbageNav/frames}"   # extracted frames + manifest.csv
# human GT from annotate_rows; the repo ships labels1.json (primary annotator),
# so fall back to it when no merged labels.json is present
LABELS="${LABELS:-$FRAMES/labels.json}"
[ -f "$LABELS" ] || LABELS="$FRAMES/labels1.json"
FPS="${FPS:-1}"                                  # frame sampling rate for extraction
MAXW="${MAXW:-640}"                              # cap frame width (robot cam is 640)
echo "[cfg] VIDEOS=$VIDEOS FRAMES=$FRAMES"

# ---- 1. extract frames from the robot videos (idempotent; FORCE_PREP=1 to redo) ----
if [ "${FORCE_PREP:-0}" = "1" ] || [ ! -d "$FRAMES" ] || [ -z "$(ls -A "$FRAMES" 2>/dev/null || true)" ]; then
  echo "[prep] extracting frames @ ${FPS}fps -> $FRAMES"
  python extract_frames.py --root "$VIDEOS" --out "$FRAMES" --fps "$FPS" --max-w "$MAXW"
else
  echo "[prep] $FRAMES present -> skip (FORCE_PREP=1 to rebuild)"
fi

# ---- 2. (manual, once) label a subset for ground truth ----
# Convention A: click along the cabbage-row centreline (near->far, >=3 pts), press SPACE/s to save.
#   python annotate_rows.py --frames "$FRAMES" --out "$LABELS" --limit 120

# ---- 3. evaluate: accuracy vs GT (if labels exist) + stability/temporal over the run(s) ----
LBL=""
if [ -f "$LABELS" ]; then
  LBL="--labels $LABELS"
else
  echo "[eval] no labels yet -> stability only (run step 2 to get the accuracy numbers)"
fi
python eval_cabbage.py --frames "$FRAMES" $LBL

echo "[done] frames in $FRAMES ; eval printed above."
