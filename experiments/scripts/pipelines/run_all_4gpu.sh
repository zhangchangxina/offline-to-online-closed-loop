#!/usr/bin/env bash

set -euo pipefail

# This script launches four pipelines in parallel, one per GPU.
# GPU0: door-binary-v0
# GPU1: kitchen-partial-v0
# GPU2: kitchen-mixed-v0
# GPU3: antmaze-umaze-diverse-v0

ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl-main"

bash ${ROOT}/experiments/scripts/pipelines/run_door_binary_pipeline.sh 0 &
bash ${ROOT}/experiments/scripts/pipelines/run_kitchen_partial_pipeline.sh 1 &
bash ${ROOT}/experiments/scripts/pipelines/run_kitchen_mixed_pipeline.sh 2 &
bash ${ROOT}/experiments/scripts/pipelines/run_antmaze_umaze_diverse_pipeline.sh 3 &

wait

echo "All 4 pipelines finished."


