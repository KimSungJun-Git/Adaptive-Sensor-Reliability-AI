#!/usr/bin/env bash
# Full Phase 7-10 experiment matrix on an already built dataset (data/episodes):
#   main       stratified run-level split, 3 seeds  -> deployed model
#   holdout_*  leave-one-world-out generalization (1 seed, key scenarios)
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="normal slip enc_drift enc_w_drift imu_bias lidar_sector combo"

python3 -m asr_evaluation.train --seeds 0 1 2
python3 -m asr_evaluation.compare --models data/models/main/seed_0 data/models/main/seed_1 data/models/main/seed_2

for w in tb3_world house stage4; do
  python3 -m asr_evaluation.train --holdout "$w" --seeds 0
  python3 -m asr_evaluation.compare --models "data/models/holdout_$w/seed_0" --scenarios $KEY
done
echo "all experiments done"
