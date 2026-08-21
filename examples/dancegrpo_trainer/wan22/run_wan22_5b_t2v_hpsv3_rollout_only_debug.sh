#!/bin/bash
# Reproduce Wan2.2 rollout CPU RSS growth without reward or training.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROLLOUT_ONLY_STEPS=${ROLLOUT_ONLY_STEPS:-30}
ROLLOUT_ONLY_MEMORY_TRIM=${ROLLOUT_ONLY_MEMORY_TRIM:-false}

bash "$SCRIPT_DIR/run_wan22_5b_t2v_hpsv3_auto.sh" \
    +trainer.rollout_only=true \
    +trainer.rollout_only_memory_trim="$ROLLOUT_ONLY_MEMORY_TRIM" \
    trainer.val_before_train=false \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger='["console"]' \
    trainer.experiment_name=wan22_5b_t2v_rollout_only_debug \
    trainer.total_training_steps="$ROLLOUT_ONLY_STEPS" \
    "$@"
