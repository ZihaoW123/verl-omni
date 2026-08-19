#!/bin/bash
# Wan2.2 LoRA RL with DanceGRPO
#
# Model: Wan-AI/Wan2.2-TI2V-5B-Diffusers (text+image-to-video, used in T2V mode)
# Algorithm: DanceGRPO (reuses FlowGRPO's advantage estimator and loss)
# Reward: HPSv3 (Human Preference Score v3) - custom reward model
#
# Auto-detects NPU or GPU and runs with appropriate configuration.
#
# Reference: https://github.com/XueZeyue/DanceGRPO and https://github.com/verl-project/verl-recipe/blob/main/dance_grpo/dance_grpo_mindspeed_mm/
#
set -x

export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export MALLOC_ARENA_MAX=2

if npu-smi info &>/dev/null; then
    DEVICE="npu"
elif nvidia-smi &>/dev/null; then
    DEVICE="gpu"
else
    echo "Error: Neither NPU (npu-smi) nor GPU (nvidia-smi) detected." >&2
    exit 1
fi
echo "Detected device: $DEVICE"

if [ "$DEVICE" = "npu" ]; then
    # Ray workers preload jemalloc on aarch64. Purge freed rollout buffers
    # immediately instead of leaving tens of GiB per worker as LazyFree pages.
    export MALLOC_CONF=${MALLOC_CONF:-background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0}

    # Diagnose A3 anonymous RSS growth and trim free glibc pages at diffusion
    # agent step boundaries. Disable diagnostics after the leak is identified.
    export VERL_OMNI_AGENT_MALLOC_TRIM=${VERL_OMNI_AGENT_MALLOC_TRIM:-1}
    export VERL_OMNI_AGENT_MEMORY_DIAGNOSTICS=${VERL_OMNI_AGENT_MEMORY_DIAGNOSTICS:-1}

    ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
    source $ASCEND_HOME_PATH/set_env.sh
    source $ASCEND_HOME_PATH/../nnal/atb/set_env.sh
    export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:=2}

    ATTENTION_BACKEND='native'
    ROLLOUT_ATTN_BACKEND='TORCH_SDPA'
    NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
    MICRO_BATCH_SIZE=4
    EXPERIMENT_NAME="wan22_5b_t2v_hpsv3_npu"
    PROJECT_NAME="dance_grpo"
else
    ATTENTION_BACKEND='native'
    ROLLOUT_ATTN_BACKEND='TORCH_SDPA'
    NUM_GPUS_ACTOR_ROLLOUT_REWARD=8
    MICRO_BATCH_SIZE=4
    EXPERIMENT_NAME="wan22_5b_t2v_hpsv3_gpu"
    PROJECT_NAME="dance_grpo"
fi

WORKSPACE=${WORKSPACE:-$HOME}

train_files_path=${TRAIN_FILES_PATH:-$WORKSPACE/data/hpsv3/train.parquet}
val_files_path=${VAL_FILES_PATH:-$WORKSPACE/data/hpsv3/test.parquet}

model_name=${MODEL_NAME:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}
export custom_reward_model_path=${CUSTOM_REWARD_MODEL_PATH:-$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors}
custom_reward_function_path=verl_omni/utils/reward_score/hpsv3_reward.py

ROLLOUT_TP=${ROLLOUT_TP:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}

ENGINE=vllm_omni

HOSTMEM_COLLECTOR_PID=""

stop_host_memory_collector() {
    if [ -n "$HOSTMEM_COLLECTOR_PID" ] && kill -0 "$HOSTMEM_COLLECTOR_PID" 2>/dev/null; then
        kill "$HOSTMEM_COLLECTOR_PID" 2>/dev/null || true
        wait "$HOSTMEM_COLLECTOR_PID" 2>/dev/null || true
    fi
}

trap stop_host_memory_collector EXIT

if [ "$DEVICE" = "npu" ] && [ "${A3_HOST_MEMORY_COLLECT:-1}" = "1" ]; then
    SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    HOSTMEM_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    A3_HOST_MEMORY_LOG=${A3_HOST_MEMORY_LOG:-a3_hostmem_${HOSTMEM_TIMESTAMP}.log}
    A3_HOST_MEMORY_INTERVAL_SECONDS=${A3_HOST_MEMORY_INTERVAL_SECONDS:-60}
    A3_HOST_MEMORY_COMPACT=${A3_HOST_MEMORY_COMPACT:-1}
    A3_HOST_MEMORY_TOP_PSS_PROCESSES=${A3_HOST_MEMORY_TOP_PSS_PROCESSES:-40}

    COMPACT=$A3_HOST_MEMORY_COMPACT \
        SAMPLES=0 \
        INTERVAL_SECONDS=$A3_HOST_MEMORY_INTERVAL_SECONDS \
        TOP_PSS_PROCESSES=$A3_HOST_MEMORY_TOP_PSS_PROCESSES \
        "$SCRIPT_DIR/collect_a3_host_memory.sh" "$A3_HOST_MEMORY_LOG" \
        >/dev/null 2>&1 &
    HOSTMEM_COLLECTOR_PID=$!
    echo "A3 host-memory collection started: pid=$HOSTMEM_COLLECTOR_PID log=$A3_HOST_MEMORY_LOG"
fi

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=$DEVICE \
    algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    data.train_files=$train_files_path \
    data.val_files=$val_files_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.seed=42 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.attn_backend=$ATTENTION_BACKEND \
    actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.height=704 \
    actor_rollout_ref.rollout.pipeline.width=1280 \
    actor_rollout_ref.rollout.pipeline.num_frames=8 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="dance_sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward.num_workers=1 \
    reward.max_inflight_per_agent=1 \
    reward.tool_extra_fields='[]' \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$custom_reward_function_path \
    reward.custom_reward_function.name=compute_score_hpsv3 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=120 "$@" \
    2>&1 | tee run_wan22_5b_t2v_hpsv3_${DEVICE}.log
