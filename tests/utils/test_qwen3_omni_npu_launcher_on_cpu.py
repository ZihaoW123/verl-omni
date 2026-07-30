# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path


def test_avqa_npu_launcher_uses_v1_without_changing_legacy_launcher():
    launcher_dir = Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni"
    legacy_launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu.sh").read_text(encoding="utf-8")
    avqa_launcher = (launcher_dir / "run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh").read_text(encoding="utf-8")

    assert "python3 -m verl.trainer.main_ppo" in legacy_launcher
    assert "qwen3_omni_thinker_only_npu.yaml" in legacy_launcher

    assert "python3 -m verl_omni.trainer.main_omni" in avqa_launcher
    assert "python3 -m verl.trainer.main_ppo" not in avqa_launcher
    assert "--config-path" not in avqa_launcher
    assert "--config-name" not in avqa_launcher
    assert "models.transformers" not in avqa_launcher
    assert "external_lib" not in avqa_launcher
    assert "VERL_USE_EXTERNAL_MODULES=verl_omni" in avqa_launcher
    assert "stage_configs_path" not in avqa_launcher
    assert "qwen3_omni_thinker_only_npu.yaml" not in avqa_launcher
    assert "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe" in avqa_launcher
    assert "actor_rollout_ref.model.lora_rank=0" in avqa_launcher
    assert "actor_rollout_ref.actor.strategy=fsdp2" in avqa_launcher
    assert "data.custom_cls.name=OmniRLHFDataset" in avqa_launcher
    assert "reward.reward_manager.source=register" in avqa_launcher
    assert "reward.reward_manager.name=naive" in avqa_launcher
    assert "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py" in avqa_launcher
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.6" in avqa_launcher
    assert "data.train_batch_size=128" in avqa_launcher
    assert "data.max_prompt_length=4096" in avqa_launcher
    assert "data.max_response_length=8192" in avqa_launcher
    assert "data.val_max_samples=-1" in avqa_launcher
    assert "data.truncation=error" in avqa_launcher
    assert "actor_rollout_ref.actor.ppo_mini_batch_size=16" in avqa_launcher
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480" in avqa_launcher
    assert "actor_rollout_ref.actor.entropy_from_logits_with_chunking=true" in avqa_launcher
    assert "actor_rollout_ref.actor.entropy_from_logits_chunk_size=2048" in avqa_launcher
    assert "actor_rollout_ref.rollout.n=16" in avqa_launcher
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}" in avqa_launcher
    assert "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480" in avqa_launcher
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0" in avqa_launcher
    assert "actor_rollout_ref.rollout.val_kwargs.top_p=1.0" in avqa_launcher
    assert "actor_rollout_ref.rollout.val_kwargs.top_k=-1" in avqa_launcher
    assert "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480" in avqa_launcher
    assert "algorithm.rollout_correction.rollout_is=token" in avqa_launcher
    assert "algorithm.rollout_correction.rollout_is_threshold=2.0" in avqa_launcher
    assert "trainer.val_before_train=true" in avqa_launcher
    assert "trainer.total_epochs=10" in avqa_launcher
