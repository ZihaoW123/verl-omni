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
    assert "engine_kwargs.vllm_omni.pipeline_mode=thinker_only" in avqa_launcher
    assert "actor_rollout_ref.model.lora_rank=0" in avqa_launcher
    assert "actor_rollout_ref.actor.strategy=fsdp" in avqa_launcher
    assert "actor_rollout_ref.ref.strategy=fsdp" in avqa_launcher
    assert "data.custom_cls.name=OmniRLHFDataset" in avqa_launcher
    assert "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py" in avqa_launcher
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.8" in avqa_launcher
