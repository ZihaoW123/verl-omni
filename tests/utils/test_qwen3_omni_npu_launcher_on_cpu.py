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


def test_npu_launcher_uses_v1_with_replica_local_dynamic_deploy_config():
    launcher = (
        Path(__file__).parents[2] / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu.sh"
    ).read_text(encoding="utf-8")

    assert "python3 -m verl_omni.trainer.main_omni" in launcher
    assert "python3 -m verl.trainer.main_ppo" not in launcher
    assert "--config-path" not in launcher
    assert "--config-name" not in launcher
    assert "models.transformers" not in launcher
    assert "external_lib" not in launcher
    assert "VERL_USE_EXTERNAL_MODULES=verl_omni" in launcher
    assert "stage_configs_path" not in launcher
    assert "qwen3_omni_thinker_only_npu.yaml" not in launcher
    assert "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe" in launcher
    assert "engine_kwargs.vllm_omni.pipeline_mode=thinker_only" in launcher
    assert "actor_rollout_ref.model.lora_rank=0" in launcher
    assert "actor_rollout_ref.actor.strategy=fsdp" in launcher
    assert "actor_rollout_ref.ref.strategy=fsdp" in launcher
