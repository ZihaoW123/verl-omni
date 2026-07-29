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

import importlib.util
import sys
import types
from pathlib import Path


def _load_omni_model_config_module(monkeypatch):
    """Load the config module without importing verl_omni's accelerator stack."""
    for package_name in (
        "verl_omni",
        "verl_omni.utils",
        "verl_omni.workers",
        "verl_omni.workers.config",
        "verl_omni.workers.config.omni",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    fs_module = types.ModuleType("verl_omni.utils.fs")
    fs_module.resolve_model_local_dir = lambda path, use_shm=False: path
    monkeypatch.setitem(sys.modules, "verl_omni.utils.fs", fs_module)

    module_name = "verl_omni.workers.config.omni.model"
    module_path = Path(__file__).parents[3] / "verl_omni" / "workers" / "config" / "omni" / "model.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_v1_reference_worker_can_disable_mtp(monkeypatch):
    """Match engine_workers.init_model's mutation of the copied model config."""
    module = _load_omni_model_config_module(monkeypatch)
    model_config = module.OmniModelConfig.__new__(module.OmniModelConfig)
    model_config.mtp = module.MtpConfig(enable=True)

    model_config.mtp = module.MtpConfig(enable=False)

    assert model_config.mtp.enable is False
