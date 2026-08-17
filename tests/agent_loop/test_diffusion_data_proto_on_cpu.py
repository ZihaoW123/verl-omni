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
from pathlib import Path

import torch
from tensordict import TensorDict
from verl.protocol import deserialize_tensordict

_MODULE_PATH = Path(__file__).parents[2] / "verl_omni" / "agent_loop" / "diffusion_data_proto.py"
_SPEC = importlib.util.spec_from_file_location("diffusion_data_proto", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DiffusionDataProto = _MODULE.DiffusionDataProto


def test_numpy_serialization_skips_tensordict_consolidation(monkeypatch):
    data = DiffusionDataProto.from_dict(tensors={"payload": torch.arange(8).reshape(2, 4)})

    def fail_consolidate(self):
        raise AssertionError("NumPy serialization must not consolidate the full diffusion batch")

    monkeypatch.setenv("VERL_DATAPROTO_SERIALIZATION_METHOD", "numpy")
    monkeypatch.setattr(TensorDict, "consolidate", fail_consolidate)

    serialized_batch, _, _ = data.__getstate__()

    restored = deserialize_tensordict(serialized_batch)
    torch.testing.assert_close(restored["payload"], data.batch["payload"])
