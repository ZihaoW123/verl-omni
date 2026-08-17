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

import os

from verl.protocol import DataProto, serialize_tensordict


class DiffusionDataProto(DataProto):
    """DataProto that avoids a redundant consolidated copy in NumPy mode.

    Diffusion batches contain decoded videos and denoising trajectories, so the
    upstream pre-serialization ``consolidate()`` can allocate several gigabytes.
    NumPy serialization already handles each tensor independently and does not
    consume the consolidated TensorDict.
    """

    def __getstate__(self):
        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            batch = serialize_tensordict(self.batch) if self.batch is not None else None
            return batch, self.non_tensor_batch, self.meta_info
        return super().__getstate__()
