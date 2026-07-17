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

from typing import Literal

import numpy as np
import torch

_CHANNEL_COUNTS = {1, 3, 4}


def prepare_visual_outputs(outputs: torch.Tensor) -> tuple[Literal["image", "video"], np.ndarray]:
    """Convert batched diffusion outputs to uint8 BHWC images or BTHWC videos."""
    outputs = torch.as_tensor(outputs).detach().cpu().float()

    if outputs.ndim == 4:
        if outputs.shape[1] in _CHANNEL_COUNTS:
            media_type = "image"
            visual_outputs = outputs.permute(0, 2, 3, 1)
        elif outputs.shape[-1] in _CHANNEL_COUNTS:
            media_type = "image"
            visual_outputs = outputs
        else:
            # Some video backends return grayscale frames as BTHW.
            media_type = "video"
            visual_outputs = outputs.unsqueeze(-1)
    elif outputs.ndim == 5:
        media_type = "video"
        if outputs.shape[2] in _CHANNEL_COUNTS:  # BTCHW
            visual_outputs = outputs.permute(0, 1, 3, 4, 2)
        elif outputs.shape[1] in _CHANNEL_COUNTS:  # BCTHW
            visual_outputs = outputs.permute(0, 2, 3, 4, 1)
        elif outputs.shape[-1] in _CHANNEL_COUNTS:  # BTHWC
            visual_outputs = outputs
        else:
            raise ValueError(f"Cannot identify the channel axis of video outputs with shape {tuple(outputs.shape)}.")
    else:
        raise ValueError(f"Expected batched image/video outputs with 4 or 5 dimensions, got {tuple(outputs.shape)}.")

    visual_outputs = (visual_outputs * 255).round().clamp(0, 255).to(torch.uint8)
    if visual_outputs.shape[-1] == 1:
        visual_outputs = visual_outputs.expand(*visual_outputs.shape[:-1], 3)
    elif visual_outputs.shape[-1] == 4:
        visual_outputs = visual_outputs[..., :3]

    return media_type, visual_outputs.contiguous().numpy()
