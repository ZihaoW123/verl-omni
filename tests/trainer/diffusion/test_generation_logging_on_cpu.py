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
from PIL import Image


def _load_generation_logging_module():
    module_path = Path(__file__).parents[3] / "verl_omni/trainer/diffusion/generation_logging.py"
    spec = importlib.util.spec_from_file_location("generation_logging", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_visual_outputs_recognizes_ltx_frame_batch() -> None:
    generation_logging = _load_generation_logging_module()

    media_type, videos = generation_logging.prepare_visual_outputs(torch.zeros(2, 121, 8, 10))

    assert media_type == "video"
    assert videos.shape == (2, 121, 8, 10, 3)
    assert videos.dtype.name == "uint8"
    Image.fromarray(videos[0, 0])


def test_prepare_visual_outputs_preserves_image_batch_support() -> None:
    generation_logging = _load_generation_logging_module()

    media_type, images = generation_logging.prepare_visual_outputs(torch.zeros(2, 3, 8, 10))

    assert media_type == "image"
    assert images.shape == (2, 8, 10, 3)
    Image.fromarray(images[0])
