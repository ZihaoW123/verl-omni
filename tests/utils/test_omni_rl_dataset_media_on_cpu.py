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
from types import SimpleNamespace

import pytest


def _load_dataset_module(monkeypatch):
    rl_dataset = types.ModuleType("verl.utils.dataset.rl_dataset")
    rl_dataset.RLHFDataset = object
    for package_name in ("verl", "verl.utils", "verl.utils.dataset"):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.rl_dataset", rl_dataset)

    module_path = Path(__file__).parents[2] / "verl_omni" / "utils" / "dataset" / "omni_rl_datasets.py"
    spec = importlib.util.spec_from_file_location("omni_rl_datasets_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("config", "expected_audio_from_video"),
    [({}, False), ({"use_audio_in_video": True}, True)],
)
def test_qwen_media_loader_forwards_audio_mode_and_patch_size(monkeypatch, config, expected_audio_from_video):
    module = _load_dataset_module(monkeypatch)
    calls = []
    decoder_checks = []
    expected_audios = [object()]
    expected_images = [object()]
    expected_videos = [object()]

    def fake_process_mm_info(messages, **kwargs):
        calls.append((messages, kwargs))
        return expected_audios, expected_images, expected_videos

    monkeypatch.setattr(module, "_ensure_audioread_ffmpeg", lambda: decoder_checks.append(True))
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=fake_process_mm_info))
    messages = [{"role": "user", "content": [{"type": "video", "video": "/data/sample.mp4"}]}]

    result = module.QwenOmniRLHFDataset._process_multi_modal_info(
        messages,
        image_patch_size=16,
        config=config,
    )

    assert result == (expected_images, expected_videos, expected_audios)
    assert decoder_checks == ([True] if expected_audio_from_video else [])
    assert calls == [
        (
            messages,
            {
                "use_audio_in_video": expected_audio_from_video,
                "image_patch_size": 16,
            },
        )
    ]


def test_qwen_media_loader_uses_bundled_ffmpeg_for_video_audio(monkeypatch, tmp_path):
    module = _load_dataset_module(monkeypatch)
    bundled_ffmpeg = tmp_path / "imageio_ffmpeg"
    bundled_ffmpeg.write_text("", encoding="utf-8")
    bundled_ffmpeg.chmod(0o755)

    ffdec = SimpleNamespace(COMMANDS=("ffmpeg", "avconv"))
    audioread = types.ModuleType("audioread")
    audioread.ffdec = ffdec
    imageio_ffmpeg = types.ModuleType("imageio_ffmpeg")
    imageio_ffmpeg.get_ffmpeg_exe = lambda: str(bundled_ffmpeg)
    monkeypatch.setitem(sys.modules, "audioread", audioread)
    monkeypatch.setitem(sys.modules, "audioread.ffdec", ffdec)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", imageio_ffmpeg)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    def fake_process_mm_info(messages, **kwargs):
        assert ffdec.COMMANDS == (str(bundled_ffmpeg),)
        return ["audio"], None, ["video"]

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=fake_process_mm_info))

    result = module.QwenOmniRLHFDataset._process_multi_modal_info(
        [{"role": "user", "content": [{"type": "video", "video": "/data/sample.mp4"}]}],
        image_patch_size=16,
        config={"use_audio_in_video": True},
    )

    assert result == (None, ["video"], ["audio"])
