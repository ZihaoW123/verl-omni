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


def test_qwen_media_loader_forwards_patch_size_without_video_audio(monkeypatch):
    module = _load_dataset_module(monkeypatch)
    calls = []
    expected_audios = [object()]
    expected_images = [object()]
    expected_videos = [object()]

    def fake_process_mm_info(messages, **kwargs):
        calls.append((messages, kwargs))
        return expected_audios, expected_images, expected_videos

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=fake_process_mm_info))
    messages = [{"role": "user", "content": [{"type": "video", "video": "/data/sample.mp4"}]}]

    result = module.QwenOmniRLHFDataset._process_multi_modal_info(
        messages,
        image_patch_size=16,
        config={},
    )

    assert result == (expected_images, expected_videos, expected_audios)
    assert calls == [
        (
            messages,
            {
                "use_audio_in_video": False,
                "image_patch_size": 16,
            },
        )
    ]


def test_qwen_media_loader_resolves_bundled_ffmpeg(monkeypatch, tmp_path):
    module = _load_dataset_module(monkeypatch)
    bundled_ffmpeg = tmp_path / "imageio_ffmpeg"
    bundled_ffmpeg.write_text("", encoding="utf-8")
    bundled_ffmpeg.chmod(0o755)

    imageio_ffmpeg = types.ModuleType("imageio_ffmpeg")
    imageio_ffmpeg.get_ffmpeg_exe = lambda: str(bundled_ffmpeg)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", imageio_ffmpeg)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    assert module._ffmpeg_executable() == str(bundled_ffmpeg)


def test_audio_video_loader_materializes_media_before_qwen_processing(monkeypatch):
    module = _load_dataset_module(monkeypatch)
    calls = []

    def fake_process_mm_info(messages, **kwargs):
        calls.append((messages, kwargs))
        assert messages[0]["content"][0]["video"] == ["file:///tmp/frame-1.png", "file:///tmp/frame-2.png"]
        return None, None, [["frame-1", "frame-2"]]

    qwen_omni_utils = SimpleNamespace(process_mm_info=fake_process_mm_info)
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", qwen_omni_utils)
    monkeypatch.setattr(
        module,
        "_materialize_video_item",
        lambda *args: (["file:///tmp/frame-1.png", "file:///tmp/frame-2.png"], "audio-array"),
        raising=False,
    )

    result = module.QwenOmniRLHFDataset._process_multi_modal_info(
        [{"role": "user", "content": [{"type": "video", "video": "/data/sample.mp4", "max_frames": 64}]}],
        image_patch_size=16,
        config={"use_audio_in_video": True},
    )

    assert result == (None, [["frame-1", "frame-2"]], ["audio-array"])
    assert calls[0][1] == {"use_audio_in_video": False, "image_patch_size": 16}


def test_audio_video_materialization_bounds_ffmpeg_runtime(monkeypatch, tmp_path):
    module = _load_dataset_module(monkeypatch)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if kwargs.get("text"):
            return SimpleNamespace(stderr="Duration: 00:01:00.00, start: 0.000000")
        if command[-1] == "pipe:1":
            return SimpleNamespace(stdout=b"\x00\x00\x00\x00" * 16)
        output_pattern = command[-1]
        for index in range(1, 5):
            Path(output_pattern.replace("%04d", f"{index:04d}")).write_bytes(b"frame")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(module, "_ffmpeg_executable", lambda: "/opt/ffmpeg")
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("OMNIVIDEO_INPUT_DECODE_TIMEOUT", "7.5")

    frames, audio = module._materialize_video_item(
        {"type": "video", "video": "/data/sample.mp4", "fps": 2.0, "min_frames": 4, "max_frames": 64},
        str(tmp_path),
        0,
    )

    frame_command = calls[1][0]
    assert frame_command[frame_command.index("-frames:v") + 1] == "64"
    assert frame_command[frame_command.index("-vf") + 1] == "fps=1.06666667"
    assert [kwargs["timeout"] for _, kwargs in calls] == [7.5, 7.5, 7.5]
    assert len(frames) == 4
    assert audio.shape == (16,)
