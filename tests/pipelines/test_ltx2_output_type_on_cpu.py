import importlib.util
from pathlib import Path


def _load_ltx2_common_module():
    module_path = (
        Path(__file__).parents[2]
        / "verl_omni"
        / "pipelines"
        / "ltx2_flow_grpo"
        / "common.py"
    )
    spec = importlib.util.spec_from_file_location("ltx2_flow_grpo_common", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_ltx_output_type_maps_generic_image_to_tensor():
    common = _load_ltx2_common_module()

    assert common.normalize_ltx_output_type("image") == "pt"


def test_normalize_ltx_output_type_preserves_native_ltx_values():
    common = _load_ltx2_common_module()

    assert common.normalize_ltx_output_type("np") == "np"
    assert common.normalize_ltx_output_type("pt") == "pt"
    assert common.normalize_ltx_output_type("pil") == "pil"
    assert common.normalize_ltx_output_type("latent") == "latent"
    assert common.normalize_ltx_output_type(None) is None
