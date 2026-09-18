# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 global INT8 through the offline Omni API on four Ascend devices."""

from copy import deepcopy

import numpy as np
import pytest

from tests.e2e.minimax_h3_config import (
    DURATION,
    FPS,
    HEIGHT,
    MODEL,
    NPU_INT8_OFFLOAD_CONFIG,
    NUM_INFERENCE_STEPS,
    PROMPT,
    WIDTH,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import resolve_component_quant_config

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU"),
]

# Use the existing (model, deploy_config, extra_kwargs) runner contract. These
# are Python API options, not CLI arguments and not a prebuilt QuantizationConfig.
_OMNI_RUNNER_PARAM = (
    MODEL,
    None,
    {
        "trust_remote_code": True,
        "task_type": "fl2va",
        "num_gpus": 4,
        "usp": 4,
        "ring": 1,
        "text_encoder_tp_size": 4,
        "vae_patch_parallel_size": 4,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "quantization": "int8",
        "diffusion_attention_backend": "TORCH_SDPA",
        "diffusion_offload_config": deepcopy(NPU_INT8_OFFLOAD_CONFIG),
        "stage_init_timeout": 1800,
        "init_timeout": 1800,
    },
)


@pytest.mark.parametrize(
    "omni_runner",
    [
        pytest.param(_OMNI_RUNNER_PARAM, marks=hardware_marks(res={"npu": "A2"}, num_cards=4), id="A2-int8"),
        pytest.param(_OMNI_RUNNER_PARAM, marks=hardware_marks(res={"npu": "A3"}, num_cards=4), id="A3-int8"),
    ],
    indirect=True,
)
def test_minimax_h3_global_int8_t2va_offline_npu(omni_runner: OmniRunner) -> None:
    """Check Python quantization-option propagation and real video/audio output."""
    stage_clients = omni_runner.omni.engine.stage_clients
    assert len(stage_clients) == 1
    quant_config = stage_clients[0].od_config.quantization_config
    assert quant_config is not None, "Python quantization='int8' did not reach the diffusion runtime"
    for component in ("transformer", "text_encoder"):
        component_config = resolve_component_quant_config(quant_config, component)
        assert component_config is not None, f"Global INT8 did not reach {component}"
        assert component_config.get_name() == "int8"

    sampling = OmniDiffusionSamplingParams(
        width=WIDTH,
        height=HEIGHT,
        fps=FPS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=6595,
        output_type="np",
        extra_args={
            "task": "t2va",
            "duration": DURATION,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    )
    outputs = omni_runner.omni.generate(PROMPT, [sampling], use_tqdm=False)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.error is None, output.error
    assert output.finished
    assert output.final_output_type == "video"
    assert len(output.images) == 1

    frames = np.asarray(output.images[0])
    assert frames.ndim == 4 and frames.shape[1:] == (HEIGHT, WIDTH, 3)
    assert frames.shape[0] > 0
    assert frames.dtype == np.uint8

    audio = output.multimodal_output.get("audio")
    assert audio is not None, "MiniMax-H3 returned no audio"
    audio_array = np.asarray(audio)
    assert audio_array.ndim == 3 and audio_array.shape[:2] == (1, 2)
    assert audio_array.shape[-1] > 0
    assert np.isfinite(audio_array).all()
    assert output.multimodal_output.get("audio_sample_rate") == 32000
