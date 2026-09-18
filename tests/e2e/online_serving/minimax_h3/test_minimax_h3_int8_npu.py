# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 global INT8 smoke test on four Ascend A2/A3 devices."""

import json

import pytest

from tests.e2e.minimax_h3_config import NPU_INT8_OFFLOAD_CONFIG
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler
from vllm_omni.platforms import current_omni_platform

from ._common import HEIGHT, MODEL, WIDTH, assert_h3_video, run_t2va

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU"),
]

SERVER_PARAMS = OmniServerParams(
    model=MODEL,
    server_args=[
        "--trust-remote-code",
        "--task-type",
        "fl2va",
        "--num-gpus",
        "4",
        "--usp",
        "4",
        "--ring",
        "1",
        "--text-encoder-tp-size",
        "4",
        "--vae-patch-parallel-size",
        "4",
        "--vae-parallel-mode",
        "tile",
        "--vae-use-tiling",
        # Global quantization includes the text encoder. A transformer-only
        # config would miss the quantized NPU MLP regression in #6595/#6852.
        "--quantization",
        "int8",
        "--diffusion-attention-backend",
        "TORCH_SDPA",
        "--diffusion-offload-config",
        json.dumps(NPU_INT8_OFFLOAD_CONFIG),
    ],
    stage_init_timeout=1800,
    init_timeout=1800,
    env_dict={
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_OMNI_VIDEO_SYNC_TIMEOUT": "1800",
    },
)


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(SERVER_PARAMS, marks=hardware_marks(res={"npu": "A2"}, num_cards=4), id="A2-int8"),
        pytest.param(SERVER_PARAMS, marks=hardware_marks(res={"npu": "A3"}, num_cards=4), id="A3-int8"),
    ],
    indirect=True,
)
def test_minimax_h3_global_int8_t2va_npu(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
) -> None:
    """Generate four-step T2VA and validate video metadata and decoded audio."""
    video = run_t2va(openai_client, seed=6595, timeout=1800)
    assert_h3_video(video, width=WIDTH, height=HEIGHT)
