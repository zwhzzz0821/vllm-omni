# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared MiniMax-H3 smoke-test inputs, without serving or model imports."""

import os

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_MODEL", "MiniMaxAI/MiniMax-H3")
WIDTH = 1344
HEIGHT = 768
FPS = 24
NUM_INFERENCE_STEPS = 4
DURATION = 4.0
PROMPT = (
    "A cinematic live-action scene with a clear subject moving naturally; "
    "the atmosphere includes synchronized environmental sound."
)

NPU_INT8_OFFLOAD_CONFIG = {
    "mode": "layer",
    "components": ["dit", "text_encoder"],
    "layer_options": {
        "dit": {"weight_transfer": "rank-local"},
        "text_encoder": {"weight_transfer": "rank-local"},
    },
}
