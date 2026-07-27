# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3KVRequirementPlanner,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeTokenizerWrapper:
    def __init__(self, *, stable_lens: list[int]) -> None:
        self.stable_lens = stable_lens
        self.calls: list[dict] = []

    def apply_chat_template(self, **kwargs):
        self.calls.append(kwargs)
        rows = len(self.stable_lens)
        seq_len = max(self.stable_lens) + 8
        return {
            "output": SimpleNamespace(
                tokens=torch.arange(rows * seq_len, dtype=torch.long).reshape(rows, seq_len),
                gen_timestep_scatter_index=torch.tensor(self.stable_lens, dtype=torch.long).reshape(rows, 1),
                joint_image_slices=[[] for _ in range(rows)],
                gen_image_slices=[[slice(length + 1, seq_len)] for length in self.stable_lens],
            )
        }


class _FakeImageProcessor:
    def __init__(self, image_token_length: int = 16) -> None:
        self.image_token_length = image_token_length
        self.image_sizes: list[tuple[int, int]] = []

    def build_image_info(self, image_size):
        self.image_sizes.append(image_size)
        return ImageInfo(
            image_type="gen_image",
            image_width=image_size[1],
            image_height=image_size[0],
            token_width=4,
            token_height=4,
            image_token_length=self.image_token_length,
            base_size=1024,
            ratio_index=0,
        )


def _planner(stable_lens: list[int]):
    tokenizer = _FakeTokenizerWrapper(stable_lens=stable_lens)
    image_processor = _FakeImageProcessor()
    planner = HunyuanImage3KVRequirementPlanner(
        tokenizer_wrapper=tokenizer,
        image_processor=image_processor,
        generation_config=SimpleNamespace(sequence_template="instruct", drop_think=False),
        image_base_size=1024,
    )
    return planner, tokenizer, image_processor


def _request(
    *,
    guidance_scale: float,
    prompt="draw a cat",
    request_id: str = "req",
) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            height=768,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=4,
        ),
        request_id=request_id,
    )


def test_planner_derives_stable_and_current_lengths_without_model_execution() -> None:
    planner, tokenizer, image_processor = _planner([12])

    requirement = planner.plan(_request(guidance_scale=1.0))

    assert [(branch.stable_len, branch.current_len, branch.resident_len) for branch in requirement.branches] == [
        (12, 17, 29)
    ]
    assert image_processor.image_sizes == [(768, 1024)]
    assert tokenizer.calls[0]["sequence_template"] == "instruct"
    assert tokenizer.calls[0]["cfg_factor"] == 1
    assert len(requirement.request_layout_digest) == 64


def test_planner_creates_one_requirement_per_cfg_row() -> None:
    planner, tokenizer, _ = _planner([12, 14])

    requirement = planner.plan(_request(guidance_scale=5.0))

    assert [branch.branch_id for branch in requirement.branches] == [0, 1]
    assert [branch.stable_len for branch in requirement.branches] == [12, 14]
    assert tokenizer.calls[0]["cfg_factor"] == 2


def test_planner_passes_preprocessed_reference_image_geometry_to_tokenizer() -> None:
    planner, tokenizer, _ = _planner([20])
    joint_image = JointImageInfo(
        vae_image_info=ImageInfo(
            image_type="vae",
            token_width=8,
            token_height=8,
            image_token_length=64,
        ),
        vision_image_info=ImageInfo(
            image_type="siglip2",
            token_width=4,
            token_height=4,
            image_token_length=16,
        ),
    )
    prompt = {
        "prompt": "edit this image",
        "additional_information": {"batch_cond_image_info": [joint_image]},
    }

    requirement = planner.plan(_request(guidance_scale=1.0, prompt=prompt))

    assert tokenizer.calls[0]["batch_cond_image_info"] == [[joint_image]]
    assert requirement.branches[0].stable_len == 20
    assert requirement.branches[0].current_len == 17


def test_planner_rejects_tokenizer_cfg_row_mismatch() -> None:
    planner, _, _ = _planner([12])

    with pytest.raises(ValueError, match="branch count does not match"):
        planner.plan(_request(guidance_scale=5.0))


def test_planner_digest_is_stable_for_the_same_layout() -> None:
    planner, _, _ = _planner([12])
    request = _request(guidance_scale=1.0)

    assert planner.plan(request).request_layout_digest == planner.plan(request).request_layout_digest


def test_planner_digest_changes_when_token_layout_changes() -> None:
    short_planner, _, _ = _planner([12])
    long_planner, _, _ = _planner([13])
    request = _request(guidance_scale=1.0)

    assert short_planner.plan(request).request_layout_digest != long_planner.plan(request).request_layout_digest
