# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real NPU INT8 regression for the text MLP (issues #6595 and #6852)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU"),
]


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=hardware_marks(res={"npu": "A2"}, num_cards=1), id="A2-bf16"),
        pytest.param(torch.float16, marks=hardware_marks(res={"npu": "A2"}, num_cards=1), id="A2-fp16"),
        pytest.param(torch.bfloat16, marks=hardware_marks(res={"npu": "A3"}, num_cards=1), id="A3-bf16"),
        pytest.param(torch.float16, marks=hardware_marks(res={"npu": "A3"}, num_cards=1), id="A3-fp16"),
    ],
)
@pytest.mark.parametrize("shape", [(16, 128), (2, 7, 128)])
@torch.inference_mode()
def test_text_mlp_npu_int8_matches_float_reference(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLTextMLP
    from vllm_omni.platforms.npu.models.minimax_h3 import _forward_minimax_h3_qwen3vl_text_mlp_npu
    from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

    # ModelWeightParameter consults vLLM's TP group even for this single-rank
    # encoder. Keep quantization, weight loading, and NPU operators real.
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)
    hidden_size, intermediate_size = 128, 256
    with torch.device("npu"):
        mlp = MiniMaxH3Qwen3VLTextMLP(
            group=SimpleNamespace(rank_in_group=0, world_size=1),
            config=SimpleNamespace(hidden_size=hidden_size, intermediate_size=intermediate_size),
            dtype=dtype,
            quant_config=DiffusionInt8Config(),
            prefix="text_encoder.model.layers.0.mlp",
        )

    generator = torch.Generator(device="cpu").manual_seed(6595)
    gate_weight = (torch.randn(intermediate_size, hidden_size, generator=generator) / hidden_size**0.5).to(dtype)
    up_weight = (torch.randn(intermediate_size, hidden_size, generator=generator) / hidden_size**0.5).to(dtype)
    down_weight = (torch.randn(hidden_size, intermediate_size, generator=generator) / intermediate_size**0.5).to(dtype)
    x = (torch.randn(shape, generator=generator) * 0.5).to(dtype)

    # Loading the second packed shard completes online quantization. Refresh
    # the parameter each time because lazy loading replaces the meta weight.
    for shard_id, weight in enumerate((gate_weight, up_weight)):
        param = mlp.gate_up_proj.weight
        param.weight_loader(param, weight.to("npu"), shard_id)
    param = mlp.down_proj.weight
    param.weight_loader(param, down_weight.to("npu"))

    for projection in (mlp.gate_up_proj, mlp.down_proj):
        assert isinstance(projection.quant_method, NPUInt8OnlineLinearMethod)
        assert projection.weight.dtype == torch.int8
        assert projection.weight.device.type == "npu"
        assert torch.isfinite(projection.weight_scale).all()
        assert (projection.weight_scale > 0).all()

    # Exercise the actual NPU override, even when platform patches have not
    # been installed by a pipeline. Direct F.linear on these INT8 weights
    # bypasses activation quantization and fails here on the old implementation.
    output = _forward_minimax_h3_qwen3vl_text_mlp_npu(mlp, x.to("npu"))
    assert output.shape == x.shape
    assert output.dtype == dtype
    assert output.device.type == "npu"
    assert torch.isfinite(output).all()

    # Independent float32 CPU reference using the original checkpoint weights.
    # Allow dynamic W8A8 rounding through both projections and fused SwiGLU.
    gate = F.linear(x.float(), gate_weight.float())
    up = F.linear(x.float(), up_weight.float())
    expected = F.linear(F.silu(gate) * up, down_weight.float())
    torch.testing.assert_close(output.cpu().float(), expected, atol=2e-2, rtol=5e-2)
