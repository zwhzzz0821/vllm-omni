# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.stage_kv import (
    HunyuanImage3StageKVSpecAdapter,
    StageKVCacheGroupSpec,
    StageKVCacheSpec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _group(*, head_size: int = 8) -> StageKVCacheGroupSpec:
    return StageKVCacheGroupSpec(
        group_id=0,
        layer_names=("model.layers.0.self_attn", "model.layers.1.self_attn"),
        block_size=4,
        num_kv_heads=2,
        head_size=head_size,
        dtype="float32",
        non_causal=True,
    )


def test_stage_kv_spec_builds_vllm_config_with_one_tensor_per_layer() -> None:
    spec = StageKVCacheSpec.create(model_identity="model", groups=(_group(),))

    config = spec.to_vllm_kv_cache_config(num_blocks=8)

    assert config.num_blocks == 8
    assert len(config.kv_cache_groups) == 1
    assert config.kv_cache_groups[0].layer_names == [
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    ]
    assert len(config.kv_cache_tensors) == 2
    assert all(tensor.size == spec.groups[0].page_size_bytes * 8 for tensor in config.kv_cache_tensors)
    assert [tensor.shared_by for tensor in config.kv_cache_tensors] == [
        ["model.layers.0.self_attn"],
        ["model.layers.1.self_attn"],
    ]


def test_stage_kv_layout_fingerprint_is_semantic_and_validated() -> None:
    first = StageKVCacheSpec.create(model_identity="model", groups=(_group(),))
    same = StageKVCacheSpec.create(model_identity="model", groups=(_group(),))
    changed = StageKVCacheSpec.create(model_identity="model", groups=(_group(head_size=16),))

    assert first.layout_fingerprint == same.layout_fingerprint
    assert first.layout_fingerprint != changed.layout_fingerprint
    with pytest.raises(ValueError, match="does not match"):
        replace(first, layout_fingerprint="stale")


def test_hunyuan_spec_adapter_uses_rank_local_kv_heads() -> None:
    od_config = SimpleNamespace(
        model="hunyuan",
        revision=None,
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    hf_config = SimpleNamespace(
        num_key_value_heads=8,
        attention_head_dim=128,
        num_hidden_layers=3,
    )

    spec = HunyuanImage3StageKVSpecAdapter(
        od_config=od_config,
        hf_config=hf_config,
        block_size=16,
    ).build()

    group = spec.groups[0]
    assert group.layer_names == tuple(f"model.layers.{idx}.self_attn" for idx in range(3))
    assert group.num_kv_heads == 4
    assert group.head_size == 128
    assert group.dtype == "bfloat16"
    assert group.non_causal is True
    assert spec.block_size == 16


def test_hunyuan_spec_adapter_rejects_non_divisible_tp_heads() -> None:
    od_config = SimpleNamespace(
        model="hunyuan",
        revision=None,
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(tensor_parallel_size=3),
    )
    hf_config = SimpleNamespace(
        num_key_value_heads=8,
        attention_head_dim=128,
        num_hidden_layers=3,
    )

    with pytest.raises(ValueError, match="divisible"):
        HunyuanImage3StageKVSpecAdapter(
            od_config=od_config,
            hf_config=hf_config,
            block_size=16,
        ).build()


def test_hunyuan_spec_adapter_matches_replicated_kv_heads() -> None:
    od_config = SimpleNamespace(
        model="hunyuan",
        revision=None,
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )
    hf_config = SimpleNamespace(
        num_key_value_heads=2,
        attention_head_dim=128,
        num_hidden_layers=3,
    )

    spec = HunyuanImage3StageKVSpecAdapter(
        od_config=od_config,
        hf_config=hf_config,
        block_size=16,
    ).build()

    assert spec.groups[0].num_kv_heads == 1


def test_hunyuan_spec_adapter_matches_model_head_dim_precedence() -> None:
    od_config = SimpleNamespace(
        model="hunyuan",
        revision=None,
        dtype=torch.bfloat16,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    hf_config = SimpleNamespace(
        num_key_value_heads=8,
        head_dim=96,
        attention_head_dim=128,
        num_hidden_layers=3,
    )

    spec = HunyuanImage3StageKVSpecAdapter(
        od_config=od_config,
        hf_config=hf_config,
        block_size=16,
    ).build()

    assert spec.groups[0].head_size == 96
