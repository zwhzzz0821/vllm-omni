# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.transformers_utils.config import get_config

from vllm_omni.diffusion.stage_kv.spec import (
    StageKVCacheGroupSpec,
    StageKVCacheSpec,
    stage_kv_dtype_name,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig


class HunyuanImage3StageKVSpecAdapter:
    """Build the static paged-KV geometry for HunyuanImage3 self-attention."""

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        hf_config: Any,
        block_size: int,
    ) -> None:
        self._od_config = od_config
        self._hf_config = hf_config
        self._block_size = block_size

    @classmethod
    def from_config(
        cls,
        od_config: OmniDiffusionConfig,
        *,
        block_size: int,
        hf_config: Any | None = None,
    ) -> HunyuanImage3StageKVSpecAdapter:
        return cls(
            od_config=od_config,
            hf_config=hf_config or get_config(od_config.model, trust_remote_code=True),
            block_size=block_size,
        )

    def build(self) -> StageKVCacheSpec:
        if self._block_size <= 0:
            raise ValueError(f"Hunyuan Stage KV block_size must be positive, got {self._block_size}")

        tp_size = int(getattr(self._od_config.parallel_config, "tensor_parallel_size", 1))
        if tp_size <= 0:
            raise ValueError(f"tensor_parallel_size must be positive, got {tp_size}")

        total_kv_heads = int(self._hf_config.num_key_value_heads)
        if total_kv_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {total_kv_heads}")
        if total_kv_heads >= tp_size:
            if total_kv_heads % tp_size != 0:
                raise ValueError(
                    "Hunyuan KV heads must be divisible by tensor parallel size: "
                    f"num_key_value_heads={total_kv_heads}, tensor_parallel_size={tp_size}"
                )
            num_kv_heads = total_kv_heads // tp_size
        else:
            if tp_size % total_kv_heads != 0:
                raise ValueError(
                    "Hunyuan tensor parallel size must be divisible by replicated KV heads: "
                    f"num_key_value_heads={total_kv_heads}, tensor_parallel_size={tp_size}"
                )
            num_kv_heads = 1
        head_size = getattr(self._hf_config, "head_dim", None)
        if not head_size:
            head_size = getattr(self._hf_config, "attention_head_dim", None)
        if not head_size:
            hidden_size = int(self._hf_config.hidden_size)
            num_attention_heads = int(self._hf_config.num_attention_heads)
            if num_attention_heads <= 0:
                raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
            if hidden_size % num_attention_heads != 0:
                raise ValueError(
                    "Hunyuan hidden size must be divisible by attention heads: "
                    f"hidden_size={hidden_size}, num_attention_heads={num_attention_heads}"
                )
            head_size = hidden_size // num_attention_heads
        head_size = int(head_size)
        num_hidden_layers = int(self._hf_config.num_hidden_layers)
        if num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {num_hidden_layers}")

        layer_names = tuple(f"model.layers.{layer_idx}.self_attn" for layer_idx in range(num_hidden_layers))
        group = StageKVCacheGroupSpec(
            group_id=0,
            layer_names=layer_names,
            block_size=self._block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=stage_kv_dtype_name(self._od_config.dtype),
            non_causal=True,
        )
        revision = getattr(self._od_config, "revision", None) or "default"
        return StageKVCacheSpec.create(
            model_identity=f"hunyuan_image3:{self._od_config.model}@{revision}",
            groups=(group,),
        )
