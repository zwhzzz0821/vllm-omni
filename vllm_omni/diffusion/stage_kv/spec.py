# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
}


def stage_kv_dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _DTYPES:
        raise ValueError(f"Unsupported Stage KV dtype: {dtype}")
    return name


def stage_kv_torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Stage KV dtype name: {name!r}") from exc


@dataclass(frozen=True)
class StageKVCacheGroupSpec:
    """Serializable static geometry for one vLLM KV cache group."""

    group_id: int
    layer_names: tuple[str, ...]
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: str
    non_causal: bool
    tensor_layout: str = "separate_kv_block_token_head_dim"

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError(f"group_id must be non-negative, got {self.group_id}")
        if not self.layer_names:
            raise ValueError("at least one layer name is required")
        if len(self.layer_names) != len(set(self.layer_names)):
            raise ValueError(f"layer names must be unique, got {self.layer_names}")
        if any(not layer_name for layer_name in self.layer_names):
            raise ValueError("layer names must be non-empty")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        stage_kv_torch_dtype(self.dtype)
        if self.tensor_layout != "separate_kv_block_token_head_dim":
            raise ValueError(f"Unsupported Stage KV tensor layout: {self.tensor_layout!r}")

    def to_vllm_spec(self) -> FullAttentionSpec:
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=stage_kv_torch_dtype(self.dtype),
            non_causal=self.non_causal,
        )

    @property
    def page_size_bytes(self) -> int:
        return self.to_vllm_spec().page_size_bytes

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "layer_names": self.layer_names,
            "block_size": self.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "dtype": self.dtype,
            "non_causal": self.non_causal,
            "tensor_layout": self.tensor_layout,
        }


@dataclass(frozen=True)
class StageKVCacheSpec:
    """Model-level Stage KV geometry shared by Scheduler and Workers."""

    model_identity: str
    groups: tuple[StageKVCacheGroupSpec, ...]
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_identity:
            raise ValueError("model_identity must be non-empty")
        if not self.groups:
            raise ValueError("at least one Stage KV cache group is required")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"group IDs must be unique, got {group_ids}")
        all_layers = [layer for group in self.groups for layer in group.layer_names]
        if len(all_layers) != len(set(all_layers)):
            raise ValueError("a layer cannot belong to multiple Stage KV cache groups")
        expected = self.compute_layout_fingerprint(self.model_identity, self.groups)
        if self.layout_fingerprint != expected:
            raise ValueError("layout_fingerprint does not match the Stage KV cache geometry")

    @classmethod
    def create(
        cls,
        *,
        model_identity: str,
        groups: tuple[StageKVCacheGroupSpec, ...],
    ) -> StageKVCacheSpec:
        return cls(
            model_identity=model_identity,
            groups=groups,
            layout_fingerprint=cls.compute_layout_fingerprint(model_identity, groups),
        )

    @staticmethod
    def compute_layout_fingerprint(
        model_identity: str,
        groups: tuple[StageKVCacheGroupSpec, ...],
    ) -> str:
        payload = {
            "version": 1,
            "model_identity": model_identity,
            "groups": [group.fingerprint_payload() for group in groups],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def block_size(self) -> int:
        block_sizes = {group.block_size for group in self.groups}
        if len(block_sizes) != 1:
            raise ValueError(f"Stage KV Scheduler requires one logical block size, got {sorted(block_sizes)}")
        return block_sizes.pop()

    @property
    def bytes_per_block(self) -> int:
        return sum(group.page_size_bytes * len(group.layer_names) for group in self.groups)

    def to_vllm_kv_cache_config(self, *, num_blocks: int) -> KVCacheConfig:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")

        group_specs: list[KVCacheGroupSpec] = []
        tensors: list[KVCacheTensor] = []
        for group in self.groups:
            vllm_spec = group.to_vllm_spec()
            group_specs.append(
                KVCacheGroupSpec(
                    layer_names=list(group.layer_names),
                    kv_cache_spec=vllm_spec,
                )
            )
            tensors.extend(
                KVCacheTensor(
                    size=vllm_spec.page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in group.layer_names
            )
        return KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=tensors,
            kv_cache_groups=group_specs,
        )
