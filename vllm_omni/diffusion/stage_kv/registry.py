# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vllm.transformers_utils.config import get_config
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_omni.diffusion.stage_kv.interface import (
    StageKVCacheMode,
    StageKVRequirement,
)
from vllm_omni.diffusion.stage_kv.manager import DiTKVCacheManager
from vllm_omni.diffusion.stage_kv.spec import StageKVCacheSpec

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest


class StageKVRequirementPlanner(Protocol):
    def plan(self, request: OmniDiffusionRequest) -> StageKVRequirement: ...


@dataclass(frozen=True)
class StageKVSchedulerRuntime:
    """Scheduler-local objects for the paged Stage KV control plane."""

    cache_mode: StageKVCacheMode
    cache_spec: StageKVCacheSpec
    kv_cache_config: KVCacheConfig
    max_model_len: int
    planner: StageKVRequirementPlanner
    manager: DiTKVCacheManager


def get_stage_kv_cache_mode(od_config: OmniDiffusionConfig) -> StageKVCacheMode:
    omni_kv_config = getattr(od_config, "omni_kv_config", None) or {}
    if not isinstance(omni_kv_config, dict):
        raise TypeError("omni_kv_config must be a dict for diffusion Stage KV configuration")
    raw_mode = omni_kv_config.get("cache_mode", StageKVCacheMode.DENSE_LEGACY.value)
    try:
        return StageKVCacheMode(raw_mode)
    except ValueError as exc:
        supported = ", ".join(mode.value for mode in StageKVCacheMode)
        raise ValueError(f"Unsupported Stage KV cache_mode {raw_mode!r}; expected one of: {supported}") from exc


def create_stage_kv_scheduler_runtime(od_config: OmniDiffusionConfig) -> StageKVSchedulerRuntime | None:
    cache_mode = get_stage_kv_cache_mode(od_config)
    if cache_mode is not StageKVCacheMode.PAGED_SCHEDULER:
        return None

    model_class_name = getattr(od_config, "model_class_name", None)
    if model_class_name not in {"HunyuanImage3Pipeline", "HunyuanImage3ForCausalMM"}:
        raise ValueError(
            f"paged_scheduler Stage KV currently supports only HunyuanImage3Pipeline, got {model_class_name!r}"
        )

    omni_kv_config = od_config.omni_kv_config
    num_blocks = int(omni_kv_config.get("paged_kv_num_blocks", 0))
    if num_blocks <= 0:
        raise ValueError("paged_scheduler requires omni_kv_config.paged_kv_num_blocks to be a positive integer")
    block_size = int(omni_kv_config.get("paged_kv_block_size", 16))
    if block_size <= 0:
        raise ValueError(f"paged_kv_block_size must be positive, got {block_size}")
    max_model_len = int(omni_kv_config.get("paged_kv_max_model_len", num_blocks * block_size))
    if max_model_len <= 0:
        raise ValueError(f"paged_kv_max_model_len must be positive, got {max_model_len}")

    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        HunyuanImage3KVRequirementPlanner,
    )
    from vllm_omni.diffusion.stage_kv.hunyuan_image3 import (
        HunyuanImage3StageKVSpecAdapter,
    )

    hf_config = get_config(od_config.model, trust_remote_code=True)
    planner = HunyuanImage3KVRequirementPlanner.from_config(od_config, hf_config=hf_config)
    cache_spec = HunyuanImage3StageKVSpecAdapter.from_config(
        od_config,
        block_size=block_size,
        hf_config=hf_config,
    ).build()
    kv_cache_config = cache_spec.to_vllm_kv_cache_config(num_blocks=num_blocks)
    native_manager = KVCacheManager(
        kv_cache_config,
        max_model_len=max_model_len,
        scheduler_block_size=cache_spec.block_size,
        hash_block_size=cache_spec.block_size,
        max_num_batched_tokens=max_model_len,
        enable_caching=False,
    )
    return StageKVSchedulerRuntime(
        cache_mode=cache_mode,
        cache_spec=cache_spec,
        kv_cache_config=kv_cache_config,
        max_model_len=max_model_len,
        planner=planner,
        manager=DiTKVCacheManager(
            native_manager,
            cache_layout_fingerprint=cache_spec.layout_fingerprint,
        ),
    )
