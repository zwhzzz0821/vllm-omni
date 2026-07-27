# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

from vllm_omni.diffusion.stage_kv import (
    DiTKVCacheManager,
    StablePrefixRequestAdapter,
    StageKVBranchRequirement,
    StageKVRequirement,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4
CACHE_LAYOUT_FINGERPRINT = "cache-layout"


def _make_upstream_manager(*, num_blocks: int = 16) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
        non_causal=True,
    )
    layer_names = ["layer0"]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=spec.page_size_bytes * num_blocks, shared_by=layer_names)],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)],
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        max_num_batched_tokens=1024,
        enable_caching=False,
    )


def _requirement(
    request_id: str = "req",
    *,
    branches: tuple[StageKVBranchRequirement, ...] | None = None,
) -> StageKVRequirement:
    return StageKVRequirement(
        request_id=request_id,
        branches=branches
        or (
            StageKVBranchRequirement(branch_id=0, stable_len=5, current_len=3),
            StageKVBranchRequirement(branch_id=1, stable_len=5, current_len=3),
        ),
        request_layout_digest="layout-digest",
    )


def test_branch_requirement_counts_stable_and_current_as_resident() -> None:
    branch = StageKVBranchRequirement(branch_id=0, stable_len=5, current_len=3)

    assert branch.resident_len == 8


def test_adapter_conforms_to_vllm_025_manager_surface() -> None:
    upstream = _make_upstream_manager()
    adapter = StablePrefixRequestAdapter("req/branch-0", stable_len=5, current_len=3)
    free_before = upstream.block_pool.get_num_free_blocks()

    computed_blocks, num_computed_tokens = upstream.get_computed_blocks(adapter)
    assert num_computed_tokens == 0
    assert computed_blocks.get_block_ids() == ([],)

    blocks = upstream.allocate_slots(adapter, num_new_tokens=adapter.num_tokens, full_sequence_must_fit=True)
    assert blocks is not None
    assert len(upstream.get_block_ids(adapter.request_id)[0]) == 2

    adapter.mark_stable_computed()
    assert adapter.num_computed_tokens == 5
    upstream.cache_blocks(adapter, adapter.num_computed_tokens)
    upstream.free(adapter)
    assert upstream.block_pool.get_num_free_blocks() == free_before


def test_manager_allocates_one_block_table_per_cfg_branch() -> None:
    upstream = _make_upstream_manager()
    manager = DiTKVCacheManager(upstream, cache_layout_fingerprint=CACHE_LAYOUT_FINGERPRINT)
    free_before = upstream.block_pool.get_num_free_blocks()

    allocation = manager.allocate(_requirement())

    assert allocation is not None
    assert allocation.allocation_id == 1
    assert allocation.cache_layout_fingerprint == CACHE_LAYOUT_FINGERPRINT
    assert len(allocation.branches) == 2
    assert all(len(branch.block_ids[0]) == 2 for branch in allocation.branches)
    assert allocation.branches[0].block_ids != allocation.branches[1].block_ids
    assert allocation.to_metadata().branches[0].current_len == 3
    assert upstream.block_pool.get_num_free_blocks() == free_before - 4

    assert manager.free("req") is True
    assert manager.free("req") is False
    assert upstream.block_pool.get_num_free_blocks() == free_before


def test_manager_rolls_back_partial_branch_allocation() -> None:
    upstream = _make_upstream_manager(num_blocks=3)
    manager = DiTKVCacheManager(upstream, cache_layout_fingerprint=CACHE_LAYOUT_FINGERPRINT)
    free_before = upstream.block_pool.get_num_free_blocks()

    allocation = manager.allocate(_requirement())

    assert allocation is None
    assert manager.get_allocation("req") is None
    assert upstream.block_pool.get_num_free_blocks() == free_before


def test_manager_rejects_requirement_larger_than_upstream_max_model_len() -> None:
    upstream = _make_upstream_manager()
    manager = DiTKVCacheManager(upstream, cache_layout_fingerprint=CACHE_LAYOUT_FINGERPRINT)
    free_before = upstream.block_pool.get_num_free_blocks()
    requirement = _requirement(
        branches=(
            StageKVBranchRequirement(
                branch_id=0,
                stable_len=5,
                current_len=3,
            ),
            StageKVBranchRequirement(
                branch_id=1,
                stable_len=upstream.max_model_len,
                current_len=1,
            ),
        )
    )

    with pytest.raises(ValueError, match="exceeds the configured maximum sequence length"):
        manager.allocate(requirement)

    assert manager.get_allocation("req") is None
    assert upstream.block_pool.get_num_free_blocks() == free_before


def test_manager_rejects_duplicate_public_request() -> None:
    manager = DiTKVCacheManager(
        _make_upstream_manager(),
        cache_layout_fingerprint=CACHE_LAYOUT_FINGERPRINT,
    )
    requirement = _requirement()
    assert manager.allocate(requirement) is not None

    with pytest.raises(ValueError, match="already has a KV allocation"):
        manager.allocate(requirement)

    manager.close()


def test_manager_requires_prefix_caching_disabled() -> None:
    upstream = _make_upstream_manager()
    upstream.enable_caching = True

    with pytest.raises(ValueError, match="enable_caching=False"):
        DiTKVCacheManager(upstream, cache_layout_fingerprint=CACHE_LAYOUT_FINGERPRINT)
