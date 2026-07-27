# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import RequestStatus

from vllm_omni.diffusion.stage_kv.interface import (
    StageKVBranchAllocation,
    StageKVBranchRequirement,
    StageKVRequestAllocation,
    StageKVRequirement,
)


class StablePrefixRequestAdapter:
    """Minimal vLLM request surface for one DiT execution branch.

    Phase 1 disables cross-request prefix caching. The adapter therefore uses
    upstream ``KVCacheManager`` only for capacity admission, request-to-block
    ownership, block-table lookup, and release.
    """

    def __init__(
        self,
        request_id: str,
        *,
        stable_len: int,
        current_len: int,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if stable_len < 0:
            raise ValueError(f"stable_len must be non-negative, got {stable_len}")
        if current_len <= 0:
            raise ValueError(f"current_len must be positive, got {current_len}")

        self.request_id = request_id
        self._stable_len = stable_len
        self._current_len = current_len
        self._num_computed_tokens = 0

        self.block_hashes: list = []
        self.skip_reading_prefix_cache = True
        self.num_preemptions = 0
        self.status = RequestStatus.WAITING

    @property
    def num_computed_tokens(self) -> int:
        return self._num_computed_tokens

    @property
    def num_tokens(self) -> int:
        return self._stable_len + self._current_len

    @property
    def num_prompt_tokens(self) -> int:
        return self._stable_len

    @property
    def stable_len(self) -> int:
        return self._stable_len

    @property
    def current_len(self) -> int:
        return self._current_len

    def mark_stable_computed(self) -> None:
        """Record that the write-once stable span has been materialized."""

        self._num_computed_tokens = self._stable_len


class DiTKVCacheManager:
    """Atomic multi-branch wrapper around upstream ``KVCacheManager``."""

    def __init__(self, kv_cache_manager: KVCacheManager, *, cache_layout_fingerprint: str) -> None:
        if kv_cache_manager.enable_caching:
            raise ValueError("Phase 1 DiT KV management requires enable_caching=False")
        if not cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        self._kv_cache_manager = kv_cache_manager
        self._cache_layout_fingerprint = cache_layout_fingerprint
        self._allocations: dict[str, StageKVRequestAllocation] = {}
        self._adapters: dict[str, tuple[StablePrefixRequestAdapter, ...]] = {}
        self._next_allocation_id = 1

    @property
    def usage(self) -> float:
        return self._kv_cache_manager.usage

    @property
    def num_free_blocks(self) -> int:
        return self._kv_cache_manager.block_pool.get_num_free_blocks()

    def allocate(self, requirement: StageKVRequirement) -> StageKVRequestAllocation | None:
        """Allocate every branch or roll back the entire public request."""

        if requirement.request_id in self._allocations:
            raise ValueError(f"request {requirement.request_id!r} already has a KV allocation")

        adapters: list[StablePrefixRequestAdapter] = []
        branch_allocations: list[StageKVBranchAllocation] = []
        allocation: StageKVRequestAllocation | None = None
        try:
            for branch in requirement.branches:
                if branch.resident_len > self._kv_cache_manager.max_model_len:
                    raise ValueError(
                        "DiT KV requirement exceeds the configured maximum sequence length: "
                        f"request={requirement.request_id!r}, branch={branch.branch_id}, "
                        f"resident_len={branch.resident_len}, "
                        f"max_model_len={self._kv_cache_manager.max_model_len}"
                    )
                adapter = self._make_adapter(requirement.request_id, branch)
                computed_blocks, num_computed_tokens = self._kv_cache_manager.get_computed_blocks(adapter)
                if num_computed_tokens != 0 or any(computed_blocks.get_block_ids()):
                    raise RuntimeError("Phase 1 DiT requests must not read cross-request prefix-cache blocks")

                blocks = self._kv_cache_manager.allocate_slots(
                    adapter,
                    num_new_tokens=branch.resident_len,
                    full_sequence_must_fit=True,
                )
                if blocks is None:
                    break

                adapters.append(adapter)
                block_ids = self._kv_cache_manager.get_block_ids(adapter.request_id)
                branch_allocations.append(
                    StageKVBranchAllocation(
                        branch_id=branch.branch_id,
                        adapter_request_id=adapter.request_id,
                        block_ids=tuple(tuple(group) for group in block_ids),
                        stable_len=branch.stable_len,
                        current_len=branch.current_len,
                    )
                )
            if len(branch_allocations) == len(requirement.branches):
                allocation = StageKVRequestAllocation(
                    request_id=requirement.request_id,
                    allocation_id=self._next_allocation_id,
                    cache_layout_fingerprint=self._cache_layout_fingerprint,
                    request_layout_digest=requirement.request_layout_digest,
                    branches=tuple(branch_allocations),
                )
        except Exception:
            self._free_adapters(adapters)
            raise

        if allocation is None:
            self._free_adapters(adapters)
            return None
        self._allocations[requirement.request_id] = allocation
        self._adapters[requirement.request_id] = tuple(adapters)
        self._next_allocation_id += 1
        return allocation

    def get_allocation(self, request_id: str) -> StageKVRequestAllocation | None:
        return self._allocations.get(request_id)

    def mark_stable_computed(self, request_id: str) -> None:
        adapters = self._adapters.get(request_id)
        if adapters is None:
            raise KeyError(f"request {request_id!r} has no KV allocation")
        for adapter in adapters:
            adapter.mark_stable_computed()

    def free(self, request_id: str) -> bool:
        adapters = self._adapters.pop(request_id, None)
        allocation = self._allocations.pop(request_id, None)
        if adapters is None:
            assert allocation is None
            return False
        assert allocation is not None
        self._free_adapters(adapters)
        return True

    def close(self) -> None:
        for request_id in tuple(self._allocations):
            self.free(request_id)

    @staticmethod
    def _adapter_request_id(request_id: str, branch_id: int) -> str:
        return f"{request_id}/dit-kv-branch-{branch_id}"

    def _make_adapter(
        self,
        request_id: str,
        branch: StageKVBranchRequirement,
    ) -> StablePrefixRequestAdapter:
        return StablePrefixRequestAdapter(
            self._adapter_request_id(request_id, branch.branch_id),
            stable_len=branch.stable_len,
            current_len=branch.current_len,
        )

    def _free_adapters(
        self, adapters: tuple[StablePrefixRequestAdapter, ...] | list[StablePrefixRequestAdapter]
    ) -> None:
        for adapter in reversed(adapters):
            self._kv_cache_manager.free(adapter)
