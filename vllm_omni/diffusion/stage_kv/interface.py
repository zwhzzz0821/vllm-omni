# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StageKVCacheMode(str, Enum):
    """Migration mode for diffusion-stage KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    PAGED_WORKER_LOCAL = "paged_worker_local"
    PAGED_SCHEDULER = "paged_scheduler"


@dataclass(frozen=True)
class StageKVBranchRequirement:
    """Paged KV capacity required by one local execution branch.

    ``stable_len`` covers request-local K/V that is written once and reused
    across denoising steps. ``current_len`` covers current-step K/V that lives
    in the same physical page pool and block table but is overwritten on every
    step and is never semantically committed or transferred.
    """

    branch_id: int
    stable_len: int
    current_len: int

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError(f"branch_id must be non-negative, got {self.branch_id}")
        if self.stable_len < 0:
            raise ValueError(f"stable_len must be non-negative, got {self.stable_len}")
        if self.current_len <= 0:
            raise ValueError(f"current_len must be positive, got {self.current_len}")

    @property
    def resident_len(self) -> int:
        """Total logical tokens that must have physical slots."""

        return self.stable_len + self.current_len


@dataclass(frozen=True)
class StageKVRequirement:
    """Model-derived KV capacity plan consumed by the diffusion scheduler."""

    request_id: str
    branches: tuple[StageKVBranchRequirement, ...]
    request_layout_digest: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.branches:
            raise ValueError("at least one branch requirement is required")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError(f"branch IDs must be unique, got {branch_ids}")
        if not self.request_layout_digest:
            raise ValueError("request_layout_digest must be non-empty")


@dataclass(frozen=True)
class StageKVBranchAllocation:
    """Scheduler-owned block allocation for one execution branch."""

    branch_id: int
    adapter_request_id: str
    block_ids: tuple[tuple[int, ...], ...]
    stable_len: int
    current_len: int

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError(f"branch_id must be non-negative, got {self.branch_id}")
        if not self.adapter_request_id:
            raise ValueError("adapter_request_id must be non-empty")
        if not self.block_ids:
            raise ValueError("at least one KV cache group is required")
        if any(not group for group in self.block_ids):
            raise ValueError("every KV cache group must contain at least one block")
        if self.stable_len < 0:
            raise ValueError(f"stable_len must be non-negative, got {self.stable_len}")
        if self.current_len <= 0:
            raise ValueError(f"current_len must be positive, got {self.current_len}")

    @property
    def resident_len(self) -> int:
        return self.stable_len + self.current_len


@dataclass(frozen=True)
class StageKVRequestAllocation:
    """Atomic collection of all branch allocations for one public request."""

    request_id: str
    allocation_id: int
    cache_layout_fingerprint: str
    request_layout_digest: str
    branches: tuple[StageKVBranchAllocation, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.allocation_id <= 0:
            raise ValueError(f"allocation_id must be positive, got {self.allocation_id}")
        if not self.cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        if not self.request_layout_digest:
            raise ValueError("request_layout_digest must be non-empty")
        if not self.branches:
            raise ValueError("at least one branch allocation is required")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError(f"branch IDs must be unique, got {branch_ids}")
        adapter_request_ids = [branch.adapter_request_id for branch in self.branches]
        if len(adapter_request_ids) != len(set(adapter_request_ids)):
            raise ValueError(f"adapter request IDs must be unique, got {adapter_request_ids}")

    def to_metadata(self) -> StageKVMetadata:
        return StageKVMetadata(
            request_id=self.request_id,
            allocation_id=self.allocation_id,
            cache_mode=StageKVCacheMode.PAGED_SCHEDULER,
            cache_layout_fingerprint=self.cache_layout_fingerprint,
            request_layout_digest=self.request_layout_digest,
            branches=tuple(
                StageKVBranchMetadata(
                    branch_id=branch.branch_id,
                    block_ids=branch.block_ids,
                    stable_len=branch.stable_len,
                    current_len=branch.current_len,
                )
                for branch in self.branches
            ),
        )


@dataclass(frozen=True)
class StageKVBranchMetadata:
    """Serializable Scheduler-to-Worker allocation for one CFG branch."""

    branch_id: int
    block_ids: tuple[tuple[int, ...], ...]
    stable_len: int
    current_len: int

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError(f"branch_id must be non-negative, got {self.branch_id}")
        if not self.block_ids or any(not group for group in self.block_ids):
            raise ValueError("every KV cache group must contain at least one block")
        if self.stable_len < 0:
            raise ValueError(f"stable_len must be non-negative, got {self.stable_len}")
        if self.current_len <= 0:
            raise ValueError(f"current_len must be positive, got {self.current_len}")

    @property
    def resident_len(self) -> int:
        return self.stable_len + self.current_len


@dataclass(frozen=True)
class StageKVMetadata:
    """Serializable paged-KV metadata emitted for a new Scheduler allocation."""

    request_id: str
    allocation_id: int
    cache_mode: StageKVCacheMode
    cache_layout_fingerprint: str
    request_layout_digest: str
    branches: tuple[StageKVBranchMetadata, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.allocation_id <= 0:
            raise ValueError(f"allocation_id must be positive, got {self.allocation_id}")
        if self.cache_mode is not StageKVCacheMode.PAGED_SCHEDULER:
            raise ValueError(f"StageKVMetadata requires paged_scheduler mode, got {self.cache_mode.value!r}")
        if not self.cache_layout_fingerprint:
            raise ValueError("cache_layout_fingerprint must be non-empty")
        if not self.request_layout_digest:
            raise ValueError("request_layout_digest must be non-empty")
        if not self.branches:
            raise ValueError("at least one branch metadata entry is required")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError(f"branch IDs must be unique, got {branch_ids}")
