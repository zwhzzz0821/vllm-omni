# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.v1.core.kv_cache_manager import KVCacheManager

import vllm_omni.diffusion.sched.base_scheduler as base_scheduler_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.diffusion.stage_kv import (
    DiTKVCacheManager,
    StageKVBranchRequirement,
    StageKVCacheGroupSpec,
    StageKVCacheMode,
    StageKVCacheSpec,
    StageKVRequirement,
    StageKVSchedulerRuntime,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Planner:
    def __init__(self, *, stable_len: int = 5, current_len: int = 3) -> None:
        self.stable_len = stable_len
        self.current_len = current_len
        self.calls: list[str] = []

    def plan(self, request: OmniDiffusionRequest) -> StageKVRequirement:
        self.calls.append(request.request_id)
        return StageKVRequirement(
            request_id=request.request_id,
            branches=(
                StageKVBranchRequirement(
                    branch_id=0,
                    stable_len=self.stable_len,
                    current_len=self.current_len,
                ),
            ),
            request_layout_digest=f"layout-{request.request_id}",
        )


def _runtime(*, num_blocks: int, planner: _Planner | None = None) -> StageKVSchedulerRuntime:
    cache_spec = StageKVCacheSpec.create(
        model_identity="test-model",
        groups=(
            StageKVCacheGroupSpec(
                group_id=0,
                layer_names=("layer0",),
                block_size=4,
                num_kv_heads=2,
                head_size=8,
                dtype="float32",
                non_causal=True,
            ),
        ),
    )
    kv_cache_config = cache_spec.to_vllm_kv_cache_config(num_blocks=num_blocks)
    native_manager = KVCacheManager(
        kv_cache_config,
        max_model_len=num_blocks * cache_spec.block_size,
        scheduler_block_size=cache_spec.block_size,
        hash_block_size=cache_spec.block_size,
        max_num_batched_tokens=num_blocks * cache_spec.block_size,
        enable_caching=False,
    )
    return StageKVSchedulerRuntime(
        cache_mode=StageKVCacheMode.PAGED_SCHEDULER,
        cache_spec=cache_spec,
        kv_cache_config=kv_cache_config,
        max_model_len=num_blocks * cache_spec.block_size,
        planner=planner or _Planner(),
        manager=DiTKVCacheManager(
            native_manager,
            cache_layout_fingerprint=cache_spec.layout_fingerprint,
        ),
    )


def _config(*, max_num_seqs: int = 1):
    return SimpleNamespace(
        max_num_seqs=max_num_seqs,
        omni_kv_config={"cache_mode": "paged_scheduler"},
    )


def _request(request_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=f"prompt-{request_id}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=request_id,
    )


def _finished_output(request_id: str) -> RunnerOutput:
    return RunnerOutput(
        request_id=request_id,
        step_index=None,
        finished=True,
        result=DiffusionOutput(output=None),
    )


def _initialize_scheduler(monkeypatch: pytest.MonkeyPatch, runtime: StageKVSchedulerRuntime, *, max_num_seqs: int = 1):
    monkeypatch.setattr(base_scheduler_module, "create_stage_kv_scheduler_runtime", lambda _config: runtime)
    scheduler = RequestScheduler()
    scheduler.initialize(_config(max_num_seqs=max_num_seqs))
    return scheduler


def test_scheduler_plans_allocates_emits_metadata_and_frees(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = _Planner()
    runtime = _runtime(num_blocks=4, planner=planner)
    scheduler = _initialize_scheduler(monkeypatch, runtime)
    free_before = runtime.manager.num_free_blocks

    request_id = scheduler.add_request(_request("req"))
    first = scheduler.schedule()

    assert planner.calls == [request_id]
    assert first.scheduled_request_ids == [request_id]
    assert set(first.stage_kv_metadata) == {request_id}
    metadata = first.stage_kv_metadata[request_id]
    assert metadata.request_id == request_id
    assert metadata.allocation_id == 1
    assert metadata.cache_layout_fingerprint == runtime.cache_spec.layout_fingerprint
    assert metadata.branches[0].stable_len == 5
    assert metadata.branches[0].current_len == 3
    assert len(metadata.branches[0].block_ids[0]) == 2
    assert runtime.manager.num_free_blocks == free_before - 2

    cached = scheduler.schedule()
    assert cached.scheduled_cached_reqs.request_ids == [request_id]
    assert cached.stage_kv_metadata == {}

    assert scheduler.update_from_output(first, _finished_output(request_id)) == {request_id}
    assert runtime.manager.num_free_blocks == free_before


def test_scheduler_capacity_failure_keeps_fifo_request_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(num_blocks=3)
    scheduler = _initialize_scheduler(monkeypatch, runtime, max_num_seqs=2)
    first_id = scheduler.add_request(_request("first"))
    second_id = scheduler.add_request(_request("second"))

    first = scheduler.schedule()

    assert first.scheduled_request_ids == [first_id]
    assert scheduler.get_request_state(second_id).status == DiffusionRequestStatus.WAITING
    assert first.num_waiting_reqs == 1

    scheduler.update_from_output(first, _finished_output(first_id))
    second = scheduler.schedule()
    assert second.scheduled_request_ids == [second_id]
    assert second.stage_kv_metadata[second_id].allocation_id == 2


def test_scheduler_preemption_retains_allocation_and_resends_only_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(num_blocks=4)
    scheduler = _initialize_scheduler(monkeypatch, runtime)
    request_id = scheduler.add_request(_request("req"))
    scheduler.schedule()
    allocation = scheduler.get_request_state(request_id).stage_kv_allocation

    assert scheduler.preempt_request(request_id) is True
    resumed = scheduler.schedule()

    assert scheduler.get_request_state(request_id).stage_kv_allocation is allocation
    assert resumed.scheduled_cached_reqs.request_ids == [request_id]
    assert resumed.stage_kv_metadata == {}
    scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)


def test_scheduler_close_releases_all_allocations(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(num_blocks=4)
    scheduler = _initialize_scheduler(monkeypatch, runtime)
    free_before = runtime.manager.num_free_blocks
    scheduler.add_request(_request("req"))
    scheduler.schedule()
    assert runtime.manager.num_free_blocks < free_before

    scheduler.close()

    assert runtime.manager.num_free_blocks == free_before


def test_scheduler_pop_request_state_releases_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(num_blocks=4)
    scheduler = _initialize_scheduler(monkeypatch, runtime)
    free_before = runtime.manager.num_free_blocks
    request_id = scheduler.add_request(_request("req"))
    scheduler.schedule()

    state = scheduler.pop_request_state(request_id)

    assert state is not None
    assert state.stage_kv_allocation is None
    assert runtime.manager.num_free_blocks == free_before


def test_scheduler_initialize_failure_does_not_publish_partial_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(num_blocks=4)
    scheduler = _initialize_scheduler(monkeypatch, runtime)
    free_before = runtime.manager.num_free_blocks
    scheduler.add_request(_request("req"))
    scheduler.schedule()
    assert runtime.manager.num_free_blocks < free_before

    def fail_runtime_creation(_config):
        raise RuntimeError("runtime creation failed")

    monkeypatch.setattr(base_scheduler_module, "create_stage_kv_scheduler_runtime", fail_runtime_creation)
    with pytest.raises(RuntimeError, match="runtime creation failed"):
        scheduler.initialize(_config())

    assert scheduler.stage_kv_cache_mode is StageKVCacheMode.DENSE_LEGACY
    assert scheduler._stage_kv_runtime is None
    assert runtime.manager.num_free_blocks == free_before
