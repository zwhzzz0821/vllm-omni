# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.stage_kv import StageKVCacheMode
from vllm_omni.diffusion.stage_kv.registry import (
    create_stage_kv_scheduler_runtime,
    get_stage_kv_cache_mode,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_dense_legacy_is_default_and_does_not_build_runtime() -> None:
    config = SimpleNamespace(omni_kv_config={})

    assert get_stage_kv_cache_mode(config) is StageKVCacheMode.DENSE_LEGACY
    assert create_stage_kv_scheduler_runtime(config) is None


def test_paged_worker_local_does_not_enable_scheduler_ownership() -> None:
    config = SimpleNamespace(omni_kv_config={"cache_mode": "paged_worker_local"})

    assert get_stage_kv_cache_mode(config) is StageKVCacheMode.PAGED_WORKER_LOCAL
    assert create_stage_kv_scheduler_runtime(config) is None


def test_unknown_cache_mode_is_rejected() -> None:
    config = SimpleNamespace(omni_kv_config={"cache_mode": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Stage KV cache_mode"):
        get_stage_kv_cache_mode(config)


def test_paged_scheduler_requires_explicit_logical_capacity() -> None:
    config = SimpleNamespace(
        model_class_name="HunyuanImage3Pipeline",
        omni_kv_config={"cache_mode": "paged_scheduler"},
    )

    with pytest.raises(ValueError, match="paged_kv_num_blocks"):
        create_stage_kv_scheduler_runtime(config)


def test_paged_scheduler_rejects_unsupported_models_before_loading_config() -> None:
    config = SimpleNamespace(
        model_class_name="OtherPipeline",
        omni_kv_config={"cache_mode": "paged_scheduler", "paged_kv_num_blocks": 8},
    )

    with pytest.raises(ValueError, match="supports only HunyuanImage3Pipeline"):
        create_stage_kv_scheduler_runtime(config)
