from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from maniskill_myws.rlt.backend import require_resolved_backend


@dataclass
class _Device:
    text: str
    cuda: bool
    cuda_id: int = -1

    def is_cuda(self) -> bool:
        return self.cuda

    def __str__(self) -> str:
        return self.text


def _env(*, sim: str = "physx_cpu", render: str = "sapien_cuda"):
    sim_cuda = sim == "physx_cuda"
    render_cuda = render == "sapien_cuda"
    backend = SimpleNamespace(
        sim_backend=sim,
        render_backend=render,
        sim_device=_Device(
            "cuda:0" if sim_cuda else "cpu", sim_cuda, 0 if sim_cuda else -1
        ),
        render_device=_Device(
            "cuda:0" if render_cuda else "cpu",
            render_cuda,
            0 if render_cuda else -1,
        ),
    )
    unwrapped = SimpleNamespace(
        backend=backend,
        device=torch.device("cuda:0" if sim_cuda else "cpu"),
        gpu_sim_enabled=sim_cuda,
        num_envs=1,
    )
    return SimpleNamespace(unwrapped=unwrapped)


def test_cpu_physx_with_gpu_renderer_is_attested() -> None:
    metadata = require_resolved_backend(
        _env(),
        expected_sim_backend="physx_cpu",
        expected_render_backend="sapien_cuda:0",
    )
    assert metadata == {
        "actual_sim_backend": "physx_cpu",
        "actual_render_backend": "sapien_cuda",
        "actual_sim_device": "cpu",
        "actual_render_device": "cuda:0",
        "actual_env_device": "cpu",
        "actual_gpu_sim_enabled": False,
        "actual_num_envs": 1,
    }


def test_backend_attestation_rejects_gpu_physx_and_renderer_fallback() -> None:
    with pytest.raises(RuntimeError, match="sim backend resolved"):
        require_resolved_backend(
            _env(sim="physx_cuda"),
            expected_sim_backend="physx_cpu",
            expected_render_backend="sapien_cuda:0",
        )
    with pytest.raises(RuntimeError, match="render backend resolved"):
        require_resolved_backend(
            _env(render="sapien_cpu"),
            expected_sim_backend="physx_cpu",
            expected_render_backend="sapien_cuda:0",
        )
