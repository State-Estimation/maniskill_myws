"""Fail-closed simulator/backend provenance for RLT artifacts.

Command-line backend strings record intent. These helpers also attest the
backend ManiSkill actually resolved after environment construction, so a
renderer fallback or accidental GPU PhysX environment cannot enter a CPU
replay unnoticed.
"""

from __future__ import annotations

from typing import Any


_SIM_ALIASES = {
    "cpu": "physx_cpu",
    "physx_cpu": "physx_cpu",
    "cuda": "physx_cuda",
    "gpu": "physx_cuda",
    "physx_cuda": "physx_cuda",
}
_RENDER_ALIASES = {
    "cpu": "sapien_cpu",
    "sapien_cpu": "sapien_cpu",
    "cuda": "sapien_cuda",
    "gpu": "sapien_cuda",
    "sapien_cuda": "sapien_cuda",
    "none": "none",
}


def _requested_backend(
    value: str | None,
    *,
    aliases: dict[str, str],
    label: str,
) -> tuple[str, int | None]:
    raw = "none" if value is None else str(value)
    name, separator, index_text = raw.partition(":")
    normalized = aliases.get(name, name)
    if not separator:
        return normalized, None
    try:
        index = int(index_text)
    except ValueError as error:
        raise ValueError(f"{label} has an invalid device index: {value!r}") from error
    if index < 0:
        raise ValueError(f"{label} device index must be non-negative: {value!r}")
    return normalized, index


def resolved_backend_metadata(env: Any) -> dict[str, object]:
    """Return JSON-safe backend facts from a constructed ManiSkill env."""

    unwrapped = env.unwrapped
    backend = unwrapped.backend
    render_device = backend.render_device
    return {
        "actual_sim_backend": str(backend.sim_backend),
        "actual_render_backend": str(backend.render_backend),
        "actual_sim_device": str(backend.sim_device),
        "actual_render_device": None if render_device is None else str(render_device),
        "actual_env_device": str(unwrapped.device),
        "actual_gpu_sim_enabled": bool(unwrapped.gpu_sim_enabled),
        "actual_num_envs": int(unwrapped.num_envs),
    }


def require_resolved_backend(
    env: Any,
    *,
    expected_sim_backend: str,
    expected_render_backend: str | None,
) -> dict[str, object]:
    """Attest requested versus resolved backends and reject silent fallback."""

    expected_sim, expected_sim_index = _requested_backend(
        expected_sim_backend,
        aliases=_SIM_ALIASES,
        label="sim backend",
    )
    expected_render, expected_render_index = _requested_backend(
        expected_render_backend,
        aliases=_RENDER_ALIASES,
        label="render backend",
    )
    if expected_sim not in {"physx_cpu", "physx_cuda"}:
        raise ValueError(
            "RLT provenance requires an explicit physx_cpu or physx_cuda backend, "
            f"got {expected_sim_backend!r}"
        )

    unwrapped = env.unwrapped
    backend = unwrapped.backend
    actual = resolved_backend_metadata(env)
    violations: list[str] = []
    if actual["actual_sim_backend"] != expected_sim:
        violations.append(
            f"sim backend resolved to {actual['actual_sim_backend']!r}, "
            f"expected {expected_sim!r}"
        )
    if actual["actual_render_backend"] != expected_render:
        violations.append(
            f"render backend resolved to {actual['actual_render_backend']!r}, "
            f"expected {expected_render!r}"
        )
    if actual["actual_num_envs"] != 1:
        violations.append(
            f"num_envs resolved to {actual['actual_num_envs']!r}, expected 1"
        )

    sim_is_cuda = bool(backend.sim_device.is_cuda())
    expected_sim_cuda = expected_sim == "physx_cuda"
    if sim_is_cuda != expected_sim_cuda:
        violations.append(
            f"sim device CUDA state is {sim_is_cuda}, expected {expected_sim_cuda}"
        )
    if bool(unwrapped.gpu_sim_enabled) != expected_sim_cuda:
        violations.append(
            f"gpu_sim_enabled={bool(unwrapped.gpu_sim_enabled)}, "
            f"expected {expected_sim_cuda}"
        )
    expected_env_type = "cuda" if expected_sim_cuda else "cpu"
    if str(unwrapped.device.type) != expected_env_type:
        violations.append(
            f"env device type is {unwrapped.device.type!r}, "
            f"expected {expected_env_type!r}"
        )
    if expected_sim_index is not None and int(backend.sim_device.cuda_id) != (
        expected_sim_index
    ):
        violations.append(
            f"sim CUDA index is {int(backend.sim_device.cuda_id)}, "
            f"expected {expected_sim_index}"
        )

    if expected_render == "none":
        if backend.render_device is not None:
            violations.append("render device is enabled, expected no renderer")
    elif backend.render_device is None:
        violations.append("render device is disabled")
    else:
        render_is_cuda = bool(backend.render_device.is_cuda())
        expected_render_cuda = expected_render == "sapien_cuda"
        if render_is_cuda != expected_render_cuda:
            violations.append(
                f"render device CUDA state is {render_is_cuda}, "
                f"expected {expected_render_cuda}"
            )
        if (
            expected_render_index is not None
            and int(backend.render_device.cuda_id) != expected_render_index
        ):
            violations.append(
                f"render CUDA index is {int(backend.render_device.cuda_id)}, "
                f"expected {expected_render_index}"
            )

    if violations:
        raise RuntimeError(
            "resolved ManiSkill backend violates the requested RLT protocol: "
            + "; ".join(violations)
        )
    return actual
