from __future__ import annotations


def apply_env_device_kwargs(env_kwargs: dict, env_device: str | None) -> None:
    """Translate a simple device string into this ManiSkill version's backend args."""
    if env_device is None:
        return
    device = str(env_device).strip().lower()
    if not device:
        return
    if device == "cpu":
        env_kwargs.setdefault("sim_backend", "physx_cpu")
        env_kwargs.setdefault("render_backend", "sapien_cpu")
        return
    if device == "cuda":
        env_kwargs.setdefault("sim_backend", "physx_cuda")
        env_kwargs.setdefault("render_backend", "sapien_cuda")
        return
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        env_kwargs.setdefault("sim_backend", f"physx_cuda:{index}")
        env_kwargs.setdefault("render_backend", f"sapien_cuda:{index}")
        return
    raise ValueError(
        f"Unsupported --env-device {env_device!r}; expected 'cpu', 'cuda', or 'cuda:N'."
    )
