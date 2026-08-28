from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


BASE_REPLAY_SCHEMA = "rlt_base_chunks_v3"
EPISODE_RESET_STRATEGY = "reconfigure_each_episode_v1"
BRANCH_PREFIX_RESET_STRATEGY = "reconfigure_each_prefix_v1"
FRESH_SCENE_RESET_OPTIONS: Mapping[str, bool] = MappingProxyType(
    {"reconfigure": True}
)
EPISODE_SOLVER_HISTORY_ISOLATION = "fresh_physx_scene_per_episode_v1"
BRANCH_PREFIX_SOLVER_HISTORY_ISOLATION = "fresh_physx_scene_per_prefix_v1"


def reset_env_fresh_scene(
    env: Any,
    *,
    seed: int,
    operation: str = "RLT episode",
) -> tuple[Any, Mapping[str, Any]]:
    """Rebuild an environment and prove that the requested reset occurred.

    GPU PhysX can retain solver/contact history across an ordinary reset. RLT
    therefore never falls back to a plain reset: both a reset exception and an
    environment that does not attest ``reconfigure=True`` are fatal.
    """

    options = dict(FRESH_SCENE_RESET_OPTIONS)
    try:
        result = env.reset(seed=seed, options=options)
    except Exception as error:
        raise RuntimeError(
            f"{operation} requires env.reset(seed=..., "
            "options={'reconfigure': True}); fresh-scene reset failed"
        ) from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            f"{operation} fresh-scene reset must return an (observation, info) tuple"
        )
    observation, info = result
    if not isinstance(info, Mapping):
        raise RuntimeError(
            f"{operation} fresh-scene reset returned non-mapping info: "
            f"{type(info).__name__}"
        )
    if info.get("reconfigure") is not True:
        raise RuntimeError(
            f"{operation} fresh-scene reset was not acknowledged: "
            f"info['reconfigure']={info.get('reconfigure')!r}"
        )
    return observation, info
