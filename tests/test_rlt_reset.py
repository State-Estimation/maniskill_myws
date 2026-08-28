from __future__ import annotations

from collections import UserDict
import pytest

from maniskill_myws.rlt.reset import (
    EPISODE_RESET_STRATEGY,
    FRESH_SCENE_RESET_OPTIONS,
    reset_env_fresh_scene,
)


class _ResetEnv:
    def __init__(self, info) -> None:
        self.info = info
        self.calls: list[tuple[int, dict[str, bool]]] = []

    def reset(self, *, seed: int, options: dict[str, bool]):
        self.calls.append((seed, options))
        return {"seed": seed}, self.info


def test_fresh_scene_reset_requires_positive_reconfigure_attestation() -> None:
    env = _ResetEnv(UserDict(reconfigure=True))

    observation, info = reset_env_fresh_scene(env, seed=17, operation="test")

    assert observation == {"seed": 17}
    assert info["reconfigure"] is True
    assert env.calls == [(17, {"reconfigure": True})]
    assert dict(FRESH_SCENE_RESET_OPTIONS) == {"reconfigure": True}
    assert EPISODE_RESET_STRATEGY == "reconfigure_each_episode_v1"


@pytest.mark.parametrize("info", [None, {}, {"reconfigure": False}, {"reconfigure": 1}])
def test_fresh_scene_reset_rejects_missing_or_ambiguous_attestation(info) -> None:
    with pytest.raises(RuntimeError, match="fresh-scene reset"):
        reset_env_fresh_scene(_ResetEnv(info), seed=0, operation="test")


def test_fresh_scene_reset_never_falls_back_after_reset_error() -> None:
    class BrokenEnv:
        def __init__(self) -> None:
            self.calls = 0

        def reset(self, *, seed: int, options: dict[str, bool]):
            self.calls += 1
            raise ValueError("cannot rebuild")

    env = BrokenEnv()
    with pytest.raises(RuntimeError, match="fresh-scene reset failed") as caught:
        reset_env_fresh_scene(env, seed=2, operation="test")

    assert env.calls == 1
    assert isinstance(caught.value.__cause__, ValueError)
