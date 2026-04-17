import importlib

import pytest


@pytest.mark.smoke
def test_env_make_if_available():
    gym_spec = importlib.util.find_spec("gymnasium")
    ms_spec = importlib.util.find_spec("mani_skill")
    if gym_spec is None or ms_spec is None:
        pytest.skip("gymnasium/mani_skill not installed in this environment")

    import gymnasium as gym

    import maniskill_myws

    maniskill_myws.register()
    try:
        env = gym.make(
            "TurnGlobeValve-v1",
            obs_mode="state",
            reward_mode="none",
            render_mode=None,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "supported physical device" in msg or "Rendering might be disabled" in msg:
            pytest.skip(f"render device unavailable in this environment: {exc}")
        raise
    env.reset(seed=0)
    env.close()

