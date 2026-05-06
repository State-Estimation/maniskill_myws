from maniskill_myws.task_prompts import get_task_prompt, get_task_prompts


def test_joint_training_tasks_have_twenty_prompt_variants():
    for env_id in (
        "OpenSafeDoor-v2",
        "TurnGlobeValve-v1",
        "SolarPanelStatic-v1",
        "TakeSafetyHook-v1",
    ):
        assert len(get_task_prompts(env_id)) == 20


def test_default_prompt_is_first_variant():
    prompts = get_task_prompts("OpenSafeDoor-v2")

    assert get_task_prompt("OpenSafeDoor-v2") == prompts[0]


def test_prompt_variant_is_stable():
    key = "episode_0001"

    assert get_task_prompt("TakeSafetyHook-v1", variant=key) == get_task_prompt(
        "TakeSafetyHook-v1", variant=key
    )


def test_unknown_prompt_returns_none():
    assert get_task_prompt("MissingTask-v1") is None
