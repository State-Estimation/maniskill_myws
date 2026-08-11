from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rlt"
    / "train_frozen_latent_residual.py"
)
SPEC = importlib.util.spec_from_file_location("train_frozen_latent_residual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


@pytest.mark.parametrize(
    ("reward", "success", "grasp_event"),
    [
        (0.0, False, False),
        (0.25, False, True),
        (1.0, True, False),
        (1.25, True, True),
    ],
)
def test_dense_reward_schema_accepts_exact_components(reward, success, grasp_event):
    result = TRAIN._validate_environment_reward(
        reward_mode="dense",
        reward_value=reward,
        info={"success": success, "grasp_reward_event": grasp_event},
        grasp_process_reward=0.25,
        task_success_reward=1.0,
    )
    assert result == (success, grasp_event)


def test_dense_reward_schema_rejects_hidden_or_mismatched_reward():
    with pytest.raises(ValueError, match="does not match"):
        TRAIN._validate_environment_reward(
            reward_mode="dense",
            reward_value=1.0,
            info={"success": False, "grasp_reward_event": True},
            grasp_process_reward=0.25,
            task_success_reward=1.0,
        )
    with pytest.raises(ValueError, match="missing grasp_reward_event"):
        TRAIN._validate_environment_reward(
            reward_mode="dense",
            reward_value=0.0,
            info={"success": False},
            grasp_process_reward=0.25,
            task_success_reward=1.0,
        )


@pytest.mark.parametrize(
    ("reward", "success", "grasp_event"),
    [
        (0.0, False, False),
        (0.25, False, True),
        (1.0, True, False),
        (1.25, True, True),
    ],
)
def test_sparse_reward_schema_includes_grasp_and_success(
    reward, success, grasp_event
):
    assert TRAIN._validate_environment_reward(
        reward_mode="sparse",
        reward_value=reward,
        info={"success": success, "grasp_reward_event": grasp_event},
        grasp_process_reward=0.25,
        task_success_reward=1.0,
    ) == (success, grasp_event)


def test_sparse_reward_schema_rejects_success_only_legacy_semantics():
    with pytest.raises(ValueError, match="does not match"):
        TRAIN._validate_environment_reward(
            reward_mode="sparse",
            reward_value=0.0,
            info={"success": False, "grasp_reward_event": True},
            grasp_process_reward=0.25,
            task_success_reward=1.0,
        )


def test_independent_online_exploration_can_override_accepted_actor() -> None:
    assert TRAIN._should_explore_online(
        eligible=True,
        accepted=True,
        independent=True,
        probability=1.0,
        rng=np.random.default_rng(1),
    )
    assert not TRAIN._should_explore_online(
        eligible=True,
        accepted=True,
        independent=False,
        probability=1.0,
        rng=np.random.default_rng(1),
    )


def test_independent_online_exploration_consumes_accepted_context_rng() -> None:
    independent = np.random.default_rng(7)
    expected_independent = np.random.default_rng(7)
    assert not TRAIN._should_explore_online(
        eligible=True,
        accepted=True,
        independent=True,
        probability=0.0,
        rng=independent,
    )
    expected_independent.random()
    assert independent.random() == expected_independent.random()

    dependent = np.random.default_rng(7)
    expected_dependent = np.random.default_rng(7)
    assert not TRAIN._should_explore_online(
        eligible=True,
        accepted=True,
        independent=False,
        probability=1.0,
        rng=dependent,
    )
    assert dependent.random() == expected_dependent.random()


def test_online_exploration_respects_eligibility() -> None:
    assert not TRAIN._should_explore_online(
        eligible=False,
        accepted=False,
        independent=True,
        probability=1.0,
        rng=np.random.default_rng(2),
    )


def test_legacy_budget_exhaustion_is_opt_in_and_exact() -> None:
    assert not TRAIN._legacy_budget_exhausted(
        env_steps=49_999, total_env_steps=50_000, enabled=True
    )
    assert TRAIN._legacy_budget_exhausted(
        env_steps=50_000, total_env_steps=50_000, enabled=True
    )
    assert not TRAIN._legacy_budget_exhausted(
        env_steps=50_000, total_env_steps=50_000, enabled=False
    )


def test_trainer_state_round_trip_restores_rng_and_episode_boundary(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    torch.manual_seed(456)
    payload = TRAIN._trainer_state_payload(
        env_steps=50000,
        episode=162,
        last_save=50000,
        recent_successes=[0, 1, 1],
        warmup_successes=173,
        warmup_failures=122,
        warmup_nonzero_transitions=302,
        rng=rng,
        torch=torch,
        wandb_run_id="example-run",
        snapshot_id="generation-a",
    )
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = copy.deepcopy(payload["trainer_rng_state"])
    expected_rng_value = expected_rng.random()
    expected_torch_value = torch.rand(4)

    path = tmp_path / "trainer_state.json"
    TRAIN._write_json_atomic(path, payload)
    rng.random(10)
    torch.manual_seed(999)
    restored = TRAIN._load_trainer_state(path, rng=rng, torch=torch)

    assert restored["env_steps"] == 50000
    assert restored["episode"] == 162
    assert restored["resume_mode"] == "generation_bound_state_restore"
    assert restored["snapshot_id"] == "generation-a"
    assert restored["wandb_run_id"] == "example-run"
    assert rng.random() == expected_rng_value
    assert torch.equal(torch.rand(4), expected_torch_value)


def test_snapshot_generation_rejects_mixed_files() -> None:
    progress = {
        "resume_mode": "generation_bound_state_restore",
        "snapshot_id": "generation-a",
    }
    TRAIN._validate_snapshot_generation(
        resume_progress=progress,
        checkpoint_snapshot_id="generation-a",
        replay_snapshot_id="generation-a",
    )
    with pytest.raises(ValueError, match="different generations"):
        TRAIN._validate_snapshot_generation(
            resume_progress=progress,
            checkpoint_snapshot_id="generation-b",
            replay_snapshot_id="generation-a",
        )


@pytest.mark.parametrize(
    ("checkpoint_id", "replay_id"),
    ((None, None), ("generation-a", None), (None, "generation-a")),
)
def test_policy_fork_requires_explicit_snapshot_pair(
    checkpoint_id: str | None, replay_id: str | None
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TRAIN._require_checkpoint_replay_snapshot_pair(
            checkpoint_snapshot_id=checkpoint_id,
            replay_snapshot_id=replay_id,
            source_name="test fork",
        )
    with pytest.raises(ValueError, match="generations differ"):
        TRAIN._require_checkpoint_replay_snapshot_pair(
            checkpoint_snapshot_id="generation-a",
            replay_snapshot_id="generation-b",
            source_name="test fork",
        )
    assert (
        TRAIN._require_checkpoint_replay_snapshot_pair(
            checkpoint_snapshot_id="generation-a",
            replay_snapshot_id="generation-a",
            source_name="test fork",
        )
        == "generation-a"
    )
