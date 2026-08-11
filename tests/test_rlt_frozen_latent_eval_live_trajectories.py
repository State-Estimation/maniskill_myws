from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rlt" / "eval_frozen_latent_residual.py"
SPEC = importlib.util.spec_from_file_location("eval_frozen_latent_residual", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def test_checkpoint_environment_mismatch_fails_before_rollout() -> None:
    with pytest.raises(ValueError, match="trained for 'TakeSafetyHook-v1'"):
        evaluator._assert_checkpoint_environment(
            "TakeSafetyHook-v2",
            {"env_id": "TakeSafetyHook-v1"},
        )


def test_terminal_sparse_reward_schema_fallback_is_checkpoint_gated() -> None:
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(SUPPORTED_REWARD_MODES=["sparse", "none"])
    )

    schema = evaluator._environment_reward_schema(
        env,
        reward_mode="sparse",
        expected_schema=evaluator.TERMINAL_SUCCESS_SPARSE_REWARD_SCHEMA,
    )

    assert schema == evaluator.TERMINAL_SUCCESS_SPARSE_REWARD_SCHEMA
    assert schema is not evaluator.TERMINAL_SUCCESS_SPARSE_REWARD_SCHEMA


def test_terminal_sparse_reward_schema_fallback_fails_closed() -> None:
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(SUPPORTED_REWARD_MODES=["sparse", "none"])
    )

    with pytest.raises(RuntimeError, match="does not declare the exact"):
        evaluator._environment_reward_schema(
            env,
            reward_mode="sparse",
            expected_schema={"schema": "unknown"},
        )


def test_explicit_environment_reward_schema_takes_precedence() -> None:
    explicit = {"schema": "explicit_process_reward_v1"}
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            SUPPORTED_REWARD_MODES=["sparse"],
            grasp_reward_schema=explicit,
        )
    )

    schema = evaluator._environment_reward_schema(
        env,
        reward_mode="sparse",
        expected_schema={"schema": "different_checkpoint_schema"},
    )

    assert schema == explicit
    assert schema is not explicit


def test_tcp_position_accepts_single_environment_numpy_and_torch() -> None:
    numpy_obs = {
        "extra": {
            "tcp_pose": np.asarray(
                [[0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            )
        }
    }
    torch_obs = {
        "extra": {
            "tcp_pose": torch.tensor(
                [[0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            )
        }
    }

    np.testing.assert_array_equal(
        evaluator._tcp_position(numpy_obs, "extra/tcp_pose"),
        np.asarray([0.1, -0.2, 0.3], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        evaluator._tcp_position(torch_obs, "obs/extra/tcp_pose"),
        np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "pose",
    [
        np.zeros((2, 7), dtype=np.float32),
        np.zeros((1, 6), dtype=np.float32),
        np.asarray([[0.0, 0.0, np.nan, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    ],
)
def test_tcp_position_rejects_invalid_schema(pose: np.ndarray) -> None:
    with pytest.raises(ValueError, match="single-environment shape"):
        evaluator._tcp_position({"extra": {"tcp_pose": pose}}, "extra/tcp_pose")


def test_trajectory_geometry_builds_colored_segments() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        dtype=np.float32,
    )
    color = np.asarray([0.1, 0.2, 0.3, 1.0], dtype=np.float32)

    vertices, colors = evaluator._trajectory_line_geometry(points, color)

    np.testing.assert_array_equal(
        vertices,
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(colors, np.repeat(color[None], 4, axis=0))


def test_aggregate_rows_includes_grasp_and_intervention_distribution() -> None:
    rows = [
        {
            "base": {"success": False, "grasped": False},
            "rlt": {"success": True, "grasped": True, "intervention_chunks": 1},
        },
        {
            "base": {"success": True, "grasped": True},
            "rlt": {"success": True, "grasped": True, "intervention_chunks": 0},
        },
        {
            "base": {"success": True, "grasped": True},
            "rlt": {"success": False, "grasped": False, "intervention_chunks": 2},
        },
    ]

    summary = evaluator._aggregate_rows(rows, bootstrap_seed=7)

    assert summary["base_successes"] == 2
    assert summary["rlt_successes"] == 2
    assert summary["base_grasped"] == 2
    assert summary["rlt_grasped"] == 2
    assert summary["base_fail_rlt_success"] == 1
    assert summary["base_success_rlt_fail"] == 1
    assert summary["intervention_distribution"]["0"]["count"] == 1
    assert summary["intervention_distribution"]["1"]["count"] == 1
    assert summary["intervention_distribution"]["2_plus"]["count"] == 1
    assert len(summary["paired_delta_bootstrap_95ci"]) == 2


def test_trajectory_geometry_handles_short_paths() -> None:
    vertices, colors = evaluator._trajectory_line_geometry(
        np.zeros((1, 3), dtype=np.float32),
        evaluator.BASE_TRAJECTORY_COLOR,
    )
    assert vertices.shape == (0, 3)
    assert colors.shape == (0, 4)


def test_overlay_draws_blue_base_and_orange_rl_without_gray() -> None:
    class FakeContext:
        def create_line_set(self, vertices, colors):
            return SimpleNamespace(vertices=vertices.copy(), colors=colors.copy())

    class FakeScene:
        def __init__(self):
            self.added = []
            self.removed = []

        def add_line_set(self, primitive):
            node = SimpleNamespace(primitive=primitive, line_width=None)
            self.added.append(node)
            return node

        def remove_node(self, node):
            self.removed.append(node)

    scene = FakeScene()
    viewer = SimpleNamespace(
        closed=False,
        render_scene=scene,
        renderer_context=FakeContext(),
    )
    overlay = evaluator._LivePairedTrajectoryOverlay(viewer, line_width=3.0)
    base = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    rlt = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)

    overlay.update(base, rlt)

    np.testing.assert_array_equal(
        overlay.base_node.primitive.colors[0], evaluator.BASE_TRAJECTORY_COLOR
    )
    np.testing.assert_array_equal(
        overlay.rlt_node.primitive.colors[0], evaluator.RLT_TRAJECTORY_COLOR
    )
    assert not np.array_equal(
        evaluator.BASE_TRAJECTORY_COLOR,
        np.asarray([0.72, 0.72, 0.72, 1.0], dtype=np.float32),
    )
    assert overlay.base_node.line_width == 5.0
    assert overlay.rlt_node.line_width == 3.0

    first_base = overlay.base_node
    first_rlt = overlay.rlt_node
    overlay.update(base, rlt)
    assert any(node is first_base for node in scene.removed)
    assert any(node is first_rlt for node in scene.removed)
    overlay.clear()
    assert overlay.base_node is None
    assert overlay.rlt_node is None


def test_q_gate_statuses_match_executed_policy() -> None:
    assert evaluator._gate_decision(
        q_advantage=0.09,
        min_q_advantage=0.10,
        interventions=0,
        max_interventions=1,
    ) == (False, "Q_REJECTED")
    assert evaluator._gate_decision(
        q_advantage=0.10,
        min_q_advantage=0.10,
        interventions=0,
        max_interventions=1,
    ) == (True, "EXECUTED")
    assert evaluator._gate_decision(
        q_advantage=1.0,
        min_q_advantage=0.10,
        interventions=1,
        max_interventions=1,
    ) == (False, "BUDGET_EXHAUSTED")
    assert evaluator._gate_decision(
        q_advantage=1.0,
        min_q_advantage=0.10,
        interventions=1,
        max_interventions=2,
        boundary_index=1,
        last_intervention_boundary=0,
        cooldown_chunks=1,
    ) == (False, "COOLDOWN")
    assert evaluator._gate_decision(
        q_advantage=1.0,
        min_q_advantage=0.10,
        interventions=1,
        max_interventions=2,
        boundary_index=2,
        last_intervention_boundary=0,
        cooldown_chunks=1,
    ) == (True, "EXECUTED")
