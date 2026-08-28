from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "pi0" / "serve.py"
_SPEC = importlib.util.spec_from_file_location("pi0_serve_identity", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SERVE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SERVE
_SPEC.loader.exec_module(_SERVE)


def _write_tree(root: Path, files: list[tuple[str, bytes]]) -> None:
    for relative_path, content in files:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_checkpoint_hash_is_deterministic_and_covers_assets_and_paths(
    tmp_path: Path,
) -> None:
    files = [
        ("params/z.bin", b"z-parameters"),
        ("params/a.bin", b"a-parameters"),
        ("assets/local/demo/norm_stats.json", b'{"mean": [1, 2]}'),
    ]
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_tree(left, files)
    _write_tree(right, list(reversed(files)))

    left_identity = _SERVE._hash_content_path(left)
    right_identity = _SERVE._hash_content_path(right)

    assert left_identity.sha256 == right_identity.sha256
    assert left_identity.file_count == 3
    assert left_identity.total_bytes == sum(len(content) for _, content in files)

    (right / "assets/local/demo/norm_stats.json").write_bytes(
        b'{"mean": [1, 3]}'
    )
    assert _SERVE._hash_content_path(right).sha256 != left_identity.sha256

    renamed = tmp_path / "renamed"
    _write_tree(
        renamed,
        [
            ("params/z.bin", b"z-parameters"),
            ("params/renamed.bin", b"a-parameters"),
            ("assets/local/demo/norm_stats.json", b'{"mean": [1, 2]}'),
        ],
    )
    assert _SERVE._hash_content_path(renamed).sha256 != left_identity.sha256


def test_external_norm_stats_adds_a_separate_content_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_tree(
        checkpoint,
        [
            ("params/model.bin", b"model"),
            ("assets/local/demo/norm_stats.json", b"inside"),
        ],
    )
    external = tmp_path / "external_norm_stats"
    _write_tree(external, [("norm_stats.json", b"outside")])

    checkpoint_identity = _SERVE._hash_content_path(checkpoint)
    external_identity = _SERVE._hash_content_path(external)
    metadata = _SERVE._content_identity_metadata(
        checkpoint_identity,
        external_identity,
    )

    assert metadata == {
        "resolved_checkpoint": str(checkpoint.resolve()),
        "checkpoint_content_sha256": checkpoint_identity.sha256,
        "norm_stats_content_sha256": external_identity.sha256,
    }
    assert _SERVE._is_within(
        checkpoint / "assets/local/demo/norm_stats.json",
        checkpoint,
    )
    assert not _SERVE._is_within(external, checkpoint)
    assert "norm_stats_content_sha256" not in _SERVE._content_identity_metadata(
        checkpoint_identity,
        None,
    )


def test_hash_fails_closed_when_tree_changes_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_tree(
        checkpoint,
        [("a.bin", b"a"), ("z.bin", b"z")],
    )
    original = _SERVE._hash_regular_file
    target = checkpoint / "z.bin"
    mutated = False

    def mutate_then_hash(path, expected, digest):
        nonlocal mutated
        if not mutated:
            target.unlink()
            mutated = True
        return original(path, expected, digest)

    monkeypatch.setattr(_SERVE, "_hash_regular_file", mutate_then_hash)
    with pytest.raises(RuntimeError, match="changed or became unreadable"):
        _SERVE._hash_content_path(checkpoint)


def test_identity_recheck_rejects_post_hash_change_and_nested_symlink(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _write_tree(checkpoint, [("params/model.bin", b"before")])
    identity = _SERVE._hash_content_path(checkpoint)

    (checkpoint / "params/model.bin").write_bytes(b"after")
    with pytest.raises(RuntimeError, match="changed after hashing"):
        _SERVE._assert_content_unchanged(identity)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (checkpoint / "linked.bin").symlink_to(outside)
    with pytest.raises(RuntimeError, match="refusing symlink or special entry"):
        _SERVE._hash_content_path(checkpoint)


def test_hash_rejects_missing_content_path(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        _SERVE._hash_content_path(tmp_path / "missing")
