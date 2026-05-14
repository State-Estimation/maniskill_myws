#!/usr/bin/env python
"""Launch PLD residual SAC training from a small YAML config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML is required. Install it or use the raw command line.") from e
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a YAML mapping in {path}")
    return data


def _arg_tokens(args: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in args.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                tokens.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            tokens.append(flag)
            tokens.extend(str(v) for v in value)
            continue
        tokens.extend([flag, str(value)])
    return tokens


def _cuda_index(value: Any) -> int | None:
    text = str(value)
    if text == "cuda":
        return 0
    if not text.startswith("cuda:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except ValueError:
        return None


def _validate_visible_cuda(train_args: dict[str, Any], env: dict[str, str]) -> None:
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return
    visible_count = len([part for part in visible.split(",") if part.strip()])
    if visible_count <= 0:
        return
    for key in ("device", "env-device", "env_device"):
        if key not in train_args:
            continue
        idx = _cuda_index(train_args[key])
        if idx is not None and idx >= visible_count:
            raise SystemExit(
                f"{key}={train_args[key]} is invalid with CUDA_VISIBLE_DEVICES={visible!r}. "
                f"Use cuda:0 for the first visible GPU."
            )


def _display_command(cmd: list[str], extra_env: dict[str, Any] | None) -> str:
    env_parts: list[str] = []
    if extra_env:
        env_parts = [f"{key}={shlex.quote(str(value))}" for key, value in extra_env.items()]
    return " ".join([*env_parts, *(shlex.quote(part) for part in cmd)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    cfg = _load_yaml(config_path)

    script = cfg.get("script", "scripts/pld/train_residual_sac.py")
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = Path.cwd() / script_path

    train_args = cfg.get("args", {})
    if not isinstance(train_args, dict):
        raise SystemExit("'args' must be a YAML mapping")

    env = os.environ.copy()
    extra_env = cfg.get("env", {})
    if extra_env is not None:
        if not isinstance(extra_env, dict):
            raise SystemExit("'env' must be a YAML mapping")
        env.update({str(k): str(v) for k, v in extra_env.items()})
    _validate_visible_cuda(train_args, env)

    cmd = [sys.executable, str(script_path), *_arg_tokens(train_args)]
    print(_display_command(cmd, extra_env), flush=True)
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
