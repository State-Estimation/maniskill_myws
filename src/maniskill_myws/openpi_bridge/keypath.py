from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_by_path(d: Mapping[str, Any], path: str, *, sep: str = "/") -> Any:
    """
    Read nested dict values with a slash-separated path.

    Example:
      get_by_path(obs, "observation/image")
    """
    cur: Any = d
    for part in path.split(sep):
        if not isinstance(cur, Mapping):
            raise KeyError(f"Path '{path}' failed at '{part}': not a mapping (got {type(cur)})")
        if part not in cur:
            raise KeyError(f"Path '{path}' missing key '{part}'. Available keys: {list(cur.keys())}")
        cur = cur[part]
    return cur


