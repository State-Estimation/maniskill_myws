"""
maniskill_myws: a ManiSkill local-source workspace for custom tasks + imitation learning.

Usage (after `pip install -e .`):

```python
import maniskill_myws
maniskill_myws.register()
```
"""

from __future__ import annotations


def register() -> None:
    """Import task modules to trigger ManiSkill's @register_env decorators."""
    from .tasks import turn_globe_valve as _turn_globe_valve  # noqa: F401
    from .tasks import open_safe_door as _open_safe_door  # noqa: F401


