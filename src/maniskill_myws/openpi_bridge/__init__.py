"""
ManiSkill <-> openpi bridge utilities.

This package intentionally contains ONLY glue code:
- read nested obs dicts produced by ManiSkill
- convert obs to openpi policy input dict
- convert openpi action chunks to step-wise ManiSkill actions
"""

from .keypath import get_by_path as get_by_path
from .obs_to_openpi import ObsAdapter as ObsAdapter
from .remote_policy import RemoteWebsocketChunkPolicy as RemoteWebsocketChunkPolicy


