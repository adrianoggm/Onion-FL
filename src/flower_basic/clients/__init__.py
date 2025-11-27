"""Client entrypoints for different workflows."""

from .swell import main as run_swell_client
from .fog_bridge_swell import main as run_swell_bridge

__all__ = ["run_swell_client", "run_swell_bridge"]
