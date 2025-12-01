"""Client entrypoints for different workflows."""

from .fog_bridge_swell import main as run_swell_bridge
from .swell import main as run_swell_client

__all__ = ["run_swell_client", "run_swell_bridge"]
