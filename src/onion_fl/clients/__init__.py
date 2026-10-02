"""Client entrypoints for different workflows."""

from .fog_bridge_sweet import main as run_sweet_bridge
from .fog_bridge_swell import main as run_swell_bridge
from .sweet import main as run_sweet_client
from .swell import main as run_swell_client

__all__ = [
    "run_swell_client",
    "run_swell_bridge",
    "run_sweet_client",
    "run_sweet_bridge",
]
