from __future__ import annotations

"""Deprecated alias of ``onion_fl.continual`` (M1, #193), kept for one version.

``import onion_fl.continuum.memory`` gives the very module
``onion_fl.continual.memory``, so plugin references such as
``onion_fl.continuum.memory:Reservoir`` still resolve to the same classes.
"""

import importlib
import pkgutil
import sys
import warnings

import onion_fl.continual as _continual

warnings.warn(
    "onion_fl.continuum is now onion_fl.continual; the old name goes in a later "
    "version",
    DeprecationWarning,
    stacklevel=2,
)

for _module in pkgutil.iter_modules(_continual.__path__):
    _loaded = importlib.import_module(f"{_continual.__name__}.{_module.name}")
    sys.modules[f"{__name__}.{_module.name}"] = _loaded
    globals()[_module.name] = _loaded
