"""Make the ``black_hole`` package importable straight from a source checkout.

The distribution maps ``black_hole`` onto ``src/``, so that name only resolves
once the project is installed.  The root wrapper scripts import this module
first to bind it to the local ``src/`` directory instead.
"""

import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"


def install():
    """Bind ``black_hole`` to ./src, unless it is already imported."""
    init = _SRC / "__init__.py"
    if "black_hole" in sys.modules or not init.exists():
        return
    spec = importlib.util.spec_from_file_location(
        "black_hole", init, submodule_search_locations=[str(_SRC)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["black_hole"] = module
    spec.loader.exec_module(module)


install()
