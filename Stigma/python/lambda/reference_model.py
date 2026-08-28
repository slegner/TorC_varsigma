"""
Loader for the reference (lam = 0) model, ``python/equations_Hz.py``.

That file starts with the IPython magic ``%matplotlib inline`` and imports
matplotlib/camb at the top even though none of its functions use them, so it
cannot simply be imported from a plain script.  This module execs it with those
lines stripped, leaving the user's file untouched.

    from reference_model import ref
    z, hubble, P_DE, rho_DE, asol = ref.rhopsol(cpr, omega_m, h, OmL, stigma, mups2)
"""

import importlib.util
import os
import re
import sys

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "python", "equations_Hz.py")

# IPython magics, and top-level imports that equations_Hz.py never actually uses
# (they only exist because the file doubles as a notebook cell).
_SKIP = re.compile(r"^\s*(%|import\s+matplotlib|import\s+camb\b|from\s+matplotlib)")


def load_reference(path=REF_PATH, name="equations_Hz_ref"):
    """Import python/equations_Hz.py as a module, dropping notebook-only lines."""
    with open(path) as f:
        src = "".join(line for line in f if not _SKIP.match(line))
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = os.path.abspath(path)
    sys.modules[name] = module
    exec(compile(src, os.path.abspath(path), "exec"), module.__dict__)
    return module


ref = load_reference()
