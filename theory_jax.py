"""Back-compat shim: the canonical theory now lives in torc/theory.py
(noH0 anchoring).  Kept so tutorial/ and legacy/ imports keep working."""
from torc.theory import *                                    # noqa: F401,F403
from torc.theory import (_TERM_TAU, _TERM_LNA, _SOLVER,      # noqa: F401
                         _CTRL, TAU_SPAN)
