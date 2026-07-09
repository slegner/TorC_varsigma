"""Back-compat shim: the scipy reference equations now live in
torc/scipy_reference.py.  Kept so tutorial/ and legacy/ imports work."""
from torc.scipy_reference import *                           # noqa: F401,F403
