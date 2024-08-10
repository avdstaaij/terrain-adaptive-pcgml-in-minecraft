"""
Amulet depends on PyMCTranslate, which, for *some* reason, calls basicConfig() on the root logger
and then logs something, *on import*.

Importing amulet via this wrapper suppresses that log message and undoes the basicConfig() call.
"""

from contextlib import redirect_stderr
import logging

with redirect_stderr(None):
    import amulet


rootLogger = logging.getLogger()
for handler in rootLogger.handlers:
    rootLogger.removeHandler(handler)
