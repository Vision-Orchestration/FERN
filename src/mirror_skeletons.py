"""
FERN v2 — DEPRECATED: left-right skeleton mirror (MediaPipe 33-joint).

This script is DEPRECATED. Use mirror_10joint.py instead which operates on
the project's standard 10-joint skeleton format.

Reason: mirror_10joint.py was used for all existing training data (confirmed
by 12-decimal-precision CSV output). Keeping both active creates a silent
data corruption risk when running the wrong script on the wrong joint format.

See mirror_10joint.py for the active mirror implementation.
"""

raise DeprecationWarning(
    "mirror_skeletons.py is deprecated. Use mirror_10joint.py instead."
)
