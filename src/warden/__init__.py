"""Warden: a runtime trust layer for LLM agents.

Information-flow control and capability enforcement at the tool boundary, built on
a content-addressed provenance graph. See WARDEN_DESIGN_v0.2.txt (the RFC) and
WARDEN_ARCHITECTURE_v0.1.txt (the build spec).

The public surface is the Guard. Declare a capability policy, label external inputs
with ``Guard.source``, and wrap tools with ``@guard.tool``; the monitor mediates
every call and a denial raises ``WardenPolicyViolation``::

    guard = Guard("allow send_email if body.integrity == trusted")

    @guard.tool
    def send_email(body, recipient): ...
"""

from warden.harness import Recorder, Recording, Replayer, ReplayError
from warden.intercept import Guard, Handle
from warden.labels import Confidentiality, Label, Taint
from warden.monitor import WardenPolicyViolation
from warden.policy import ToolClass, compile_policy

__all__ = [
    "Confidentiality",
    "Guard",
    "Handle",
    "Label",
    "Recorder",
    "Recording",
    "ReplayError",
    "Replayer",
    "Taint",
    "ToolClass",
    "WardenPolicyViolation",
    "compile_policy",
]

__version__ = "0.0.0"
