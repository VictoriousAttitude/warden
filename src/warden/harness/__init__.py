"""The Harness: record / replay / fork / diff over the content provenance DAG.

The Harness is the substrate that turns a run into a reproducible artifact: it
records a live execution as immutable content nodes plus a replay cassette, replays
it deterministically against that cassette, and (in a later slice) forks and
counterfactually perturbs it. It depends only on ``warden.core`` -- never on the
Guard (INV-9) -- so the Guard ships standalone and the Harness records CONTENT
(Layer A) while labels stay the Guard's derived overlay (Layer B). See
WARDEN_ARCHITECTURE_v0.1.txt section 10.
"""

from warden.harness.record import BoundaryEvent, Recorder, Recording
from warden.harness.replay import (
    BoundaryOracle,
    CounterfactualReplayer,
    Replayer,
    ReplayError,
)

__all__ = [
    "BoundaryEvent",
    "BoundaryOracle",
    "CounterfactualReplayer",
    "Recorder",
    "Recording",
    "ReplayError",
    "Replayer",
]
