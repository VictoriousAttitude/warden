"""The evaluation harness: utility cost vs. attack coverage (arch section 12 + 13.4).

Two layers share one metric contract (``GateReport``):

* ``warden.eval.gate`` -- the offline release gate. A scripted label corpus run
  straight through the Monitor, no agent and no I/O, so it runs in CI on every
  commit. It pins the two release-blocking numbers: false-positive rate (label
  creep) must be 0 and attack-block rate must be 1.
* ``warden.eval.echoleak`` -- a real tool-using agent run through the Guard and
  recorded on the provenance DAG (the M2 upgrade the gate always anticipated). It
  reports against the same ``GateReport``, now from an actual end-to-end flow.

Only the offline gate API is re-exported here; concrete scenarios live in their own
modules (``from warden.eval.echoleak import run``) so each suite stays self-contained.
"""

from warden.eval.gate import (
    GateReport,
    Scenario,
    ScenarioResult,
    Step,
    evaluate,
    run_scenario,
)

__all__ = [
    "GateReport",
    "Scenario",
    "ScenarioResult",
    "Step",
    "evaluate",
    "run_scenario",
]
