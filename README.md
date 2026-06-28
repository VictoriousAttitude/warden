# Warden

A runtime trust layer for LLM agents: **information-flow control** and
**capability enforcement** at the tool boundary, built on a content-addressed
provenance graph that also yields deterministic replay.

Prompt injection is a *flow* problem, not a detection problem. Scanners that try
to spot "malicious" prompts are bypassable. Warden takes the principled route from
classical security engineering: label every value with integrity and
confidentiality, propagate those labels along the agent's data-flow graph, and
enforce a capability policy at a single reference monitor that mediates every
consequential action — fail-closed, with an explainable provenance path for every
denial.

Warden is framework-agnostic and local-first. It is not a model, not a scanner,
and does not claim to "solve" prompt injection; it bounds the blast radius and
makes exploitation provably hard given correct policies.

## Status

Pre-alpha. The design is in `WARDEN_DESIGN_v0.2.txt` (the RFC) and
`WARDEN_ARCHITECTURE_v0.1.txt` (the engineering build spec).

Implemented so far:

- **Core (M0)** — a content-addressed provenance graph: a deterministic CBOR
  encoding profile, self-describing multihash content ids with a pluggable hash
  algorithm, node identity, an object store, and run-level fork/diff.
- **The Guard (M1)** — the label algebra (integrity + confidentiality on the
  Denning lattice), taint propagation, a small capability-policy DSL, and a
  reference monitor enforcing complete mediation, fail-closed, with an explainable
  provenance path on every denial. Mode 2 in-process interception via a
  `@guard.tool` decorator, plus a static bypass-lint.
- **The Harness (M2)** — record, deterministic replay over a logical-sequence
  boundary scheduler, and counterfactual injection on the provenance graph.
- **Evaluation** — an offline release gate, the EchoLeak exfiltration scenario,
  and an AgentDojo integration (a vendored workspace suite plus an adapter that
  mediates the real benchmark's tool boundary, with record/replay for hermetic
  runs).

## Development

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check . && mypy src && pytest
```

## License

Apache-2.0.
