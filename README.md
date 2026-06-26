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

Implemented so far (milestone M0, the content-addressed substrate):

- A deterministic CBOR encoding profile for stable content-addressing.
- Self-describing content hashes (multihash) with a pluggable hash algorithm.
- Content-addressed node identity for the provenance graph.

## Development

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check . && mypy src && pytest
```

## License

Apache-2.0.
