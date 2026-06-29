# Security Policy

Warden is a security tool, and is itself pre-alpha. It does not yet carry any
production security guarantee. Please do not rely on it as the sole control on a
high-value system.

## Reporting a vulnerability

Please report security issues **privately**, not as public issues or pull
requests. Use GitHub's private vulnerability reporting on this repository:
**Security → Advisories → Report a vulnerability**
(<https://github.com/VictoriousAttitude/warden/security/advisories/new>).

Include enough to reproduce: the policy, the tool wiring, and the flow that was
allowed or denied unexpectedly. We aim to acknowledge a report within a few days
and will coordinate disclosure with you before publishing a fix.

## In scope

Defects in the trust boundary itself:

- a consequential sink that executes without mediation (a complete-mediation
  bypass, INV-4);
- a label that is dropped or weakened along a derivation (a monotonicity
  violation, INV-3);
- a denial that fails open instead of closed (INV-5);
- non-determinism that breaks replay or content identity (INV-1, INV-7);
- policy-DSL parsing or evaluation that admits a flow the policy forbids.

## Out of scope

These are documented limitations, not vulnerabilities (see the README, "What
Warden does *not* defend"):

- weaknesses that follow from an incorrect or overly permissive user policy;
- tools deliberately wired around the Guard, or buggy tool implementations;
- structural laundering through opaque container arguments, and semantic
  laundering by a free-typing model (findings F4/F5) — known residuals on the
  M3 roadmap;
- issues in third-party dependencies (e.g. the optional `agentdojo` extra);
  report those upstream.
