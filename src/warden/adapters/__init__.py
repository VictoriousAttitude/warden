"""Framework adapters: drop-in attach points for the agents people already run.

This is the layer INV-9 reserves above the interception shim (``adapters <-
intercept``). Each adapter wires a real agent framework's tool-execution step
through the Guard, so an existing graph or pipeline gains complete mediation and
the dual-plane masking defense without rewriting its tools.

Adapters are heavyweight and framework-specific, so none is imported at package
import: an adapter pulls its framework only when its own module is imported (the
convention the ``eval`` package follows). Import the one you need directly, e.g.
``from warden.adapters.langgraph import WardenToolNode``; nothing here re-exports
it to the top level.
"""
