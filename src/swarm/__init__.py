"""Shared-memory agent coordination on a MongoDB blackboard.

Two coordination patterns over one substrate:

* Phase 1 — stigmergy: agents leave observations, hypotheses and open questions.
  Coordination emerges from the traces. Nobody assigns work.
* Phase 2 — negotiation-by-editing: agents make typed structural edits to one
  shared document. Disagreement is data, not dialogue.

There is no orchestrator anywhere in this package. Phase transitions happen via
atomic conditional writes: the database's consistency guarantee does the job you
would normally reach for a coordinator to do.
"""
