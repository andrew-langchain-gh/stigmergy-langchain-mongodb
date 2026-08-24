# Shared memory as a coordination mechanism

Demo for a talk about agent memory. Eight agents work one production incident. They never
call each other. All coordination happens through writes to a shared MongoDB blackboard,
in two distinct styles:

**Phase 1 — stigmergy.** Four investigators leave traces: observations, hypotheses and
open questions. Nobody assigns work. An open question gets picked up by whichever agent's
domain matches, purely because the trace is sitting in shared memory.

**Phase 2 — negotiation by editing.** Four agents make typed structural edits to one
shared document. Disagreement shows up as data — objections carrying a severity and a
blocking flag — rather than as dialogue.

Both phases share one substrate and one document. A single `_id`'s revision history runs
from bare trigger → corroborated hypothesis → negotiated plan.

## The claim, stated carefully

The coordination layer is framework-agnostic; you could build it with raw `pymongo`.
LangGraph and the LangChain agent harness earn their place by making each *individual*
agent robust enough to trust with it: durable execution, semantic memory access, tracing.
Nothing here claims the framework invented the blackboard pattern.

## Setup

```bash
docker compose up -d          # local Atlas: replica set + Atlas Vector Search
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY
uv sync
uv run swarm seed             # load INC-1042 evidence, build the vector index
uv run swarm doctor           # verify everything before an audience is watching
```

`mongodb/mongodb-atlas-local` is used because the demo needs two MongoDB features that a
plain `mongo` image does not offer together: change streams (which require a replica set)
and Atlas Vector Search (which requires `mongot`). The host port is **27018**, to avoid
colliding with a mongo you may already be running on 27017.

## Running it

Three terminals.

```bash
uv run swarm dashboard        # 1 — the audience-facing view
uv run swarm run              # 2 — supervises 8 agent processes + the watchdog
uv run swarm trigger          # 3 — one insert; the only thing that starts anything
```

Presenter controls, from terminal 3:

```bash
uv run swarm status           # swarm state, read straight out of MongoDB
uv run swarm show             # the focal document
uv run swarm timeline         # what happened, in order, with T+ offsets
uv run swarm kill metrics     # real SIGKILL to a real pid
uv run swarm start metrics    # rejoins, replays what it missed, finishes pending work
uv run swarm procedures       # private per-agent memory, separate namespace
uv run swarm reset            # clear the board; evidence stays loaded
```

To export the focal document's revision history as files — one per write, for slides — start
`swarm capture` **before** the trigger. Like the dashboard it is only a change-stream
subscriber, so it can only export writes it witnessed:

```bash
uv run swarm capture --out docs/document-evolution/run-d
```

### Stepping through it

Eight agents converge in about a minute, which is faster than you can narrate. Step mode
puts you in control: agents park before each turn until you release one.

```bash
uv run swarm run --step       # start paused
uv run swarm next             # release one agent turn
uv run swarm next 3           # release three
uv run swarm hold             # pause again mid-run
uv run swarm free             # let it run at full speed
```

Or drive it from the dashboard, which also scrubs the document's revision history:

| Key | Does |
|---|---|
| `n` / `space` | release one agent turn |
| `1`–`9` | release that many turns |
| `h` / `r` | hold / run free |
| `↑` `↓` / `←` `→` | step back/forward one document revision |
| `PgUp` `PgDn` | jump 10 revisions |
| `Home` | oldest revision |
| `=` / `End` | back to the live document |
| `q` | quit |

The gate is a token in MongoDB claimed with a conditional write, so "next" releases
exactly one turn no matter how many agents are waiting — the database picks the winner.
Revision history is reconstructed from change-stream post-images, so the dashboard still
writes nothing; start it before the swarm, since it can only show writes it witnessed.

Pacing matters: eight agents on change streams converge faster than anyone can narrate.
`SWARM_PACE_MS` (default 2500) throttles how quickly an agent acts on a wake-up.

## Layout

| Path | What it is |
|---|---|
| `src/swarm/blackboard.py` | Typed writes and **both convergence rules** as atomic conditional writes |
| `src/swarm/watch.py` | Change streams, resume tokens, self-write filtering |
| `src/swarm/agent_base.py` | Per-agent runtime: `create_agent` + `MongoDBSaver`, `durability="sync"` |
| `src/swarm/tools/phase1.py` | Stigmergic traces. Note there is no `promote` and no `assign` tool |
| `src/swarm/tools/phase2.py` | Role-scoped typed edits — the permission model |
| `src/swarm/watchdog.py` | Exogenous trigger and both escape valves |
| `src/swarm/tui.py` | Dashboard. A change-stream subscriber with no IPC to any agent |
| `src/swarm/capture.py` | Exports every version of the focal document to disk |
| `src/swarm/revisions.py` | What one write did to the document — shared by the dashboard and capture |
| `data/inc-1042/` | The fixed evidence. Reasoning over it stays live |
| `docs/demo-beats.md` | Beat-by-beat run sheet for the live portion |
| `docs/document-evolution/` | Captured revision histories + slide notes for the deck |

## Tests

```bash
uv run pytest
```

These cover the claims the talk makes out loud: that three distinct evidence types are
required, that an open question blocks promotion, that a blocking objection is scoped to
its own option rather than the whole document, and that when four agents cross the
threshold simultaneously exactly one wins.
