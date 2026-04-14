# Watchdog Layer — Observer-Agent Architecture

## The problem (first principles)

The orchestrator is **blind** to most of what happens in the system it runs.

A live experiment generates rich information across many streams — the agent log, device layer log, perception traces, event bus, embryo state files, timeline. We've made all of it accessible, but accessibility isn't comprehension. The orchestrator, with finite context and a job to do, can't be the one checking all of this on every turn — that's context rot, and it trades orchestration quality for surveillance duty.

Many problems, however, are obvious to a narrow observer. If an embryo's top focus failed calibration (low R²), that single signal says "recalibrate". If perception confidence has collapsed for one embryo while others are fine, that's a per-embryo imaging problem. These don't require the orchestrator's full reasoning — they need something narrower and cheaper that **watches, filters, and decides when to speak up**.

## The shape of the solution

A **watchdog layer** sits between the raw data streams and the orchestrator. Its job: consume noisy streams, distill them into observations the orchestrator can act on, and deliver them without polluting the orchestrator's context.

The orchestrator stays focused on orchestration. The watchdog stays focused on watching. They meet at a narrow, well-defined interface.

## Architecture: the observer agent

Two agents, two contexts. A newsroom model.

```
                     raw streams                    curated signal
 ┌──────────────────────────────┐      ┌─────────────┐      ┌──────────────┐
 │ channels (cheap pre-digest)  │ ───▶ │   Observer  │ ───▶ │ Orchestrator │
 │ progress / perception /      │      │ (Haiku-ish) │      │    (Opus)    │
 │ hardware / system health     │      │ own context │      │  user-facing │
 └──────────────────────────────┘      └─────────────┘      └──────────────┘
            ▲                                 │                     ▲
            │                                 ▼                     │
       event bus + logs          narrative state (bounded)     pull: get_system_status()
                                                               push: critical handoffs only
```

**Channels** (the reporters) are cheap, stateless digests of specific streams. Each one takes a window of raw data and produces a small structured observation.

**Observer** (the editor) is a small-model agent with its own conversation context. It receives channel outputs, maintains a running narrative of what's happening in the experiment, and decides what — if anything — to tell the orchestrator. Its context is bounded and rotating; it summarizes itself as it grows.

**Orchestrator** (the front page) gets one of two things from the observer:
- **Pull** — on demand, via `get_system_status()` tool: returns the observer's current narrative
- **Push** — rare, only for critical observations the orchestrator needs to know *now*

Most of the watchdog's writing never makes the orchestrator's context. That's correct.

## Why this beats direct injection

Direct injection (every observation goes into the orchestrator's history) fails for three compounding reasons over a long timelapse:

1. **Signal dilution** — user intent competes with 50+ watchdog messages
2. **Unbounded cost growth** — every observation lives in every future turn's input
3. **Stale context** — resolved observations sit in history forever

The observer absorbs the pollution. Its context is allowed to grow and rotate; the orchestrator's stays pristine. When a human reads the orchestrator's conversation later, they see a clean task thread, not a forensic log.

## Watchdog channels

Different information has different cadences and voices. The channels compose:

| Channel | Watches | Cadence | Sample observation |
|---|---|---|---|
| **Progress** | Round count, stage distribution, ETA | Per round | "T12: 8/10 alive, 3 bean, 2 hatched, 1 arrested" |
| **Perception** | VLM confidence, stage regressions, no_object streaks | Per round | "embryo_5: no_object 3 rounds" |
| **Hardware** | Calibration R², plan timing, device errors | Pre-flight + on-event | "embryo_5 top R²=0.0 — calibration failed" |
| **System health** | FileStore errors, VLM cache, mesh state | On burst | "FileStore failed 50× in 5 min" |

Each channel is a small, isolated component. Adding a new one (e.g., "mesh partner disconnected") is dropping in a watcher, not rewiring the system. Channels feed the observer — not the orchestrator directly.

## Severity and handoff

Observer outputs fall into three buckets:

| Severity | Delivery | Rationale |
|---|---|---|
| **info** | Stays in observer narrative, surfaced on pull | Orchestrator can ask, observer has the answer ready |
| **warning** | Stays in narrative; TUI toast | User sees it; orchestrator pulls if relevant |
| **critical** | Pushed as attributed message into orchestrator conversation | Needs attention now |

**Pull path** (always available):
```
orchestrator: get_system_status()
observer → returns:
  {
    "summary": "T12, 8/10 embryos active, 1 concerning (embryo_5).",
    "channels": {
      "progress": "...",
      "perception": "embryo_5 no_object 3 rounds, others nominal",
      "hardware": "all calibrations within threshold except embryo_5 top (R²=0.0)",
      "system_health": "nominal"
    },
    "open_concerns": [
      {"id": "obs_42", "severity": "warning", "about": "embryo_5", "recommend": "recenter or skip"}
    ]
  }
```

**Push path** (rare, critical only): observer synthesizes and injects a single attributed message at the next turn boundary. Format:
```
[watchdog/observer — critical] embryo_5 top focus failed calibration (R²=0.0)
and has now shown no_object for 3 rounds. Recommend stop_timelapse_embryo
or recalibrate.
```

## Observer's own context hygiene

The observer has a conversation too, and it must not rot. Policies:

- **Bounded history** — observer keeps last N channel digests verbatim; older material gets summarized into a narrative state it maintains
- **Self-summary turns** — periodically the observer emits "narrative so far: ..." replacing older detail
- **No raw log dumps** — channels pre-digest; observer never sees 10MB of log tail
- **Cheap model** — Haiku-class, maybe Sonnet for synthesis passes; cost stays linear and small

This means the observer can run for an 8-hour timelapse without unbounded growth. It's a different beast from the orchestrator because its job is different — it's a running state machine with narrative, not a task-focused reasoner.

## Feedback loop safety

When the orchestrator acts on an observation, the action itself shows up in the event bus and logs — which the observer also watches. Without care, this loops: observer emits → orchestrator stops embryo → observer sees "embryo stopped" → observer emits again.

The observer has a short-term cooldown keyed on entity (embryo_id) and action event type. When it sees STATUS_CHANGED or ACQUISITION_STOPPED for an embryo, it suppresses further observations about that embryo for a few minutes. Simple and local to the observer — orchestrator doesn't need to know.

## Critical-severity timing

The push path has to respect Anthropic's role alternation (tool_use ↔ tool_result balanced) — cannot inject mid-stream. Rules:

- Inject at turn boundaries only
- If orchestrator is idle and the observation is critical, trigger an **autonomous turn** (bridge streams with `user_message=None` and the critical handoff as the prompt-bearing message)
- Autonomous turns are rate-limited (1 per 10 minutes) and disabled in plan mode
- TUI notification fires immediately regardless

If you want to punt on autonomous turns for v1, the critical path waits for the user's next message and relies on the TUI for immediacy. That's acceptable.

## Persistence

Observer narrative state and pending critical handoffs survive session resume via a session-dir file (e.g., `observer_state.yaml`). On resume, the observer is re-hydrated so it picks up where it left off. Channels are stateless so they need no persistence.

## Why this is the first loop-closure step

The orchestrator currently acts only on what the **user** tells it and what its **tools** return. The observer gives it peripheral vision — not by making it look everywhere, but by giving the system a voice that speaks in its language. Once this interface exists, further closures become tractable:

- **Learning feedback** — "this intervention worked, this didn't" flows through the same observer
- **Cross-session patterns** — a campaign-level channel says "this nickname usually regresses at T9"
- **User-elevated concerns** — the user tags something, channel/observer tracks it

Keep the observer interface small and opinionated. It will get reused.

---

## Concrete grounding — session 2f4472c3 replay

What the layer would produce on a real run:

| Observation | Channel → Observer → Orchestrator |
|---|---|
| embryo_5 top R²=0.0 | Hardware → observer (warning) → pull reveals + TUI toast |
| embryo_6 bottom R²=0.46 | Hardware → observer (warning) → pull reveals |
| embryo_5 no_object by T3 | Perception → observer (promotes to **critical** because hardware also flagged this embryo) → push |
| embryo_2 bean→early at T9 | Perception → observer (warning) → pull |
| FileStore write failures (498×) | System health → observer (**critical**) → push |
| Round duration drifting | Hardware → observer (info) → narrative |

Notice the observer can *correlate* across channels (embryo_5 has both a calibration issue and a perception issue → promoted severity). A flat detector can't do this; the observer's context is exactly where that synthesis should live.

---

## Critical files (implementation targets)

- **NEW** `gently/app/orchestration/watchdog/channels.py` — channel implementations (progress, perception, hardware, system health)
- **NEW** `gently/app/orchestration/watchdog/observer.py` — observer agent loop, narrative state, pull/push API
- **NEW** `gently/app/tools/watchdog_tools.py` — `get_system_status()` tool for orchestrator
- **MODIFY** `gently/app/orchestration/timelapse.py` — spawn observer alongside timelapse; pass event bus + session dir
- **MODIFY** `gently/app/agent.py` — register watchdog_tools; wire observer push path to conversation drain
- **MODIFY** `gently/harness/conversation.py` — `inject_critical_handoff(source, message)` (small, targeted — just critical path)
- **MODIFY** `gently/harness/bridge.py` — support `user_message=None` for autonomous turns (critical-severity only)
- **MODIFY** `gently/core/file_store.py` — persist/load observer state
- **MODIFY** `gently/settings.py` — `WatchdogSettings` (channels enabled, observer model, cadence, rate limits)

## Existing utilities to reuse

- `gently_perception.Perceiver` pattern for context accumulation (observer is similar: long-running, accumulating context)
- `EventBus` subscription pattern already used by orchestrator
- `FileStore` YAML read/write (now sanitized, safe)
- Prompt caching pattern already used in `bridge.py` (apply to observer's system prompt)
- Tool-result attribution pattern (observer's critical push mirrors tool_result formatting)

## Verification

1. **Replay test** — feed session 2f4472c3 events/logs/traces into the channels in dry-run, confirm six concrete observations fire correctly and the observer correlates embryo_5 cross-channel into a critical push
2. **Observer context bound** — run an 8-hour simulation; confirm observer's context does not grow past its cap
3. **Feedback loop** — emit an observation, trigger action event, confirm observer suppresses re-emission during cooldown
4. **Pull path** — orchestrator calls `get_system_status()` mid-experiment; confirm narrative is current and well-formed
5. **Push path** — synthesize a critical observation while orchestrator is idle; confirm autonomous turn fires (or, if v1 without autonomous turns, confirm TUI toast + queued handoff for next turn)
6. **Persistence** — stop observer mid-experiment, restart, confirm narrative resumes
7. **Live run** — small timelapse with channels enabled; verify orchestrator's conversation stays clean (no push except on genuine critical) and pull returns useful state

## Open design questions

1. **Autonomous turns in v1?** Or is TUI-plus-waiting-for-user acceptable for first pass?
2. **Observer model** — Haiku for throughput, Sonnet for synthesis passes, or just one?
3. **`modify_timelapse_interval` tool gap** — referenced in tool descriptions but not registered. Some recommendations ("slow everything down") need this. Prerequisite or documented limitation?
4. **Channel extensibility** — should channels be plugin-shaped (class registration) or hard-coded for v1?
5. **Observer narrative format** — freeform markdown or structured schema? (Freeform is easier for synthesis, structured is easier for pull API.)
