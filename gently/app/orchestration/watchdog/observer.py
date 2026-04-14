"""Observer agent — the editor of the watchdog newsroom.

Consumes channel digests, maintains a bounded, rotating narrative of the
experiment, and decides what — if anything — the orchestrator needs to know.
Exposes two paths to the orchestrator:

* **Pull** via :meth:`Observer.get_status` — returns the current narrative and
  open concerns. The ``get_system_status`` tool wraps this.
* **Push** via :attr:`Observer.on_critical_push` — a callback invoked for each
  critical handoff. In production this is wired to
  :meth:`ConversationManager.inject_critical_handoff`.

The observer is deliberately frugal with the LLM:

* Uses Haiku (settings.watchdog.model or settings.models.fast)
* Only calls the LLM when there are new or changed observations
* Passes its own short narrative back to itself on each turn — there is no
  growing conversation history beyond that narrative
* Applies dedup and per-embryo action cooldowns before anything reaches the
  LLM, so the LLM's input is bounded
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from gently.core.event_bus import EventBus, EventType, get_event_bus
from gently.settings import settings

from .channels import (
    Channel,
    Observation,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    build_default_channels,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------

OBSERVER_SYSTEM_PROMPT = """You are the WATCHDOG OBSERVER for a live C. elegans microscopy timelapse.

You watch structured observations streaming in from four channels:
- PROGRESS: round count, stage distribution, elapsed time
- PERCEPTION: VLM stage calls, regressions, no_object streaks, arrest
- HARDWARE: calibration quality, device errors
- SYSTEM_HEALTH: file store / VLM / error bursts

Your job is NOT to act. You maintain a short, evolving narrative of what is
happening, and you decide which — if any — observations need the orchestrator's
immediate attention.

You speak in two voices:

1. NARRATIVE — a concise markdown paragraph (≤ 600 characters) describing the
   current state of the experiment. You rewrite this fresh each turn, folding
   new observations into it. Do not accumulate — trim aggressively. This is
   the orchestrator's on-demand view of the lab.

2. CRITICAL_PUSH — used only when an observation genuinely needs the
   orchestrator to act now and cannot wait to be pulled. Examples: an embryo
   has failed calibration AND is showing no_object (cross-channel correlation),
   all acquisitions are failing, FileStore is dead. Be conservative — most
   observations do not warrant a push.

Respond in this exact structured format:

NARRATIVE:
<one-paragraph markdown>

OPEN_CONCERNS:
- <short line per unresolved concern, include embryo id or subject>
(or "none" if there are no open concerns)

CRITICAL_PUSH:
<one-line critical message to the orchestrator, max 280 chars — or "none">

Rules:
- CRITICAL_PUSH fires only when severity is "critical" AND the observation is
  novel (not a restatement of something already in the narrative).
- Never invent data. If a channel emits nothing, the narrative should reflect
  calm.
- Do not repeat the raw observations verbatim — synthesize.
- Do not recommend global interval changes (the orchestrator has no tool for
  that). Recommend per-embryo actions: stop_timelapse_embryo, skip_embryo,
  calibrate_embryo, modify_parameters.
"""


# ----------------------------------------------------------------------
# State containers
# ----------------------------------------------------------------------


class _ObserverState:
    """In-memory state of the observer — serializable for persistence."""

    def __init__(self) -> None:
        self.narrative: str = "Experiment just started; no observations yet."
        self.open_concerns: List[str] = []
        # key -> last_seen_timestamp
        self.recent_keys: Dict[str, float] = {}
        # embryo_id -> cooldown_until_timestamp
        self.action_cooldowns: Dict[str, float] = {}
        # pending critical handoffs that haven't drained into the orchestrator yet
        self.pending_handoffs: List[Dict[str, Any]] = []
        # timestamp of last critical push (for rate limit)
        self.last_critical_push_at: Optional[float] = None
        # digest history (for debugging / replay)
        self.tick_count: int = 0
        # timestamp of observer start
        self.started_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative,
            "open_concerns": list(self.open_concerns),
            "recent_keys": dict(self.recent_keys),
            "action_cooldowns": dict(self.action_cooldowns),
            "pending_handoffs": list(self.pending_handoffs),
            "last_critical_push_at": self.last_critical_push_at,
            "tick_count": self.tick_count,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "_ObserverState":
        state = cls()
        if not data:
            return state
        state.narrative = str(data.get("narrative") or state.narrative)
        state.open_concerns = list(data.get("open_concerns") or [])
        state.recent_keys = {str(k): float(v) for k, v in (data.get("recent_keys") or {}).items()}
        state.action_cooldowns = {
            str(k): float(v) for k, v in (data.get("action_cooldowns") or {}).items()
        }
        state.pending_handoffs = list(data.get("pending_handoffs") or [])
        last = data.get("last_critical_push_at")
        state.last_critical_push_at = float(last) if last is not None else None
        state.tick_count = int(data.get("tick_count") or 0)
        state.started_at = data.get("started_at")
        return state


# ----------------------------------------------------------------------
# Observer
# ----------------------------------------------------------------------


class Observer:
    """Background watchdog observer — own context, own conversation with Haiku.

    Parameters
    ----------
    claude_client
        The shared ``anthropic.Anthropic`` client (reused from the agent).
    model
        Model id. Defaults to ``settings.watchdog.model`` or
        ``settings.models.fast``.
    channels
        List of channels to digest. If ``None``, the default set is built via
        :func:`build_default_channels`.
    store
        ``FileStore`` instance — used for persisting observer state. May be
        ``None`` for headless tests.
    session_id
        Session id under which state is persisted.
    on_critical_push
        Callback invoked for each critical handoff. Signature:
        ``(handoff: dict) -> None``. The production wire-up passes
        ``ConversationManager.inject_critical_handoff``.
    on_notification
        Callback invoked for each observation that should surface in the TUI.
        Signature: ``(event: dict) -> Awaitable``. Optional.
    event_bus
        Optional explicit event bus (for tests).
    tick_interval_s
        How often to digest channels. Defaults to
        ``settings.watchdog.tick_interval_s``.
    """

    def __init__(
        self,
        claude_client,
        *,
        model: Optional[str] = None,
        channels: Optional[List[Channel]] = None,
        store=None,
        session_id: Optional[str] = None,
        on_critical_push: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_notification: Optional[Callable[[Dict[str, Any]], Any]] = None,
        event_bus: Optional[EventBus] = None,
        tick_interval_s: Optional[float] = None,
        embryo_ids: Optional[List[str]] = None,
    ) -> None:
        self._claude = claude_client
        self._model = (
            model
            or settings.watchdog.model
            or settings.models.fast
        )
        self._store = store
        self._session_id = session_id
        self._bus = event_bus or get_event_bus()
        self._tick_interval_s = (
            tick_interval_s
            if tick_interval_s is not None
            else settings.watchdog.tick_interval_s
        )
        self._on_critical_push = on_critical_push
        self._on_notification = on_notification

        # Channels: use provided or build the default set
        if channels is not None:
            self._channels = channels
        else:
            self._channels = build_default_channels(
                store=store,
                session_id=session_id or "",
                embryo_ids=list(embryo_ids or []),
                event_bus=self._bus,
            )

        self._state = _ObserverState()
        self._task: Optional[asyncio.Task] = None
        self._stop_requested = False
        self._action_event_unsubs: List[Callable[[], None]] = []

    # ---- lifecycle ----

    async def start(self) -> None:
        """Hydrate state, start channels, kick the tick loop."""
        if self._state.started_at is None:
            self._state.started_at = datetime.now().isoformat()
        self._load_state()
        for ch in self._channels:
            ch.start()
        # Subscribe to action events for cooldown tracking
        for et in (
            EventType.ACQUISITION_STOPPED,
            EventType.EMBRYO_SKIPPED,
            EventType.EMBRYO_CENTERED,
            EventType.EMBRYO_CALIBRATED,
            EventType.STATUS_CHANGED,
        ):
            unsub = self._bus.subscribe(et, self._on_action_event)
            self._action_event_unsubs.append(unsub)
        logger.info(
            "watchdog observer started (model=%s, interval=%ss, channels=%s)",
            self._model,
            self._tick_interval_s,
            [c.name for c in self._channels],
        )
        self._stop_requested = False
        self._task = asyncio.create_task(self._run_loop(), name="watchdog-observer")

    async def stop(self) -> None:
        self._stop_requested = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("watchdog observer: loop raised on stop")
            self._task = None
        for unsub in self._action_event_unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._action_event_unsubs.clear()
        for ch in self._channels:
            try:
                ch.stop()
            except Exception:
                logger.debug("channel %s: stop failed", ch.name, exc_info=True)
        self._save_state()
        logger.info("watchdog observer stopped")

    # ---- pull API ----

    def get_status(self) -> Dict[str, Any]:
        """Return the current narrative and open concerns. Used by the
        ``get_system_status`` tool.
        """
        return {
            "narrative": self._state.narrative,
            "open_concerns": list(self._state.open_concerns),
            "last_updated": datetime.now().isoformat(),
            "tick_count": self._state.tick_count,
            "pending_handoffs": len(self._state.pending_handoffs),
            "channels": [c.name for c in self._channels],
            "session_id": self._session_id,
        }

    # ---- main loop ----

    async def _run_loop(self) -> None:
        # One immediate tick so the pre-flight calibration flags surface early
        try:
            await self._tick()
        except Exception:
            logger.exception("watchdog observer: first tick failed")

        while not self._stop_requested:
            try:
                await asyncio.sleep(self._tick_interval_s)
                if self._stop_requested:
                    break
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("watchdog observer: tick failed; continuing")

    async def _tick(self) -> None:
        self._state.tick_count += 1
        # 1. Collect observations from every channel
        all_obs: List[Observation] = []
        for ch in self._channels:
            try:
                all_obs.extend(ch.digest())
            except Exception:
                logger.exception("watchdog observer: channel %s.digest failed", ch.name)

        # 2. Apply cooldown (per-embryo) and dedup (per-key)
        now = time.time()
        self._prune_expired(now)
        filtered: List[Observation] = []
        for obs in all_obs:
            if self._is_in_action_cooldown(obs, now):
                continue
            if self._is_deduped(obs, now):
                continue
            filtered.append(obs)
            # Mark the key as seen
            self._state.recent_keys[obs.key] = now

        # 3. Surface all observations to the TUI (optional push to notification sink)
        if self._on_notification:
            for obs in filtered:
                try:
                    await _maybe_await(self._on_notification({
                        "type": "watchdog_observation",
                        "observation": obs.to_dict(),
                    }))
                except Exception:
                    logger.debug("notification callback failed", exc_info=True)

        # 4. If nothing filtered through, keep narrative but skip LLM
        if not filtered and self._state.tick_count > 1:
            self._save_state()
            return

        # 5. Call Haiku to fold observations into narrative + decide on push
        try:
            synthesis = await self._synthesize(filtered)
        except Exception:
            logger.exception("watchdog observer: synthesis failed; preserving prior narrative")
            self._save_state()
            return

        # 6. Apply synthesis
        if synthesis.get("narrative"):
            self._state.narrative = synthesis["narrative"][:2000]
        if "open_concerns" in synthesis:
            self._state.open_concerns = synthesis["open_concerns"][:20]

        # 7. Critical push?
        critical = synthesis.get("critical_push")
        if critical and self._can_critical_push(now, filtered):
            handoff = {
                "source": "watchdog_observer",
                "severity": SEVERITY_CRITICAL,
                "message": critical,
                "timestamp": datetime.now().isoformat(),
                "tick": self._state.tick_count,
            }
            self._state.pending_handoffs.append(handoff)
            self._state.last_critical_push_at = now
            try:
                if self._on_critical_push:
                    self._on_critical_push(handoff)
            except Exception:
                logger.exception("watchdog observer: critical push callback failed")

        self._save_state()

    # ---- synthesis (LLM call) ----

    async def _synthesize(self, observations: List[Observation]) -> Dict[str, Any]:
        """Call Haiku with the current narrative + new observations.

        Returns a dict with keys: ``narrative``, ``open_concerns``,
        ``critical_push``.
        """
        obs_block = _format_observations_for_prompt(observations)
        user_prompt = (
            f"CURRENT NARRATIVE:\n{self._state.narrative}\n\n"
            f"CURRENT OPEN CONCERNS:\n"
            + ("\n".join(f"- {c}" for c in self._state.open_concerns) or "(none)")
            + f"\n\nNEW OBSERVATIONS (tick {self._state.tick_count}):\n{obs_block}\n\n"
            "Update narrative & open concerns. Decide critical_push."
        )
        # Run the sync Anthropic call off the event loop
        response_text = await asyncio.to_thread(self._call_claude, user_prompt)
        return _parse_synthesis(response_text)

    def _call_claude(self, user_prompt: str) -> str:
        """Blocking Anthropic call; wrapped in ``to_thread`` by the caller."""
        resp = self._claude.messages.create(
            model=self._model,
            max_tokens=600,
            system=[
                {
                    "type": "text",
                    "text": OBSERVER_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts).strip()

    # ---- cooldowns, dedup, rate limits ----

    def _prune_expired(self, now: float) -> None:
        dedup_window = settings.watchdog.dedup_window_s
        cooldown_window = settings.watchdog.action_cooldown_s
        self._state.recent_keys = {
            k: t for k, t in self._state.recent_keys.items() if now - t < dedup_window
        }
        self._state.action_cooldowns = {
            k: t for k, t in self._state.action_cooldowns.items() if t > now - cooldown_window
        }

    def _is_deduped(self, obs: Observation, now: float) -> bool:
        last = self._state.recent_keys.get(obs.key)
        if last is None:
            return False
        return (now - last) < settings.watchdog.dedup_window_s

    def _is_in_action_cooldown(self, obs: Observation, now: float) -> bool:
        until = self._state.action_cooldowns.get(obs.about)
        if until is None:
            return False
        return now < until

    def _can_critical_push(self, now: float, observations: List[Observation]) -> bool:
        has_crit = any(o.severity == SEVERITY_CRITICAL for o in observations)
        if not has_crit:
            # Observer can still promote to critical based on cross-channel
            # correlation (warning+warning on same embryo). We allow that too.
            by_about: Dict[str, int] = defaultdict(int)
            for o in observations:
                if o.severity in (SEVERITY_WARNING, SEVERITY_CRITICAL):
                    by_about[o.about] += 1
            has_crit = any(v >= 2 for v in by_about.values() if by_about)
        if not has_crit:
            return False
        if self._state.last_critical_push_at is None:
            return True
        return (now - self._state.last_critical_push_at) > settings.watchdog.critical_push_min_gap_s

    # ---- event handlers ----

    def _on_action_event(self, event) -> None:
        """Set a cooldown when the orchestrator acts on an embryo.

        Prevents the feedback loop where observer flags an embryo → orchestrator
        stops it → observer sees the stop as a new anomaly.
        """
        data = dict(event.data or {})
        embryo_id = (
            data.get("embryo_id")
            or data.get("about")
            or data.get("subject")
        )
        if not embryo_id:
            return
        until = time.time() + settings.watchdog.action_cooldown_s
        self._state.action_cooldowns[str(embryo_id)] = until
        logger.debug(
            "watchdog observer: cooldown set for %s until %s (from %s)",
            embryo_id,
            until,
            getattr(event, "event_type", ""),
        )

    # ---- persistence ----

    def _state_path(self):
        if self._store is None or self._session_id is None:
            return None
        try:
            return self._store.observer_state_path(self._session_id)
        except Exception:
            return None

    def _load_state(self) -> None:
        if self._store is None or self._session_id is None:
            return
        try:
            data = self._store.load_observer_state(self._session_id)
            if data:
                self._state = _ObserverState.from_dict(data)
                logger.info(
                    "watchdog observer: rehydrated state (tick %d, %d pending handoffs)",
                    self._state.tick_count,
                    len(self._state.pending_handoffs),
                )
        except Exception:
            logger.warning("watchdog observer: failed to load state", exc_info=True)

    def _save_state(self) -> None:
        if self._store is None or self._session_id is None:
            return
        try:
            self._store.save_observer_state(self._session_id, self._state.to_dict())
        except Exception:
            logger.warning("watchdog observer: failed to save state", exc_info=True)

    # ---- handoff drain ----

    def drain_handoffs(self) -> List[Dict[str, Any]]:
        """Return and clear pending critical handoffs.

        Called by the conversation drain path so handoffs are consumed into the
        orchestrator's conversation exactly once.
        """
        out = list(self._state.pending_handoffs)
        self._state.pending_handoffs.clear()
        self._save_state()
        return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _format_observations_for_prompt(observations: List[Observation]) -> str:
    if not observations:
        return "(no new observations this tick)"
    lines = []
    for o in observations:
        lines.append(
            f"- [{o.channel}/{o.severity}] about={o.about} :: {o.summary}"
            + (f" | recommend: {o.recommend}" if o.recommend else "")
        )
    return "\n".join(lines)


_SECTION_RE = re.compile(r"^(NARRATIVE|OPEN_CONCERNS|CRITICAL_PUSH)\s*:", re.MULTILINE)


def _parse_synthesis(text: str) -> Dict[str, Any]:
    """Parse the observer's structured response into a dict.

    Format we expect:
        NARRATIVE:
        <markdown paragraph>

        OPEN_CONCERNS:
        - line 1
        - line 2
        (or "none")

        CRITICAL_PUSH:
        <one-line critical message> (or "none")
    """
    result: Dict[str, Any] = {
        "narrative": "",
        "open_concerns": [],
        "critical_push": None,
    }
    if not text:
        return result

    # Split by section headers
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    result["narrative"] = sections.get("NARRATIVE", "").strip()

    concerns_raw = sections.get("OPEN_CONCERNS", "").strip()
    if concerns_raw and concerns_raw.lower() != "none":
        concerns = []
        for line in concerns_raw.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line:
                concerns.append(line)
        result["open_concerns"] = concerns

    critical_raw = sections.get("CRITICAL_PUSH", "").strip()
    if critical_raw and critical_raw.lower() not in {"none", "(none)", "n/a"}:
        # Trim to single line
        first_line = critical_raw.splitlines()[0].strip()
        if first_line:
            result["critical_push"] = first_line[:400]

    return result


async def _maybe_await(x):
    if asyncio.iscoroutine(x):
        return await x
    return x
