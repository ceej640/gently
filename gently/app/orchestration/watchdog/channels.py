"""Watchdog channels — cheap, stateless digesters of specific data streams.

Each channel watches one aspect of the running experiment (progress, perception,
hardware, system health), maintains a small sliding window of state, and
produces structured :class:`Observation` objects on demand.

Channels never speak to the orchestrator directly — they feed the
:class:`~gently.app.orchestration.watchdog.observer.Observer`, which synthesizes
and decides what the orchestrator sees.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from gently.core.event_bus import EventBus, EventType, get_event_bus
from gently.settings import settings

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Observation — the common output type of every channel
# ----------------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

_SEVERITY_ORDER = {
    SEVERITY_INFO: 0,
    SEVERITY_WARNING: 1,
    SEVERITY_CRITICAL: 2,
}


@dataclass
class Observation:
    """A structured observation emitted by a channel.

    Attributes
    ----------
    channel : str
        Name of the channel that produced this observation ("progress",
        "perception", "hardware", "system_health").
    severity : str
        One of "info", "warning", "critical".
    about : str
        What the observation concerns — e.g. ``"embryo_5"``, ``"filestore"``,
        ``"global"``. Used for cooldown keying and correlation.
    summary : str
        Short human-readable summary (one line).
    detail : dict
        Structured payload — numbers, lists, sub-fields the observer may quote.
    recommend : str
        Suggested action the orchestrator could take (free-form, optional).
    key : str
        Dedup key. Observations with the same key within the dedup window are
        collapsed. Default: ``f"{channel}:{type}:{about}"``.
    timestamp : datetime
        When the observation was produced.
    """

    channel: str
    severity: str
    about: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    recommend: str = ""
    key: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.key:
            # Best-effort default — channels can override
            self.key = f"{self.channel}:{self.about}:{self.summary[:40]}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @property
    def severity_rank(self) -> int:
        return _SEVERITY_ORDER.get(self.severity, 0)


# ----------------------------------------------------------------------
# Channel base class
# ----------------------------------------------------------------------


class Channel:
    """Base class for watchdog channels.

    A channel subscribes to event-bus events (and/or reads state files) on
    :meth:`start`, accumulates internal state, and emits observations on
    :meth:`digest`. Channels are intended to be cheap — pure digesters, no
    LLM calls.
    """

    name: str = "base"

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._bus = event_bus or get_event_bus()
        self._unsubscribers: List[Callable[[], None]] = []
        self._started = False

    # ---- lifecycle ----

    def start(self) -> None:
        """Subscribe to events and perform any pre-flight checks.

        Subclasses override :meth:`_on_start` for hook-based setup.
        """
        if self._started:
            return
        try:
            self._on_start()
        except Exception:
            logger.exception("channel %s: error during start", self.name)
        self._started = True

    def stop(self) -> None:
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception:
                logger.debug("channel %s: unsubscribe error", self.name, exc_info=True)
        self._unsubscribers.clear()
        try:
            self._on_stop()
        except Exception:
            logger.exception("channel %s: error during stop", self.name)
        self._started = False

    def digest(self) -> List[Observation]:
        """Return the current set of observations this channel wants to flag.

        Called periodically by the observer. Must return only *current*
        observations — stale ones should be dropped.
        """
        try:
            return self._on_digest()
        except Exception:
            logger.exception("channel %s: digest failed", self.name)
            return []

    # ---- hooks for subclasses ----

    def _on_start(self) -> None:
        pass

    def _on_stop(self) -> None:
        pass

    def _on_digest(self) -> List[Observation]:  # pragma: no cover - default
        return []

    # ---- helpers ----

    def _subscribe(self, event_type: EventType, handler: Callable) -> None:
        unsub = self._bus.subscribe(event_type, handler)
        self._unsubscribers.append(unsub)


# ----------------------------------------------------------------------
# Hardware channel — calibration quality + device errors
# ----------------------------------------------------------------------


class HardwareChannel(Channel):
    """Watches hardware fitness: pre-flight calibration quality and device errors.

    Pre-flight: reads each embryo's ``embryo.yaml``, inspects
    ``calibration.r_squared_top`` / ``r_squared_bottom``, flags any below the
    configured threshold. These checks happen once at :meth:`start`.

    During runtime: subscribes to :data:`EventType.ERROR_OCCURRED` from device
    sources and tracks error bursts keyed by embryo (if present).
    """

    name = "hardware"

    def __init__(
        self,
        store,
        session_id: str,
        embryo_ids: List[str],
        *,
        min_r_squared: Optional[float] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._store = store
        self._session_id = session_id
        self._embryo_ids = list(embryo_ids)
        self._min_r2 = (
            min_r_squared
            if min_r_squared is not None
            else settings.watchdog.min_calibration_r_squared
        )
        self._calibration_flags: Dict[str, Dict[str, Any]] = {}
        self._device_errors: Deque[Tuple[float, str, Dict[str, Any]]] = deque(maxlen=200)

    def _on_start(self) -> None:
        self._scan_calibrations()
        self._subscribe(EventType.ERROR_OCCURRED, self._on_error)
        self._subscribe(EventType.ACQUISITION_FAILED, self._on_error)

    def _scan_calibrations(self) -> None:
        """Read each embryo.yaml and flag poor calibrations."""
        for embryo_id in self._embryo_ids:
            try:
                rec = self._store.get_embryo(self._session_id, embryo_id)
            except Exception:
                logger.debug("hardware channel: failed to read %s", embryo_id, exc_info=True)
                continue
            if not rec:
                continue
            calib = rec.get("calibration") or {}
            r2_top = _safe_float(calib.get("r_squared_top"))
            r2_bot = _safe_float(calib.get("r_squared_bottom"))
            flags: Dict[str, Any] = {}
            if r2_top is not None and r2_top < self._min_r2:
                flags["top"] = r2_top
            if r2_bot is not None and r2_bot < self._min_r2:
                flags["bottom"] = r2_bot
            if flags:
                self._calibration_flags[embryo_id] = {
                    "r_squared_top": r2_top,
                    "r_squared_bottom": r2_bot,
                    "threshold": self._min_r2,
                    "failed": flags,
                }

    def _on_error(self, event) -> None:
        data = dict(event.data or {})
        source = getattr(event, "source", "") or ""
        # Only keep device-origin errors
        if "device" not in source.lower() and "hardware" not in source.lower():
            # Still keep — may be useful for system_health channel
            return
        self._device_errors.append((time.time(), source, data))

    def _on_digest(self) -> List[Observation]:
        out: List[Observation] = []
        # Calibration flags (persistent — re-emitted each tick; dedup collapses)
        for embryo_id, info in self._calibration_flags.items():
            failed = info["failed"]
            sides = ", ".join(f"{k}=R²{v:.2f}" for k, v in failed.items())
            severity = SEVERITY_WARNING if len(failed) == 1 else SEVERITY_CRITICAL
            out.append(
                Observation(
                    channel=self.name,
                    severity=severity,
                    about=embryo_id,
                    summary=f"{embryo_id}: calibration quality low ({sides}, threshold R²>{info['threshold']})",
                    detail=info,
                    recommend="Consider stop_timelapse_embryo or calibrate_embryo",
                    key=f"hardware:calib:{embryo_id}",
                )
            )

        # Device error burst in last 5 min
        cutoff = time.time() - 300
        recent = [e for e in self._device_errors if e[0] >= cutoff]
        if len(recent) >= 10:
            out.append(
                Observation(
                    channel=self.name,
                    severity=SEVERITY_WARNING if len(recent) < 30 else SEVERITY_CRITICAL,
                    about="device",
                    summary=f"Device errors: {len(recent)} in last 5 min",
                    detail={"count": len(recent), "window_s": 300},
                    recommend="Check device layer logs; pause if hardware unresponsive",
                    key="hardware:device_errors",
                )
            )
        return out


# ----------------------------------------------------------------------
# Perception channel — stage regressions, no_object streaks, arrest
# ----------------------------------------------------------------------


# Approximate developmental ordering for C. elegans. Values closer to zero
# are earlier; higher values are later. Used only to detect monotonicity
# regressions — not for any biological claim.
_STAGE_ORDER = {
    "no_object": -1,
    "unknown": -1,
    "1-cell": 0,
    "2-cell": 1,
    "4-cell": 2,
    "8-cell": 3,
    "early": 4,
    "morula": 5,
    "bean": 6,
    "comma": 7,
    "1.5-fold": 8,
    "2-fold": 9,
    "3-fold": 10,
    "pretzel": 11,
    "late": 12,
    "hatching": 13,
    "hatched": 14,
}


def _stage_rank(stage: Optional[str]) -> Optional[int]:
    if not stage:
        return None
    return _STAGE_ORDER.get(stage.lower())


class PerceptionChannel(Channel):
    """Watches VLM stage calls for regressions, no_object streaks, and arrest.

    Subscribes to :data:`EventType.DETECTOR_EVALUATED` and parses the
    ``temporal_analysis`` payload. State: a small rolling window of recent
    stage calls per embryo.
    """

    name = "perception"

    def __init__(
        self,
        *,
        no_object_streak_warning: Optional[int] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._streak_warn = (
            no_object_streak_warning
            if no_object_streak_warning is not None
            else settings.watchdog.no_object_streak_warning
        )
        self._history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=20))
        self._last_temporal: Dict[str, Dict[str, Any]] = {}

    def _on_start(self) -> None:
        self._subscribe(EventType.DETECTOR_EVALUATED, self._on_detector_eval)

    def _on_detector_eval(self, event) -> None:
        data = dict(event.data or {})
        embryo_id = data.get("embryo_id")
        if not embryo_id:
            return
        rec = {
            "timepoint": data.get("timepoint"),
            "stage": (data.get("stage") or data.get("predicted_stage") or "").lower(),
            "stability": data.get("stability"),
            "timestamp": time.time(),
        }
        self._history[embryo_id].append(rec)
        temporal = data.get("temporal_analysis")
        if isinstance(temporal, dict):
            self._last_temporal[embryo_id] = temporal

    def _on_digest(self) -> List[Observation]:
        out: List[Observation] = []
        for embryo_id, hist in self._history.items():
            if not hist:
                continue
            # no_object streak
            trailing = list(hist)[-self._streak_warn:] if len(hist) >= self._streak_warn else []
            if trailing and all((r["stage"] == "no_object") for r in trailing):
                count = 0
                for r in reversed(hist):
                    if r["stage"] == "no_object":
                        count += 1
                    else:
                        break
                severity = SEVERITY_WARNING if count < self._streak_warn + 2 else SEVERITY_CRITICAL
                out.append(
                    Observation(
                        channel=self.name,
                        severity=severity,
                        about=embryo_id,
                        summary=f"{embryo_id}: no_object for {count} consecutive rounds",
                        detail={"streak": count, "recent": [r["stage"] for r in list(hist)[-5:]]},
                        recommend="Recenter, recalibrate, or skip the embryo",
                        key=f"perception:no_object:{embryo_id}",
                    )
                )

            # Stage regression (monotonicity break)
            ranked = [(r["timepoint"], _stage_rank(r["stage"])) for r in hist]
            ranked = [(t, rk) for t, rk in ranked if t is not None and rk is not None and rk >= 0]
            if len(ranked) >= 3:
                # Look for a drop in the last 3 datapoints that is > 1 rank and
                # doesn't recover
                last3 = ranked[-3:]
                if last3[-1][1] < last3[0][1] - 1 and last3[-1][1] < last3[-2][1]:
                    out.append(
                        Observation(
                            channel=self.name,
                            severity=SEVERITY_WARNING,
                            about=embryo_id,
                            summary=f"{embryo_id}: stage regressed (timepoints {last3[0][0]}→{last3[-1][0]})",
                            detail={"trajectory": [(t, r) for t, r in ranked[-5:]]},
                            recommend="Verify imaging — regression usually indicates a focus or perception problem",
                            key=f"perception:regression:{embryo_id}:{last3[-1][0]}",
                        )
                    )

            # Temporal arrest
            temporal = self._last_temporal.get(embryo_id)
            if isinstance(temporal, dict) and temporal.get("is_potentially_arrested"):
                out.append(
                    Observation(
                        channel=self.name,
                        severity=SEVERITY_WARNING,
                        about=embryo_id,
                        summary=(
                            f"{embryo_id}: potentially arrested in "
                            f"{temporal.get('current_stage')} "
                            f"(overtime ratio {float(temporal.get('overtime_ratio') or 0):.1f}x)"
                        ),
                        detail=temporal,
                        recommend="Watch for recovery; consider stop if further regress",
                        key=f"perception:arrest:{embryo_id}",
                    )
                )
        return out


# ----------------------------------------------------------------------
# Progress channel — round-over-round roll-up
# ----------------------------------------------------------------------


class ProgressChannel(Channel):
    """Summarizes experiment progress: rounds, stage distribution, ETA."""

    name = "progress"

    def __init__(self, *, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus=event_bus)
        self._started_at: Optional[float] = None
        self._current_round: int = 0
        self._latest_stage: Dict[str, str] = {}
        self._embryo_ids: List[str] = []
        self._interval_seconds: Optional[float] = None
        self._stop_condition: Optional[str] = None

    def _on_start(self) -> None:
        self._subscribe(EventType.ACQUISITION_STARTED, self._on_started)
        self._subscribe(EventType.VOLUME_ACQUIRED, self._on_volume)
        self._subscribe(EventType.DETECTOR_EVALUATED, self._on_detector)
        self._subscribe(EventType.ACQUISITION_COMPLETED, self._on_completed)
        self._subscribe(EventType.ACQUISITION_STOPPED, self._on_completed)

    def _on_started(self, event) -> None:
        data = dict(event.data or {})
        self._started_at = time.time()
        self._embryo_ids = list(data.get("embryo_ids") or [])
        self._interval_seconds = _safe_float(data.get("interval_seconds"))
        self._stop_condition = data.get("stop_condition")

    def _on_volume(self, event) -> None:
        data = dict(event.data or {})
        rnd = _safe_int(data.get("round") or data.get("timepoint"))
        if rnd is not None:
            self._current_round = max(self._current_round, rnd)

    def _on_detector(self, event) -> None:
        data = dict(event.data or {})
        embryo_id = data.get("embryo_id")
        stage = data.get("stage") or data.get("predicted_stage")
        if embryo_id and stage:
            self._latest_stage[embryo_id] = stage

    def _on_completed(self, event) -> None:
        # Lock in current round
        pass

    def _on_digest(self) -> List[Observation]:
        if not self._started_at:
            return []
        elapsed = time.time() - self._started_at
        hours = elapsed / 3600.0
        stages = defaultdict(int)
        for s in self._latest_stage.values():
            stages[s] += 1
        active = len(self._latest_stage)
        total = max(len(self._embryo_ids), active)
        summary_stages = ", ".join(f"{n} {st}" for st, n in sorted(stages.items()))
        return [
            Observation(
                channel=self.name,
                severity=SEVERITY_INFO,
                about="global",
                summary=(
                    f"Round {self._current_round}: {active}/{total} embryos reporting — "
                    f"{summary_stages or 'no stage data yet'} (elapsed {hours:.1f}h)"
                ),
                detail={
                    "round": self._current_round,
                    "elapsed_hours": round(hours, 2),
                    "interval_seconds": self._interval_seconds,
                    "stop_condition": self._stop_condition,
                    "stages": dict(stages),
                    "active": active,
                    "total": total,
                },
                key="progress:global",
            )
        ]


# ----------------------------------------------------------------------
# System health channel — FileStore / VLM / error bursts
# ----------------------------------------------------------------------


class SystemHealthChannel(Channel):
    """Watches system-level error bursts (FileStore, VLM, generic ERROR_OCCURRED)."""

    name = "system_health"

    def __init__(self, *, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(event_bus=event_bus)
        self._errors: Deque[Tuple[float, str, Dict[str, Any]]] = deque(maxlen=500)
        self._warnings: Deque[Tuple[float, str, Dict[str, Any]]] = deque(maxlen=500)

    def _on_start(self) -> None:
        self._subscribe(EventType.ERROR_OCCURRED, self._on_error)
        self._subscribe(EventType.WARNING_ISSUED, self._on_warning)

    def _on_error(self, event) -> None:
        data = dict(event.data or {})
        self._errors.append((time.time(), getattr(event, "source", "") or "", data))

    def _on_warning(self, event) -> None:
        data = dict(event.data or {})
        self._warnings.append((time.time(), getattr(event, "source", "") or "", data))

    def _on_digest(self) -> List[Observation]:
        out: List[Observation] = []
        cutoff = time.time() - 300  # 5 min window

        # Categorize by source substring
        by_source: Dict[str, int] = defaultdict(int)
        for t, src, data in self._errors:
            if t < cutoff:
                continue
            key = _coarse_source(src, data)
            by_source[key] += 1

        for src, count in by_source.items():
            if count < 10:
                continue
            severity = SEVERITY_WARNING if count < 50 else SEVERITY_CRITICAL
            out.append(
                Observation(
                    channel=self.name,
                    severity=severity,
                    about=src,
                    summary=f"{src}: {count} errors in last 5 min",
                    detail={"count": count, "window_s": 300, "source": src},
                    recommend=(
                        "Investigate root cause — repeated errors here may indicate "
                        "disk pressure, a lost service, or a misconfiguration"
                    ),
                    key=f"system_health:errors:{src}",
                )
            )
        return out


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_default_channels(
    *,
    store,
    session_id: str,
    embryo_ids: List[str],
    event_bus: Optional[EventBus] = None,
) -> List[Channel]:
    """Return the stock set of channels for a timelapse session."""
    bus = event_bus or get_event_bus()
    return [
        ProgressChannel(event_bus=bus),
        PerceptionChannel(event_bus=bus),
        HardwareChannel(
            store=store,
            session_id=session_id,
            embryo_ids=embryo_ids,
            event_bus=bus,
        ),
        SystemHealthChannel(event_bus=bus),
    ]


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _coarse_source(source: str, data: Dict[str, Any]) -> str:
    s = (source or "").lower()
    if "filestore" in s or "file_store" in s:
        return "filestore"
    if "vlm" in s or "perception" in s:
        return "perception"
    if "device" in s or "hardware" in s:
        return "device"
    if "mesh" in s:
        return "mesh"
    # fallback to the data's own source hint if any
    hint = (data.get("component") or data.get("source") or "").lower()
    if hint:
        return hint
    return source or "unknown"
