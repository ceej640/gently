"""Dry-run replay test for the watchdog channels.

Feeds session 2f4472c3's embryo calibration data + synthetic events that
mimic what actually happened in that session, and verifies that each
channel emits the expected observations. Does NOT call the LLM — only
exercises the channel logic.

Run from the repo root::

    python scripts/watchdog_replay.py

Exits non-zero if any expected observation is missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

# Windows console defaults to cp1252; the observation summaries contain
# unicode arrows ("\u2192") and R\u00b2 glyphs. Reconfigure stdio if we can.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Ensure the repo root is on the path when the script is run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gently.core.event_bus import EventBus, EventType  # noqa: E402
from gently.core.file_store import FileStore  # noqa: E402
from gently.settings import settings  # noqa: E402
from gently.app.orchestration.watchdog.channels import (  # noqa: E402
    Observation,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    build_default_channels,
)


SESSION_ID = "2f4472c3"
EMBRYO_IDS = [f"embryo_{i}" for i in range(1, 7)]


def _print_observations(title: str, obs: List[Observation]) -> None:
    print(f"\n--- {title} ({len(obs)} observations) ---")
    for o in obs:
        print(f"  [{o.channel}/{o.severity}] about={o.about} :: {o.summary}")


def _must_contain(obs: List[Observation], *, channel: str, about: str, needle: str) -> bool:
    for o in obs:
        if o.channel == channel and o.about == about and needle.lower() in o.summary.lower():
            return True
    return False


def _emit(bus: EventBus, event_type: EventType, data: dict, source: str = "test") -> None:
    bus.publish(event_type=event_type, data=data, source=source)


def main() -> int:
    # Use an isolated event bus so we don't race with anything.
    bus = EventBus()

    # FileStore root from settings (D:/Gently3 by default)
    store = FileStore(settings.storage.base_path)

    # Sanity check
    embryos = store.list_embryos(SESSION_ID)
    if not embryos:
        print(f"FATAL: session {SESSION_ID} has no embryos in FileStore")
        return 2
    print(
        f"Session {SESSION_ID}: {len(embryos)} embryos "
        f"({[e.get('embryo_id') for e in embryos]})"
    )

    channels = build_default_channels(
        store=store,
        session_id=SESSION_ID,
        embryo_ids=EMBRYO_IDS,
        event_bus=bus,
    )
    for ch in channels:
        ch.start()

    by_name = {ch.name: ch for ch in channels}

    expected_failures: List[str] = []

    # ------------------------------------------------------------------
    # 1. Hardware channel: pre-flight calibration scan should have flagged
    #    embryo_5 (r_squared_top=0.0) and embryo_6 (r_squared_bottom<0.75).
    # ------------------------------------------------------------------
    hw_obs = by_name["hardware"].digest()
    _print_observations("hardware (pre-flight)", hw_obs)
    if not _must_contain(hw_obs, channel="hardware", about="embryo_5", needle="calibration"):
        expected_failures.append("hardware: no calibration observation for embryo_5")
    if not _must_contain(hw_obs, channel="hardware", about="embryo_6", needle="calibration"):
        expected_failures.append("hardware: no calibration observation for embryo_6")

    # ------------------------------------------------------------------
    # 2. Perception channel: emit DETECTOR_EVALUATED events for
    #    embryo_5 (no_object 3x), embryo_2 (bean→early), and embryo_1 (normal).
    # ------------------------------------------------------------------
    for t in range(1, 4):
        _emit(bus, EventType.DETECTOR_EVALUATED, {
            "embryo_id": "embryo_5",
            "timepoint": t,
            "stage": "no_object",
            "stability": 1,
        }, source="timelapse_orchestrator")
    # embryo_2 at T1-T8 bean, then regress to early at T9
    for t in range(1, 9):
        _emit(bus, EventType.DETECTOR_EVALUATED, {
            "embryo_id": "embryo_2",
            "timepoint": t,
            "stage": "bean",
            "stability": t,
        }, source="timelapse_orchestrator")
    _emit(bus, EventType.DETECTOR_EVALUATED, {
        "embryo_id": "embryo_2",
        "timepoint": 9,
        "stage": "early",
        "stability": 1,
        "temporal_analysis": {
            "current_stage": "early",
            "time_in_stage_min": 1.0,
            "observations_in_stage": 1,
            "expected_duration_min": 60.0,
            "overtime_ratio": 0.0,
            "is_potentially_arrested": False,
            "total_observations": 9,
        },
    }, source="timelapse_orchestrator")
    # embryo_1 normal: 1-cell, 2-cell, 4-cell
    for t, stage in enumerate(["1-cell", "2-cell", "4-cell", "early"], start=1):
        _emit(bus, EventType.DETECTOR_EVALUATED, {
            "embryo_id": "embryo_1",
            "timepoint": t,
            "stage": stage,
            "stability": 1,
        }, source="timelapse_orchestrator")

    # embryo_3 potentially arrested
    _emit(bus, EventType.DETECTOR_EVALUATED, {
        "embryo_id": "embryo_3",
        "timepoint": 12,
        "stage": "bean",
        "temporal_analysis": {
            "current_stage": "bean",
            "time_in_stage_min": 200.0,
            "observations_in_stage": 35,
            "expected_duration_min": 45.0,
            "overtime_ratio": 4.4,
            "is_potentially_arrested": True,
            "total_observations": 35,
        },
    }, source="timelapse_orchestrator")

    perc_obs = by_name["perception"].digest()
    _print_observations("perception", perc_obs)
    if not _must_contain(perc_obs, channel="perception", about="embryo_5", needle="no_object"):
        expected_failures.append("perception: missing no_object observation for embryo_5")
    if not _must_contain(perc_obs, channel="perception", about="embryo_2", needle="regress"):
        expected_failures.append("perception: missing regression observation for embryo_2")
    if not _must_contain(perc_obs, channel="perception", about="embryo_3", needle="arrest"):
        expected_failures.append("perception: missing arrest observation for embryo_3")

    # ------------------------------------------------------------------
    # 3. System health channel: emit a burst of filestore errors
    # ------------------------------------------------------------------
    for i in range(50):
        _emit(bus, EventType.ERROR_OCCURRED, {
            "message": "YAML write failed",
            "component": "filestore",
        }, source="core.filestore")
        # Stagger timestamps very slightly so deque ordering is stable
        time.sleep(0.0005)

    sh_obs = by_name["system_health"].digest()
    _print_observations("system_health", sh_obs)
    if not _must_contain(sh_obs, channel="system_health", about="filestore", needle="errors"):
        expected_failures.append("system_health: missing filestore burst observation")

    # ------------------------------------------------------------------
    # 4. Progress channel: emit ACQUISITION_STARTED + VOLUME_ACQUIRED
    # ------------------------------------------------------------------
    _emit(bus, EventType.ACQUISITION_STARTED, {
        "embryo_ids": EMBRYO_IDS,
        "stop_condition": "manual",
        "interval_seconds": 120.0,
    }, source="timelapse_orchestrator")
    for t in range(1, 10):
        _emit(bus, EventType.VOLUME_ACQUIRED, {
            "embryo_id": "embryo_1",
            "round": t,
            "timepoint": t,
        }, source="timelapse_orchestrator")

    prog_obs = by_name["progress"].digest()
    _print_observations("progress", prog_obs)
    if not prog_obs:
        expected_failures.append("progress: no progress observation emitted")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    for ch in channels:
        ch.stop()

    if expected_failures:
        print("\nREPLAY FAILED:")
        for f in expected_failures:
            print(f"  - {f}")
        return 1

    print("\nOK — all channels produced the expected observations for session 2f4472c3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
