"""
Daemon — The main loop that keeps the agent thinking.

The daemon:
- Runs continuously in the background
- Listens for events from the event bus
- Executes tasks through the scheduler (cognitive/physical/interaction)
- Updates context based on what it learns
- Monitors expectations and triggers on approaching/expired
- Supports escalation from fast scans to deeper thinking

The scheduler is the daemon's execution engine. Events, expectations, and
time generate typed tasks; the scheduler picks the highest-priority task
and executes it through the appropriate path (LLM, hardware, or interaction).
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..core.event_bus import Event, EventBus, EventType, get_event_bus
from ..context import Context, ContextStore, ContextUpdates
from .clock import (
    Clock,
    ThinkTrigger,
    ThinkingMode,
    AROUSAL_SIGNALS,
    select_model,
    should_escalate,
    EscalationRequest,
    EscalationReason,
)
from .types import WorldState, ThinkResult
from .task import (
    Task,
    TaskCategory,
    TaskPriority,
    TaskQueue,
    TaskResult,
    TaskStatus,
    TaskType,
    make_task,
)
from .scheduler import Scheduler

logger = logging.getLogger(__name__)


class ExpectationMonitor:
    """
    Monitors expectations and generates triggers when they approach or expire.

    Runs periodically to check:
    - Expectations approaching their expected time
    - Expectations that have expired without resolution
    """

    def __init__(
        self,
        context_store: ContextStore,
        clock: Clock,
        approach_threshold: timedelta = timedelta(minutes=10),
    ):
        self.context_store = context_store
        self.clock = clock
        self.approach_threshold = approach_threshold
        self.last_check: Optional[datetime] = None
        self.notified_approaching: set = set()  # Expectation IDs we've already notified about

    def check(self) -> List[Dict[str, Any]]:
        """
        Check expectations and return any that need attention.

        Returns
        -------
        List of dicts with expectation info for triggers
        """
        now = datetime.now()
        self.last_check = now

        results = []
        expectations = self.context_store.get_pending_expectations()

        for exp in expectations:
            exp_id = exp.id
            expected_time = exp.expected_time

            # Check if approaching
            time_until = expected_time - now
            if timedelta(0) < time_until <= self.approach_threshold:
                if exp_id not in self.notified_approaching:
                    self.notified_approaching.add(exp_id)
                    results.append({
                        "type": "approaching",
                        "expectation_id": exp_id,
                        "target": exp.target,
                        "prediction": exp.prediction,
                        "time_until_minutes": time_until.total_seconds() / 60,
                    })
                    self.clock.raise_arousal(
                        AROUSAL_SIGNALS["expectation_approaching"],
                        f"expectation approaching: {exp.target}"
                    )

            # Check if expired
            elif time_until <= timedelta(0):
                results.append({
                    "type": "expired",
                    "expectation_id": exp_id,
                    "target": exp.target,
                    "prediction": exp.prediction,
                    "overdue_minutes": abs(time_until.total_seconds()) / 60,
                })
                self.clock.raise_arousal(
                    AROUSAL_SIGNALS["expectation_expired"],
                    f"expectation expired: {exp.target}"
                )

        return results


class Daemon:
    """
    The continuously thinking agent daemon.

    Runs async loops:
    - scheduler: Task-driven heartbeat (replaces the old think loop)
    - decay_loop: Decays arousal over time
    - expectation_loop: Monitors expectations, adds tasks to queue

    The scheduler is the execution engine. Events generate tasks in the
    queue; the scheduler picks the highest-priority task and executes it.
    """

    def __init__(
        self,
        context_store: ContextStore,
        event_bus: Optional[EventBus] = None,
        think_fn: Optional[Callable] = None,
        capabilities: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        context_store : ContextStore
            Storage for agent's context
        event_bus : EventBus, optional
            Event bus for receiving events (defaults to global)
        think_fn : callable, optional
            Function that performs thinking. Signature:
            async def think(context, world, trigger, mode) -> ThinkResult
        capabilities : Any, optional
            Capabilities wrapper for executing actions
        """
        self.context_store = context_store
        self.event_bus = event_bus or get_event_bus()
        self.think_fn = think_fn
        self.capabilities = capabilities
        self.clock = Clock()

        # Task queue and scheduler
        self.queue = TaskQueue()
        self.scheduler = Scheduler(
            queue=self.queue,
            context_store=context_store,
            clock=self.clock,
            think_fn=think_fn,
            capabilities=capabilities,
            event_bus=self.event_bus,
        )

        # Expectation monitor
        self.expectation_monitor = ExpectationMonitor(context_store, self.clock)

        # State
        self.alive = False
        self.user_present = False
        self.current_session_id: Optional[str] = None

        # Event subscriptions (unsubscribe functions)
        self._unsubs: List[Callable] = []

    async def start(self):
        """Start the daemon."""
        if self.alive:
            logger.warning("Daemon already running")
            return

        logger.info("Starting daemon")
        self.alive = True

        # Subscribe to events
        self._subscribe_events()

        # Set event loop for async handlers
        self.event_bus.set_event_loop(asyncio.get_running_loop())

        # Run all loops
        try:
            await asyncio.gather(
                self.scheduler.run(),
                self._decay_loop(),
                self._expectation_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Daemon cancelled")
        finally:
            self._unsubscribe_events()

    async def stop(self):
        """Stop the daemon."""
        logger.info("Stopping daemon")
        self.alive = False
        await self.scheduler.stop()

    def _subscribe_events(self):
        """Subscribe to relevant events."""
        handlers = [
            (EventType.VOLUME_ACQUIRED, self._on_volume_acquired),
            (EventType.STAGE_DETECTED, self._on_stage_detected),
            (EventType.HATCHING_DETECTED, self._on_hatching_detected),
            (EventType.USER_INPUT, self._on_user_input),
            (EventType.ACQUISITION_STARTED, self._on_acquisition_started),
            (EventType.ACQUISITION_COMPLETED, self._on_acquisition_completed),
            (EventType.SESSION_STARTED, self._on_session_started),
            (EventType.SESSION_ENDED, self._on_session_ended),
            (EventType.ANOMALY_DETECTED, self._on_anomaly_detected),
        ]
        for event_type, handler in handlers:
            unsub = self.event_bus.subscribe(event_type, handler)
            self._unsubs.append(unsub)

    def _unsubscribe_events(self):
        """Unsubscribe from all events."""
        for unsub in self._unsubs:
            unsub()
        self._unsubs.clear()

    # ================================================================
    # Event Handlers
    # ================================================================

    def _on_volume_acquired(self, event: Event):
        """Handle volume acquisition."""
        self.clock.add_trigger(ThinkTrigger.EVENT, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["acquisition_complete"],
            "volume acquired"
        )

    def _on_stage_detected(self, event: Event):
        """Handle stage detection."""
        embryo_id = event.data.get("embryo_id")
        detected_stage = event.data.get("stage")

        # Check if this matches expectations
        expected = self.context_store.get_expectation_for(embryo_id)
        if expected and expected.prediction != detected_stage:
            # Surprise!
            self.clock.add_trigger(ThinkTrigger.SURPRISE, {
                "embryo_id": embryo_id,
                "expected": expected.prediction,
                "actual": detected_stage,
                "expectation_id": expected.id,
            })
            self.clock.raise_arousal(AROUSAL_SIGNALS["surprise"], "stage surprise")
        else:
            self.clock.add_trigger(ThinkTrigger.EVENT, event.data)
            self.clock.raise_arousal(
                AROUSAL_SIGNALS["stage_transition"],
                "stage detected"
            )

    def _on_hatching_detected(self, event: Event):
        """Handle hatching detection - high priority."""
        self.clock.add_trigger(ThinkTrigger.WATCHPOINT, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["hatching_detected"],
            "hatching detected"
        )

    def _on_anomaly_detected(self, event: Event):
        """Handle anomaly detection."""
        self.clock.add_trigger(ThinkTrigger.SURPRISE, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["surprise"],
            "anomaly detected"
        )

    def _on_user_input(self, event: Event):
        """Handle user input."""
        self.user_present = True
        self.clock.add_trigger(ThinkTrigger.USER, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["user_message"],
            "user input"
        )

    def _on_acquisition_started(self, event: Event):
        """Handle acquisition start."""
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["routine_event"],
            "acquisition started"
        )

    def _on_acquisition_completed(self, event: Event):
        """Handle acquisition completion."""
        self.clock.add_trigger(ThinkTrigger.EVENT, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["acquisition_complete"],
            "acquisition completed"
        )

    def _on_session_started(self, event: Event):
        """Handle session start."""
        self.current_session_id = event.data.get("session_id")
        self.clock.add_trigger(ThinkTrigger.EVENT, event.data)
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["session_start"],
            "session started"
        )

    def _on_session_ended(self, event: Event):
        """Handle session end."""
        self.current_session_id = None
        self.clock.raise_arousal(
            AROUSAL_SIGNALS["session_end"],
            "session ended"
        )

    # ================================================================
    # Support Loops
    # ================================================================

    async def _decay_loop(self):
        """Decay arousal over time."""
        last = time.time()
        while self.alive:
            now = time.time()
            self.clock.decay_arousal(now - last)
            last = now
            await asyncio.sleep(1.0)

    async def _expectation_loop(self):
        """Monitor expectations periodically, adding tasks to queue."""
        while self.alive:
            # Check every 30 seconds
            await asyncio.sleep(30.0)

            if not self.alive:
                break

            try:
                results = self.expectation_monitor.check()
                for result in results:
                    if result["type"] in ("approaching", "expired"):
                        # Add trigger for the clock (pacing)
                        self.clock.add_trigger(ThinkTrigger.EXPECTATION, result)
                        logger.info(
                            f"Expectation {result['type']}: {result['target']} - {result['prediction']}"
                        )
            except Exception as e:
                logger.error(f"Expectation monitor error: {e}")

    # ================================================================
    # External Interface
    # ================================================================

    def status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return {
            "alive": self.alive,
            "scheduler": self.scheduler.status(),
            "clock": self.clock.status(),
            "user_present": self.user_present,
            "session_id": self.current_session_id,
        }

    def inject_task(self, task: Task):
        """Inject a task from an external source (CLI, test, etc.)."""
        self.queue.inject(task)

    def inject_trigger(self, trigger: ThinkTrigger, data: Optional[Dict] = None):
        """Inject a trigger from external source."""
        self.clock.add_trigger(trigger, data)

    def set_user_present(self, present: bool):
        """Update user presence on daemon and capabilities."""
        self.user_present = present
        if self.capabilities:
            self.capabilities.set_user_present(present)
        if present:
            self.clock.raise_arousal(0.3, "user arrived")
