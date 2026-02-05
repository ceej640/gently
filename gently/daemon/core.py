"""
Daemon — The main loop that keeps the agent thinking.

The daemon:
- Runs continuously in the background
- Listens for events from the event bus
- Thinks at a pace determined by arousal
- Executes actions through capabilities
- Updates context based on what it learns
- Monitors expectations and triggers on approaching/expired
- Supports escalation from fast scans to deeper thinking
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
    - think_loop: Main heartbeat, calls LLM
    - decay_loop: Decays arousal over time
    - expectation_loop: Monitors expectations

    Enhanced features:
    - Escalation from fast scans to deeper thinking
    - Expectation monitoring with triggers
    - Context richness calculation for model selection
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

        # Expectation monitor
        self.expectation_monitor = ExpectationMonitor(context_store, self.clock)

        # State
        self.alive = False
        self.user_present = False
        self.current_session_id: Optional[str] = None

        # Event subscriptions (unsubscribe functions)
        self._unsubs: List[Callable] = []

        # Statistics
        self.think_count = 0
        self.think_by_mode: Dict[ThinkingMode, int] = {
            ThinkingMode.FAST: 0,
            ThinkingMode.MODERATE: 0,
            ThinkingMode.DEEP: 0,
        }
        self.escalation_count = 0
        self.last_think_result: Optional[ThinkResult] = None

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
                self._think_loop(),
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
    # Main Loops
    # ================================================================

    async def _think_loop(self):
        """Main heartbeat loop."""
        while self.alive:
            should, trigger, trigger_data = self.clock.should_think()

            if should:
                try:
                    await self._think_cycle(trigger, trigger_data)
                except Exception as e:
                    logger.error(f"Think cycle error: {e}", exc_info=True)

            # Sleep briefly before checking again
            await asyncio.sleep(0.5)

    async def _think_cycle(self, trigger: ThinkTrigger, trigger_data: Optional[Dict] = None):
        """Execute one thinking cycle."""
        start_time = time.time()

        # 1. Gather inputs
        context = self.context_store.load_active()
        world = await self._sample_world(context)

        # 2. Calculate context richness for model selection
        context_richness = self._calculate_context_richness(context)
        world.context_richness = context_richness

        # 3. Select thinking depth
        mode = select_model(
            trigger=trigger,
            arousal=self.clock.arousal.level,
            context_richness=context_richness,
            has_pending_expectations=len(context.pending_expectations) > 0,
            has_watchpoints=len(context.active_watchpoints) > 0,
            user_present=self.user_present,
            can_deep_think=self.clock.can_deep_think(),
            trigger_data=trigger_data,
        )

        logger.info(
            f"Think cycle: trigger={trigger.value}, mode={mode.value}, "
            f"arousal={self.clock.arousal.level:.2f}, richness={context_richness:.2f}"
        )

        # 4. Think (call LLM if available)
        if self.think_fn:
            result = await self.think_fn(context, world, trigger, mode, trigger_data)
        else:
            # Placeholder result when no think function
            result = ThinkResult(
                reasoning="No think function configured",
                model_used="none",
            )

        result.duration_ms = (time.time() - start_time) * 1000
        result.mode = mode
        result.trigger = trigger

        # 5. Execute actions
        if self.capabilities and result.actions:
            for action in result.actions:
                try:
                    await self._execute_action(action)
                except Exception as e:
                    logger.error(f"Action execution error: {e}")

        # 6. Update context
        self.context_store.apply_updates(result.context_updates)

        # 7. Check for escalation (only from fast scans)
        escalation = should_escalate(result, context, trigger, mode)
        if escalation:
            self.clock.request_escalation(escalation)
            self.escalation_count += 1
            logger.info(f"Escalation triggered: {escalation.reason.value}")

        # 8. Record
        self.clock.record_think(mode)
        self.think_count += 1
        self.think_by_mode[mode] += 1
        self.last_think_result = result

        logger.debug(
            f"Think complete: {result.duration_ms:.0f}ms, mode={mode.value}, "
            f"{len(result.actions)} actions, {len(result.observations_noted)} observations"
        )

    def _calculate_context_richness(self, context: Context) -> float:
        """
        Calculate how rich the current context is.

        Higher richness suggests more thorough thinking is valuable.
        """
        score = 0.0

        # Active campaigns add context
        if context.active_campaigns:
            score += 0.2

        # Tracked embryos add context
        embryo_count = len(context.understanding.embryo_states)
        score += min(0.3, embryo_count * 0.05)

        # Pending expectations need checking
        exp_count = len(context.pending_expectations)
        score += min(0.2, exp_count * 0.1)

        # Active watchpoints need attention
        wp_count = len(context.active_watchpoints)
        score += min(0.2, wp_count * 0.1)

        # Recent observations provide context
        obs_count = len(context.observations)
        score += min(0.1, obs_count * 0.01)

        return min(1.0, score)

    async def _execute_action(self, action: Dict[str, Any]):
        """Execute a single action through capabilities."""
        action_type = action.get("type")
        params = action.get("params", {})

        if self.capabilities:
            await self.capabilities.execute(action_type, params)
        else:
            logger.warning(f"No capabilities to execute action: {action_type}")

    async def _sample_world(self, context: Context) -> WorldState:
        """Sample current world state."""
        recent = self.event_bus.get_history(limit=10)

        return WorldState(
            current_time=datetime.now(),
            user_present=self.user_present,
            microscope_status=None,  # TODO: Get from capabilities
            recent_events=recent,
            session_id=self.current_session_id,
        )

    async def _decay_loop(self):
        """Decay arousal over time."""
        last = time.time()
        while self.alive:
            now = time.time()
            self.clock.decay_arousal(now - last)
            last = now
            await asyncio.sleep(1.0)

    async def _expectation_loop(self):
        """Monitor expectations periodically."""
        while self.alive:
            # Check every 30 seconds
            await asyncio.sleep(30.0)

            if not self.alive:
                break

            try:
                results = self.expectation_monitor.check()
                for result in results:
                    if result["type"] in ("approaching", "expired"):
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
            "think_count": self.think_count,
            "think_by_mode": {k.value: v for k, v in self.think_by_mode.items()},
            "escalation_count": self.escalation_count,
            "clock": self.clock.status(),
            "user_present": self.user_present,
            "session_id": self.current_session_id,
            "last_think": {
                "model": self.last_think_result.model_used if self.last_think_result else None,
                "mode": self.last_think_result.mode.value if self.last_think_result else None,
                "trigger": self.last_think_result.trigger.value if self.last_think_result else None,
                "duration_ms": self.last_think_result.duration_ms if self.last_think_result else None,
                "actions": len(self.last_think_result.actions) if self.last_think_result else 0,
            } if self.last_think_result else None,
        }

    def inject_trigger(self, trigger: ThinkTrigger, data: Optional[Dict] = None):
        """Inject a trigger from external source."""
        self.clock.add_trigger(trigger, data)

    def set_user_present(self, present: bool):
        """Update user presence."""
        self.user_present = present
        if present:
            self.clock.raise_arousal(0.3, "user arrived")
