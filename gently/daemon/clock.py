"""
Clock — When and how to think.

The clock manages:
- Arousal level (0.0 = resting, 1.0 = max alert)
- Thinking pace (slower when resting, faster when alert)
- Pending triggers (events that should cause immediate thinking)
- Model selection (haiku for quick scans, sonnet/opus for deep thought)
- Escalation (fast scan can trigger deeper thinking if needed)
"""

import math
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThinkTrigger(Enum):
    """What triggered a thinking cycle."""
    INTERVAL = "interval"      # Time-based resting pace
    EVENT = "event"            # Something happened
    SURPRISE = "surprise"      # Expectation violated
    WATCHPOINT = "watchpoint"  # Watched condition triggered
    USER = "user"              # User interaction
    ESCALATION = "escalation"  # Fast scan recommended deeper thinking
    EXPECTATION = "expectation"  # Expectation approaching or expired


class ThinkingMode(Enum):
    """Depth of thinking."""
    FAST = "fast"              # Haiku - routine scans
    MODERATE = "moderate"      # Sonnet - integration, reasoning
    DEEP = "deep"              # Opus - planning, synthesis


class EscalationReason(Enum):
    """Why a fast scan escalated to deeper thinking."""
    UNCERTAINTY = "uncertainty"           # Low confidence in assessment
    MULTIPLE_CONCERNS = "multiple_concerns"  # Several things need attention
    PATTERN_DETECTED = "pattern_detected"   # Noticed something worth investigating
    USER_CONTEXT = "user_context"          # User is present and engaged
    GOAL_RELEVANT = "goal_relevant"        # Relates to active campaign goals
    ANOMALY = "anomaly"                    # Something unexpected


# Arousal signals - how much different events raise arousal
AROUSAL_SIGNALS = {
    "surprise": 0.8,
    "user_message": 0.7,
    "watchpoint_triggered": 0.6,
    "hatching_detected": 0.7,
    "stage_transition": 0.4,
    "acquisition_complete": 0.3,
    "expectation_approaching": 0.2,
    "expectation_expired": 0.4,
    "escalation": 0.3,
    "routine_event": 0.1,
    "session_start": 0.3,
    "session_end": 0.1,
}


@dataclass
class EscalationRequest:
    """Request from fast scan to escalate to deeper thinking."""
    reason: EscalationReason
    context: str  # What specifically triggered escalation
    urgency: float = 0.5  # 0.0-1.0, affects how quickly to escalate
    suggested_mode: ThinkingMode = ThinkingMode.MODERATE


@dataclass
class ArousalState:
    """
    Full arousal state with momentum and history.

    Arousal has two components:
    - Level: Current arousal (0.0-1.0)
    - Momentum: Rate of change (positive = rising, negative = falling)

    This allows for more natural arousal dynamics where sudden events
    cause a spike that gradually settles.
    """
    level: float = 0.0
    momentum: float = 0.0

    # Recent arousal history for pattern detection
    history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, level)
    max_history: int = 60  # Keep last 60 samples

    def record(self):
        """Record current level in history."""
        now = time.time()
        self.history.append((now, self.level))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def average_recent(self, seconds: float = 30.0) -> float:
        """Get average arousal over recent period."""
        if not self.history:
            return self.level
        cutoff = time.time() - seconds
        recent = [level for ts, level in self.history if ts >= cutoff]
        return sum(recent) / len(recent) if recent else self.level

    def is_rising(self, threshold: float = 0.1) -> bool:
        """Check if arousal has been rising recently."""
        return self.momentum > threshold

    def is_falling(self, threshold: float = 0.1) -> bool:
        """Check if arousal has been falling recently."""
        return self.momentum < -threshold


@dataclass
class Clock:
    """
    Manages when and how the agent thinks.

    The clock has two dimensions:
    - Pace: How often to think (based on arousal)
    - Depth: How deeply to think (based on trigger and context)

    Enhanced features:
    - Non-linear arousal decay (faster decay at high arousal)
    - Momentum tracking for arousal
    - Escalation support (fast scan → deep thinking)
    - Expectation monitoring
    """

    # Arousal state (enhanced)
    arousal: ArousalState = field(default_factory=ArousalState)

    # Timing
    last_think_time: Optional[float] = None
    last_deep_think_time: Optional[float] = None

    # Pending triggers (priority queue - higher priority triggers first)
    pending_triggers: List[Tuple[int, ThinkTrigger, Optional[Dict]]] = field(default_factory=list)

    # Escalation state
    pending_escalation: Optional[EscalationRequest] = None
    escalation_cooldown: float = 0.0  # Don't escalate too frequently

    # Pace bounds (seconds)
    resting_interval: float = 60.0
    elevated_interval: float = 5.0
    min_deep_interval: float = 120.0  # Minimum time between deep thinks

    # Arousal dynamics
    base_decay_rate: float = 0.02  # per second at low arousal
    high_arousal_decay_multiplier: float = 2.0  # Faster decay when aroused
    momentum_decay: float = 0.1  # How fast momentum decays

    # Trigger priorities (higher = more urgent)
    TRIGGER_PRIORITY = {
        ThinkTrigger.USER: 100,
        ThinkTrigger.SURPRISE: 90,
        ThinkTrigger.WATCHPOINT: 80,
        ThinkTrigger.ESCALATION: 70,
        ThinkTrigger.EXPECTATION: 60,
        ThinkTrigger.EVENT: 50,
        ThinkTrigger.INTERVAL: 10,
    }

    def current_pace(self) -> float:
        """
        Interval between thinks based on arousal.

        Uses a smooth curve rather than linear interpolation:
        - Very responsive at high arousal
        - Gradual slowdown as arousal decreases
        """
        level = self.arousal.level
        # Use exponential curve for more natural feel
        # At 0: resting_interval, at 1: elevated_interval
        t = 1 - math.exp(-3 * level)  # Steeper response at high arousal
        return self.resting_interval - t * (self.resting_interval - self.elevated_interval)

    def should_think(self) -> Tuple[bool, Optional[ThinkTrigger], Optional[Dict]]:
        """
        Check if it's time to think.

        Returns
        -------
        (should_think, trigger, trigger_data)
            Whether to think, what triggered it, and any associated data.
        """
        # Check for pending escalation first
        if self.pending_escalation and self.escalation_cooldown <= 0:
            escalation = self.pending_escalation
            self.pending_escalation = None
            self.escalation_cooldown = 30.0  # Cooldown before next escalation
            return True, ThinkTrigger.ESCALATION, {
                "reason": escalation.reason.value,
                "context": escalation.context,
                "suggested_mode": escalation.suggested_mode,
            }

        # Priority: pending triggers (sorted by priority)
        if self.pending_triggers:
            self.pending_triggers.sort(key=lambda x: -x[0])  # Highest priority first
            _, trigger, data = self.pending_triggers.pop(0)
            return True, trigger, data

        # First think
        if self.last_think_time is None:
            return True, ThinkTrigger.INTERVAL, None

        # Time-based
        elapsed = time.time() - self.last_think_time
        if elapsed >= self.current_pace():
            return True, ThinkTrigger.INTERVAL, None

        return False, None, None

    def record_think(self, mode: ThinkingMode):
        """Record that a thinking cycle completed."""
        now = time.time()
        self.last_think_time = now
        if mode == ThinkingMode.DEEP:
            self.last_deep_think_time = now
        self.arousal.record()

    def raise_arousal(self, amount: float, reason: str):
        """
        Raise arousal level with momentum.

        Parameters
        ----------
        amount : float
            How much to raise (0.0 to 1.0)
        reason : str
            Why arousal is being raised (for logging)
        """
        old = self.arousal.level
        self.arousal.level = min(1.0, self.arousal.level + amount)

        # Add positive momentum
        self.arousal.momentum = min(0.5, self.arousal.momentum + amount * 0.5)

        if self.arousal.level != old:
            logger.debug(
                f"Arousal raised: {old:.2f} -> {self.arousal.level:.2f} ({reason})"
            )

    def decay_arousal(self, dt: float):
        """
        Decay arousal over time with non-linear dynamics.

        Parameters
        ----------
        dt : float
            Time elapsed in seconds
        """
        old = self.arousal.level

        # Non-linear decay: faster at high arousal
        decay_rate = self.base_decay_rate * (
            1 + (self.high_arousal_decay_multiplier - 1) * self.arousal.level
        )

        # Apply momentum
        self.arousal.level += self.arousal.momentum * dt
        self.arousal.level = max(0.0, min(1.0, self.arousal.level))

        # Decay momentum towards negative (settling)
        target_momentum = -decay_rate
        self.arousal.momentum += (target_momentum - self.arousal.momentum) * self.momentum_decay * dt

        # Apply base decay
        self.arousal.level = max(0.0, self.arousal.level - decay_rate * dt)

        # Decay escalation cooldown
        if self.escalation_cooldown > 0:
            self.escalation_cooldown = max(0.0, self.escalation_cooldown - dt)

        # Only log significant changes
        if abs(self.arousal.level - old) > 0.05:
            logger.debug(f"Arousal decayed: {old:.2f} -> {self.arousal.level:.2f}")

    def add_trigger(self, trigger: ThinkTrigger, data: Optional[Dict] = None):
        """Add a trigger to the pending queue with priority."""
        priority = self.TRIGGER_PRIORITY.get(trigger, 50)
        self.pending_triggers.append((priority, trigger, data))
        logger.debug(f"Added trigger: {trigger.value} (priority={priority})")

    def request_escalation(self, request: EscalationRequest):
        """
        Request escalation from a fast scan to deeper thinking.

        The escalation will be processed on the next think cycle
        if cooldown has elapsed.
        """
        if self.escalation_cooldown <= 0:
            self.pending_escalation = request
            self.raise_arousal(
                AROUSAL_SIGNALS["escalation"] * request.urgency,
                f"escalation: {request.reason.value}"
            )
            logger.info(f"Escalation requested: {request.reason.value} - {request.context}")
        else:
            logger.debug(f"Escalation blocked by cooldown ({self.escalation_cooldown:.1f}s remaining)")

    def can_deep_think(self) -> bool:
        """Check if enough time has passed for deep thinking."""
        if self.last_deep_think_time is None:
            return True
        elapsed = time.time() - self.last_deep_think_time
        return elapsed >= self.min_deep_interval

    def time_until_next_think(self) -> float:
        """
        Time until next scheduled think.

        Returns
        -------
        float
            Seconds until next think, or 0 if should think now.
        """
        if self.pending_triggers or self.pending_escalation:
            return 0.0
        if self.last_think_time is None:
            return 0.0
        elapsed = time.time() - self.last_think_time
        remaining = self.current_pace() - elapsed
        return max(0.0, remaining)

    def status(self) -> dict:
        """Get current clock status."""
        return {
            "arousal": round(self.arousal.level, 3),
            "arousal_momentum": round(self.arousal.momentum, 3),
            "arousal_avg_30s": round(self.arousal.average_recent(30), 3),
            "pace_seconds": round(self.current_pace(), 1),
            "pending_triggers": len(self.pending_triggers),
            "pending_escalation": self.pending_escalation.reason.value if self.pending_escalation else None,
            "escalation_cooldown": round(self.escalation_cooldown, 1),
            "time_until_next": round(self.time_until_next_think(), 1),
            "can_deep_think": self.can_deep_think(),
        }


def select_model(
    trigger: ThinkTrigger,
    arousal: float,
    context_richness: float = 0.5,
    has_pending_expectations: bool = False,
    has_watchpoints: bool = False,
    user_present: bool = False,
    can_deep_think: bool = True,
    trigger_data: Optional[Dict] = None,
) -> ThinkingMode:
    """
    Select thinking depth based on situation.

    Parameters
    ----------
    trigger : ThinkTrigger
        What triggered this think
    arousal : float
        Current arousal level
    context_richness : float
        How much context is available (0.0-1.0)
    has_pending_expectations : bool
        Whether there are expectations to check
    has_watchpoints : bool
        Whether there are active watchpoints
    user_present : bool
        Whether the user is currently present
    can_deep_think : bool
        Whether enough time has passed since last deep think
    trigger_data : dict, optional
        Additional data from the trigger

    Returns
    -------
    ThinkingMode
        Selected thinking depth
    """
    # Escalation trigger: use suggested mode
    if trigger == ThinkTrigger.ESCALATION and trigger_data:
        suggested = trigger_data.get("suggested_mode", ThinkingMode.MODERATE)
        if suggested == ThinkingMode.DEEP and not can_deep_think:
            return ThinkingMode.MODERATE
        return suggested

    # Surprise always gets deep thinking (if allowed)
    if trigger == ThinkTrigger.SURPRISE:
        return ThinkingMode.DEEP if can_deep_think else ThinkingMode.MODERATE

    # User interaction: depends on arousal and context
    if trigger == ThinkTrigger.USER:
        if arousal > 0.7:
            return ThinkingMode.FAST  # Quick response first
        elif arousal > 0.4 or context_richness > 0.7:
            return ThinkingMode.MODERATE
        else:
            return ThinkingMode.FAST

    # Watchpoint triggered: moderate to investigate
    if trigger == ThinkTrigger.WATCHPOINT:
        if arousal > 0.6:
            return ThinkingMode.DEEP if can_deep_think else ThinkingMode.MODERATE
        return ThinkingMode.MODERATE

    # Expectation-related: check thoroughly
    if trigger == ThinkTrigger.EXPECTATION:
        return ThinkingMode.MODERATE

    # Resting interval: depends on context
    if trigger == ThinkTrigger.INTERVAL:
        if arousal < 0.2 and not has_pending_expectations and not has_watchpoints:
            return ThinkingMode.FAST
        elif arousal < 0.4:
            return ThinkingMode.FAST
        else:
            return ThinkingMode.MODERATE

    # Event: depends on arousal
    if trigger == ThinkTrigger.EVENT:
        if arousal > 0.6:
            return ThinkingMode.MODERATE
        return ThinkingMode.FAST

    # Default to moderate
    return ThinkingMode.MODERATE


def should_escalate(
    result: Any,  # ThinkResult
    context: Any,  # Context
    trigger: ThinkTrigger,
    mode: ThinkingMode,
) -> Optional[EscalationRequest]:
    """
    Determine if a fast scan should escalate to deeper thinking.

    Parameters
    ----------
    result : ThinkResult
        Result from the fast scan
    context : Context
        Current context
    trigger : ThinkTrigger
        What triggered the scan
    mode : ThinkingMode
        Mode that was used

    Returns
    -------
    EscalationRequest or None
        Request to escalate, or None if no escalation needed
    """
    # Only escalate from fast scans
    if mode != ThinkingMode.FAST:
        return None

    # Check for escalation indicators in the result
    escalation_reasons = []

    # Multiple observations noted suggests something interesting
    if len(result.observations_noted) >= 3:
        escalation_reasons.append((
            EscalationReason.MULTIPLE_CONCERNS,
            f"Noted {len(result.observations_noted)} observations"
        ))

    # Actions requested from fast scan might need deeper consideration
    if len(result.actions) >= 2:
        escalation_reasons.append((
            EscalationReason.MULTIPLE_CONCERNS,
            f"Fast scan suggested {len(result.actions)} actions"
        ))

    # New expectations or watchpoints suggest pattern detection
    updates = result.context_updates
    if updates.new_expectations or updates.new_watchpoints:
        escalation_reasons.append((
            EscalationReason.PATTERN_DETECTED,
            "Fast scan identified patterns worth tracking"
        ))

    # New questions suggest uncertainty
    if updates.new_questions:
        escalation_reasons.append((
            EscalationReason.UNCERTAINTY,
            f"Fast scan raised {len(updates.new_questions)} questions"
        ))

    # Check reasoning for escalation keywords
    reasoning_lower = result.reasoning.lower()
    escalation_keywords = [
        ("investigate", EscalationReason.UNCERTAINTY),
        ("unusual", EscalationReason.ANOMALY),
        ("unexpected", EscalationReason.ANOMALY),
        ("surprising", EscalationReason.ANOMALY),
        ("pattern", EscalationReason.PATTERN_DETECTED),
        ("concerning", EscalationReason.MULTIPLE_CONCERNS),
        ("important", EscalationReason.GOAL_RELEVANT),
        ("significant", EscalationReason.GOAL_RELEVANT),
    ]

    for keyword, reason in escalation_keywords:
        if keyword in reasoning_lower:
            escalation_reasons.append((reason, f"Reasoning mentioned '{keyword}'"))
            break  # Only add one keyword-based reason

    # If we have escalation reasons, create request
    if escalation_reasons:
        # Pick the most urgent reason
        primary_reason, context_str = escalation_reasons[0]

        # Determine urgency based on number of reasons
        urgency = min(1.0, 0.3 + 0.2 * len(escalation_reasons))

        return EscalationRequest(
            reason=primary_reason,
            context=context_str,
            urgency=urgency,
            suggested_mode=ThinkingMode.MODERATE,
        )

    return None


def model_name_for_mode(mode: ThinkingMode) -> str:
    """
    Get the Claude model name for a thinking mode.

    Returns
    -------
    str
        Model name for API calls
    """
    mapping = {
        ThinkingMode.FAST: "claude-3-5-haiku-20241022",
        ThinkingMode.MODERATE: "claude-sonnet-4-20250514",
        ThinkingMode.DEEP: "claude-opus-4-20250514",
    }
    return mapping[mode]
