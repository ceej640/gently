"""
Shared types for the daemon and agent core.

These are data types that flow between daemon/core, agent_core, and capabilities.
Separated from daemon/core.py to break the bidirectional dependency between
daemon and agent_core.

Dependency direction:
    daemon/types → context, core/event_bus, daemon/clock
    daemon/core  → daemon/types
    agent_core   → daemon/types
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.event_bus import Event
from ..context import ContextUpdates
from .clock import ThinkTrigger, ThinkingMode


@dataclass
class WorldState:
    """
    Current state of the world as seen by the agent.

    Sampled fresh each thinking cycle.
    """
    current_time: datetime
    user_present: bool = False
    microscope_status: Optional[Dict[str, Any]] = None
    recent_events: List[Event] = field(default_factory=list)
    session_id: Optional[str] = None

    # Derived metrics
    context_richness: float = 0.5  # How much context is available (0.0-1.0)

    def summary(self) -> str:
        """One-line summary of world state."""
        parts = [f"time={self.current_time.strftime('%H:%M:%S')}"]
        if self.user_present:
            parts.append("user_present")
        if self.microscope_status:
            status = self.microscope_status.get("status", "unknown")
            parts.append(f"microscope={status}")
        parts.append(f"recent_events={len(self.recent_events)}")
        parts.append(f"richness={self.context_richness:.2f}")
        return " ".join(parts)


@dataclass
class ThinkResult:
    """Result from a thinking cycle."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    context_updates: ContextUpdates = field(default_factory=ContextUpdates)
    reasoning: str = ""
    observations_noted: List[str] = field(default_factory=list)
    model_used: str = ""
    duration_ms: float = 0.0
    mode: ThinkingMode = ThinkingMode.MODERATE
    trigger: ThinkTrigger = ThinkTrigger.INTERVAL
