"""
Daemon — The continuously thinking agent.

The daemon runs as a background process, executing tasks at a pace determined
by arousal level. It:
- Holds the researcher's intentions
- Builds understanding from observations
- Makes predictions and notices surprises
- Adapts its pace based on what's happening
- Escalates from quick scans to deep thinking when needed
- Monitors expectations for approaching/expired deadlines
- Bootstraps its context via gap assessment and onboarding (cold start doctrine)

Key components:
- Clock: When and how to think (arousal, pacing, escalation)
- Types: Shared data types (WorldState, ThinkResult)
- Task: Typed tasks and priority queue
- Scheduler: Task-driven heartbeat (cognitive/physical/interaction dispatch)
- Daemon: Lifecycle, event handling, expectation monitoring
- Onboarding: Cold start conversation logic and context seeding
- Runner: Entry point, startup
"""

from .clock import (
    Clock,
    ArousalState,
    ThinkTrigger,
    ThinkingMode,
    EscalationReason,
    EscalationRequest,
    AROUSAL_SIGNALS,
    select_model,
    should_escalate,
    model_name_for_mode,
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
from .core import Daemon, ExpectationMonitor
from .onboarding import generate_onboarding_tasks, apply_ingestion_to_context

__all__ = [
    # Clock
    "Clock",
    "ArousalState",
    "ThinkTrigger",
    "ThinkingMode",
    "EscalationReason",
    "EscalationRequest",
    "AROUSAL_SIGNALS",
    "select_model",
    "should_escalate",
    "model_name_for_mode",
    # Types
    "WorldState",
    "ThinkResult",
    # Task
    "Task",
    "TaskCategory",
    "TaskPriority",
    "TaskQueue",
    "TaskResult",
    "TaskStatus",
    "TaskType",
    "make_task",
    # Scheduler
    "Scheduler",
    # Daemon
    "Daemon",
    "ExpectationMonitor",
    # Onboarding
    "generate_onboarding_tasks",
    "apply_ingestion_to_context",
]
