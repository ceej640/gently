"""
Task model and queue — typed tasks that drive the scheduler.

Tasks are the unit of work in the scheduler. Each task has:
- A type that determines how it's executed (cognitive/physical/interaction)
- A priority that determines when it runs
- Relationships to parent tasks (for tracing execution chains)
- A result slot for what happened

Tasks are ephemeral — they live in memory and are pruned when completed.
Context (understanding, expectations, etc.) is what persists.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..context import ContextUpdates

logger = logging.getLogger(__name__)


# ================================================================
# Enums
# ================================================================

class TaskType(Enum):
    """What kind of work this task represents."""

    # Cognitive (need LLM)
    OBSERVE = "observe"          # Sample state, check on something
    SYNTHESIZE = "synthesize"    # Combine observations into understanding
    PREDICT = "predict"          # Update expectations
    COMPARE = "compare"          # Relate observations to each other
    REFLECT = "reflect"          # Step back, look at bigger picture
    QUESTION = "question"        # Formulate open questions
    RETRIEVE = "retrieve"        # Pull relevant context for situation

    # Physical (need hardware)
    IMAGE = "image"              # Acquire data
    MOVE = "move"                # Stage movement
    CONFIGURE = "configure"      # Set parameters
    CALIBRATE = "calibrate"      # Focus, alignment

    # Interaction (need user)
    SURFACE = "surface"          # Bring insight to user's attention
    ASK = "ask"                  # Request input or clarification
    NOTIFY = "notify"            # Alert about something important


class TaskCategory(Enum):
    """Derived from TaskType — determines execution path."""
    COGNITIVE = "cognitive"
    PHYSICAL = "physical"
    INTERACTION = "interaction"


# Map each type to its category
_TYPE_TO_CATEGORY = {
    TaskType.OBSERVE: TaskCategory.COGNITIVE,
    TaskType.SYNTHESIZE: TaskCategory.COGNITIVE,
    TaskType.PREDICT: TaskCategory.COGNITIVE,
    TaskType.COMPARE: TaskCategory.COGNITIVE,
    TaskType.REFLECT: TaskCategory.COGNITIVE,
    TaskType.QUESTION: TaskCategory.COGNITIVE,
    TaskType.RETRIEVE: TaskCategory.COGNITIVE,

    TaskType.IMAGE: TaskCategory.PHYSICAL,
    TaskType.MOVE: TaskCategory.PHYSICAL,
    TaskType.CONFIGURE: TaskCategory.PHYSICAL,
    TaskType.CALIBRATE: TaskCategory.PHYSICAL,

    TaskType.SURFACE: TaskCategory.INTERACTION,
    TaskType.ASK: TaskCategory.INTERACTION,
    TaskType.NOTIFY: TaskCategory.INTERACTION,
}


class TaskStatus(Enum):
    """Lifecycle of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority:
    """Named priority levels. Tasks use float 0-100."""
    CRITICAL = 100    # Surprise, user request
    HIGH = 75         # Watchpoint, escalation
    NORMAL = 50       # Routine observations
    LOW = 25          # Reflection, background synthesis
    IDLE = 10         # Housekeeping


# ================================================================
# Task
# ================================================================

@dataclass
class Task:
    """A unit of work for the scheduler."""

    id: str
    type: TaskType
    priority: float                         # 0-100
    status: TaskStatus = TaskStatus.PENDING

    # What to do
    target: Optional[str] = None            # embryo_id, position, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""                        # Why this task was created

    # Relationships
    parent_id: Optional[str] = None         # Task that spawned this one

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None  # Don't run before this time
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Result
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def category(self) -> TaskCategory:
        """Derive category from type."""
        return _TYPE_TO_CATEGORY[self.type]

    @property
    def is_terminal(self) -> bool:
        """Whether this task is done (completed, failed, or cancelled)."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        )

    def __repr__(self) -> str:
        parts = [
            f"Task({self.type.value}",
            f"pri={self.priority:.0f}",
            f"status={self.status.value}",
        ]
        if self.target:
            parts.append(f"target={self.target}")
        if self.reason:
            parts.append(f"reason={self.reason!r}")
        return " ".join(parts) + ")"


def make_task(
    type: TaskType,
    priority: float = TaskPriority.NORMAL,
    target: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    reason: str = "",
    parent_id: Optional[str] = None,
    scheduled_for: Optional[datetime] = None,
) -> Task:
    """Create a task with a fresh ID."""
    return Task(
        id=uuid.uuid4().hex[:12],
        type=type,
        priority=priority,
        target=target,
        params=params or {},
        reason=reason,
        parent_id=parent_id,
        scheduled_for=scheduled_for,
    )


# ================================================================
# TaskResult
# ================================================================

@dataclass
class TaskResult:
    """Result from executing a task."""

    task_id: str
    success: bool
    context_updates: ContextUpdates = field(default_factory=ContextUpdates)
    new_tasks: List[Task] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


# ================================================================
# TaskQueue
# ================================================================

class TaskQueue:
    """
    Priority queue of tasks with inspection and manipulation.

    Tasks are stored in a dict keyed by ID. get_next() picks the
    highest-priority pending task whose scheduled_for has passed.
    """

    def __init__(self, max_size: int = 200):
        self._tasks: Dict[str, Task] = {}
        self._max_size = max_size

    def add(self, task: Task) -> str:
        """
        Add a task to the queue.

        Returns the task ID.
        """
        self._tasks[task.id] = task
        logger.debug(f"Task added: {task}")
        self._prune_if_needed()
        return task.id

    def get_next(self, now: Optional[datetime] = None) -> Optional[Task]:
        """
        Get the highest-priority pending task that's ready to run.

        Parameters
        ----------
        now : datetime, optional
            Current time (defaults to datetime.now())

        Returns
        -------
        Task or None
        """
        now = now or datetime.now()
        candidates = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
            and (t.scheduled_for is None or t.scheduled_for <= now)
        ]
        if not candidates:
            return None

        # Highest priority first, then oldest
        candidates.sort(key=lambda t: (-t.priority, t.created_at))
        return candidates[0]

    def complete(self, task_id: str, result: Optional[Dict] = None):
        """Mark a task as completed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

    def fail(self, task_id: str, error: str):
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error = error

    def cancel(self, task_id: str):
        """Cancel a pending task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()

    # ----------------------------------------------------------------
    # Inspection
    # ----------------------------------------------------------------

    def pending(self) -> List[Task]:
        """All pending tasks, sorted by priority."""
        tasks = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        tasks.sort(key=lambda t: (-t.priority, t.created_at))
        return tasks

    def running(self) -> List[Task]:
        """All currently running tasks."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]

    def by_category(self, cat: TaskCategory) -> List[Task]:
        """All pending tasks of a given category."""
        return [
            t for t in self._tasks.values()
            if t.category == cat and t.status == TaskStatus.PENDING
        ]

    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    @property
    def size(self) -> int:
        """Total number of tasks (all statuses)."""
        return len(self._tasks)

    @property
    def pending_count(self) -> int:
        """Number of pending tasks."""
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def status(self) -> Dict[str, Any]:
        """Queue status summary."""
        by_status = {}
        by_category = {}
        for t in self._tasks.values():
            by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
            if t.status == TaskStatus.PENDING:
                cat = t.category.value
                by_category[cat] = by_category.get(cat, 0) + 1

        top_pending = self.pending()[:5]
        return {
            "total": len(self._tasks),
            "by_status": by_status,
            "pending_by_category": by_category,
            "top_pending": [
                {"id": t.id, "type": t.type.value, "priority": t.priority, "reason": t.reason}
                for t in top_pending
            ],
        }

    # ----------------------------------------------------------------
    # Manipulation
    # ----------------------------------------------------------------

    def reprioritize(self, task_id: str, new_priority: float):
        """Change a pending task's priority."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            old = task.priority
            task.priority = new_priority
            logger.debug(f"Task {task_id} reprioritized: {old} -> {new_priority}")

    def inject(self, task: Task):
        """
        Inject a task from an external source (user, test).

        Same as add() but logged differently.
        """
        logger.info(f"Task injected: {task}")
        self.add(task)

    def clear_category(self, cat: TaskCategory):
        """Cancel all pending tasks of a category."""
        for t in self._tasks.values():
            if t.category == cat and t.status == TaskStatus.PENDING:
                t.status = TaskStatus.CANCELLED
                t.completed_at = datetime.now()

    def clear_all_pending(self):
        """Cancel all pending tasks."""
        for t in self._tasks.values():
            if t.status == TaskStatus.PENDING:
                t.status = TaskStatus.CANCELLED
                t.completed_at = datetime.now()

    # ----------------------------------------------------------------
    # Overflow protection
    # ----------------------------------------------------------------

    def _prune_if_needed(self):
        """Remove old terminal tasks if queue is over max size."""
        if len(self._tasks) <= self._max_size:
            return

        # Remove oldest completed/failed/cancelled tasks
        terminal = [
            t for t in self._tasks.values()
            if t.is_terminal
        ]
        terminal.sort(key=lambda t: t.completed_at or t.created_at)

        remove_count = len(self._tasks) - self._max_size
        for t in terminal[:remove_count]:
            del self._tasks[t.id]

        if remove_count > 0:
            logger.debug(f"Pruned {min(remove_count, len(terminal))} terminal tasks")
