"""
Scheduler — The heartbeat that drives the agent.

Replaces the daemon's _think_loop with a task-driven execution model.
Each heartbeat:
1. Pick next task from queue
2. Execute it (LLM for cognitive, hardware for physical, etc.)
3. Apply context updates from result
4. Generate follow-up tasks from result
5. Pace based on arousal

Tasks are born from:
- Other tasks (synthesis found pattern -> schedule comparison)
- Time passing (2 hours -> schedule reflection)
- Expectations coming due (embryo expected at comma -> schedule observation)
- External events (hatching detected -> schedule integration)
- User injection (CLI -> queue.inject())
- Watchpoints (embryo crossed threshold -> schedule attention)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.event_bus import EventBus
from ..context import Context, ContextStore
from .clock import Clock, ThinkTrigger, ThinkingMode, select_model
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

logger = logging.getLogger(__name__)


# Map trigger types to the tasks they should create
_TRIGGER_TO_TASK = {
    ThinkTrigger.INTERVAL: (TaskType.OBSERVE, TaskPriority.NORMAL, "routine interval scan"),
    ThinkTrigger.EVENT: (TaskType.OBSERVE, TaskPriority.NORMAL, "event occurred"),
    ThinkTrigger.SURPRISE: (TaskType.SYNTHESIZE, TaskPriority.HIGH, "surprise — investigate"),
    ThinkTrigger.USER: (TaskType.RETRIEVE, TaskPriority.CRITICAL, "user interaction"),
    ThinkTrigger.ESCALATION: (TaskType.REFLECT, TaskPriority.HIGH, "escalation from fast scan"),
    ThinkTrigger.EXPECTATION: (TaskType.OBSERVE, TaskPriority.HIGH, "expectation check"),
    ThinkTrigger.WATCHPOINT: (TaskType.OBSERVE, TaskPriority.HIGH, "watchpoint triggered"),
}

# Map task types to thinking modes
_TASK_TO_MODE = {
    TaskType.OBSERVE: ThinkingMode.FAST,
    TaskType.RETRIEVE: ThinkingMode.FAST,
    TaskType.SYNTHESIZE: ThinkingMode.MODERATE,
    TaskType.COMPARE: ThinkingMode.MODERATE,
    TaskType.PREDICT: ThinkingMode.MODERATE,
    TaskType.REFLECT: ThinkingMode.DEEP,
    TaskType.QUESTION: ThinkingMode.DEEP,
}

# Map LLM action type strings to TaskTypes
_ACTION_TO_TASK_TYPE = {
    "observe": TaskType.OBSERVE,
    "synthesize": TaskType.SYNTHESIZE,
    "predict": TaskType.PREDICT,
    "compare": TaskType.COMPARE,
    "reflect": TaskType.REFLECT,
    "question": TaskType.QUESTION,
    "retrieve": TaskType.RETRIEVE,
    "image": TaskType.IMAGE,
    "acquire": TaskType.IMAGE,
    "move": TaskType.MOVE,
    "configure": TaskType.CONFIGURE,
    "calibrate": TaskType.CALIBRATE,
    "speak": TaskType.SURFACE,
    "surface": TaskType.SURFACE,
    "ask": TaskType.ASK,
    "notify": TaskType.NOTIFY,
}


class Scheduler:
    """
    The heartbeat that drives the agent.

    Each heartbeat picks the next task from the queue, executes it,
    applies context updates, and generates follow-up tasks. The clock
    controls pacing — arousal modulates how fast tasks execute, not
    what tasks exist.
    """

    def __init__(
        self,
        queue: TaskQueue,
        context_store: ContextStore,
        clock: Clock,
        think_fn: Optional[Callable],
        capabilities: Optional[Any],
        event_bus: EventBus,
    ):
        self.queue = queue
        self.context_store = context_store
        self.clock = clock
        self.think_fn = think_fn
        self.capabilities = capabilities
        self.event_bus = event_bus
        self.alive = False

        # Stats
        self.tasks_executed = 0
        self.tasks_by_category: Dict[TaskCategory, int] = {
            TaskCategory.COGNITIVE: 0,
            TaskCategory.PHYSICAL: 0,
            TaskCategory.INTERACTION: 0,
        }
        self.last_result: Optional[TaskResult] = None
        self.last_task: Optional[Task] = None

    async def run(self):
        """Main heartbeat loop."""
        self.alive = True
        logger.info("Scheduler started")

        while self.alive:
            try:
                should, trigger, trigger_data = self.clock.should_think()

                if should:
                    # Convert trigger to task if no matching task in queue
                    self._ensure_trigger_task(trigger, trigger_data)

                    # Execute next task
                    task = self.queue.get_next()
                    if task:
                        result = await self._execute(task)
                        self.last_result = result
                        self.last_task = task

                    self.clock.record_think(
                        self._mode_for_last_task() or ThinkingMode.FAST
                    )

            except Exception as e:
                logger.error(f"Scheduler heartbeat error: {e}", exc_info=True)

            await asyncio.sleep(0.5)

        logger.info("Scheduler stopped")

    async def stop(self):
        """Stop the scheduler."""
        self.alive = False

    # ================================================================
    # Task Execution
    # ================================================================

    async def _execute(self, task: Task) -> TaskResult:
        """Execute a single task and handle its result."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        logger.info(f"Executing {task}")

        try:
            match task.category:
                case TaskCategory.COGNITIVE:
                    result = await self._execute_cognitive(task)
                case TaskCategory.PHYSICAL:
                    result = await self._execute_physical(task)
                case TaskCategory.INTERACTION:
                    result = await self._execute_interaction(task)

            # Apply context updates
            if result.context_updates:
                self.context_store.apply_updates(result.context_updates)

            # Schedule follow-up tasks
            for new_task in result.new_tasks:
                self.queue.add(new_task)

            # Complete the task
            self.queue.complete(task.id, result.data)

            # Stats
            self.tasks_executed += 1
            self.tasks_by_category[task.category] += 1

            logger.debug(
                f"Task {task.id} completed: "
                f"{len(result.new_tasks)} follow-ups, "
                f"success={result.success}"
            )

            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}", exc_info=True)
            self.queue.fail(task.id, str(e))
            return TaskResult(task_id=task.id, success=False)

    async def _execute_cognitive(self, task: Task) -> TaskResult:
        """Execute via LLM call with task-specific prompt framing."""
        if not self.think_fn:
            return TaskResult(
                task_id=task.id,
                success=True,
                reasoning="No think function configured",
            )

        context = self.context_store.load_active()
        world = self._sample_world()
        mode = self._select_mode_for_task(task)

        # Build trigger info from task for the think function
        trigger = self._task_to_trigger(task)
        trigger_data = self._task_to_trigger_data(task)

        result = await self.think_fn(context, world, trigger, mode, trigger_data)

        # Convert ThinkResult actions -> follow-up tasks
        new_tasks = self._actions_to_tasks(result.actions, parent=task)

        return TaskResult(
            task_id=task.id,
            success=True,
            context_updates=result.context_updates,
            new_tasks=new_tasks,
            data={"reasoning": result.reasoning, "model": result.model_used},
            reasoning=result.reasoning,
        )

    async def _execute_physical(self, task: Task) -> TaskResult:
        """Execute via capabilities — no LLM needed."""
        if not self.capabilities:
            return TaskResult(
                task_id=task.id,
                success=False,
                data={"error": "No capabilities configured"},
            )

        cap_result = await self.capabilities.execute(task.type.value, task.params)

        new_tasks = []
        # Physical tasks may generate cognitive follow-ups
        # e.g., after imaging, schedule an OBSERVE to check the result
        if task.type == TaskType.IMAGE and cap_result.success:
            new_tasks.append(make_task(
                type=TaskType.OBSERVE,
                priority=TaskPriority.NORMAL,
                target=task.target,
                reason=f"check result of imaging {task.target or 'unknown'}",
                parent_id=task.id,
            ))

        return TaskResult(
            task_id=task.id,
            success=cap_result.success,
            new_tasks=new_tasks,
            data=cap_result.data or {},
        )

    async def _execute_interaction(self, task: Task) -> TaskResult:
        """Execute via interaction capability."""
        if not self.capabilities:
            return TaskResult(
                task_id=task.id,
                success=False,
                data={"error": "No capabilities configured"},
            )

        # Map task types to capability action names
        action_map = {
            TaskType.SURFACE: "speak",
            TaskType.ASK: "ask",
            TaskType.NOTIFY: "notify",
        }
        action_type = action_map.get(task.type, task.type.value)

        cap_result = await self.capabilities.execute(action_type, task.params)

        new_tasks = []
        # If we asked the user something and got a response, schedule retrieval
        if task.type == TaskType.ASK and cap_result.success and cap_result.data:
            new_tasks.append(make_task(
                type=TaskType.RETRIEVE,
                priority=TaskPriority.HIGH,
                params={"user_response": cap_result.data.get("response", "")},
                reason="process user response",
                parent_id=task.id,
            ))

        return TaskResult(
            task_id=task.id,
            success=cap_result.success,
            new_tasks=new_tasks,
            data=cap_result.data or {},
        )

    # ================================================================
    # Task Generation
    # ================================================================

    def _ensure_trigger_task(self, trigger: ThinkTrigger, data: Optional[Dict]):
        """Convert a clock trigger into a task if the queue needs it."""
        mapping = _TRIGGER_TO_TASK.get(trigger)
        if not mapping:
            return

        task_type, priority, reason = mapping

        # Include trigger data in the task
        params = {}
        target = None
        if data:
            params["trigger_data"] = data
            target = data.get("target") or data.get("embryo_id")

            # Refine reason from trigger data
            if trigger == ThinkTrigger.SURPRISE:
                expected = data.get("expected", "?")
                actual = data.get("actual", "?")
                reason = f"surprise: expected {expected}, got {actual}"
            elif trigger == ThinkTrigger.EXPECTATION:
                exp_type = data.get("type", "")
                exp_target = data.get("target", "")
                reason = f"expectation {exp_type}: {exp_target}"
            elif trigger == ThinkTrigger.USER:
                reason = "user interaction"

        task = make_task(
            type=task_type,
            priority=priority,
            target=target,
            params=params,
            reason=reason,
        )
        self.queue.add(task)

    def _actions_to_tasks(self, actions: List[Dict], parent: Task) -> List[Task]:
        """Convert LLM-output actions into typed tasks."""
        tasks = []
        for action in actions:
            action_type = action.get("type", "")
            task_type = _ACTION_TO_TASK_TYPE.get(action_type)
            if not task_type:
                logger.warning(f"Unknown action type '{action_type}', skipping")
                continue

            params = action.get("params", {})
            if isinstance(params, str):
                params = {"target": params}
            target = params.pop("target", None) or params.pop("embryo_id", None)
            reason = action.get("reason", f"follow-up from {parent.type.value}")

            # Physical/interaction tasks get normal priority from LLM actions
            # Cognitive follow-ups get slightly lower priority than parent
            if task_type in _ACTION_TO_TASK_TYPE.values():
                category = task_type_category(task_type)
                if category == TaskCategory.COGNITIVE:
                    priority = max(TaskPriority.LOW, parent.priority - 10)
                else:
                    priority = TaskPriority.NORMAL
            else:
                priority = TaskPriority.NORMAL

            tasks.append(make_task(
                type=task_type,
                priority=priority,
                target=target,
                params=params,
                reason=reason,
                parent_id=parent.id,
            ))

        return tasks

    # ================================================================
    # Mode Selection
    # ================================================================

    def _select_mode_for_task(self, task: Task) -> ThinkingMode:
        """Select LLM depth based on task type and priority."""
        base_mode = _TASK_TO_MODE.get(task.type, ThinkingMode.MODERATE)

        # High-priority tasks get at least moderate
        if task.priority >= TaskPriority.HIGH and base_mode == ThinkingMode.FAST:
            base_mode = ThinkingMode.MODERATE

        # Deep thinking needs cooldown
        if base_mode == ThinkingMode.DEEP and not self.clock.can_deep_think():
            base_mode = ThinkingMode.MODERATE

        return base_mode

    def _mode_for_last_task(self) -> Optional[ThinkingMode]:
        """Get the mode used for the last task (for clock recording)."""
        if self.last_task and self.last_task.category == TaskCategory.COGNITIVE:
            return self._select_mode_for_task(self.last_task)
        return ThinkingMode.FAST

    # ================================================================
    # Helpers
    # ================================================================

    def _sample_world(self) -> WorldState:
        """Sample current world state."""
        recent = self.event_bus.get_history(limit=10)
        return WorldState(
            current_time=datetime.now(),
            recent_events=recent,
        )

    def _task_to_trigger(self, task: Task) -> ThinkTrigger:
        """Map task type back to a trigger for the think function."""
        # The think function expects a trigger, but we're task-driven now.
        # Map task types to the closest trigger for prompt framing.
        mapping = {
            TaskType.OBSERVE: ThinkTrigger.INTERVAL,
            TaskType.SYNTHESIZE: ThinkTrigger.SURPRISE,
            TaskType.PREDICT: ThinkTrigger.INTERVAL,
            TaskType.COMPARE: ThinkTrigger.EVENT,
            TaskType.REFLECT: ThinkTrigger.ESCALATION,
            TaskType.QUESTION: ThinkTrigger.ESCALATION,
            TaskType.RETRIEVE: ThinkTrigger.USER,
        }
        return mapping.get(task.type, ThinkTrigger.INTERVAL)

    def _task_to_trigger_data(self, task: Task) -> Optional[Dict]:
        """Extract trigger data from a task for the think function."""
        data = task.params.get("trigger_data")
        if data:
            return data

        # Build basic trigger data from task fields
        result = {}
        if task.target:
            result["target"] = task.target
            result["embryo_id"] = task.target
        if task.reason:
            result["reason"] = task.reason
        return result or None

    def status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "alive": self.alive,
            "tasks_executed": self.tasks_executed,
            "tasks_by_category": {k.value: v for k, v in self.tasks_by_category.items()},
            "queue": self.queue.status(),
            "last_task": {
                "id": self.last_task.id,
                "type": self.last_task.type.value,
                "success": self.last_result.success if self.last_result else None,
            } if self.last_task else None,
        }


def task_type_category(task_type: TaskType) -> TaskCategory:
    """Get the category for a task type."""
    from .task import _TYPE_TO_CATEGORY
    return _TYPE_TO_CATEGORY[task_type]
