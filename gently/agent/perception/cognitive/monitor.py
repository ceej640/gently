"""
Monitor Mode for Lightweight Perception.

Uses Haiku for quick "still X stage?" checks between deep assessments.
Optimized for speed and cost.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

import anthropic

from .world_model import WorldModel
from .trace import MonitorResult
from .context_manager import ContextManager, VLMContext
from .memory_manager import MemoryManager


@dataclass
class MonitorCheckResult:
    """Result from a monitor check with timing info."""

    result: MonitorResult
    should_escalate: bool
    escalation_reason: Optional[str]
    duration_ms: int
    raw_output: str


class Monitor:
    """
    Lightweight perception mode using Haiku.

    Performs quick "still X stage?" checks that are:
    - Fast (~100ms)
    - Cheap (~$0.0001 per call)
    - Simple (no world model context, no multi-turn)

    When uncertain, signals escalation to deep mode.
    """

    # Haiku model ID
    MODEL = "claude-3-5-haiku-latest"

    # Max tokens for monitor response (keep small for speed)
    MAX_TOKENS = 200

    def __init__(
        self,
        client: anthropic.Anthropic,
        context_manager: ContextManager,
        memory_manager: MemoryManager,
    ):
        """
        Parameters
        ----------
        client : anthropic.Anthropic
            Anthropic API client
        context_manager : ContextManager
            For building monitor context
        memory_manager : MemoryManager
            For parsing output and recording results
        """
        self.client = client
        self.context_manager = context_manager
        self.memory_manager = memory_manager

    def check(
        self,
        image_b64: str,
        world_model: WorldModel,
        embryo_id: str,
        timepoint: int,
        media_type: str = "image/jpeg",
    ) -> MonitorCheckResult:
        """
        Perform a lightweight stage confirmation check.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded image
        world_model : WorldModel
            Current world model (only stage is used)
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint
        media_type : str
            Image media type

        Returns
        -------
        MonitorCheckResult
            Result with escalation decision
        """
        start_time = time.time()

        # Build lightweight context
        context = self.context_manager.build_monitor_context(
            image_b64=image_b64,
            world_model=world_model,
            media_type=media_type,
        )

        # Call Haiku
        raw_output = self._call_haiku(context)

        # Parse output
        result = self.memory_manager.parse_monitor_output(raw_output)

        # Record result
        self.memory_manager.record_monitor_result(embryo_id, result, timepoint)

        # Determine escalation
        should_escalate, escalation_reason = self._should_escalate(result)

        duration_ms = int((time.time() - start_time) * 1000)

        return MonitorCheckResult(
            result=result,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
            duration_ms=duration_ms,
            raw_output=raw_output,
        )

    def _call_haiku(self, context: VLMContext) -> str:
        """Make API call to Haiku."""
        messages = context.to_anthropic_messages()

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=messages,
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def _should_escalate(self, result: MonitorResult) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should escalate to deep mode.

        Returns
        -------
        should_escalate : bool
            Whether to escalate
        reason : str or None
            Reason for escalation
        """
        # Uncertain confirmation -> escalate
        if result.confirmation == "uncertain":
            return True, f"Monitor uncertain: {result.brief_observation[:50]}..."

        # No confirmation (stage change detected) -> escalate
        if result.confirmation == "no":
            return True, f"Monitor detected change: {result.brief_observation[:50]}..."

        # Red flags -> escalate
        if result.red_flags:
            return True, f"Monitor flagged: {result.red_flags[0]}"

        # Low confidence -> escalate
        if result.confidence < 0.6:
            return True, f"Low monitor confidence: {result.confidence:.0%}"

        # All good -> no escalation
        return False, None


class MonitorHistory:
    """
    Tracks recent monitor results for pattern detection.

    Used by mode selector to detect declining confidence trends
    or repeated uncertainty.
    """

    def __init__(self, max_history: int = 10):
        """
        Parameters
        ----------
        max_history : int
            Maximum number of results to keep
        """
        self.max_history = max_history
        self._results: dict[str, list[MonitorResult]] = {}

    def add(self, embryo_id: str, result: MonitorResult) -> None:
        """Add a monitor result for an embryo."""
        if embryo_id not in self._results:
            self._results[embryo_id] = []

        self._results[embryo_id].append(result)

        # Trim to max history
        if len(self._results[embryo_id]) > self.max_history:
            self._results[embryo_id] = self._results[embryo_id][-self.max_history:]

    def get(self, embryo_id: str) -> list[MonitorResult]:
        """Get monitor history for an embryo."""
        return self._results.get(embryo_id, [])

    def clear(self, embryo_id: str) -> None:
        """Clear history for an embryo (e.g., after deep perception)."""
        if embryo_id in self._results:
            del self._results[embryo_id]

    def get_confidence_trend(self, embryo_id: str, n: int = 3) -> list[float]:
        """Get recent confidence values for an embryo."""
        results = self.get(embryo_id)
        if len(results) < n:
            return [r.confidence for r in results]
        return [r.confidence for r in results[-n:]]

    def has_recent_uncertainty(self, embryo_id: str) -> bool:
        """Check if recent monitors showed uncertainty."""
        results = self.get(embryo_id)
        if not results:
            return False
        return results[-1].confirmation in ("uncertain", "no")

    def is_confidence_declining(self, embryo_id: str, n: int = 3) -> bool:
        """Check if confidence is monotonically declining."""
        trend = self.get_confidence_trend(embryo_id, n)
        if len(trend) < n:
            return False
        return all(trend[i] > trend[i+1] for i in range(len(trend)-1))
