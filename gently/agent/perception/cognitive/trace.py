"""
Cognitive Trace Structures for Observability.

Every perception produces a traceable record for visualization
and debugging of the VLM's cognitive processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any


@dataclass
class WorldModelSnapshot:
    """Snapshot of world model state for tracing."""

    stage: str
    confidence: float
    stage_entry_observed: bool
    observations_in_stage: int
    time_in_stage_minutes: Optional[float]
    predictions: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "confidence": self.confidence,
            "stage_entry_observed": self.stage_entry_observed,
            "observations_in_stage": self.observations_in_stage,
            "time_in_stage_minutes": self.time_in_stage_minutes,
            "predictions": self.predictions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModelSnapshot":
        return cls(
            stage=data["stage"],
            confidence=data["confidence"],
            stage_entry_observed=data["stage_entry_observed"],
            observations_in_stage=data["observations_in_stage"],
            time_in_stage_minutes=data.get("time_in_stage_minutes"),
            predictions=data.get("predictions"),
        )

    @classmethod
    def from_world_model(cls, model: Any) -> "WorldModelSnapshot":
        """Create snapshot from a WorldModel instance."""
        return cls(
            stage=model.current_stage,
            confidence=model.stage_confidence,
            stage_entry_observed=model.stage_entry_observed,
            observations_in_stage=model.observations_in_stage,
            time_in_stage_minutes=None,  # Would need current timepoint
            predictions=model.predictions.to_dict() if model.predictions else None,
        )


@dataclass
class MonitorResult:
    """Result from lightweight monitoring."""

    confirmation: Literal["yes", "no", "uncertain"]
    confidence: float
    brief_observation: str  # 1-2 sentences
    red_flags: List[str] = field(default_factory=list)
    duration_ms: int = 0

    def should_escalate(self) -> bool:
        """Check if this result should trigger deep perception."""
        return self.confirmation in ("no", "uncertain")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirmation": self.confirmation,
            "confidence": self.confidence,
            "brief_observation": self.brief_observation,
            "red_flags": self.red_flags,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitorResult":
        return cls(
            confirmation=data["confirmation"],
            confidence=data["confidence"],
            brief_observation=data["brief_observation"],
            red_flags=data.get("red_flags", []),
            duration_ms=data.get("duration_ms", 0),
        )


@dataclass
class TurnTrace:
    """Trace of a single cognitive turn in multi-turn processing."""

    turn_number: int
    operation: Literal[
        "initial",        # First observation
        "monitor",        # Lightweight check
        "single_turn",    # Single deep assessment
        "escalation",     # Escalation from monitor
        "perceive",       # Multi-turn: unbiased observation
        "compare",        # Multi-turn: compare to model
        "verify",         # Multi-turn: verify with references
        "integrate",      # Multi-turn: final integration
        "reject_backward" # Backward jump rejection
    ]

    # Context summary (human-readable)
    context_summary: str
    task_prompt: str

    # Output
    output_summary: str
    raw_output: str

    # References shown (for verify turn)
    references_shown: Optional[List[str]] = None

    # Timing
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_number": self.turn_number,
            "operation": self.operation,
            "context_summary": self.context_summary,
            "task_prompt": self.task_prompt,
            "output_summary": self.output_summary,
            "raw_output": self.raw_output,
            "references_shown": self.references_shown,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnTrace":
        return cls(
            turn_number=data["turn_number"],
            operation=data["operation"],
            context_summary=data["context_summary"],
            task_prompt=data["task_prompt"],
            output_summary=data["output_summary"],
            raw_output=data["raw_output"],
            references_shown=data.get("references_shown"),
            duration_ms=data.get("duration_ms", 0),
        )


@dataclass
class CognitiveTrace:
    """Full trace of cognitive processing for visualization."""

    # Identification
    embryo_id: str
    timepoint: int
    timestamp: datetime

    # Processing mode
    mode: Literal["monitor", "single_turn", "multi_turn"]
    num_turns: int
    total_duration_ms: int

    # Surprise assessment (only for deep modes)
    surprise_level: Optional[Literal["none", "low", "medium", "high", "anomaly"]] = None
    surprise_reason: Optional[str] = None

    # World model snapshots
    prior_model: Optional[WorldModelSnapshot] = None
    posterior_model: Optional[WorldModelSnapshot] = None

    # Turn-by-turn trace (for multi-turn)
    turns: List[TurnTrace] = field(default_factory=list)

    # Monitor result (for monitor mode)
    monitor_result: Optional[MonitorResult] = None

    # Decision summary
    final_stage: str = ""
    final_confidence: float = 0.0
    stage_changed: bool = False
    backward_rejected: bool = False
    narrative_update: str = ""

    # Cost tracking
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_event_payload(self) -> Dict[str, Any]:
        """Format for DETECTOR_EVALUATED event."""
        return {
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "stage": self.final_stage,
            "confidence": self.final_confidence,
            "cognitive_trace": {
                "mode": self.mode,
                "num_turns": self.num_turns,
                "duration_ms": self.total_duration_ms,
                "surprise_level": self.surprise_level,
                "surprise_reason": self.surprise_reason,
                "world_model": {
                    "prior": self.prior_model.to_dict() if self.prior_model else None,
                    "posterior": self.posterior_model.to_dict() if self.posterior_model else None,
                },
                "turns": [t.to_dict() for t in self.turns],
                "monitor_result": self.monitor_result.to_dict() if self.monitor_result else None,
                "decision": {
                    "stage_changed": self.stage_changed,
                    "backward_rejected": self.backward_rejected,
                    "confidence_delta": (
                        self.posterior_model.confidence - self.prior_model.confidence
                        if self.prior_model and self.posterior_model
                        else 0.0
                    ),
                },
                "cost": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "estimated_cost_usd": self.estimated_cost_usd,
                },
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode,
            "num_turns": self.num_turns,
            "total_duration_ms": self.total_duration_ms,
            "surprise_level": self.surprise_level,
            "surprise_reason": self.surprise_reason,
            "prior_model": self.prior_model.to_dict() if self.prior_model else None,
            "posterior_model": self.posterior_model.to_dict() if self.posterior_model else None,
            "turns": [t.to_dict() for t in self.turns],
            "monitor_result": self.monitor_result.to_dict() if self.monitor_result else None,
            "final_stage": self.final_stage,
            "final_confidence": self.final_confidence,
            "stage_changed": self.stage_changed,
            "backward_rejected": self.backward_rejected,
            "narrative_update": self.narrative_update,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveTrace":
        return cls(
            embryo_id=data["embryo_id"],
            timepoint=data["timepoint"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            mode=data["mode"],
            num_turns=data["num_turns"],
            total_duration_ms=data["total_duration_ms"],
            surprise_level=data.get("surprise_level"),
            surprise_reason=data.get("surprise_reason"),
            prior_model=WorldModelSnapshot.from_dict(data["prior_model"]) if data.get("prior_model") else None,
            posterior_model=WorldModelSnapshot.from_dict(data["posterior_model"]) if data.get("posterior_model") else None,
            turns=[TurnTrace.from_dict(t) for t in data.get("turns", [])],
            monitor_result=MonitorResult.from_dict(data["monitor_result"]) if data.get("monitor_result") else None,
            final_stage=data.get("final_stage", ""),
            final_confidence=data.get("final_confidence", 0.0),
            stage_changed=data.get("stage_changed", False),
            backward_rejected=data.get("backward_rejected", False),
            narrative_update=data.get("narrative_update", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            estimated_cost_usd=data.get("estimated_cost_usd", 0.0),
        )

    @classmethod
    def create_monitor_trace(
        cls,
        embryo_id: str,
        timepoint: int,
        prior_model: WorldModelSnapshot,
        monitor_result: MonitorResult,
        duration_ms: int,
    ) -> "CognitiveTrace":
        """Create trace for monitor mode."""
        return cls(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode="monitor",
            num_turns=1,
            total_duration_ms=duration_ms,
            prior_model=prior_model,
            posterior_model=prior_model,  # No change in monitor mode
            monitor_result=monitor_result,
            final_stage=prior_model.stage,
            final_confidence=monitor_result.confidence,
            stage_changed=False,
            backward_rejected=False,
        )

    @classmethod
    def create_single_turn_trace(
        cls,
        embryo_id: str,
        timepoint: int,
        prior_model: WorldModelSnapshot,
        posterior_model: WorldModelSnapshot,
        surprise_level: str,
        surprise_reason: str,
        duration_ms: int,
        stage_changed: bool,
        narrative_update: str = "",
    ) -> "CognitiveTrace":
        """Create trace for single-turn deep perception."""
        return cls(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode="single_turn",
            num_turns=1,
            total_duration_ms=duration_ms,
            surprise_level=surprise_level,
            surprise_reason=surprise_reason,
            prior_model=prior_model,
            posterior_model=posterior_model,
            final_stage=posterior_model.stage,
            final_confidence=posterior_model.confidence,
            stage_changed=stage_changed,
            narrative_update=narrative_update,
        )

    @classmethod
    def create_multi_turn_trace(
        cls,
        embryo_id: str,
        timepoint: int,
        prior_model: WorldModelSnapshot,
        posterior_model: WorldModelSnapshot,
        turns: List[TurnTrace],
        surprise_level: str,
        surprise_reason: str,
        stage_changed: bool,
        backward_rejected: bool = False,
        narrative_update: str = "",
    ) -> "CognitiveTrace":
        """Create trace for multi-turn deep perception."""
        total_duration = sum(t.duration_ms for t in turns)
        return cls(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode="multi_turn",
            num_turns=len(turns),
            total_duration_ms=total_duration,
            surprise_level=surprise_level,
            surprise_reason=surprise_reason,
            prior_model=prior_model,
            posterior_model=posterior_model,
            turns=turns,
            final_stage=posterior_model.stage,
            final_confidence=posterior_model.confidence,
            stage_changed=stage_changed,
            backward_rejected=backward_rejected,
            narrative_update=narrative_update,
        )


@dataclass
class BeliefSnapshot:
    """Minimal snapshot for belief evolution charting."""

    timepoint: int
    timestamp: datetime
    stage: str
    confidence: float
    mode: Literal["monitor", "single_turn", "multi_turn"]
    stage_changed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timepoint": self.timepoint,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "confidence": self.confidence,
            "mode": self.mode,
            "stage_changed": self.stage_changed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefSnapshot":
        return cls(
            timepoint=data["timepoint"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            stage=data["stage"],
            confidence=data["confidence"],
            mode=data["mode"],
            stage_changed=data["stage_changed"],
        )

    @classmethod
    def from_cognitive_trace(cls, trace: CognitiveTrace) -> "BeliefSnapshot":
        """Create belief snapshot from cognitive trace."""
        return cls(
            timepoint=trace.timepoint,
            timestamp=trace.timestamp,
            stage=trace.final_stage,
            confidence=trace.final_confidence,
            mode=trace.mode,
            stage_changed=trace.stage_changed,
        )
