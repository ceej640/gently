"""
Memory Structures for Embryo Perception.

Narratives compress observation history for efficient context loading.
They serve as external memory that persists across VLM calls.

Structure:
    BatchNarrative (experiment-wide)
        └── EmbryoNarrative (per embryo)
                ├── WorldModel (current beliefs)
                ├── StageNarrative[] (per stage visited)
                └── TransitionNarrative[] (per observed transition)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class TimestampedSummary:
    """A summary with its timestamp."""

    timestamp: datetime
    timepoint: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "timepoint": self.timepoint,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimestampedSummary":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            timepoint=data["timepoint"],
            summary=data["summary"],
        )


@dataclass
class StageMetrics:
    """Quantitative metrics for a stage."""

    observation_count: int
    duration_minutes: Optional[float]  # None if entry not observed

    # Confidence tracking
    confidence_trend: List[float] = field(default_factory=list)
    min_confidence: float = 1.0
    max_confidence: float = 0.0
    avg_confidence: float = 0.0

    # Feature tracking
    feature_observations: List[str] = field(default_factory=list)

    def update_from_observation(self, confidence: float, features: Optional[List[str]] = None) -> None:
        """Update metrics with new observation."""
        self.observation_count += 1
        self.confidence_trend.append(confidence)
        self.min_confidence = min(self.min_confidence, confidence)
        self.max_confidence = max(self.max_confidence, confidence)
        self.avg_confidence = sum(self.confidence_trend) / len(self.confidence_trend)

        if features:
            self.feature_observations.extend(features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "duration_minutes": self.duration_minutes,
            "confidence_trend": self.confidence_trend,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "avg_confidence": self.avg_confidence,
            "feature_observations": self.feature_observations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageMetrics":
        return cls(
            observation_count=data["observation_count"],
            duration_minutes=data.get("duration_minutes"),
            confidence_trend=data.get("confidence_trend", []),
            min_confidence=data.get("min_confidence", 1.0),
            max_confidence=data.get("max_confidence", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            feature_observations=data.get("feature_observations", []),
        )

    @classmethod
    def create_empty(cls) -> "StageMetrics":
        return cls(observation_count=0, duration_minutes=None)


@dataclass
class StageNarrative:
    """Compressed narrative for time spent in a stage."""

    stage_id: str  # Unique ID like "embryo_A_comma_002"
    embryo_id: str
    stage: str

    # Temporal bounds
    entered_at: Optional[datetime]  # None if entry not observed
    exited_at: Optional[datetime]
    entry_observed: bool

    # Observation tracking
    observation_timepoints: List[int] = field(default_factory=list)
    first_timepoint: int = 0
    last_timepoint: int = 0

    # Metrics
    metrics: StageMetrics = field(default_factory=StageMetrics.create_empty)

    # LLM-generated summaries
    entry_summary: Optional[str] = None
    progress_summaries: List[TimestampedSummary] = field(default_factory=list)
    exit_summary: Optional[str] = None

    def add_observation(
        self,
        timepoint: int,
        confidence: float,
        features: Optional[List[str]] = None,
    ) -> None:
        """Record a new observation."""
        self.observation_timepoints.append(timepoint)
        self.last_timepoint = timepoint
        self.metrics.update_from_observation(confidence, features)

    def add_progress_summary(self, timepoint: int, summary: str) -> None:
        """Add a periodic progress summary."""
        self.progress_summaries.append(
            TimestampedSummary(
                timestamp=datetime.now(),
                timepoint=timepoint,
                summary=summary,
            )
        )

    def to_context_text(self) -> str:
        """Format for inclusion in VLM context."""
        lines = [f"Stage: {self.stage.upper()}"]

        if self.entry_observed and self.entered_at:
            lines.append(f"  Entry: Observed at T{self.first_timepoint}")
        else:
            lines.append(f"  Entry: Not observed (started mid-stage at T{self.first_timepoint})")

        lines.append(f"  Observations: {self.metrics.observation_count}")
        lines.append(f"  Confidence: avg {self.metrics.avg_confidence:.0%} (range {self.metrics.min_confidence:.0%}-{self.metrics.max_confidence:.0%})")

        if self.entry_summary:
            lines.append(f"  Entry notes: {self.entry_summary}")

        if self.progress_summaries:
            latest = self.progress_summaries[-1]
            lines.append(f"  Latest summary (T{latest.timepoint}): {latest.summary}")

        if self.exit_summary:
            lines.append(f"  Exit notes: {self.exit_summary}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "embryo_id": self.embryo_id,
            "stage": self.stage,
            "entered_at": self.entered_at.isoformat() if self.entered_at else None,
            "exited_at": self.exited_at.isoformat() if self.exited_at else None,
            "entry_observed": self.entry_observed,
            "observation_timepoints": self.observation_timepoints,
            "first_timepoint": self.first_timepoint,
            "last_timepoint": self.last_timepoint,
            "metrics": self.metrics.to_dict(),
            "entry_summary": self.entry_summary,
            "progress_summaries": [s.to_dict() for s in self.progress_summaries],
            "exit_summary": self.exit_summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageNarrative":
        return cls(
            stage_id=data["stage_id"],
            embryo_id=data["embryo_id"],
            stage=data["stage"],
            entered_at=datetime.fromisoformat(data["entered_at"]) if data.get("entered_at") else None,
            exited_at=datetime.fromisoformat(data["exited_at"]) if data.get("exited_at") else None,
            entry_observed=data["entry_observed"],
            observation_timepoints=data.get("observation_timepoints", []),
            first_timepoint=data.get("first_timepoint", 0),
            last_timepoint=data.get("last_timepoint", 0),
            metrics=StageMetrics.from_dict(data["metrics"]) if data.get("metrics") else StageMetrics.create_empty(),
            entry_summary=data.get("entry_summary"),
            progress_summaries=[TimestampedSummary.from_dict(s) for s in data.get("progress_summaries", [])],
            exit_summary=data.get("exit_summary"),
        )

    @classmethod
    def create_initial(
        cls,
        embryo_id: str,
        stage: str,
        timepoint: int,
        confidence: float,
        entry_observed: bool = False,
    ) -> "StageNarrative":
        """Create initial stage narrative."""
        stage_num = 1  # First stage
        stage_id = f"{embryo_id}_{stage}_{stage_num:03d}"

        narrative = cls(
            stage_id=stage_id,
            embryo_id=embryo_id,
            stage=stage,
            entered_at=datetime.now() if entry_observed else None,
            exited_at=None,
            entry_observed=entry_observed,
            observation_timepoints=[timepoint],
            first_timepoint=timepoint,
            last_timepoint=timepoint,
            metrics=StageMetrics.create_empty(),
        )
        narrative.metrics.update_from_observation(confidence)
        return narrative


@dataclass
class TransitionNarrative:
    """Record of a stage transition."""

    transition_id: str  # Unique ID like "embryo_A_bean_to_comma_001"
    embryo_id: str

    from_stage: str
    to_stage: str

    # Timing
    started_at: datetime
    completed_at: datetime
    trigger_timepoint: int
    confirmation_timepoint: int

    # Evidence
    key_evidence: List[str] = field(default_factory=list)
    transitional_timepoints: List[int] = field(default_factory=list)

    # Summary
    summary: str = ""

    def to_context_text(self) -> str:
        """Format for inclusion in VLM context."""
        lines = [f"Transition: {self.from_stage} -> {self.to_stage}"]
        lines.append(f"  At timepoint: T{self.trigger_timepoint}")
        if self.key_evidence:
            lines.append(f"  Key evidence: {', '.join(self.key_evidence[:3])}")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "embryo_id": self.embryo_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "trigger_timepoint": self.trigger_timepoint,
            "confirmation_timepoint": self.confirmation_timepoint,
            "key_evidence": self.key_evidence,
            "transitional_timepoints": self.transitional_timepoints,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionNarrative":
        return cls(
            transition_id=data["transition_id"],
            embryo_id=data["embryo_id"],
            from_stage=data["from_stage"],
            to_stage=data["to_stage"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            trigger_timepoint=data["trigger_timepoint"],
            confirmation_timepoint=data["confirmation_timepoint"],
            key_evidence=data.get("key_evidence", []),
            transitional_timepoints=data.get("transitional_timepoints", []),
            summary=data.get("summary", ""),
        )

    @classmethod
    def create(
        cls,
        embryo_id: str,
        from_stage: str,
        to_stage: str,
        timepoint: int,
        key_evidence: List[str],
        transition_number: int = 1,
    ) -> "TransitionNarrative":
        """Create a new transition narrative."""
        now = datetime.now()
        transition_id = f"{embryo_id}_{from_stage}_to_{to_stage}_{transition_number:03d}"

        return cls(
            transition_id=transition_id,
            embryo_id=embryo_id,
            from_stage=from_stage,
            to_stage=to_stage,
            started_at=now,
            completed_at=now,
            trigger_timepoint=timepoint,
            confirmation_timepoint=timepoint,
            key_evidence=key_evidence,
            transitional_timepoints=[timepoint],
            summary=f"Transitioned from {from_stage} to {to_stage} at T{timepoint}",
        )


@dataclass
class EmbryoNarrative:
    """Complete narrative for an embryo across its development."""

    embryo_id: str
    created_at: datetime
    updated_at: datetime

    # Current status
    current_stage: str
    is_terminal: bool  # hatched or arrested

    # Stage histories
    stage_narratives: List[StageNarrative] = field(default_factory=list)
    transition_narratives: List[TransitionNarrative] = field(default_factory=list)

    # Overall trajectory
    stages_visited: List[str] = field(default_factory=list)
    total_observations: int = 0

    # High-level summaries
    lifecycle_summary: Optional[str] = None
    notable_events: List[str] = field(default_factory=list)

    def get_current_stage_narrative(self) -> Optional[StageNarrative]:
        """Get the narrative for the current stage."""
        if not self.stage_narratives:
            return None
        return self.stage_narratives[-1]

    def add_stage_narrative(self, narrative: StageNarrative) -> None:
        """Add a new stage narrative."""
        self.stage_narratives.append(narrative)
        if narrative.stage not in self.stages_visited:
            self.stages_visited.append(narrative.stage)
        self.updated_at = datetime.now()

    def add_transition_narrative(self, narrative: TransitionNarrative) -> None:
        """Add a new transition narrative."""
        self.transition_narratives.append(narrative)
        self.updated_at = datetime.now()

    def to_context_text(self, include_full_history: bool = False) -> str:
        """Format for inclusion in VLM context."""
        lines = [f"EMBRYO {self.embryo_id} NARRATIVE:"]
        lines.append(f"  Trajectory: {' -> '.join(self.stages_visited)}")
        lines.append(f"  Total observations: {self.total_observations}")

        if self.is_terminal:
            lines.append(f"  Status: TERMINAL ({self.current_stage})")
        else:
            lines.append(f"  Status: Active in {self.current_stage}")

        # Current stage details
        current = self.get_current_stage_narrative()
        if current:
            lines.append("")
            lines.append(current.to_context_text())

        # Recent transitions
        if self.transition_narratives:
            lines.append("")
            lines.append("Recent transitions:")
            for trans in self.transition_narratives[-3:]:
                lines.append(f"  - {trans.from_stage} -> {trans.to_stage} at T{trans.trigger_timepoint}")

        if include_full_history:
            lines.append("")
            lines.append("Full stage history:")
            for stage_narr in self.stage_narratives:
                lines.append(stage_narr.to_context_text())

        if self.lifecycle_summary:
            lines.append("")
            lines.append(f"Summary: {self.lifecycle_summary}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embryo_id": self.embryo_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "current_stage": self.current_stage,
            "is_terminal": self.is_terminal,
            "stage_narratives": [s.to_dict() for s in self.stage_narratives],
            "transition_narratives": [t.to_dict() for t in self.transition_narratives],
            "stages_visited": self.stages_visited,
            "total_observations": self.total_observations,
            "lifecycle_summary": self.lifecycle_summary,
            "notable_events": self.notable_events,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbryoNarrative":
        return cls(
            embryo_id=data["embryo_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            current_stage=data["current_stage"],
            is_terminal=data["is_terminal"],
            stage_narratives=[StageNarrative.from_dict(s) for s in data.get("stage_narratives", [])],
            transition_narratives=[TransitionNarrative.from_dict(t) for t in data.get("transition_narratives", [])],
            stages_visited=data.get("stages_visited", []),
            total_observations=data.get("total_observations", 0),
            lifecycle_summary=data.get("lifecycle_summary"),
            notable_events=data.get("notable_events", []),
        )

    @classmethod
    def create_initial(
        cls,
        embryo_id: str,
        initial_stage: str,
        timepoint: int,
        confidence: float,
    ) -> "EmbryoNarrative":
        """Create initial embryo narrative."""
        now = datetime.now()

        # Create initial stage narrative
        stage_narrative = StageNarrative.create_initial(
            embryo_id=embryo_id,
            stage=initial_stage,
            timepoint=timepoint,
            confidence=confidence,
            entry_observed=False,  # First observation, entry not observed
        )

        return cls(
            embryo_id=embryo_id,
            created_at=now,
            updated_at=now,
            current_stage=initial_stage,
            is_terminal=False,
            stage_narratives=[stage_narrative],
            transition_narratives=[],
            stages_visited=[initial_stage],
            total_observations=1,
            lifecycle_summary=None,
            notable_events=[],
        )


@dataclass
class BatchNarrative:
    """Cross-embryo patterns for the batch."""

    batch_id: str
    started_at: datetime
    updated_at: datetime

    # Embryo tracking
    embryo_ids: List[str] = field(default_factory=list)
    stage_distribution: Dict[str, int] = field(default_factory=dict)  # stage -> count

    # Status counts
    active_count: int = 0
    hatched_count: int = 0
    arrested_count: int = 0

    # Summaries
    population_summary: Optional[str] = None
    outlier_notes: List[str] = field(default_factory=list)

    def update_stage_distribution(self, embryo_stages: Dict[str, str]) -> None:
        """Update stage distribution from current embryo stages."""
        self.stage_distribution = {}
        for stage in embryo_stages.values():
            self.stage_distribution[stage] = self.stage_distribution.get(stage, 0) + 1
        self.updated_at = datetime.now()

    def update_status_counts(
        self,
        active: int,
        hatched: int,
        arrested: int,
    ) -> None:
        """Update status counts."""
        self.active_count = active
        self.hatched_count = hatched
        self.arrested_count = arrested
        self.updated_at = datetime.now()

    def to_context_text(self) -> str:
        """Format for inclusion in VLM context."""
        lines = [f"BATCH {self.batch_id} OVERVIEW:"]
        lines.append(f"  Total embryos: {len(self.embryo_ids)}")
        lines.append(f"  Active: {self.active_count} | Hatched: {self.hatched_count} | Arrested: {self.arrested_count}")

        if self.stage_distribution:
            dist_parts = [f"{stage}: {count}" for stage, count in sorted(self.stage_distribution.items())]
            lines.append(f"  Stage distribution: {', '.join(dist_parts)}")

        if self.population_summary:
            lines.append(f"  Summary: {self.population_summary}")

        if self.outlier_notes:
            lines.append("  Outliers:")
            for note in self.outlier_notes[-3:]:
                lines.append(f"    - {note}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embryo_ids": self.embryo_ids,
            "stage_distribution": self.stage_distribution,
            "active_count": self.active_count,
            "hatched_count": self.hatched_count,
            "arrested_count": self.arrested_count,
            "population_summary": self.population_summary,
            "outlier_notes": self.outlier_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchNarrative":
        return cls(
            batch_id=data["batch_id"],
            started_at=datetime.fromisoformat(data["started_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            embryo_ids=data.get("embryo_ids", []),
            stage_distribution=data.get("stage_distribution", {}),
            active_count=data.get("active_count", 0),
            hatched_count=data.get("hatched_count", 0),
            arrested_count=data.get("arrested_count", 0),
            population_summary=data.get("population_summary"),
            outlier_notes=data.get("outlier_notes", []),
        )

    @classmethod
    def create_initial(cls, batch_id: str) -> "BatchNarrative":
        """Create initial batch narrative."""
        now = datetime.now()
        return cls(
            batch_id=batch_id,
            started_at=now,
            updated_at=now,
        )
