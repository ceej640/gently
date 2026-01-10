"""
World Model for Embryo Perception.

The world model represents the VLM's beliefs about each embryo.
It is loaded into context for each cognitive operation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any

from ..stages import DevelopmentalStage, STAGE_CRITERIA


@dataclass
class TransitionRecord:
    """Record of an observed stage transition."""

    from_stage: str
    to_stage: str
    timepoint: int
    confidence: float
    key_evidence: List[str]
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "timepoint": self.timepoint,
            "confidence": self.confidence,
            "key_evidence": self.key_evidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransitionRecord":
        """Deserialize from dictionary."""
        return cls(
            from_stage=data["from_stage"],
            to_stage=data["to_stage"],
            timepoint=data["timepoint"],
            confidence=data["confidence"],
            key_evidence=data["key_evidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
        )


@dataclass
class Predictions:
    """What the VLM should expect for the next observation."""

    mode: Literal["feature_based", "time_based"]
    timing_available: bool

    # Always available
    expected_stage_distribution: Dict[str, float]  # e.g., {"comma": 0.7, "1.5fold": 0.3}
    expected_features: List[str]
    transition_markers: List[str]  # What to watch for

    # Only if timing_available
    progress_ratio: Optional[float] = None  # time_in_stage / expected_duration
    transition_probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": self.mode,
            "timing_available": self.timing_available,
            "expected_stage_distribution": self.expected_stage_distribution,
            "expected_features": self.expected_features,
            "transition_markers": self.transition_markers,
            "progress_ratio": self.progress_ratio,
            "transition_probability": self.transition_probability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Predictions":
        """Deserialize from dictionary."""
        return cls(
            mode=data["mode"],
            timing_available=data["timing_available"],
            expected_stage_distribution=data["expected_stage_distribution"],
            expected_features=data["expected_features"],
            transition_markers=data["transition_markers"],
            progress_ratio=data.get("progress_ratio"),
            transition_probability=data.get("transition_probability"),
        )

    @classmethod
    def empty(cls) -> "Predictions":
        """Create an empty predictions object."""
        return cls(
            mode="feature_based",
            timing_available=False,
            expected_stage_distribution={},
            expected_features=[],
            transition_markers=[],
            progress_ratio=None,
            transition_probability=None,
        )

    def to_context_text(self) -> str:
        """Format predictions for inclusion in VLM context."""
        lines = ["PREDICTIONS:"]
        lines.append(f"- Expected features: {', '.join(self.expected_features)}")
        lines.append(f"- Watch for: {', '.join(self.transition_markers)}")

        if self.timing_available and self.progress_ratio is not None:
            lines.append(f"- Stage progress: {self.progress_ratio:.0%} of expected duration")
            lines.append(f"- Transition probability: {self.transition_probability:.0%}")
        else:
            lines.append("- Timing: NOT AVAILABLE (entry into stage not observed)")

        # Stage distribution
        dist_parts = [f"{stage}: {prob:.0%}" for stage, prob in self.expected_stage_distribution.items()]
        lines.append(f"- Expected stage: {' | '.join(dist_parts)}")

        return "\n".join(lines)


@dataclass
class WorldModel:
    """
    The VLM's beliefs about an embryo - loaded into context each call.

    Key Design: Entry Observation Flag
    Since experiments start with embryos at various stages, we cannot know
    how long an embryo has been in its initial stage. The world model tracks
    this explicitly with `stage_entry_observed`.
    """

    # Identification
    embryo_id: str

    # Current beliefs
    current_stage: str
    stage_confidence: float

    # Entry tracking - CRITICAL for timing reliability
    stage_entry_observed: bool  # Did we SEE the transition INTO this stage?
    entered_at_timepoint: Optional[int]  # None if entry not observed

    # What we can always track (regardless of entry observation)
    observations_in_stage: int
    first_observation_timepoint: int
    last_observation_timepoint: int
    confidence_trend: List[float] = field(default_factory=list)
    feature_observations: List[str] = field(default_factory=list)

    # Trajectory (only includes transitions we observed)
    stages_visited: List[str] = field(default_factory=list)
    transitions_observed: List[TransitionRecord] = field(default_factory=list)

    # Predictions for next observation
    predictions: Optional[Predictions] = None

    # Monitor mode tracking
    monitors_since_last_deep: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def time_in_stage_minutes(
        self, current_timepoint: int, interval_minutes: float
    ) -> Optional[float]:
        """
        Calculate time in current stage.

        Only returns time if we observed entry. None otherwise.
        """
        if not self.stage_entry_observed or self.entered_at_timepoint is None:
            return None
        return (current_timepoint - self.entered_at_timepoint) * interval_minutes

    def observations_since_entry(self) -> Optional[int]:
        """Number of observations since stage entry. None if entry not observed."""
        if not self.stage_entry_observed or self.entered_at_timepoint is None:
            return None
        return self.last_observation_timepoint - self.entered_at_timepoint

    def to_context_text(
        self,
        current_timepoint: Optional[int] = None,
        interval_minutes: float = 5.0,
    ) -> str:
        """Format for inclusion in VLM context."""
        lines = ["CURRENT WORLD MODEL:"]
        lines.append(
            f"- Believed stage: {self.current_stage} (confidence: {self.stage_confidence:.0%})"
        )

        if self.stage_entry_observed and current_timepoint is not None:
            time = self.time_in_stage_minutes(current_timepoint, interval_minutes)
            if time is not None:
                lines.append(
                    f"- Time in stage: {time:.0f} minutes (entry observed at T{self.entered_at_timepoint})"
                )
        else:
            lines.append("- Time in stage: UNKNOWN (imaging started mid-stage)")
            lines.append(f"- Observations in stage: {self.observations_in_stage}")

        lines.append(f"- Trajectory: {' -> '.join(self.stages_visited)}")

        # Recent confidence trend
        if len(self.confidence_trend) >= 3:
            recent = self.confidence_trend[-3:]
            trend_str = " -> ".join([f"{c:.0%}" for c in recent])
            lines.append(f"- Recent confidence: {trend_str}")

        # Predictions
        if self.predictions:
            lines.append("")
            lines.append(self.predictions.to_context_text())

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "embryo_id": self.embryo_id,
            "current_stage": self.current_stage,
            "stage_confidence": self.stage_confidence,
            "stage_entry_observed": self.stage_entry_observed,
            "entered_at_timepoint": self.entered_at_timepoint,
            "observations_in_stage": self.observations_in_stage,
            "first_observation_timepoint": self.first_observation_timepoint,
            "last_observation_timepoint": self.last_observation_timepoint,
            "confidence_trend": self.confidence_trend,
            "feature_observations": self.feature_observations,
            "stages_visited": self.stages_visited,
            "transitions_observed": [t.to_dict() for t in self.transitions_observed],
            "predictions": self.predictions.to_dict() if self.predictions else None,
            "monitors_since_last_deep": self.monitors_since_last_deep,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldModel":
        """Deserialize from dictionary."""
        return cls(
            embryo_id=data["embryo_id"],
            current_stage=data["current_stage"],
            stage_confidence=data["stage_confidence"],
            stage_entry_observed=data["stage_entry_observed"],
            entered_at_timepoint=data.get("entered_at_timepoint"),
            observations_in_stage=data["observations_in_stage"],
            first_observation_timepoint=data["first_observation_timepoint"],
            last_observation_timepoint=data["last_observation_timepoint"],
            confidence_trend=data.get("confidence_trend", []),
            feature_observations=data.get("feature_observations", []),
            stages_visited=data.get("stages_visited", []),
            transitions_observed=[
                TransitionRecord.from_dict(t) for t in data.get("transitions_observed", [])
            ],
            predictions=Predictions.from_dict(data["predictions"]) if data.get("predictions") else None,
            monitors_since_last_deep=data.get("monitors_since_last_deep", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
        )

    @classmethod
    def create_initial(
        cls,
        embryo_id: str,
        initial_stage: str,
        initial_confidence: float,
        timepoint: int,
    ) -> "WorldModel":
        """
        Create initial world model for first observation.

        Note: stage_entry_observed is False because we didn't see the
        transition INTO this stage - imaging started mid-stage.
        """
        now = datetime.now()
        return cls(
            embryo_id=embryo_id,
            current_stage=initial_stage,
            stage_confidence=initial_confidence,
            stage_entry_observed=False,  # We didn't see entry
            entered_at_timepoint=None,
            observations_in_stage=1,
            first_observation_timepoint=timepoint,
            last_observation_timepoint=timepoint,
            confidence_trend=[initial_confidence],
            feature_observations=[],
            stages_visited=[initial_stage],
            transitions_observed=[],
            predictions=None,  # Will be generated after creation
            monitors_since_last_deep=0,
            created_at=now,
            updated_at=now,
        )

    def record_observation(
        self,
        timepoint: int,
        confidence: float,
        features: Optional[List[str]] = None,
    ) -> None:
        """Record a new observation within the current stage."""
        self.observations_in_stage += 1
        self.last_observation_timepoint = timepoint
        self.stage_confidence = confidence
        self.confidence_trend.append(confidence)
        self.updated_at = datetime.now()

        if features:
            self.feature_observations.extend(features)

    def record_transition(
        self,
        new_stage: str,
        timepoint: int,
        confidence: float,
        key_evidence: List[str],
    ) -> TransitionRecord:
        """
        Record a stage transition.

        After this, stage_entry_observed becomes True for the new stage
        because we just observed the transition.
        """
        transition = TransitionRecord(
            from_stage=self.current_stage,
            to_stage=new_stage,
            timepoint=timepoint,
            confidence=confidence,
            key_evidence=key_evidence,
            timestamp=datetime.now(),
        )

        # Update state for new stage
        self.current_stage = new_stage
        self.stage_confidence = confidence
        self.stage_entry_observed = True  # NOW we observed entry!
        self.entered_at_timepoint = timepoint
        self.observations_in_stage = 1
        self.confidence_trend = [confidence]
        self.feature_observations = []
        self.stages_visited.append(new_stage)
        self.transitions_observed.append(transition)
        self.monitors_since_last_deep = 0
        self.updated_at = datetime.now()

        return transition

    def reset_deep_counter(self) -> None:
        """Reset monitors_since_last_deep after a deep perception."""
        self.monitors_since_last_deep = 0

    def increment_monitor_counter(self) -> None:
        """Increment monitors_since_last_deep after a monitor check."""
        self.monitors_since_last_deep += 1


def get_next_stage(stage: str) -> Optional[str]:
    """Get the next developmental stage."""
    ordered = DevelopmentalStage.ordered_values()
    try:
        idx = ordered.index(stage)
        if idx < len(ordered) - 1:
            return ordered[idx + 1]
        return None
    except ValueError:
        return None


def get_stage_features(stage: str) -> List[str]:
    """Get the expected features for a stage."""
    criteria = STAGE_CRITERIA.get(stage, {})
    return criteria.get("features", [])


def get_transition_markers(from_stage: str, to_stage: str) -> List[str]:
    """Get markers indicating transition between stages."""
    # Get NOT_if markers from current stage (indicators of next stage)
    current_criteria = STAGE_CRITERIA.get(from_stage, {})
    not_if = current_criteria.get("NOT_if", [])

    # Get features of next stage
    next_criteria = STAGE_CRITERIA.get(to_stage, {})
    next_features = next_criteria.get("features", [])

    # Combine: transition markers are things that would indicate we're moving forward
    markers = []
    markers.extend(not_if[:2])  # First two NOT_if conditions
    markers.extend(next_features[:2])  # First two features of next stage

    return markers


def get_expected_duration(stage: str) -> Optional[float]:
    """Get expected duration in minutes for a stage."""
    criteria = STAGE_CRITERIA.get(stage, {})
    return criteria.get("typical_duration_min")


def generate_predictions(
    model: WorldModel,
    current_timepoint: int,
    interval_minutes: float = 5.0,
) -> Predictions:
    """
    Generate predictions based on current beliefs.

    If stage_entry_observed is False, we can only make feature-based predictions.
    If True, we can add timing-based predictions.
    """
    current = model.current_stage
    next_stage = get_next_stage(current)

    # Get features and markers
    expected_features = get_stage_features(current)
    if next_stage:
        transition_markers = get_transition_markers(current, next_stage)
    else:
        transition_markers = []

    if not model.stage_entry_observed:
        # FEATURE-BASED MODE: No timing predictions
        # We don't know when embryo entered this stage
        stage_dist = {current: 0.8}
        if next_stage:
            stage_dist[next_stage] = 0.2

        return Predictions(
            mode="feature_based",
            timing_available=False,
            expected_stage_distribution=stage_dist,
            expected_features=expected_features,
            transition_markers=transition_markers,
            progress_ratio=None,
            transition_probability=None,
        )

    # TIME-BASED MODE: Full predictions with timing
    time_in_stage = model.time_in_stage_minutes(current_timepoint, interval_minutes)
    expected_duration = get_expected_duration(current)

    if time_in_stage is None or expected_duration is None:
        # Fallback to feature-based
        stage_dist = {current: 0.8}
        if next_stage:
            stage_dist[next_stage] = 0.2

        return Predictions(
            mode="feature_based",
            timing_available=False,
            expected_stage_distribution=stage_dist,
            expected_features=expected_features,
            transition_markers=transition_markers,
            progress_ratio=None,
            transition_probability=None,
        )

    progress = time_in_stage / expected_duration

    # Transition probability increases as we approach/exceed expected duration
    if progress < 0.5:
        trans_prob = 0.05
        stage_dist = {current: 0.95}
        if next_stage:
            stage_dist[next_stage] = 0.05
    elif progress < 0.7:
        trans_prob = 0.15
        stage_dist = {current: 0.85}
        if next_stage:
            stage_dist[next_stage] = 0.15
    elif progress < 1.0:
        trans_prob = 0.4
        stage_dist = {current: 0.6}
        if next_stage:
            stage_dist[next_stage] = 0.4
    else:  # Overtime
        trans_prob = 0.7
        stage_dist = {current: 0.3}
        if next_stage:
            stage_dist[next_stage] = 0.7

    return Predictions(
        mode="time_based",
        timing_available=True,
        expected_stage_distribution=stage_dist,
        expected_features=expected_features,
        transition_markers=transition_markers,
        progress_ratio=progress,
        transition_probability=trans_prob,
    )
