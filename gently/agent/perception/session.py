"""
Core data structures for the perception system.

PerceptionSession maintains continuous context per embryo, accumulating
understanding over time through BeliefState and EvidenceHistory.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BeliefState:
    """
    Current beliefs about an embryo's developmental state.

    Unlike boolean flags, this maintains probabilistic beliefs that
    can be updated with new evidence over time.
    """

    # Developmental stage (probabilistic distribution)
    stage_distribution: Dict[str, float] = field(default_factory=lambda: {
        "early": 0.3,
        "comma": 0.2,
        "pretzel": 0.2,
        "3fold": 0.15,
        "hatching": 0.1,
        "hatched": 0.05,
    })
    most_likely_stage: str = "early"
    stage_confidence: float = 0.3

    # Anomaly flags
    possibly_dead: bool = False
    dead_confidence: float = 0.0
    dead_evidence: List[str] = field(default_factory=list)

    technical_issue_suspected: bool = False
    technical_issue_type: Optional[str] = None  # "blank_frame", "focus_drift", etc.

    # Temporal dynamics
    last_significant_change_timepoint: Optional[int] = None
    hours_since_change: float = 0.0
    movement_detected: bool = False

    # Hatching state machine (one-way progression)
    hatching_in_progress: bool = False
    breach_detected: bool = False
    worm_exiting: bool = False
    hatching_complete: bool = False
    hatching_timepoint: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage"""
        return {
            "stage_distribution": self.stage_distribution,
            "most_likely_stage": self.most_likely_stage,
            "stage_confidence": self.stage_confidence,
            "possibly_dead": self.possibly_dead,
            "dead_confidence": self.dead_confidence,
            "dead_evidence": self.dead_evidence,
            "technical_issue_suspected": self.technical_issue_suspected,
            "technical_issue_type": self.technical_issue_type,
            "last_significant_change_timepoint": self.last_significant_change_timepoint,
            "hours_since_change": self.hours_since_change,
            "movement_detected": self.movement_detected,
            "hatching_in_progress": self.hatching_in_progress,
            "breach_detected": self.breach_detected,
            "worm_exiting": self.worm_exiting,
            "hatching_complete": self.hatching_complete,
            "hatching_timepoint": self.hatching_timepoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeliefState":
        """Deserialize from JSON"""
        return cls(
            stage_distribution=data.get("stage_distribution", {}),
            most_likely_stage=data.get("most_likely_stage", "early"),
            stage_confidence=data.get("stage_confidence", 0.3),
            possibly_dead=data.get("possibly_dead", False),
            dead_confidence=data.get("dead_confidence", 0.0),
            dead_evidence=data.get("dead_evidence", []),
            technical_issue_suspected=data.get("technical_issue_suspected", False),
            technical_issue_type=data.get("technical_issue_type"),
            last_significant_change_timepoint=data.get("last_significant_change_timepoint"),
            hours_since_change=data.get("hours_since_change", 0.0),
            movement_detected=data.get("movement_detected", False),
            hatching_in_progress=data.get("hatching_in_progress", False),
            breach_detected=data.get("breach_detected", False),
            worm_exiting=data.get("worm_exiting", False),
            hatching_complete=data.get("hatching_complete", False),
            hatching_timepoint=data.get("hatching_timepoint"),
        )


@dataclass
class EvidencePoint:
    """
    A single observation that supports or contradicts beliefs.

    Evidence accumulates over time to build a reasoning trace
    that explains why we believe what we believe.
    """

    timepoint: int
    timestamp: datetime
    observation: str  # Natural language description
    supports: List[str]  # Beliefs this evidence supports
    contradicts: List[str]  # Beliefs this evidence contradicts
    confidence: float  # 0-1 confidence in this observation
    image_uid: Optional[str] = None  # Reference to the source image

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage"""
        return {
            "timepoint": self.timepoint,
            "timestamp": self.timestamp.isoformat(),
            "observation": self.observation,
            "supports": self.supports,
            "contradicts": self.contradicts,
            "confidence": self.confidence,
            "image_uid": self.image_uid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidencePoint":
        """Deserialize from JSON"""
        return cls(
            timepoint=data["timepoint"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            observation=data["observation"],
            supports=data.get("supports", []),
            contradicts=data.get("contradicts", []),
            confidence=data.get("confidence", 0.5),
            image_uid=data.get("image_uid"),
        )


@dataclass
class PerceptionSession:
    """
    Long-running perception context for one embryo.

    Unlike one-shot detection calls, this accumulates understanding
    over the entire timelapse, enabling reasoning like:
    "I saw hatching start last round, and now the field is empty,
    therefore hatching is complete."
    """

    embryo_id: str
    created_at: datetime

    # Current beliefs (updated each round)
    beliefs: BeliefState = field(default_factory=BeliefState)

    # Evidence history (grows over time)
    evidence_history: List[EvidencePoint] = field(default_factory=list)

    # Full reasoning trace (for debugging/review)
    reasoning_trace: List[str] = field(default_factory=list)

    # Reference examples loaded for this session
    loaded_examples: Dict[str, List[str]] = field(default_factory=dict)

    # Anomaly tracking
    unchanged_count: int = 0  # Consecutive rounds with no significant change
    blank_frame_count: int = 0  # Consecutive blank frames

    # Session metadata
    rounds_processed: int = 0
    last_processed_timepoint: Optional[int] = None

    def add_evidence(self, evidence: EvidencePoint) -> None:
        """Add new evidence to history"""
        self.evidence_history.append(evidence)

    def add_reasoning(self, reasoning: str, timepoint: Optional[int] = None) -> None:
        """Add reasoning trace entry"""
        prefix = f"[T{timepoint}] " if timepoint else ""
        self.reasoning_trace.append(f"{prefix}{reasoning}")

    def get_recent_evidence(self, n: int = 5) -> List[EvidencePoint]:
        """Get the N most recent evidence points"""
        return self.evidence_history[-n:] if self.evidence_history else []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage"""
        return {
            "embryo_id": self.embryo_id,
            "created_at": self.created_at.isoformat(),
            "beliefs": self.beliefs.to_dict(),
            "evidence_history": [e.to_dict() for e in self.evidence_history],
            "reasoning_trace": self.reasoning_trace[-50:],  # Keep last 50 entries
            "unchanged_count": self.unchanged_count,
            "blank_frame_count": self.blank_frame_count,
            "rounds_processed": self.rounds_processed,
            "last_processed_timepoint": self.last_processed_timepoint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionSession":
        """Deserialize from JSON"""
        session = cls(
            embryo_id=data["embryo_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            beliefs=BeliefState.from_dict(data.get("beliefs", {})),
            unchanged_count=data.get("unchanged_count", 0),
            blank_frame_count=data.get("blank_frame_count", 0),
            rounds_processed=data.get("rounds_processed", 0),
            last_processed_timepoint=data.get("last_processed_timepoint"),
        )
        session.evidence_history = [
            EvidencePoint.from_dict(e) for e in data.get("evidence_history", [])
        ]
        session.reasoning_trace = data.get("reasoning_trace", [])
        return session


@dataclass
class PerceptionRoundInput:
    """Input to a single perception round"""

    # Current observation
    current_image_b64: str
    current_timepoint: int
    current_timestamp: datetime

    # Temporal context (recent images)
    recent_images: List[Tuple[int, str]]  # (timepoint, b64_image)

    # Session context
    prior_beliefs: BeliefState
    recent_evidence: List[EvidencePoint]

    # Few-shot examples (stage -> list of example images)
    stage_examples: Dict[str, List[str]]

    # Anomaly reference examples
    anomaly_examples: Dict[str, List[str]]  # "dead_embryo", "blank_technical", etc.

    # Hardware context (for technical failure detection)
    recent_errors: List[str]


@dataclass
class PerceptionRoundOutput:
    """Output from a single perception round"""

    # Updated beliefs
    updated_beliefs: BeliefState

    # New evidence observed this round
    new_evidence: List[Dict[str, Any]]

    # Natural language reasoning (for trace)
    reasoning: str

    # Recommended actions
    recommended_actions: List[str]  # "stop_imaging", "increase_frequency", "alert_user"

    # Confidence in this round's analysis
    analysis_confidence: float

    # Anomaly alerts
    anomaly_alerts: List[Dict[str, Any]]

    # Model tier used for this round
    model_tier: str = "full"  # "fast", "full", "deep"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage"""
        return {
            "updated_beliefs": self.updated_beliefs.to_dict(),
            "new_evidence": self.new_evidence,
            "reasoning": self.reasoning,
            "recommended_actions": self.recommended_actions,
            "analysis_confidence": self.analysis_confidence,
            "anomaly_alerts": self.anomaly_alerts,
            "model_tier": self.model_tier,
        }
