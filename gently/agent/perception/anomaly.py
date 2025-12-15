"""
Anomaly Detection for Perception System.

Detects:
- Dead embryos (no change/movement for extended periods)
- Technical failures (blank frames, focus issues)
- Distinguishes technical vs biological blank frames
"""

import logging
from typing import Any, Dict, List, Optional

from .session import BeliefState, PerceptionSession

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in embryo development.

    Uses a combination of:
    - VLM analysis (from perception round)
    - Temporal patterns (unchanged over time)
    - Context (hatching progression, hardware errors)
    """

    def __init__(
        self,
        dead_threshold_hours: float = 2.0,
        blank_threshold_frames: int = 2,
        stasis_observation_threshold: int = 5,
    ):
        """
        Parameters
        ----------
        dead_threshold_hours : float
            Hours without change before suspecting dead embryo
        blank_threshold_frames : int
            Consecutive blank frames before flagging
        stasis_observation_threshold : int
            Number of stasis observations to confirm dead
        """
        self.dead_threshold_hours = dead_threshold_hours
        self.blank_threshold_frames = blank_threshold_frames
        self.stasis_observation_threshold = stasis_observation_threshold

    def check_dead_embryo(
        self,
        session: PerceptionSession,
    ) -> Dict[str, Any]:
        """
        Check if embryo appears dead (no development/movement).

        Criteria:
        1. No significant morphological change over threshold time
        2. No visible movement in late stages (when movement expected)
        3. Evidence history shows repeated stasis observations

        Returns
        -------
        Dict with:
            - is_dead: bool
            - confidence: float
            - evidence: List[str]
            - recommendation: str
        """
        beliefs = session.beliefs
        evidence = []
        confidence = 0.0

        # Already marked as dead by perception round
        if beliefs.possibly_dead:
            confidence = beliefs.dead_confidence
            evidence.extend(beliefs.dead_evidence)

        # Check temporal stasis
        if beliefs.hours_since_change > self.dead_threshold_hours:
            evidence.append(
                f"No significant change for {beliefs.hours_since_change:.1f} hours"
            )
            confidence += 0.3

        # Check movement in late stages
        if beliefs.most_likely_stage in ["pretzel", "3fold"]:
            if not beliefs.movement_detected:
                evidence.append("No movement detected in late stage (expected)")
                confidence += 0.3

        # Check evidence history for stasis
        recent_observations = session.evidence_history[-10:]
        stasis_count = sum(
            1 for e in recent_observations
            if any(word in e.observation.lower() for word in [
                "unchanged", "static", "no change", "same as",
                "identical", "no movement", "no progress"
            ])
        )
        if stasis_count >= self.stasis_observation_threshold:
            evidence.append(
                f"{stasis_count} recent observations show no change"
            )
            confidence += 0.2

        # Can't be dead if hatching is in progress
        if beliefs.hatching_in_progress or beliefs.hatching_complete:
            return {
                "is_dead": False,
                "confidence": 0.0,
                "evidence": ["Hatching in progress - embryo is alive"],
                "recommendation": "continue_imaging",
            }

        # Determine result
        is_dead = confidence > 0.5
        confidence = min(confidence, 0.95)  # Cap confidence

        if is_dead:
            recommendation = "stop_imaging" if confidence > 0.7 else "alert_user"
        else:
            recommendation = "continue_imaging"

        return {
            "is_dead": is_dead,
            "confidence": confidence,
            "evidence": evidence,
            "recommendation": recommendation,
        }

    def classify_blank_frame(
        self,
        session: PerceptionSession,
        hardware_errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Classify a blank/empty frame as technical or biological.

        Technical blank indicators:
        - Hardware errors in recent history
        - Sudden blank without prior hatching signs
        - Blank frame isolated (surrounded by normal frames)

        Biological blank (hatching) indicators:
        - Prior evidence of hatching progression
        - Worm was exiting in previous frames
        - Field shows deflated eggshell (not just empty)

        Parameters
        ----------
        session : PerceptionSession
            Current perception session
        hardware_errors : List[str], optional
            Recent hardware error messages

        Returns
        -------
        Dict with:
            - classification: "technical" | "biological_hatching" | "unknown"
            - confidence: float
            - reasoning: str
            - recommendation: str
        """
        beliefs = session.beliefs
        hardware_errors = hardware_errors or []

        # Check for hatching progression - strongest indicator
        if beliefs.hatching_in_progress or beliefs.worm_exiting:
            return {
                "classification": "biological_hatching",
                "confidence": 0.85,
                "reasoning": "Hatching was in progress, empty field indicates worm has left",
                "recommendation": "stop_imaging",
            }

        if beliefs.breach_detected:
            return {
                "classification": "biological_hatching",
                "confidence": 0.75,
                "reasoning": "Breach was detected, empty field likely indicates hatching completion",
                "recommendation": "verify_hatching",
            }

        # Check for hardware errors
        if hardware_errors:
            error_keywords = [
                "timeout", "error", "fail", "lost", "disconnect",
                "position", "stage", "camera", "acquisition"
            ]
            relevant_errors = [
                e for e in hardware_errors
                if any(kw in e.lower() for kw in error_keywords)
            ]
            if relevant_errors:
                return {
                    "classification": "technical",
                    "confidence": 0.7,
                    "reasoning": f"Hardware errors present: {relevant_errors[:2]}",
                    "recommendation": "retry_acquisition",
                }

        # Check evidence history for hatching signs
        recent_evidence = session.evidence_history[-5:]
        hatching_evidence = [
            e for e in recent_evidence
            if any(s in e.supports for s in ["hatching", "hatched", "breach", "exiting"])
        ]

        if hatching_evidence:
            return {
                "classification": "biological_hatching",
                "confidence": 0.75,
                "reasoning": "Recent evidence supports hatching progression",
                "recommendation": "verify_hatching",
            }

        # Check blank frame history
        if session.blank_frame_count > self.blank_threshold_frames:
            return {
                "classification": "technical",
                "confidence": 0.6,
                "reasoning": f"Multiple consecutive blank frames ({session.blank_frame_count})",
                "recommendation": "check_hardware",
            }

        # Early stage embryo going blank is suspicious
        if beliefs.most_likely_stage in ["early", "comma"]:
            return {
                "classification": "technical",
                "confidence": 0.6,
                "reasoning": "Early stage embryo should not be blank - likely technical issue",
                "recommendation": "retry_acquisition",
            }

        # Unknown - need more context
        return {
            "classification": "unknown",
            "confidence": 0.3,
            "reasoning": "Insufficient evidence to classify blank frame",
            "recommendation": "continue_monitoring",
        }

    def check_stage_regression(
        self,
        session: PerceptionSession,
        new_stage: str,
    ) -> Dict[str, Any]:
        """
        Check if detected stage represents impossible regression.

        Embryo development is one-way: early -> comma -> pretzel -> 3fold -> hatching -> hatched

        Parameters
        ----------
        session : PerceptionSession
            Current perception session
        new_stage : str
            Newly detected stage

        Returns
        -------
        Dict with:
            - is_regression: bool
            - from_stage: str
            - to_stage: str
            - recommendation: str
        """
        stage_order = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]

        current_stage = session.beliefs.most_likely_stage

        try:
            current_idx = stage_order.index(current_stage)
            new_idx = stage_order.index(new_stage)
        except ValueError:
            # Unknown stage
            return {
                "is_regression": False,
                "from_stage": current_stage,
                "to_stage": new_stage,
                "recommendation": "unknown_stage",
            }

        if new_idx < current_idx:
            return {
                "is_regression": True,
                "from_stage": current_stage,
                "to_stage": new_stage,
                "recommendation": "verify_detection",
            }

        return {
            "is_regression": False,
            "from_stage": current_stage,
            "to_stage": new_stage,
            "recommendation": "continue",
        }

    def get_anomaly_summary(
        self,
        session: PerceptionSession,
        hardware_errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive anomaly status for an embryo.

        Returns
        -------
        Dict with all anomaly checks:
            - dead_embryo: result from check_dead_embryo
            - blank_frame: result if blank_frame_count > 0
            - overall_status: "healthy" | "warning" | "critical"
            - recommended_action: str
        """
        results = {}

        # Dead embryo check
        dead_result = self.check_dead_embryo(session)
        results["dead_embryo"] = dead_result

        # Blank frame check (if applicable)
        if session.blank_frame_count > 0:
            blank_result = self.classify_blank_frame(session, hardware_errors)
            results["blank_frame"] = blank_result

        # Determine overall status
        if dead_result["is_dead"] and dead_result["confidence"] > 0.7:
            status = "critical"
            action = "stop_imaging"
        elif dead_result["is_dead"] and dead_result["confidence"] > 0.5:
            status = "warning"
            action = "alert_user"
        elif session.blank_frame_count > self.blank_threshold_frames:
            status = "warning"
            action = "check_hardware"
        else:
            status = "healthy"
            action = "continue_imaging"

        results["overall_status"] = status
        results["recommended_action"] = action

        return results
