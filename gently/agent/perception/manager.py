"""
Perception Manager - Orchestrates perception sessions for all embryos.

Replaces:
- DetectionQueue (replaced by perception rounds)
- DetectionVerifier (replaced by continuous belief tracking)

Integrates with existing EmbryoState, EventBus, and ImageManager.
"""

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import anthropic

from .session import (
    BeliefState,
    EvidencePoint,
    PerceptionSession,
    PerceptionRoundInput,
    PerceptionRoundOutput,
)
from .engine import PerceptionEngine
from .example_store import ExampleStore
from .belief_updater import BeliefUpdater
from .anomaly import AnomalyDetector
from .scheduler import AdaptivePerceptionScheduler

logger = logging.getLogger(__name__)


class PerceptionManager:
    """
    Manages perception sessions for all embryos.

    Provides a unified interface for running perception on volumes,
    replacing the old detection queue + verifier pattern.
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        examples_path: Path,
        event_bus: Optional[Any] = None,
        fast_model: Optional[str] = None,
        full_model: Optional[str] = None,
        deep_model: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Claude API client
        examples_path : Path
            Root directory for few-shot example images
        event_bus : EventBus, optional
            Event bus for emitting perception events
        fast_model : str, optional
            Model for routine checks
        full_model : str, optional
            Model for standard analysis
        deep_model : str, optional
            Model for critical decisions
        """
        self.example_store = ExampleStore(examples_path)
        self.engine = PerceptionEngine(
            claude_client=claude_client,
            example_store=self.example_store,
            fast_model=fast_model,
            full_model=full_model,
            deep_model=deep_model,
        )
        self.belief_updater = BeliefUpdater()
        self.anomaly_detector = AnomalyDetector()
        self.scheduler = AdaptivePerceptionScheduler()
        self._event_bus = event_bus

        # Active sessions (one per embryo)
        self.sessions: Dict[str, PerceptionSession] = {}

        # Callbacks
        self._on_hatching_detected: Optional[Callable] = None
        self._on_dead_embryo_detected: Optional[Callable] = None
        self._on_anomaly_detected: Optional[Callable] = None

    def get_or_create_session(self, embryo_id: str) -> PerceptionSession:
        """
        Get existing session or create new one.

        Parameters
        ----------
        embryo_id : str
            Embryo identifier

        Returns
        -------
        PerceptionSession
            Active session for this embryo
        """
        if embryo_id not in self.sessions:
            self.sessions[embryo_id] = PerceptionSession(
                embryo_id=embryo_id,
                created_at=datetime.now(),
                beliefs=BeliefState(),
            )
            logger.info(f"Created new perception session for {embryo_id}")

        return self.sessions[embryo_id]

    async def process_volume(
        self,
        embryo_id: str,
        timepoint: int,
        current_image_b64: str,
        recent_images: List[tuple],  # (timepoint, b64_image)
        hardware_errors: Optional[List[str]] = None,
        force_tier: Optional[str] = None,
    ) -> PerceptionRoundOutput:
        """
        Process a new volume through the perception system.

        This replaces DetectionQueue.run_detectors().

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint number
        current_image_b64 : str
            Base64-encoded current image (max projection)
        recent_images : List[tuple]
            List of (timepoint, b64_image) for temporal context
        hardware_errors : List[str], optional
            Recent hardware error messages
        force_tier : str, optional
            Force a specific model tier ("fast", "full", "deep")

        Returns
        -------
        PerceptionRoundOutput
            Analysis results with updated beliefs
        """
        session = self.get_or_create_session(embryo_id)

        # Check if we should run full analysis
        if not force_tier and not self.scheduler.should_run_full_analysis(
            embryo_id, session, timepoint
        ):
            force_tier = "fast"

        # Build round input
        round_input = self._build_round_input(
            session=session,
            timepoint=timepoint,
            current_image_b64=current_image_b64,
            recent_images=recent_images,
            hardware_errors=hardware_errors or [],
        )

        # Run perception round
        try:
            output = await self.engine.run_perception_round(
                session=session,
                round_input=round_input,
                force_tier=force_tier,
            )
        except Exception as e:
            logger.error(f"Perception round failed for {embryo_id}: {e}")
            # Return safe default
            output = PerceptionRoundOutput(
                updated_beliefs=session.beliefs,
                new_evidence=[],
                reasoning=f"Error: {e}",
                recommended_actions=[],
                analysis_confidence=0.0,
                anomaly_alerts=[],
                model_tier=force_tier or "fast",
            )

        # Calculate time delta
        time_delta_hours = 0.0
        if session.last_processed_timepoint is not None:
            # Assume ~2 min per timepoint
            time_delta_hours = (timepoint - session.last_processed_timepoint) * 2 / 60

        # Update beliefs
        self.belief_updater.update_beliefs(
            session=session,
            round_output=output,
            timepoint=timepoint,
            time_delta_hours=time_delta_hours,
        )

        # Mark scheduler
        if output.model_tier != "fast":
            self.scheduler.mark_full_analysis_run(embryo_id)

        # Handle blank frames
        if self._is_blank_image(current_image_b64):
            session.blank_frame_count += 1
            blank_result = self.anomaly_detector.classify_blank_frame(
                session, hardware_errors
            )
            output.anomaly_alerts.append({
                "type": "blank_frame",
                **blank_result,
            })
        else:
            session.blank_frame_count = 0

        # Emit events
        self._emit_perception_event(session, output, timepoint)

        # Handle recommended actions
        await self._handle_recommended_actions(embryo_id, session, output)

        return output

    def _build_round_input(
        self,
        session: PerceptionSession,
        timepoint: int,
        current_image_b64: str,
        recent_images: List[tuple],
        hardware_errors: List[str],
    ) -> PerceptionRoundInput:
        """Build input for perception round"""

        # Select relevant stage examples based on current beliefs
        stage_examples = self._get_relevant_stage_examples(session.beliefs)

        # Get anomaly examples if needed
        anomaly_examples = {}
        if session.beliefs.hours_since_change > 1 or session.beliefs.possibly_dead:
            anomaly_examples["dead_embryo"] = self.example_store.get_anomaly_examples(
                "dead_embryo", max_examples=2
            )
        if session.blank_frame_count > 0:
            anomaly_examples["blank_technical"] = self.example_store.get_anomaly_examples(
                "blank_technical", max_examples=2
            )
            anomaly_examples["blank_biological"] = self.example_store.get_anomaly_examples(
                "blank_biological", max_examples=2
            )

        return PerceptionRoundInput(
            current_image_b64=current_image_b64,
            current_timepoint=timepoint,
            current_timestamp=datetime.now(),
            recent_images=recent_images,
            prior_beliefs=session.beliefs,
            recent_evidence=session.get_recent_evidence(5),
            stage_examples=stage_examples,
            anomaly_examples=anomaly_examples,
            recent_errors=hardware_errors,
        )

    def _get_relevant_stage_examples(
        self,
        beliefs: BeliefState,
    ) -> Dict[str, List[str]]:
        """Select which stage examples to show based on current beliefs"""
        examples = {}

        # Get stage order and current position
        stage_order = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]

        try:
            current_idx = stage_order.index(beliefs.most_likely_stage)
        except ValueError:
            current_idx = 0

        # Show current stage and neighbors
        for offset in [-1, 0, 1]:
            idx = current_idx + offset
            if 0 <= idx < len(stage_order):
                stage = stage_order[idx]
                stage_examples = self.example_store.get_stage_examples(
                    stage, max_examples=2
                )
                if stage_examples:
                    examples[stage] = stage_examples

        return examples

    def _is_blank_image(self, image_b64: str) -> bool:
        """
        Simple heuristic check for blank image.

        More thorough check would use VLM or image statistics.
        """
        # If image is very short, it's probably blank
        if len(image_b64) < 1000:
            return True
        return False

    async def _handle_recommended_actions(
        self,
        embryo_id: str,
        session: PerceptionSession,
        output: PerceptionRoundOutput,
    ) -> None:
        """Handle recommended actions from perception round"""
        for action in output.recommended_actions:
            if action == "stop_imaging":
                # Hatching detected with high confidence
                logger.info(
                    f"[{embryo_id}] Perception recommends STOP IMAGING: "
                    f"hatching_complete={session.beliefs.hatching_complete}, "
                    f"confidence={output.analysis_confidence:.0%}"
                )

                if self._on_hatching_detected:
                    await self._on_hatching_detected(
                        embryo_id=embryo_id,
                        timepoint=session.last_processed_timepoint,
                        confidence=output.analysis_confidence,
                        beliefs=session.beliefs,
                    )

                self._emit_hatching_event(embryo_id, session)

            elif action == "alert_dead_embryo":
                logger.info(
                    f"[{embryo_id}] Perception suspects DEAD EMBRYO: "
                    f"confidence={session.beliefs.dead_confidence:.0%}"
                )

                if self._on_dead_embryo_detected:
                    await self._on_dead_embryo_detected(
                        embryo_id=embryo_id,
                        confidence=session.beliefs.dead_confidence,
                        evidence=session.beliefs.dead_evidence,
                    )

                self._emit_dead_embryo_event(embryo_id, session)

            elif action == "alert_user":
                logger.info(f"[{embryo_id}] Perception requests user attention")

    def _emit_perception_event(
        self,
        session: PerceptionSession,
        output: PerceptionRoundOutput,
        timepoint: int,
    ) -> None:
        """Emit perception round completed event"""
        if not self._event_bus:
            return

        try:
            from ...core import EventType

            self._event_bus.publish(
                EventType.DETECTOR_EVALUATED,  # Reuse existing event type
                {
                    "embryo_id": session.embryo_id,
                    "timepoint": timepoint,
                    "detector_name": "perception",  # Compatibility with viz
                    "beliefs": session.beliefs.to_dict(),
                    "new_evidence": output.new_evidence,
                    "analysis_confidence": output.analysis_confidence,
                    "reasoning_summary": output.reasoning[:200] if output.reasoning else "",
                    "model_tier": output.model_tier,
                    "recommended_actions": output.recommended_actions,
                },
                source="perception_manager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit perception event: {e}")

    def _emit_hatching_event(
        self,
        embryo_id: str,
        session: PerceptionSession,
    ) -> None:
        """Emit hatching detected event"""
        if not self._event_bus:
            return

        try:
            from ...core import EventType

            self._event_bus.publish(
                EventType.HATCHING_DETECTED,
                {
                    "embryo_id": embryo_id,
                    "timepoint": session.last_processed_timepoint,
                    "confidence": session.beliefs.stage_confidence,
                    "beliefs": session.beliefs.to_dict(),
                },
                source="perception_manager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit hatching event: {e}")

    def _emit_dead_embryo_event(
        self,
        embryo_id: str,
        session: PerceptionSession,
    ) -> None:
        """Emit dead embryo suspected event"""
        if not self._event_bus:
            return

        try:
            from ...core import EventType

            # Use ANOMALY_DETECTED if available, else fall back
            event_type = getattr(EventType, "DEAD_EMBRYO_SUSPECTED", None)
            if event_type is None:
                event_type = getattr(EventType, "ANOMALY_DETECTED", None)
            if event_type is None:
                return

            self._event_bus.publish(
                event_type,
                {
                    "embryo_id": embryo_id,
                    "anomaly_type": "dead_embryo",
                    "confidence": session.beliefs.dead_confidence,
                    "evidence": session.beliefs.dead_evidence,
                },
                source="perception_manager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit dead embryo event: {e}")

    def get_session(self, embryo_id: str) -> Optional[PerceptionSession]:
        """Get session for an embryo (if exists)"""
        return self.sessions.get(embryo_id)

    def get_beliefs(self, embryo_id: str) -> Optional[BeliefState]:
        """Get current beliefs for an embryo"""
        session = self.sessions.get(embryo_id)
        return session.beliefs if session else None

    def get_all_sessions(self) -> Dict[str, PerceptionSession]:
        """Get all active sessions"""
        return self.sessions.copy()

    def clear_session(self, embryo_id: str) -> bool:
        """Clear session for an embryo (reset perception)"""
        if embryo_id in self.sessions:
            del self.sessions[embryo_id]
            return True
        return False

    def set_callbacks(
        self,
        on_hatching_detected: Optional[Callable] = None,
        on_dead_embryo_detected: Optional[Callable] = None,
        on_anomaly_detected: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for perception events"""
        self._on_hatching_detected = on_hatching_detected
        self._on_dead_embryo_detected = on_dead_embryo_detected
        self._on_anomaly_detected = on_anomaly_detected

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manager state for session persistence"""
        return {
            "sessions": {
                embryo_id: session.to_dict()
                for embryo_id, session in self.sessions.items()
            },
            "example_counts": self.example_store.get_example_counts(),
        }

    def restore_sessions(self, data: Dict[str, Any]) -> None:
        """Restore sessions from serialized data"""
        sessions_data = data.get("sessions", {})
        for embryo_id, session_dict in sessions_data.items():
            try:
                self.sessions[embryo_id] = PerceptionSession.from_dict(session_dict)
                logger.info(f"Restored perception session for {embryo_id}")
            except Exception as e:
                logger.warning(f"Failed to restore session for {embryo_id}: {e}")
