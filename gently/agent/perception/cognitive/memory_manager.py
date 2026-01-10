"""
Memory Manager for Cognitive Perception.

Parses VLM output and persists state to disk.
Handles the transition from VLM cognition to external memory.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from .world_model import WorldModel, Predictions, TransitionRecord
from .memory import (
    StageNarrative,
    TransitionNarrative,
    EmbryoNarrative,
    BatchNarrative,
    StageMetrics,
    TimestampedSummary,
)
from .trace import MonitorResult, CognitiveTrace
from ..stages import DevelopmentalStage


class MemoryManager:
    """
    Manages parsing VLM output and persisting cognitive state.

    Responsibilities:
    - Parse structured JSON from VLM responses
    - Update world model with new beliefs
    - Update narratives with observations
    - Persist to disk
    - Load from disk for session recovery
    """

    def __init__(
        self,
        narratives_dir: Path,
        auto_persist: bool = True,
    ):
        """
        Parameters
        ----------
        narratives_dir : Path
            Directory for narrative persistence
        auto_persist : bool
            Whether to automatically persist after updates
        """
        self.narratives_dir = Path(narratives_dir)
        self.auto_persist = auto_persist

        # In-memory cache
        self._world_models: Dict[str, WorldModel] = {}
        self._narratives: Dict[str, EmbryoNarrative] = {}
        self._batch_narrative: Optional[BatchNarrative] = None

    def parse_monitor_output(self, raw_output: str) -> MonitorResult:
        """
        Parse output from monitor mode VLM call.

        Expected format (JSON):
        {
            "confirmation": "yes" | "no" | "uncertain",
            "confidence": 0.9,
            "brief_observation": "Still shows comma shape...",
            "red_flags": []
        }
        """
        try:
            data = self._extract_json(raw_output)
            return MonitorResult(
                confirmation=data.get("confirmation", "uncertain"),
                confidence=float(data.get("confidence", 0.5)),
                brief_observation=data.get("brief_observation", raw_output[:200]),
                red_flags=data.get("red_flags", []),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to parse as free-form text
            return self._parse_monitor_freeform(raw_output)

    def _parse_monitor_freeform(self, raw_output: str) -> MonitorResult:
        """Parse monitor output when JSON parsing fails."""
        lower = raw_output.lower()

        # Determine confirmation
        if "uncertain" in lower or "unclear" in lower or "can't tell" in lower:
            confirmation = "uncertain"
            confidence = 0.5
        elif "no" in lower[:50] or "different" in lower or "not" in lower[:50]:
            confirmation = "no"
            confidence = 0.6
        else:
            confirmation = "yes"
            confidence = 0.8

        return MonitorResult(
            confirmation=confirmation,
            confidence=confidence,
            brief_observation=raw_output[:200],
            red_flags=[],
        )

    def parse_single_turn_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from single-turn deep perception.

        Expected format (JSON):
        {
            "observation": "what you observed",
            "surprise_level": "none|low|medium|high",
            "surprise_reason": "why this level",
            "needs_verification": false,
            "candidate_stages": [],
            "updated_beliefs": {
                "stage": "stage_name",
                "confidence": 0.85,
                "transition_detected": false,
                "transition_from": null,
                "key_features": [],
                "reasoning": "explanation"
            }
        }
        """
        try:
            data = self._extract_json(raw_output)
            return self._validate_single_turn_output(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Fallback: try to extract key information from free-form text
            return self._parse_single_turn_freeform(raw_output)

    def _validate_single_turn_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize single-turn output structure."""
        # Ensure required fields
        result = {
            "observation": data.get("observation", ""),
            "surprise_level": data.get("surprise_level", "low"),
            "surprise_reason": data.get("surprise_reason", ""),
            "needs_verification": data.get("needs_verification", False),
            "candidate_stages": data.get("candidate_stages", []),
            "updated_beliefs": {},
        }

        # Validate and normalize updated_beliefs
        beliefs = data.get("updated_beliefs", data)  # Sometimes nested, sometimes flat
        result["updated_beliefs"] = {
            "stage": beliefs.get("stage", ""),
            "confidence": float(beliefs.get("confidence", 0.5)),
            "transition_detected": beliefs.get("transition_detected", False),
            "transition_from": beliefs.get("transition_from"),
            "key_features": beliefs.get("key_features", []),
            "reasoning": beliefs.get("reasoning", ""),
        }

        # Validate stage
        stage = result["updated_beliefs"]["stage"]
        if stage and not DevelopmentalStage.is_valid(stage):
            # Try to find a valid stage in the response
            for valid_stage in DevelopmentalStage.all_valid_values():
                if valid_stage in data.get("observation", "").lower():
                    result["updated_beliefs"]["stage"] = valid_stage
                    break

        return result

    def _parse_single_turn_freeform(self, raw_output: str) -> Dict[str, Any]:
        """Parse single-turn output when JSON parsing fails."""
        lower = raw_output.lower()

        # Try to extract stage
        detected_stage = None
        for stage in DevelopmentalStage.all_valid_values():
            if stage in lower:
                detected_stage = stage
                break

        # Try to extract confidence
        confidence = 0.7  # Default
        conf_match = re.search(r'(\d{1,2}(?:\.\d+)?)\s*%', raw_output)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100
        else:
            conf_match = re.search(r'confidence[:\s]+(\d\.\d+)', lower)
            if conf_match:
                confidence = float(conf_match.group(1))

        # Determine surprise level
        surprise_level = "low"
        if "high surprise" in lower or "very surprised" in lower:
            surprise_level = "high"
        elif "medium surprise" in lower or "somewhat surprised" in lower:
            surprise_level = "medium"
        elif "no surprise" in lower or "as expected" in lower:
            surprise_level = "none"

        return {
            "observation": raw_output[:500],
            "surprise_level": surprise_level,
            "surprise_reason": "",
            "needs_verification": surprise_level in ("medium", "high"),
            "candidate_stages": [],
            "updated_beliefs": {
                "stage": detected_stage or "",
                "confidence": confidence,
                "transition_detected": False,
                "transition_from": None,
                "key_features": [],
                "reasoning": raw_output[:200],
            },
        }

    def parse_perceive_output(self, raw_output: str) -> str:
        """
        Parse output from PERCEIVE turn (unbiased observation).

        This is pure text - no JSON expected.
        Just clean up and return.
        """
        # Remove any accidental JSON formatting
        text = raw_output.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "observation" in data:
                    return data["observation"]
            except json.JSONDecodeError:
                pass
        return text

    def parse_compare_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from COMPARE turn.

        Expected format (JSON):
        {
            "surprise_level": "none|low|medium|high",
            "surprise_reason": "explanation",
            "still_uncertain": true|false,
            "candidate_stages": ["stage1", "stage2"]
        }
        """
        try:
            data = self._extract_json(raw_output)
            return {
                "surprise_level": data.get("surprise_level", "low"),
                "surprise_reason": data.get("surprise_reason", ""),
                "still_uncertain": data.get("still_uncertain", False),
                "candidate_stages": data.get("candidate_stages", []),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback parsing
            lower = raw_output.lower()
            return {
                "surprise_level": "medium" if "uncertain" in lower else "low",
                "surprise_reason": raw_output[:200],
                "still_uncertain": "uncertain" in lower or "unclear" in lower,
                "candidate_stages": [],
            }

    def parse_verify_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from VERIFY turn.

        Expected format (JSON):
        {
            "verified_stage": "stage_name",
            "confidence": 0.85,
            "evidence": {
                "matches_stage": ["feature1", "feature2"],
                "differs_from_others": ["difference1", "difference2"]
            }
        }
        """
        try:
            data = self._extract_json(raw_output)
            return {
                "verified_stage": data.get("verified_stage", ""),
                "confidence": float(data.get("confidence", 0.5)),
                "evidence": data.get("evidence", {}),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Try to extract stage from free-form
            detected_stage = None
            for stage in DevelopmentalStage.all_valid_values():
                if stage in raw_output.lower():
                    detected_stage = stage
                    break

            return {
                "verified_stage": detected_stage or "",
                "confidence": 0.7,
                "evidence": {"raw": raw_output[:300]},
            }

    def parse_integrate_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from INTEGRATE turn.

        Expected format (JSON):
        {
            "updated_stage": "stage_name",
            "updated_confidence": 0.85,
            "transition": {
                "detected": true|false,
                "from": "old_stage",
                "to": "new_stage",
                "evidence": ["feature1", "feature2"]
            },
            "narrative_update": "text summary..."
        }
        """
        try:
            data = self._extract_json(raw_output)
            return {
                "updated_stage": data.get("updated_stage", ""),
                "updated_confidence": float(data.get("updated_confidence", 0.5)),
                "transition": data.get("transition", {"detected": False}),
                "narrative_update": data.get("narrative_update", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback parsing
            detected_stage = None
            for stage in DevelopmentalStage.all_valid_values():
                if stage in raw_output.lower():
                    detected_stage = stage
                    break

            return {
                "updated_stage": detected_stage or "",
                "updated_confidence": 0.7,
                "transition": {"detected": False},
                "narrative_update": raw_output[:300],
            }

    def parse_backward_rejection_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from backward rejection reasoning.

        Expected format (JSON):
        {
            "analysis": "reasoning about the situation",
            "conclusion": "A" | "B",
            "reasoning": "detailed explanation",
            "action": {
                "maintain_stage": "stage_name",
                "confidence_adjustment": -0.15,
                "flag_for_review": true,
                "backward_rejected": true
            }
        }
        """
        try:
            data = self._extract_json(raw_output)
            return {
                "analysis": data.get("analysis", ""),
                "conclusion": data.get("conclusion", "A"),
                "reasoning": data.get("reasoning", ""),
                "action": data.get("action", {}),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Default: maintain current stage
            return {
                "analysis": raw_output[:300],
                "conclusion": "A",
                "reasoning": "Failed to parse VLM output, defaulting to maintain current stage",
                "action": {
                    "maintain_stage": None,  # Will be filled by caller
                    "confidence_adjustment": -0.1,
                    "flag_for_review": True,
                    "backward_rejected": True,
                },
            }

    def parse_initial_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Parse output from initial classification (no prior model).

        Expected format (JSON):
        {
            "stage": "stage_name",
            "confidence": 0.85,
            "key_features": ["feature1", "feature2"],
            "reasoning": "explanation"
        }
        """
        try:
            data = self._extract_json(raw_output)
            stage = data.get("stage", "")
            if not DevelopmentalStage.is_valid(stage):
                # Try to find valid stage
                for valid_stage in DevelopmentalStage.all_valid_values():
                    if valid_stage in raw_output.lower():
                        stage = valid_stage
                        break

            return {
                "stage": stage,
                "confidence": float(data.get("confidence", 0.7)),
                "key_features": data.get("key_features", []),
                "reasoning": data.get("reasoning", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback parsing
            detected_stage = None
            for stage in DevelopmentalStage.all_valid_values():
                if stage in raw_output.lower():
                    detected_stage = stage
                    break

            return {
                "stage": detected_stage or "early",
                "confidence": 0.6,
                "key_features": [],
                "reasoning": raw_output[:200],
            }

    def update_world_model(
        self,
        embryo_id: str,
        parsed_output: Dict[str, Any],
        timepoint: int,
        mode: str = "single_turn",
    ) -> WorldModel:
        """
        Update world model with parsed VLM output.

        Parameters
        ----------
        embryo_id : str
            The embryo identifier
        parsed_output : Dict
            Parsed output from VLM (from parse_* methods)
        timepoint : int
            Current timepoint
        mode : str
            Which cognitive mode produced this output

        Returns
        -------
        WorldModel
            The updated world model
        """
        model = self.get_world_model(embryo_id)

        if model is None:
            # Create new world model from initial output
            return self._create_world_model(embryo_id, parsed_output, timepoint)

        # Determine the source of beliefs based on mode
        if mode == "initial":
            beliefs = parsed_output
        elif mode in ("single_turn", "escalation"):
            beliefs = parsed_output.get("updated_beliefs", parsed_output)
        elif mode == "integrate":
            beliefs = {
                "stage": parsed_output.get("updated_stage"),
                "confidence": parsed_output.get("updated_confidence"),
            }
        elif mode == "backward_rejection":
            action = parsed_output.get("action", {})
            beliefs = {
                "stage": action.get("maintain_stage", model.current_stage),
                "confidence": model.stage_confidence + action.get("confidence_adjustment", 0),
            }
        else:
            beliefs = parsed_output

        new_stage = beliefs.get("stage", model.current_stage)
        new_confidence = beliefs.get("confidence", model.stage_confidence)

        # Detect transition
        transition_detected = (
            beliefs.get("transition_detected", False) or
            (new_stage != model.current_stage and new_stage in DevelopmentalStage.all_valid_values())
        )

        if transition_detected and new_stage != model.current_stage:
            # Record the transition
            transition = TransitionRecord(
                from_stage=model.current_stage,
                to_stage=new_stage,
                timepoint=timepoint,
                confidence=new_confidence,
                key_evidence=beliefs.get("key_features", []),
            )
            model.transitions_observed.append(transition)
            model.stages_visited.append(new_stage)

            # Update stage entry tracking
            model.current_stage = new_stage
            model.stage_confidence = new_confidence
            model.stage_entry_observed = True  # We observed this transition!
            model.entered_at_timepoint = timepoint
            model.observations_in_stage = 1
            model.first_observation_timepoint = timepoint
            model.monitors_since_last_deep = 0
        else:
            # Same stage, update confidence
            model.stage_confidence = new_confidence
            model.observations_in_stage += 1

        # Update confidence trend
        model.confidence_trend.append(new_confidence)
        if len(model.confidence_trend) > 10:
            model.confidence_trend = model.confidence_trend[-10:]

        # Update feature observations
        features = beliefs.get("key_features", [])
        if features:
            model.feature_observations.extend(features)
            if len(model.feature_observations) > 20:
                model.feature_observations = model.feature_observations[-20:]

        # Cache and optionally persist
        self._world_models[embryo_id] = model
        if self.auto_persist:
            self._persist_world_model(embryo_id, model)

        return model

    def _create_world_model(
        self,
        embryo_id: str,
        parsed_output: Dict[str, Any],
        timepoint: int,
    ) -> WorldModel:
        """Create initial world model from first observation."""
        stage = parsed_output.get("stage", "early")
        confidence = parsed_output.get("confidence", 0.7)
        features = parsed_output.get("key_features", [])

        model = WorldModel(
            embryo_id=embryo_id,
            current_stage=stage,
            stage_confidence=confidence,
            stage_entry_observed=False,  # First observation - didn't see entry
            entered_at_timepoint=None,
            observations_in_stage=1,
            first_observation_timepoint=timepoint,
            confidence_trend=[confidence],
            feature_observations=features,
            stages_visited=[stage],
            transitions_observed=[],
            predictions=Predictions.empty(),
            monitors_since_last_deep=0,
        )

        self._world_models[embryo_id] = model
        if self.auto_persist:
            self._persist_world_model(embryo_id, model)

        return model

    def get_world_model(self, embryo_id: str) -> Optional[WorldModel]:
        """Get world model for an embryo (from cache or disk)."""
        if embryo_id in self._world_models:
            return self._world_models[embryo_id]

        # Try to load from disk
        model = self._load_world_model(embryo_id)
        if model:
            self._world_models[embryo_id] = model
        return model

    def update_narrative(
        self,
        embryo_id: str,
        observation_text: str,
        timepoint: int,
        stage: str,
        confidence: float,
        transition_detected: bool = False,
        transition_from: Optional[str] = None,
    ) -> EmbryoNarrative:
        """
        Update embryo narrative with new observation.

        Parameters
        ----------
        embryo_id : str
            The embryo identifier
        observation_text : str
            Text description of the observation
        timepoint : int
            Current timepoint
        stage : str
            Current stage
        confidence : float
            Current confidence
        transition_detected : bool
            Whether a transition was detected
        transition_from : str, optional
            Previous stage if transition detected

        Returns
        -------
        EmbryoNarrative
            Updated narrative
        """
        narrative = self.get_narrative(embryo_id)

        if narrative is None:
            narrative = EmbryoNarrative.create_initial(
                embryo_id=embryo_id,
                initial_stage=stage,
                timepoint=timepoint,
                confidence=confidence,
            )

        # Update current stage
        narrative.current_stage = stage

        # Get or create stage narrative
        stage_narrative = self._get_or_create_stage_narrative(
            narrative, embryo_id, stage, timepoint, confidence
        )

        # Add observation (uses StageNarrative.add_observation which updates metrics)
        stage_narrative.add_observation(timepoint, confidence)

        # Update total observations
        narrative.total_observations += 1
        narrative.updated_at = datetime.now()

        # Handle transition
        if transition_detected and transition_from:
            # Close previous stage narrative
            prev_stage_narrative = self._find_stage_narrative(narrative, transition_from)
            if prev_stage_narrative:
                prev_stage_narrative.exited_at = datetime.now()
                prev_stage_narrative.exit_summary = f"Transitioned to {stage}"

            # Create transition narrative
            transition = TransitionNarrative(
                transition_id=f"{embryo_id}_{transition_from}_to_{stage}_{timepoint}",
                embryo_id=embryo_id,
                from_stage=transition_from,
                to_stage=stage,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                trigger_timepoint=timepoint,
                confirmation_timepoint=timepoint,
                key_evidence=[observation_text[:100]],
                transitional_timepoints=[timepoint],
                summary=f"Transitioned from {transition_from} to {stage}",
            )
            narrative.transitions.append(transition)

            # Mark new stage entry as observed
            stage_narrative.entry_observed = True
            stage_narrative.entered_at = datetime.now()
            stage_narrative.entry_summary = f"Entered from {transition_from}"

        # Cache and persist
        self._narratives[embryo_id] = narrative
        if self.auto_persist:
            self._persist_narrative(embryo_id, narrative)

        return narrative

    def _get_or_create_stage_narrative(
        self,
        narrative: EmbryoNarrative,
        embryo_id: str,
        stage: str,
        timepoint: int,
        confidence: float = 0.7,
    ) -> StageNarrative:
        """Get existing stage narrative or create new one."""
        # Look for existing
        for sn in narrative.stage_narratives:
            if sn.stage == stage:
                return sn

        # Create new using factory method
        stage_narrative = StageNarrative.create_initial(
            embryo_id=embryo_id,
            stage=stage,
            timepoint=timepoint,
            confidence=confidence,
            entry_observed=False,  # Unknown unless we see transition
        )
        narrative.add_stage_narrative(stage_narrative)
        return stage_narrative

    def _find_stage_narrative(
        self,
        narrative: EmbryoNarrative,
        stage: str,
    ) -> Optional[StageNarrative]:
        """Find the most recent stage narrative for a given stage."""
        for sn in reversed(narrative.stage_narratives):
            if sn.stage == stage:
                return sn
        return None

    def get_narrative(self, embryo_id: str) -> Optional[EmbryoNarrative]:
        """Get narrative for an embryo (from cache or disk)."""
        if embryo_id in self._narratives:
            return self._narratives[embryo_id]

        # Try to load from disk
        narrative = self._load_narrative(embryo_id)
        if narrative:
            self._narratives[embryo_id] = narrative
        return narrative

    def record_monitor_result(
        self,
        embryo_id: str,
        result: MonitorResult,
        timepoint: int,
    ) -> None:
        """
        Record a monitor result (lightweight observation).

        Updates world model counter but not full narrative.
        """
        model = self.get_world_model(embryo_id)
        if model:
            model.monitors_since_last_deep += 1
            model.observations_in_stage += 1
            model.confidence_trend.append(result.confidence)
            if len(model.confidence_trend) > 10:
                model.confidence_trend = model.confidence_trend[-10:]

            self._world_models[embryo_id] = model
            if self.auto_persist:
                self._persist_world_model(embryo_id, model)

    def reset_deep_counter(self, embryo_id: str) -> None:
        """Reset the monitors_since_last_deep counter after a deep perception."""
        model = self.get_world_model(embryo_id)
        if model:
            model.monitors_since_last_deep = 0
            self._world_models[embryo_id] = model
            if self.auto_persist:
                self._persist_world_model(embryo_id, model)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _ensure_embryo_dir(self, embryo_id: str) -> Path:
        """Ensure embryo directory exists."""
        embryo_dir = self.narratives_dir / "embryos" / embryo_id
        embryo_dir.mkdir(parents=True, exist_ok=True)
        return embryo_dir

    def _persist_world_model(self, embryo_id: str, model: WorldModel) -> None:
        """Persist world model to disk."""
        embryo_dir = self._ensure_embryo_dir(embryo_id)
        path = embryo_dir / "world_model.json"
        with open(path, "w") as f:
            json.dump(model.to_dict(), f, indent=2, default=str)

    def _load_world_model(self, embryo_id: str) -> Optional[WorldModel]:
        """Load world model from disk."""
        path = self.narratives_dir / "embryos" / embryo_id / "world_model.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return WorldModel.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _persist_narrative(self, embryo_id: str, narrative: EmbryoNarrative) -> None:
        """Persist embryo narrative to disk."""
        embryo_dir = self._ensure_embryo_dir(embryo_id)
        path = embryo_dir / "narrative.json"
        with open(path, "w") as f:
            json.dump(narrative.to_dict(), f, indent=2, default=str)

    def _load_narrative(self, embryo_id: str) -> Optional[EmbryoNarrative]:
        """Load embryo narrative from disk."""
        path = self.narratives_dir / "embryos" / embryo_id / "narrative.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            return EmbryoNarrative.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def persist_all(self) -> None:
        """Persist all cached state to disk."""
        for embryo_id, model in self._world_models.items():
            self._persist_world_model(embryo_id, model)
        for embryo_id, narrative in self._narratives.items():
            self._persist_narrative(embryo_id, narrative)

    def load_all_embryos(self) -> List[str]:
        """Load all embryos from disk and return their IDs."""
        embryos_dir = self.narratives_dir / "embryos"
        if not embryos_dir.exists():
            return []

        embryo_ids = []
        for embryo_dir in embryos_dir.iterdir():
            if embryo_dir.is_dir():
                embryo_id = embryo_dir.name
                self.get_world_model(embryo_id)
                self.get_narrative(embryo_id)
                embryo_ids.append(embryo_id)

        return embryo_ids

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text that may contain markdown or other content."""
        # Try direct parse first
        text = text.strip()
        if text.startswith("{"):
            return json.loads(text)

        # Look for JSON in code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Look for bare JSON object
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))

        raise json.JSONDecodeError("No JSON found", text, 0)
