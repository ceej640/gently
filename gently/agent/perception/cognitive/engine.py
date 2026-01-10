"""
Cognitive Engine for Embryo Perception.

Main orchestrator that routes between monitor and deep perception modes,
handles multi-turn cognition, and manages the cognitive architecture.

The VLM (Claude) IS the cognitive system. This engine is scaffolding that:
1. Manages what goes into the VLM's context (working memory)
2. Persists what comes out (external memory)
3. Orchestrates multi-turn reasoning when needed
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field

import anthropic

from .world_model import WorldModel, Predictions, generate_predictions
from .memory import EmbryoNarrative
from .trace import (
    CognitiveTrace,
    TurnTrace,
    MonitorResult,
    WorldModelSnapshot,
)
from .context_manager import ContextManager, VLMContext
from .memory_manager import MemoryManager
from .monitor import Monitor, MonitorHistory, MonitorCheckResult
from .mode_selector import (
    select_perception_mode,
    should_use_multi_turn,
    is_backward_jump,
    get_mode_reason,
    get_escalation_reason,
)
from ..example_store import ExampleStore


@dataclass
class PerceptionResult:
    """Result from cognitive perception."""

    # Core result
    stage: str
    confidence: float
    is_transition: bool
    transition_from: Optional[str]

    # Mode used
    mode: Literal["initial", "monitor", "single_turn", "multi_turn", "escalation"]
    num_turns: int

    # Cognition metadata
    surprise_level: str
    surprise_reason: str
    backward_rejected: bool

    # World model
    world_model: WorldModel

    # Tracing
    trace: CognitiveTrace

    # Timing
    duration_ms: int

    def to_session_result(self) -> "SessionPerceptionResult":
        """
        Convert to session.PerceptionResult for compatibility with PerceptionManager.

        This allows the cognitive engine to be used as a drop-in replacement
        for the old perception engine.
        """
        from ..session import PerceptionResult as SessionPerceptionResult

        # Build reasoning string from trace
        reasoning = self.surprise_reason
        if self.trace.narrative_update:
            reasoning = f"{reasoning}. {self.trace.narrative_update}"

        return SessionPerceptionResult(
            stage=self.stage,
            is_hatching=(self.stage == "hatching"),
            confidence=self.confidence,
            reasoning=reasoning,
            observed_features=None,  # Could extract from trace if needed
            contrastive_reasoning=None,
            is_transitional=self.is_transition,
            transition_between=[self.transition_from, self.stage] if self.is_transition else None,
            reasoning_trace=None,  # Different trace format
            should_stop=(self.stage == "hatched"),
            verification_triggered=(self.mode == "multi_turn"),
            verification_result=None,
            candidate_stages=None,
            multi_phase_trace=None,
            phase_count=self.num_turns,
        )


# Alias for import convenience
SessionPerceptionResult = None  # Will be set on first use


class CognitiveEngine:
    """
    Main cognitive architecture engine.

    Routes perception through appropriate modes:
    - Initial: First observation, no prior beliefs
    - Monitor: Lightweight "still X stage?" checks (Haiku)
    - Deep (single-turn): Full assessment (Sonnet)
    - Deep (multi-turn): Extended reasoning with verification (Sonnet)
    - Escalation: Monitor flagged uncertainty, deep follow-up
    """

    # Sonnet model for deep perception
    DEEP_MODEL = "claude-sonnet-4-20250514"

    # Max tokens for deep responses
    DEEP_MAX_TOKENS = 1000

    def __init__(
        self,
        client: anthropic.Anthropic,
        narratives_dir: Path,
        example_store: Optional[ExampleStore] = None,
        interval_minutes: float = 5.0,
        force_deep: bool = False,
    ):
        """
        Parameters
        ----------
        client : anthropic.Anthropic
            Anthropic API client
        narratives_dir : Path
            Directory for narrative persistence
        example_store : ExampleStore, optional
            Source for reference images
        interval_minutes : float
            Time interval between observations
        force_deep : bool
            Force deep mode for all observations (debugging)
        """
        self.client = client
        self.force_deep = force_deep
        self.interval_minutes = interval_minutes

        # Initialize managers
        self.context_manager = ContextManager(
            example_store=example_store,
            max_references_per_stage=2,
            interval_minutes=interval_minutes,
        )
        self.memory_manager = MemoryManager(
            narratives_dir=narratives_dir,
            auto_persist=True,
        )
        self.monitor = Monitor(
            client=client,
            context_manager=self.context_manager,
            memory_manager=self.memory_manager,
        )
        self.monitor_history = MonitorHistory()

    def perceive(
        self,
        image_b64: str,
        embryo_id: str,
        timepoint: int,
        media_type: str = "image/jpeg",
        force_mode: Optional[Literal["monitor", "deep"]] = None,
    ) -> PerceptionResult:
        """
        Main entry point for perception.

        Routes to appropriate mode based on world model state.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded image
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint
        media_type : str
            Image media type
        force_mode : str, optional
            Force specific mode ("monitor" or "deep")

        Returns
        -------
        PerceptionResult
            Full perception result with trace
        """
        start_time = time.time()

        # Get current world model (if any)
        world_model = self.memory_manager.get_world_model(embryo_id)
        narrative = self.memory_manager.get_narrative(embryo_id)

        # First observation?
        if world_model is None:
            result = self._initial_classification(
                image_b64, embryo_id, timepoint, media_type
            )
            result.duration_ms = int((time.time() - start_time) * 1000)
            return result

        # Select mode
        if force_mode:
            mode = force_mode
        else:
            mode = select_perception_mode(
                world_model=world_model,
                monitor_history=self.monitor_history.get(embryo_id),
                force_deep=self.force_deep,
            )

        if mode == "monitor":
            result = self._monitor_perception(
                image_b64, world_model, narrative, embryo_id, timepoint, media_type
            )
        else:
            result = self._deep_perception(
                image_b64, world_model, narrative, embryo_id, timepoint, media_type
            )

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    def _initial_classification(
        self,
        image_b64: str,
        embryo_id: str,
        timepoint: int,
        media_type: str,
    ) -> PerceptionResult:
        """
        Handle first observation of an embryo.

        No prior beliefs - pure classification.
        """
        start_time = time.time()

        # Build context
        context = self.context_manager.build_initial_context(
            image_b64=image_b64,
            embryo_id=embryo_id,
            timepoint=timepoint,
            media_type=media_type,
        )

        # Call VLM
        raw_output = self._call_deep_vlm(context)

        # Parse output
        parsed = self.memory_manager.parse_initial_output(raw_output)

        # Create world model
        world_model = self.memory_manager.update_world_model(
            embryo_id=embryo_id,
            parsed_output=parsed,
            timepoint=timepoint,
            mode="initial",
        )

        # Create narrative
        self.memory_manager.update_narrative(
            embryo_id=embryo_id,
            observation_text=parsed.get("reasoning", ""),
            timepoint=timepoint,
            stage=world_model.current_stage,
            confidence=world_model.stage_confidence,
        )

        # Build trace
        turn = TurnTrace(
            turn_number=1,
            operation="initial",
            context_summary=context.context_summary,
            task_prompt="Initial classification",
            output_summary=f"Classified as {world_model.current_stage}",
            raw_output=raw_output,
            references_shown=None,
            duration_ms=int((time.time() - start_time) * 1000),
        )

        trace = CognitiveTrace(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode="single_turn",
            num_turns=1,
            total_duration_ms=turn.duration_ms,
            surprise_level="none",
            surprise_reason="First observation",
            prior_model=None,
            posterior_model=WorldModelSnapshot.from_world_model(world_model),
            turns=[turn],
            final_stage=world_model.current_stage,
            final_confidence=world_model.stage_confidence,
            stage_changed=False,
            backward_rejected=False,
            narrative_update=f"Initial observation: {world_model.current_stage}",
        )

        return PerceptionResult(
            stage=world_model.current_stage,
            confidence=world_model.stage_confidence,
            is_transition=False,
            transition_from=None,
            mode="initial",
            num_turns=1,
            surprise_level="none",
            surprise_reason="First observation",
            backward_rejected=False,
            world_model=world_model,
            trace=trace,
            duration_ms=0,  # Will be set by caller
        )

    def _monitor_perception(
        self,
        image_b64: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative],
        embryo_id: str,
        timepoint: int,
        media_type: str,
    ) -> PerceptionResult:
        """
        Lightweight monitor mode perception.

        If monitor flags uncertainty, escalates to deep.
        """
        prior_snapshot = WorldModelSnapshot.from_world_model(world_model)

        # Perform monitor check
        check_result = self.monitor.check(
            image_b64=image_b64,
            world_model=world_model,
            embryo_id=embryo_id,
            timepoint=timepoint,
            media_type=media_type,
        )

        # Track in history
        self.monitor_history.add(embryo_id, check_result.result)

        # Escalate if needed
        if check_result.should_escalate:
            return self._escalation_perception(
                image_b64=image_b64,
                world_model=world_model,
                narrative=narrative,
                embryo_id=embryo_id,
                timepoint=timepoint,
                media_type=media_type,
                monitor_result=check_result,
            )

        # Monitor confirmed - update model with observation count
        world_model = self.memory_manager.get_world_model(embryo_id)

        # Build trace
        turn = TurnTrace(
            turn_number=1,
            operation="monitor",
            context_summary="Quick check: still {world_model.current_stage}?",
            task_prompt="Monitor check",
            output_summary=f"Confirmed: {check_result.result.confirmation}",
            raw_output=check_result.raw_output,
            references_shown=None,
            duration_ms=check_result.duration_ms,
        )

        trace = CognitiveTrace(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode="single_turn",
            num_turns=1,
            total_duration_ms=check_result.duration_ms,
            surprise_level="none",
            surprise_reason="Monitor confirmed stage",
            prior_model=prior_snapshot,
            posterior_model=WorldModelSnapshot.from_world_model(world_model),
            turns=[turn],
            final_stage=world_model.current_stage,
            final_confidence=check_result.result.confidence,
            stage_changed=False,
            backward_rejected=False,
            narrative_update=check_result.result.brief_observation,
        )

        return PerceptionResult(
            stage=world_model.current_stage,
            confidence=check_result.result.confidence,
            is_transition=False,
            transition_from=None,
            mode="monitor",
            num_turns=1,
            surprise_level="none",
            surprise_reason="Monitor confirmed",
            backward_rejected=False,
            world_model=world_model,
            trace=trace,
            duration_ms=check_result.duration_ms,
        )

    def _escalation_perception(
        self,
        image_b64: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative],
        embryo_id: str,
        timepoint: int,
        media_type: str,
        monitor_result: MonitorCheckResult,
    ) -> PerceptionResult:
        """
        Escalated perception after monitor flagged uncertainty.

        Includes monitor observation in context.
        """
        prior_snapshot = WorldModelSnapshot.from_world_model(world_model)
        start_time = time.time()
        turns = []

        # Record monitor turn
        monitor_turn = TurnTrace(
            turn_number=1,
            operation="monitor",
            context_summary="Quick check (triggered escalation)",
            task_prompt="Monitor check",
            output_summary=f"Flagged: {monitor_result.escalation_reason}",
            raw_output=monitor_result.raw_output,
            references_shown=None,
            duration_ms=monitor_result.duration_ms,
        )
        turns.append(monitor_turn)

        # Build escalation context
        context = self.context_manager.build_escalation_context(
            image_b64=image_b64,
            world_model=world_model,
            monitor_observation=monitor_result.result.brief_observation,
            narrative=narrative,
            current_timepoint=timepoint,
            media_type=media_type,
        )

        # Call deep VLM
        deep_start = time.time()
        raw_output = self._call_deep_vlm(context)
        deep_duration = int((time.time() - deep_start) * 1000)

        # Parse output
        parsed = self.memory_manager.parse_single_turn_output(raw_output)
        beliefs = parsed.get("updated_beliefs", {})

        # Check for backward jump
        detected_stage = beliefs.get("stage", world_model.current_stage)
        backward_rejected = False

        if is_backward_jump(world_model.current_stage, detected_stage, world_model.stages_visited):
            backward_result = self._handle_backward_jump(
                observation_text=parsed.get("observation", ""),
                detected_stage=detected_stage,
                world_model=world_model,
                narrative=narrative,
                timepoint=timepoint,
            )
            backward_rejected = True
            parsed = backward_result["parsed"]
            turns.extend(backward_result["turns"])

        # Check if multi-turn needed
        needs_multi = should_use_multi_turn(
            surprise_level=parsed.get("surprise_level", "low"),
            needs_verification=parsed.get("needs_verification", False),
            candidate_stages=parsed.get("candidate_stages", []),
        )

        if needs_multi and not backward_rejected:
            multi_result = self._multi_turn_perception(
                image_b64=image_b64,
                observation_text=parsed.get("observation", ""),
                candidate_stages=parsed.get("candidate_stages", []),
                world_model=world_model,
                narrative=narrative,
                embryo_id=embryo_id,
                timepoint=timepoint,
                media_type=media_type,
            )
            turns.extend(multi_result["turns"])
            parsed = multi_result["parsed"]
            mode = "multi_turn"
        else:
            # Record escalation turn
            escalation_turn = TurnTrace(
                turn_number=2,
                operation="escalation",
                context_summary=context.context_summary,
                task_prompt="Escalation from monitor",
                output_summary=f"Stage: {detected_stage}, Surprise: {parsed.get('surprise_level')}",
                raw_output=raw_output,
                references_shown=None,
                duration_ms=deep_duration,
            )
            turns.append(escalation_turn)
            mode = "escalation"

        # Update world model
        updated_model = self.memory_manager.update_world_model(
            embryo_id=embryo_id,
            parsed_output=parsed,
            timepoint=timepoint,
            mode="single_turn",
        )

        # Reset deep counter
        self.memory_manager.reset_deep_counter(embryo_id)
        self.monitor_history.clear(embryo_id)

        # Detect transition
        is_transition = updated_model.current_stage != world_model.current_stage
        transition_from = world_model.current_stage if is_transition else None

        # Update narrative
        beliefs = parsed.get("updated_beliefs", parsed)
        self.memory_manager.update_narrative(
            embryo_id=embryo_id,
            observation_text=parsed.get("observation", beliefs.get("reasoning", "")),
            timepoint=timepoint,
            stage=updated_model.current_stage,
            confidence=updated_model.stage_confidence,
            transition_detected=is_transition,
            transition_from=transition_from,
        )

        total_duration = int((time.time() - start_time) * 1000)

        trace = CognitiveTrace(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode=mode,
            num_turns=len(turns),
            total_duration_ms=total_duration,
            surprise_level=parsed.get("surprise_level", "medium"),
            surprise_reason=monitor_result.escalation_reason or "Escalated from monitor",
            prior_model=prior_snapshot,
            posterior_model=WorldModelSnapshot.from_world_model(updated_model),
            turns=turns,
            final_stage=updated_model.current_stage,
            final_confidence=updated_model.stage_confidence,
            stage_changed=is_transition,
            backward_rejected=backward_rejected,
            narrative_update=parsed.get("observation", ""),
        )

        return PerceptionResult(
            stage=updated_model.current_stage,
            confidence=updated_model.stage_confidence,
            is_transition=is_transition,
            transition_from=transition_from,
            mode=mode,
            num_turns=len(turns),
            surprise_level=parsed.get("surprise_level", "medium"),
            surprise_reason=monitor_result.escalation_reason or "Escalated from monitor",
            backward_rejected=backward_rejected,
            world_model=updated_model,
            trace=trace,
            duration_ms=total_duration,
        )

    def _deep_perception(
        self,
        image_b64: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative],
        embryo_id: str,
        timepoint: int,
        media_type: str,
    ) -> PerceptionResult:
        """
        Full deep perception mode.

        Single-turn by default, extends to multi-turn if high surprise.
        """
        prior_snapshot = WorldModelSnapshot.from_world_model(world_model)
        start_time = time.time()
        turns = []

        # Build context
        context = self.context_manager.build_single_turn_context(
            image_b64=image_b64,
            world_model=world_model,
            narrative=narrative,
            current_timepoint=timepoint,
            media_type=media_type,
        )

        # Call VLM
        raw_output = self._call_deep_vlm(context)
        single_turn_duration = int((time.time() - start_time) * 1000)

        # Parse output
        parsed = self.memory_manager.parse_single_turn_output(raw_output)
        beliefs = parsed.get("updated_beliefs", {})

        # Record turn
        turn = TurnTrace(
            turn_number=1,
            operation="single_turn",
            context_summary=context.context_summary,
            task_prompt="Full cognitive assessment",
            output_summary=f"Stage: {beliefs.get('stage')}, Surprise: {parsed.get('surprise_level')}",
            raw_output=raw_output,
            references_shown=None,
            duration_ms=single_turn_duration,
        )
        turns.append(turn)

        # Check for backward jump
        detected_stage = beliefs.get("stage", world_model.current_stage)
        backward_rejected = False

        if is_backward_jump(world_model.current_stage, detected_stage, world_model.stages_visited):
            backward_result = self._handle_backward_jump(
                observation_text=parsed.get("observation", ""),
                detected_stage=detected_stage,
                world_model=world_model,
                narrative=narrative,
                timepoint=timepoint,
            )
            backward_rejected = True
            parsed = backward_result["parsed"]
            turns.extend(backward_result["turns"])

        # Check if multi-turn needed
        needs_multi = should_use_multi_turn(
            surprise_level=parsed.get("surprise_level", "low"),
            needs_verification=parsed.get("needs_verification", False),
            candidate_stages=parsed.get("candidate_stages", []),
        )

        mode = "single_turn"
        if needs_multi and not backward_rejected:
            multi_result = self._multi_turn_perception(
                image_b64=image_b64,
                observation_text=parsed.get("observation", ""),
                candidate_stages=parsed.get("candidate_stages", []),
                world_model=world_model,
                narrative=narrative,
                embryo_id=embryo_id,
                timepoint=timepoint,
                media_type=media_type,
            )
            turns.extend(multi_result["turns"])
            parsed = multi_result["parsed"]
            mode = "multi_turn"

        # Update world model
        updated_model = self.memory_manager.update_world_model(
            embryo_id=embryo_id,
            parsed_output=parsed,
            timepoint=timepoint,
            mode="single_turn" if mode == "single_turn" else "integrate",
        )

        # Reset deep counter
        self.memory_manager.reset_deep_counter(embryo_id)
        self.monitor_history.clear(embryo_id)

        # Detect transition
        is_transition = updated_model.current_stage != world_model.current_stage
        transition_from = world_model.current_stage if is_transition else None

        # Update narrative
        beliefs = parsed.get("updated_beliefs", parsed)
        self.memory_manager.update_narrative(
            embryo_id=embryo_id,
            observation_text=parsed.get("observation", beliefs.get("reasoning", "")),
            timepoint=timepoint,
            stage=updated_model.current_stage,
            confidence=updated_model.stage_confidence,
            transition_detected=is_transition,
            transition_from=transition_from,
        )

        total_duration = int((time.time() - start_time) * 1000)

        trace = CognitiveTrace(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            mode=mode,
            num_turns=len(turns),
            total_duration_ms=total_duration,
            surprise_level=parsed.get("surprise_level", "low"),
            surprise_reason=parsed.get("surprise_reason", ""),
            prior_model=prior_snapshot,
            posterior_model=WorldModelSnapshot.from_world_model(updated_model),
            turns=turns,
            final_stage=updated_model.current_stage,
            final_confidence=updated_model.stage_confidence,
            stage_changed=is_transition,
            backward_rejected=backward_rejected,
            narrative_update=parsed.get("observation", ""),
        )

        return PerceptionResult(
            stage=updated_model.current_stage,
            confidence=updated_model.stage_confidence,
            is_transition=is_transition,
            transition_from=transition_from,
            mode=mode,
            num_turns=len(turns),
            surprise_level=parsed.get("surprise_level", "low"),
            surprise_reason=parsed.get("surprise_reason", ""),
            backward_rejected=backward_rejected,
            world_model=updated_model,
            trace=trace,
            duration_ms=total_duration,
        )

    def _multi_turn_perception(
        self,
        image_b64: str,
        observation_text: str,
        candidate_stages: List[str],
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative],
        embryo_id: str,
        timepoint: int,
        media_type: str,
    ) -> Dict[str, Any]:
        """
        Extended multi-turn reasoning for uncertain cases.

        Separate API calls for:
        1. PERCEIVE (if not already done)
        2. COMPARE
        3. VERIFY
        4. INTEGRATE
        """
        turns = []

        # PERCEIVE turn (unbiased observation)
        if not observation_text:
            perceive_start = time.time()
            perceive_context = self.context_manager.build_perceive_context(
                image_b64=image_b64,
                media_type=media_type,
            )
            perceive_output = self._call_deep_vlm(perceive_context)
            observation_text = self.memory_manager.parse_perceive_output(perceive_output)

            turns.append(TurnTrace(
                turn_number=len(turns) + 1,
                operation="perceive",
                context_summary=perceive_context.context_summary,
                task_prompt="Unbiased observation",
                output_summary=observation_text[:100],
                raw_output=perceive_output,
                references_shown=None,
                duration_ms=int((time.time() - perceive_start) * 1000),
            ))

        # COMPARE turn
        compare_start = time.time()
        compare_prompt, compare_summary = self.context_manager.build_compare_context(
            observation_text=observation_text,
            world_model=world_model,
            current_timepoint=timepoint,
        )
        compare_output = self._call_deep_vlm_text(compare_prompt)
        compare_parsed = self.memory_manager.parse_compare_output(compare_output)

        turns.append(TurnTrace(
            turn_number=len(turns) + 1,
            operation="compare",
            context_summary=compare_summary,
            task_prompt="Compare observation to model",
            output_summary=f"Surprise: {compare_parsed.get('surprise_level')}, Uncertain: {compare_parsed.get('still_uncertain')}",
            raw_output=compare_output,
            references_shown=None,
            duration_ms=int((time.time() - compare_start) * 1000),
        ))

        # Get candidate stages for verification
        if compare_parsed.get("still_uncertain") and compare_parsed.get("candidate_stages"):
            candidate_stages = compare_parsed["candidate_stages"]
        elif not candidate_stages:
            # Default to current and adjacent stages
            from ..stages import get_adjacent_stages
            prev, next_s = get_adjacent_stages(world_model.current_stage)
            candidate_stages = [world_model.current_stage]
            if next_s:
                candidate_stages.append(next_s)
            if prev:
                candidate_stages.insert(0, prev)

        # VERIFY turn
        verify_start = time.time()
        verify_context = self.context_manager.build_verify_context(
            image_b64=image_b64,
            observation_text=observation_text,
            candidate_stages=candidate_stages,
            media_type=media_type,
        )
        verify_output = self._call_deep_vlm(verify_context)
        verify_parsed = self.memory_manager.parse_verify_output(verify_output)

        turns.append(TurnTrace(
            turn_number=len(turns) + 1,
            operation="verify",
            context_summary=verify_context.context_summary,
            task_prompt="Verify against references",
            output_summary=f"Verified: {verify_parsed.get('verified_stage')} ({verify_parsed.get('confidence'):.0%})",
            raw_output=verify_output,
            references_shown=[ref["label"] for ref in verify_context.reference_images],
            duration_ms=int((time.time() - verify_start) * 1000),
        ))

        # INTEGRATE turn
        integrate_start = time.time()
        integrate_prompt, integrate_summary = self.context_manager.build_integrate_context(
            perceive_output=observation_text,
            compare_output=compare_output,
            verify_output=verify_output,
            world_model=world_model,
            narrative=narrative,
            current_timepoint=timepoint,
        )
        integrate_output = self._call_deep_vlm_text(integrate_prompt)
        integrate_parsed = self.memory_manager.parse_integrate_output(integrate_output)

        turns.append(TurnTrace(
            turn_number=len(turns) + 1,
            operation="integrate",
            context_summary=integrate_summary,
            task_prompt="Integrate all evidence",
            output_summary=f"Final: {integrate_parsed.get('updated_stage')} ({integrate_parsed.get('updated_confidence'):.0%})",
            raw_output=integrate_output,
            references_shown=None,
            duration_ms=int((time.time() - integrate_start) * 1000),
        ))

        # Convert integrate output to standard format
        transition = integrate_parsed.get("transition", {})
        parsed = {
            "observation": observation_text,
            "surprise_level": compare_parsed.get("surprise_level", "medium"),
            "surprise_reason": compare_parsed.get("surprise_reason", ""),
            "needs_verification": False,
            "candidate_stages": candidate_stages,
            "updated_beliefs": {
                "stage": integrate_parsed.get("updated_stage", verify_parsed.get("verified_stage", "")),
                "confidence": integrate_parsed.get("updated_confidence", verify_parsed.get("confidence", 0.7)),
                "transition_detected": transition.get("detected", False),
                "transition_from": transition.get("from"),
                "key_features": transition.get("evidence", []),
                "reasoning": integrate_parsed.get("narrative_update", ""),
            },
        }

        return {"turns": turns, "parsed": parsed}

    def _handle_backward_jump(
        self,
        observation_text: str,
        detected_stage: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative],
        timepoint: int,
    ) -> Dict[str, Any]:
        """
        Handle biologically impossible backward developmental jump.

        The VLM reasons about whether the observation or model is wrong,
        then makes a decision.
        """
        turns = []
        start_time = time.time()

        # Build backward rejection context
        reject_prompt, reject_summary = self.context_manager.build_backward_rejection_context(
            observation_text=observation_text,
            suggested_stage=detected_stage,
            world_model=world_model,
            narrative=narrative,
            current_timepoint=timepoint,
        )

        # Call VLM for reasoning
        reject_output = self._call_deep_vlm_text(reject_prompt)
        reject_parsed = self.memory_manager.parse_backward_rejection_output(reject_output)

        # Fill in maintain_stage if needed
        action = reject_parsed.get("action", {})
        if action.get("maintain_stage") is None:
            action["maintain_stage"] = world_model.current_stage

        turns.append(TurnTrace(
            turn_number=1,  # Will be renumbered by caller
            operation="reject_backward",
            context_summary=reject_summary,
            task_prompt="Backward jump rejection",
            output_summary=f"Conclusion: {reject_parsed.get('conclusion')}, Maintain: {action.get('maintain_stage')}",
            raw_output=reject_output,
            references_shown=None,
            duration_ms=int((time.time() - start_time) * 1000),
        ))

        # Convert to standard format
        confidence = world_model.stage_confidence + action.get("confidence_adjustment", -0.1)
        confidence = max(0.1, min(1.0, confidence))  # Clamp to [0.1, 1.0]

        parsed = {
            "observation": observation_text,
            "surprise_level": "high",
            "surprise_reason": f"Backward jump detected: {detected_stage} < {world_model.current_stage}",
            "needs_verification": False,
            "candidate_stages": [],
            "updated_beliefs": {
                "stage": action.get("maintain_stage", world_model.current_stage),
                "confidence": confidence,
                "transition_detected": False,
                "transition_from": None,
                "key_features": [],
                "reasoning": reject_parsed.get("reasoning", "Backward jump rejected"),
            },
        }

        return {"turns": turns, "parsed": parsed}

    def _call_deep_vlm(self, context: VLMContext) -> str:
        """Make API call to Sonnet with image context."""
        messages = context.to_anthropic_messages()

        response = self.client.messages.create(
            model=self.DEEP_MODEL,
            max_tokens=self.DEEP_MAX_TOKENS,
            messages=messages,
        )

        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def _call_deep_vlm_text(self, prompt: str) -> str:
        """Make API call to Sonnet with text-only prompt."""
        response = self.client.messages.create(
            model=self.DEEP_MODEL,
            max_tokens=self.DEEP_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""

    def get_world_model(self, embryo_id: str) -> Optional[WorldModel]:
        """Get the current world model for an embryo."""
        return self.memory_manager.get_world_model(embryo_id)

    def get_narrative(self, embryo_id: str) -> Optional[EmbryoNarrative]:
        """Get the current narrative for an embryo."""
        return self.memory_manager.get_narrative(embryo_id)

    def persist_all(self) -> None:
        """Persist all state to disk."""
        self.memory_manager.persist_all()
