"""
Context Manager for Cognitive Perception.

Assembles VLM context for each cognitive operation by combining:
- The current image (percept)
- World model (beliefs)
- Narrative (compressed history)
- Reference examples (when needed)
- Task-specific prompts
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

from .world_model import WorldModel, generate_predictions
from .memory import EmbryoNarrative, BatchNarrative
from .prompts import (
    get_monitor_prompt,
    get_single_turn_prompt,
    get_perceive_prompt,
    get_compare_prompt,
    get_verify_prompt,
    get_integrate_prompt,
    get_backward_rejection_prompt,
    get_initial_classification_prompt,
)
from ..example_store import ExampleStore
from ..stages import get_adjacent_stages


@dataclass
class VLMContext:
    """Context assembled for a VLM call."""

    # The image to process
    image_b64: str
    media_type: str  # "image/jpeg" or "image/png"

    # System prompt (optional, usually None for Claude)
    system_prompt: Optional[str]

    # User message text
    user_message: str

    # Additional images (references)
    reference_images: List[Dict[str, str]]  # [{"b64": ..., "label": ...}, ...]

    # Metadata for tracing
    context_type: str  # "monitor", "single_turn", "perceive", "compare", "verify", etc.
    context_summary: str  # Human-readable summary

    def to_anthropic_messages(self) -> List[Dict[str, Any]]:
        """Convert to Anthropic API message format."""
        content = []

        # Add main image
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.media_type,
                "data": self.image_b64,
            },
        })

        # Add reference images if any
        for ref in self.reference_images:
            content.append({
                "type": "text",
                "text": f"\n[Reference: {ref['label']}]",
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": ref["b64"],
                },
            })

        # Add user message
        content.append({
            "type": "text",
            "text": self.user_message,
        })

        return [{"role": "user", "content": content}]


class ContextManager:
    """
    Builds VLM contexts for different cognitive operations.

    This is the "working memory" assembly - deciding what goes into
    the VLM's context window for each call.
    """

    def __init__(
        self,
        example_store: Optional[ExampleStore] = None,
        max_references_per_stage: int = 2,
        interval_minutes: float = 5.0,
    ):
        """
        Parameters
        ----------
        example_store : ExampleStore
            Source for reference images
        max_references_per_stage : int
            Maximum reference images per stage in verification
        interval_minutes : float
            Time interval between observations (for timing predictions)
        """
        self.example_store = example_store
        self.max_references = max_references_per_stage
        self.interval_minutes = interval_minutes

    def build_initial_context(
        self,
        image_b64: str,
        embryo_id: str,
        timepoint: int,
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build context for first observation (no prior beliefs).

        This is used when we have no world model yet.
        """
        prompt = get_initial_classification_prompt(embryo_id, timepoint)

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=[],
            context_type="initial",
            context_summary="First observation - no prior model",
        )

    def build_monitor_context(
        self,
        image_b64: str,
        world_model: WorldModel,
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build lightweight context for monitor mode.

        Monitor mode only needs:
        - The image
        - Current stage belief (no full model)
        - Simple yes/no/uncertain task
        """
        prompt = get_monitor_prompt(world_model.current_stage)

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=[],
            context_type="monitor",
            context_summary=f"Quick check: still {world_model.current_stage}?",
        )

    def build_single_turn_context(
        self,
        image_b64: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative] = None,
        current_timepoint: Optional[int] = None,
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build context for single-turn deep perception.

        Includes:
        - The image
        - Full world model
        - Predictions
        - Narrative summary (if available)
        """
        # Ensure predictions are up to date
        if current_timepoint is not None:
            world_model.predictions = generate_predictions(
                world_model, current_timepoint, self.interval_minutes
            )

        # Build world model context text
        model_text = world_model.to_context_text(
            current_timepoint=current_timepoint,
            interval_minutes=self.interval_minutes,
        )

        # Build narrative context text
        narrative_text = ""
        if narrative:
            narrative_text = narrative.to_context_text(include_full_history=False)

        prompt = get_single_turn_prompt(model_text, narrative_text)

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=[],
            context_type="single_turn",
            context_summary=f"Full assessment with model: {world_model.current_stage} ({world_model.stage_confidence:.0%})",
        )

    def build_perceive_context(
        self,
        image_b64: str,
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build context for PERCEIVE turn (unbiased observation).

        CRITICAL: This context has NO world model or stage labels.
        This prevents confirmation bias in multi-turn processing.
        """
        prompt = get_perceive_prompt()

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=[],
            context_type="perceive",
            context_summary="Pure observation (unbiased, no model)",
        )

    def build_compare_context(
        self,
        observation_text: str,
        world_model: WorldModel,
        current_timepoint: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Build context for COMPARE turn (observation vs predictions).

        No image needed - uses text from PERCEIVE turn.

        Returns
        -------
        prompt : str
            The comparison prompt
        context_summary : str
            Human-readable summary
        """
        # Ensure predictions are up to date
        if current_timepoint is not None:
            world_model.predictions = generate_predictions(
                world_model, current_timepoint, self.interval_minutes
            )

        model_text = world_model.to_context_text(
            current_timepoint=current_timepoint,
            interval_minutes=self.interval_minutes,
        )

        prompt = get_compare_prompt(observation_text, model_text)
        context_summary = f"Comparing observation to model: {world_model.current_stage}"

        return prompt, context_summary

    def build_verify_context(
        self,
        image_b64: str,
        observation_text: str,
        candidate_stages: List[str],
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build context for VERIFY turn (targeted comparison with references).

        Loads reference images ONLY for candidate stages.
        """
        reference_images = []

        if self.example_store:
            for stage in candidate_stages:
                examples = self.example_store.get_stage_examples_with_descriptions(
                    stage, max_examples=self.max_references
                )
                for i, ex in enumerate(examples):
                    label = f"{stage.upper()} ref {i+1}"
                    if ex.get("description"):
                        label += f": {ex['description']}"
                    reference_images.append({
                        "b64": ex["image"],
                        "label": label,
                    })

        prompt = get_verify_prompt(observation_text, candidate_stages)

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=reference_images,
            context_type="verify",
            context_summary=f"Verification with references: {', '.join(candidate_stages)}",
        )

    def build_integrate_context(
        self,
        perceive_output: str,
        compare_output: str,
        verify_output: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative] = None,
        current_timepoint: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Build context for INTEGRATE turn (final decision).

        Combines all previous turn outputs with model and narrative.

        Returns
        -------
        prompt : str
            The integration prompt
        context_summary : str
            Human-readable summary
        """
        model_text = world_model.to_context_text(
            current_timepoint=current_timepoint,
            interval_minutes=self.interval_minutes,
        )

        narrative_text = ""
        if narrative:
            narrative_text = narrative.to_context_text(include_full_history=False)

        prompt = get_integrate_prompt(
            perceive_output,
            compare_output,
            verify_output,
            model_text,
            narrative_text,
        )
        context_summary = "Integration: all turns + model + narrative"

        return prompt, context_summary

    def build_backward_rejection_context(
        self,
        observation_text: str,
        suggested_stage: str,
        world_model: WorldModel,
        narrative: Optional[EmbryoNarrative] = None,
        current_timepoint: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Build context for backward jump rejection.

        When observation suggests a previous stage (impossible regression),
        this context helps the VLM reason about what went wrong.

        Returns
        -------
        prompt : str
            The rejection prompt
        context_summary : str
            Human-readable summary
        """
        model_text = world_model.to_context_text(
            current_timepoint=current_timepoint,
            interval_minutes=self.interval_minutes,
        )

        narrative_text = ""
        if narrative:
            narrative_text = narrative.to_context_text(include_full_history=True)

        # Get trajectory
        trajectory = " -> ".join(world_model.stages_visited)

        prompt = get_backward_rejection_prompt(
            observation_text,
            suggested_stage,
            world_model.current_stage,
            trajectory,
            model_text,
            narrative_text,
        )
        context_summary = f"Backward rejection: {suggested_stage} vs current {world_model.current_stage}"

        return prompt, context_summary

    def build_escalation_context(
        self,
        image_b64: str,
        world_model: WorldModel,
        monitor_observation: str,
        narrative: Optional[EmbryoNarrative] = None,
        current_timepoint: Optional[int] = None,
        media_type: str = "image/jpeg",
    ) -> VLMContext:
        """
        Build context for escalation from monitor to deep mode.

        When monitor returns "uncertain" or "no", we escalate to deep
        perception with the monitor's observation included.
        """
        # Similar to single_turn but includes monitor observation
        if current_timepoint is not None:
            world_model.predictions = generate_predictions(
                world_model, current_timepoint, self.interval_minutes
            )

        model_text = world_model.to_context_text(
            current_timepoint=current_timepoint,
            interval_minutes=self.interval_minutes,
        )

        narrative_text = ""
        if narrative:
            narrative_text = narrative.to_context_text(include_full_history=False)

        # Modified prompt that includes monitor observation
        prompt = f"""Monitor check flagged uncertainty: "{monitor_observation}"

Perform full cognitive assessment.

{model_text}

{narrative_text if narrative_text else ""}

TASK:
1. Observe this image - what do you see?
2. Compare to your predictions - any surprises?
3. Rate surprise: none / low / medium / high
4. Update your beliefs accordingly.
5. Output your updated model state.

If surprise is MEDIUM or higher, set needs_verification: true

OUTPUT FORMAT (JSON):
{{
  "observation": "what you observed",
  "surprise_level": "none|low|medium|high",
  "surprise_reason": "why this level",
  "needs_verification": false,
  "candidate_stages": [],
  "updated_beliefs": {{
    "stage": "stage_name",
    "confidence": 0.85,
    "transition_detected": false,
    "transition_from": null,
    "key_features": [],
    "reasoning": "explanation"
  }}
}}"""

        return VLMContext(
            image_b64=image_b64,
            media_type=media_type,
            system_prompt=None,
            user_message=prompt,
            reference_images=[],
            context_type="escalation",
            context_summary=f"Escalation from monitor: {monitor_observation[:50]}...",
        )

    def get_references_for_stages(
        self,
        stages: List[str],
    ) -> List[Dict[str, str]]:
        """
        Load reference images for specified stages.

        Used for building custom contexts.
        """
        if not self.example_store:
            return []

        references = []
        for stage in stages:
            examples = self.example_store.get_stage_examples_with_descriptions(
                stage, max_examples=self.max_references
            )
            for i, ex in enumerate(examples):
                label = f"{stage.upper()} ref {i+1}"
                if ex.get("description"):
                    label += f": {ex['description']}"
                references.append({
                    "b64": ex["image"],
                    "label": label,
                })

        return references

    def get_adjacent_stage_references(
        self,
        current_stage: str,
    ) -> List[Dict[str, str]]:
        """
        Load reference images for adjacent stages (prev, current, next).

        Useful for boundary disambiguation.
        """
        prev_stage, next_stage = get_adjacent_stages(current_stage)

        stages = [current_stage]
        if prev_stage:
            stages.insert(0, prev_stage)
        if next_stage:
            stages.append(next_stage)

        return self.get_references_for_stages(stages)
