"""
Perception Engine - Core VLM reasoning for embryo analysis.

Uses tiered model selection for cost optimization:
- FAST (Haiku): Routine checks on stable embryos
- FULL (Sonnet): Stage transitions, moderate changes
- DEEP (Opus): Critical decisions (hatching, anomalies)

Supports extended thinking for multi-round reasoning.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import anthropic

from .session import (
    BeliefState,
    PerceptionSession,
    PerceptionRoundInput,
    PerceptionRoundOutput,
)
from .example_store import ExampleStore

logger = logging.getLogger(__name__)


class PerceptionEngine:
    """
    Core perception engine using tiered VLM reasoning.

    Builds multi-modal prompts with prior beliefs, few-shot examples,
    temporal context, and current images, then parses structured
    JSON output to update beliefs.
    """

    # Model tiers for cost optimization
    FAST_MODEL = "claude-haiku-4-5-20251001"
    FULL_MODEL = "claude-sonnet-4-5-20250514"
    DEEP_MODEL = "claude-opus-4-5-20251101"

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        example_store: Optional[ExampleStore] = None,
        fast_model: Optional[str] = None,
        full_model: Optional[str] = None,
        deep_model: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Claude API client
        example_store : ExampleStore, optional
            Store for few-shot example images
        fast_model : str, optional
            Model for routine checks (default: Haiku)
        full_model : str, optional
            Model for standard analysis (default: Sonnet)
        deep_model : str, optional
            Model for critical decisions (default: Opus)
        """
        self.claude = claude_client
        self.example_store = example_store
        self.fast_model = fast_model or self.FAST_MODEL
        self.full_model = full_model or self.FULL_MODEL
        self.deep_model = deep_model or self.DEEP_MODEL

    def select_tier(
        self,
        beliefs: BeliefState,
        session: PerceptionSession,
    ) -> str:
        """
        Select model tier based on current state.

        Returns
        -------
        str
            "fast", "full", or "deep"
        """
        # DEEP tier triggers - critical decisions
        if beliefs.hatching_in_progress or beliefs.worm_exiting:
            return "deep"
        if beliefs.most_likely_stage in ["pretzel", "3fold"] and beliefs.stage_confidence > 0.7:
            return "deep"  # Near hatching
        if beliefs.possibly_dead and beliefs.dead_confidence > 0.5:
            return "deep"
        if session.blank_frame_count > 0:
            return "deep"  # Need to classify blank

        # FULL tier triggers - stage transitions
        if beliefs.stage_confidence < 0.6:
            return "full"  # Uncertain about stage
        if beliefs.hours_since_change > 1:
            return "full"  # May be stuck

        # FAST tier - routine checks
        return "fast"

    def _get_model_for_tier(self, tier: str) -> str:
        """Get model name for tier"""
        if tier == "fast":
            return self.fast_model
        elif tier == "deep":
            return self.deep_model
        else:
            return self.full_model

    async def run_perception_round(
        self,
        session: PerceptionSession,
        round_input: PerceptionRoundInput,
        force_tier: Optional[str] = None,
    ) -> PerceptionRoundOutput:
        """
        Run a single perception round.

        This is NOT a one-shot classification. It reasons through:
        1. What we knew before (prior beliefs)
        2. What we see now (current + recent images)
        3. How examples compare (few-shot references)
        4. What has changed (temporal reasoning)
        5. Updated beliefs with calibrated confidence

        Parameters
        ----------
        session : PerceptionSession
            Active session for this embryo
        round_input : PerceptionRoundInput
            Input data for this round
        force_tier : str, optional
            Force a specific tier ("fast", "full", "deep")

        Returns
        -------
        PerceptionRoundOutput
            Analysis results with updated beliefs
        """
        # Select model tier
        tier = force_tier or self.select_tier(round_input.prior_beliefs, session)
        model = self._get_model_for_tier(tier)

        logger.info(
            f"[{session.embryo_id}] Running perception round T{round_input.current_timepoint} "
            f"with tier={tier} ({model})"
        )

        # Build the prompt
        if tier == "fast":
            content = self._build_fast_prompt(session, round_input)
            response = await self._call_claude(model, content, max_tokens=500)
            return self._parse_fast_output(response, round_input.prior_beliefs, tier)
        else:
            content = self._build_full_prompt(session, round_input)
            # Use extended thinking for full/deep tiers
            use_thinking = tier == "deep"
            response = await self._call_claude(
                model, content, max_tokens=4000, use_thinking=use_thinking
            )
            return self._parse_full_output(response, tier)

    async def _call_claude(
        self,
        model: str,
        content: List[Dict],
        max_tokens: int = 2000,
        use_thinking: bool = False,
    ) -> str:
        """Make Claude API call"""
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}],
            }

            if use_thinking and "opus" in model.lower():
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": 8000,
                }

            response = await asyncio.to_thread(
                self.claude.messages.create,
                **kwargs,
            )

            # Extract text from response
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

            return ""

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

    def _build_fast_prompt(
        self,
        session: PerceptionSession,
        round_input: PerceptionRoundInput,
    ) -> List[Dict]:
        """
        Build minimal prompt for fast (Haiku) checks.

        Only checks:
        - Is there significant change from prior?
        - Basic stage confirmation
        - Movement detection
        """
        content = []

        # Brief context
        beliefs = round_input.prior_beliefs
        content.append({
            "type": "text",
            "text": f"""Quick check for C. elegans embryo at timepoint {round_input.current_timepoint}.

CURRENT BELIEF: {beliefs.most_likely_stage} (confidence: {beliefs.stage_confidence:.0%})
Hours since change: {beliefs.hours_since_change:.1f}

Look at this image and answer:
1. Does the embryo look similar to before, or has something changed?
2. Is there visible movement/activity?
3. Does the current stage belief seem correct?

Respond in JSON:
{{"changed": true/false, "movement": true/false, "stage_correct": true/false, "notes": "brief observation"}}
"""
        })

        # Current image only
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": round_input.current_image_b64,
            }
        })

        return content

    def _build_full_prompt(
        self,
        session: PerceptionSession,
        round_input: PerceptionRoundInput,
    ) -> List[Dict]:
        """
        Build comprehensive prompt for full/deep analysis.

        Includes prior beliefs, few-shot examples, temporal context,
        and detailed reasoning instructions.
        """
        content = []
        beliefs = round_input.prior_beliefs

        # 1. System context
        content.append({
            "type": "text",
            "text": f"""You are analyzing C. elegans embryo development in real-time microscopy (diSPIM max projection).

This is CONTINUOUS PERCEPTION - not one-shot classification. You must:
1. Consider prior beliefs from previous rounds
2. Examine current and recent images
3. Compare to reference examples (few-shot learning)
4. Reason about what has changed
5. Update beliefs with calibrated confidence

This is perception round {session.rounds_processed + 1} for {session.embryo_id}.
"""
        })

        # 2. Prior beliefs summary
        content.append({
            "type": "text",
            "text": f"""
=== PRIOR BELIEFS ===

Developmental Stage:
- Most likely: {beliefs.most_likely_stage} ({beliefs.stage_confidence*100:.0f}% confident)
- Distribution: {json.dumps(beliefs.stage_distribution, indent=2)}

Anomaly Status:
- Possibly dead: {beliefs.possibly_dead} ({beliefs.dead_confidence*100:.0f}% confidence)
- Technical issue: {beliefs.technical_issue_suspected} ({beliefs.technical_issue_type or 'none'})
- Hours since significant change: {beliefs.hours_since_change:.1f}
- Movement detected: {beliefs.movement_detected}

Hatching Status:
- In progress: {beliefs.hatching_in_progress}
- Breach detected: {beliefs.breach_detected}
- Worm exiting: {beliefs.worm_exiting}
- Complete: {beliefs.hatching_complete}
"""
        })

        # 3. Recent evidence
        if round_input.recent_evidence:
            evidence_text = "\n=== RECENT EVIDENCE ===\n"
            for ev in round_input.recent_evidence[-5:]:
                evidence_text += f"T{ev.timepoint}: {ev.observation}\n"
            content.append({"type": "text", "text": evidence_text})

        # 4. Few-shot examples for relevant stages
        stages_to_show = self._select_relevant_stages(beliefs)
        if round_input.stage_examples:
            content.append({"type": "text", "text": "\n=== REFERENCE EXAMPLES ==="})

            for stage in stages_to_show:
                if stage in round_input.stage_examples and round_input.stage_examples[stage]:
                    content.append({"type": "text", "text": f"\nExamples of {stage.upper()} stage:"})
                    for example_b64 in round_input.stage_examples[stage][:2]:
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": example_b64,
                            }
                        })

        # 5. Anomaly examples if relevant
        if beliefs.hours_since_change > 2 or beliefs.possibly_dead:
            if "dead_embryo" in round_input.anomaly_examples:
                content.append({"type": "text", "text": "\nExamples of DEAD EMBRYOS (unchanged for hours):"})
                for example_b64 in round_input.anomaly_examples["dead_embryo"][:2]:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_b64,
                        }
                    })

        # 6. Recent temporal context images
        if round_input.recent_images:
            content.append({"type": "text", "text": "\n=== RECENT IMAGES (temporal context) ==="})
            for tp, img_b64 in round_input.recent_images[-3:]:
                content.append({"type": "text", "text": f"Timepoint {tp}:"})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    }
                })

        # 7. Current image (FOCUS)
        content.append({
            "type": "text",
            "text": f"\n=== CURRENT IMAGE (T{round_input.current_timepoint}) - ANALYZE THIS ==="
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": round_input.current_image_b64,
            }
        })

        # 8. Hardware context if relevant
        if round_input.recent_errors:
            content.append({
                "type": "text",
                "text": f"\n=== HARDWARE CONTEXT ===\nRecent errors: {round_input.recent_errors}"
            })

        # 9. Output instructions
        content.append({
            "type": "text",
            "text": """
=== YOUR ANALYSIS ===

Reason through:
1. Compare current image to reference examples - which stage does it most resemble?
2. Compare to recent images - what has changed? Is development progressing?
3. Check for anomalies - does this look like a dead embryo? Technical artifact?
4. For late stages: Is the worm INSIDE or OUTSIDE the shell? Is there a visible breach?

Output your analysis in this EXACT JSON format:

```json
{
  "updated_beliefs": {
    "stage_distribution": {"early": 0.0, "comma": 0.0, "pretzel": 0.1, "3fold": 0.1, "hatching": 0.0, "hatched": 0.8},
    "most_likely_stage": "hatched",
    "stage_confidence": 0.8,
    "possibly_dead": false,
    "dead_confidence": 0.0,
    "technical_issue_suspected": false,
    "technical_issue_type": null,
    "movement_detected": true,
    "hatching_in_progress": false,
    "breach_detected": false,
    "worm_exiting": false,
    "hatching_complete": true
  },
  "new_evidence": [
    {
      "observation": "Worm body visible outside eggshell boundary, free-floating in field",
      "supports": ["hatched"],
      "contradicts": ["pretzel", "3fold", "comma"],
      "confidence": 0.9
    }
  ],
  "reasoning": "The current image shows a clear worm body outside the deflated eggshell...",
  "recommended_actions": ["stop_imaging"],
  "analysis_confidence": 0.85,
  "anomaly_alerts": []
}
```

Important:
- Stage distribution MUST sum to 1.0
- Be calibrated - don't over-state confidence
- For hatching: worm must be OUTSIDE shell, not just moving inside
- For dead embryo: look for lack of change AND lack of movement in late stages
"""
        })

        return content

    def _select_relevant_stages(self, beliefs: BeliefState) -> List[str]:
        """Select which stage examples to show based on current beliefs"""
        stage_order = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]

        try:
            current_idx = stage_order.index(beliefs.most_likely_stage)
        except ValueError:
            current_idx = 0

        # Show current stage and neighbors
        stages = []
        for offset in [-1, 0, 1]:
            idx = current_idx + offset
            if 0 <= idx < len(stage_order):
                stages.append(stage_order[idx])

        return stages

    def _parse_fast_output(
        self,
        response: str,
        prior_beliefs: BeliefState,
        tier: str,
    ) -> PerceptionRoundOutput:
        """Parse output from fast (Haiku) check"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"changed": False, "movement": False, "stage_correct": True, "notes": response[:100]}

            # Build minimal output - keep prior beliefs mostly unchanged
            updated_beliefs = BeliefState.from_dict(prior_beliefs.to_dict())
            updated_beliefs.movement_detected = data.get("movement", False)

            new_evidence = []
            if data.get("notes"):
                new_evidence.append({
                    "observation": data["notes"],
                    "supports": [prior_beliefs.most_likely_stage] if data.get("stage_correct") else [],
                    "contradicts": [],
                    "confidence": 0.5,
                })

            return PerceptionRoundOutput(
                updated_beliefs=updated_beliefs,
                new_evidence=new_evidence,
                reasoning=f"Fast check: {data.get('notes', 'No significant change')}",
                recommended_actions=[],
                analysis_confidence=0.5,
                anomaly_alerts=[],
                model_tier=tier,
            )

        except Exception as e:
            logger.warning(f"Failed to parse fast output: {e}")
            return PerceptionRoundOutput(
                updated_beliefs=prior_beliefs,
                new_evidence=[],
                reasoning=f"Parse error: {e}",
                recommended_actions=[],
                analysis_confidence=0.3,
                anomaly_alerts=[],
                model_tier=tier,
            )

    def _parse_full_output(self, response: str, tier: str) -> PerceptionRoundOutput:
        """Parse output from full/deep analysis"""
        try:
            # Extract JSON from response (may be in code block)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Parse updated beliefs
            beliefs_data = data.get("updated_beliefs", {})
            updated_beliefs = BeliefState(
                stage_distribution=beliefs_data.get("stage_distribution", {}),
                most_likely_stage=beliefs_data.get("most_likely_stage", "early"),
                stage_confidence=beliefs_data.get("stage_confidence", 0.5),
                possibly_dead=beliefs_data.get("possibly_dead", False),
                dead_confidence=beliefs_data.get("dead_confidence", 0.0),
                technical_issue_suspected=beliefs_data.get("technical_issue_suspected", False),
                technical_issue_type=beliefs_data.get("technical_issue_type"),
                movement_detected=beliefs_data.get("movement_detected", False),
                hatching_in_progress=beliefs_data.get("hatching_in_progress", False),
                breach_detected=beliefs_data.get("breach_detected", False),
                worm_exiting=beliefs_data.get("worm_exiting", False),
                hatching_complete=beliefs_data.get("hatching_complete", False),
            )

            # Normalize stage distribution
            total = sum(updated_beliefs.stage_distribution.values())
            if total > 0:
                updated_beliefs.stage_distribution = {
                    k: v / total for k, v in updated_beliefs.stage_distribution.items()
                }

            return PerceptionRoundOutput(
                updated_beliefs=updated_beliefs,
                new_evidence=data.get("new_evidence", []),
                reasoning=data.get("reasoning", ""),
                recommended_actions=data.get("recommended_actions", []),
                analysis_confidence=data.get("analysis_confidence", 0.5),
                anomaly_alerts=data.get("anomaly_alerts", []),
                model_tier=tier,
            )

        except Exception as e:
            logger.error(f"Failed to parse full output: {e}")
            logger.debug(f"Raw response: {response[:500]}")

            # Return safe default
            return PerceptionRoundOutput(
                updated_beliefs=BeliefState(),
                new_evidence=[],
                reasoning=f"Parse error: {e}. Raw: {response[:200]}",
                recommended_actions=[],
                analysis_confidence=0.3,
                anomaly_alerts=[],
                model_tier=tier,
            )
