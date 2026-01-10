"""
Prompts for Cognitive Perception.

All task prompts for monitor and deep perception modes.
"""

from typing import List, Optional


def get_monitor_prompt(current_stage: str) -> str:
    """
    Get prompt for lightweight monitor check.

    Uses Haiku - must be concise and fast.
    """
    return f"""Quick check: Does this C. elegans embryo still look like it's in the {current_stage.upper()} stage?

Answer with ONE of:
- YES: Clearly still {current_stage}
- NO: Clearly a different stage
- UNCERTAIN: Can't tell, need deeper look

Then provide a brief observation (1-2 sentences).

Note any red flags (unusual appearance, artifacts, quality issues).

OUTPUT FORMAT (JSON):
{{
  "confirmation": "yes|no|uncertain",
  "confidence": 0.85,
  "brief_observation": "1-2 sentence description",
  "red_flags": []
}}"""


def get_initial_classification_prompt(embryo_id: str, timepoint: int) -> str:
    """
    Get prompt for first observation (no prior model).
    """
    return f"""You are observing C. elegans embryo {embryo_id} at timepoint T{timepoint}.
This is your FIRST observation of this embryo. You have no prior beliefs about its stage.

TASK:
1. Observe the embryo morphology carefully
2. Classify it into one of these developmental stages:
   - early: Round/oval shape, grainy texture, many visible nuclei
   - bean: Elongated oval "bean-shaped", clear axis, pre-comma curvature
   - comma: Clear C-curve, pronounced ventral bend, head/tail distinguishable
   - 1.5fold: Elongated ~1.5x shell width, body starting to fold back
   - 2fold: Body folded twice, two clear bends visible
   - pretzel: Tightly coiled, 3+ body segments, maximum compaction
   - hatching: Shell breach visible, part inside/part outside
   - hatched: Fully emerged L1 larva

3. Rate your confidence (0-1)
4. List the key features you observed

OUTPUT FORMAT (JSON):
{{
  "stage": "stage_name",
  "confidence": 0.85,
  "key_features": ["feature1", "feature2"],
  "reasoning": "brief explanation of classification"
}}"""


def get_single_turn_prompt(model_text: str, narrative_text: str) -> str:
    """
    Get prompt for single-turn deep perception.
    """
    narrative_section = f"\n\nNARRATIVE:\n{narrative_text}" if narrative_text else ""

    return f"""You are maintaining a world model of this C. elegans embryo.

{model_text}
{narrative_section}

TASK:
1. Observe this image - what do you see?
2. Compare to your predictions - any surprises?
3. Rate surprise: none / low / medium / high
4. Update your beliefs accordingly.
5. Output your updated model state.

SURPRISE LEVELS:
- none: Observation matches predictions perfectly
- low: Minor differences, forward progression detected
- medium: Features ambiguous between stages, need verification
- high: Backward jump detected or stage skip - biologically unusual

If surprise is MEDIUM or higher, set needs_verification: true

OUTPUT FORMAT (JSON):
{{
  "observation": "what you observed in detail",
  "surprise_level": "none|low|medium|high",
  "surprise_reason": "why this level of surprise",
  "needs_verification": false,
  "candidate_stages": [],
  "updated_beliefs": {{
    "stage": "stage_name",
    "confidence": 0.85,
    "transition_detected": false,
    "transition_from": null,
    "key_features": ["feature1", "feature2"],
    "reasoning": "explanation of your assessment"
  }}
}}"""


def get_perceive_prompt() -> str:
    """
    Get prompt for PERCEIVE turn (unbiased observation).

    CRITICAL: No stage labels or model information.
    """
    return """Describe the morphological features of this C. elegans embryo.

Focus on:
- Overall shape (round, oval, elongated, curved, coiled)
- Curvature (none, slight, C-shaped, folded, tightly coiled)
- Body length relative to eggshell width
- Number of visible folds or bends
- Head/tail differentiation
- Any visible structures or textures

Do NOT classify into a stage. Just describe what you see.

OUTPUT FORMAT (text):
Provide a detailed morphological description in 3-5 sentences."""


def get_compare_prompt(observation_text: str, model_text: str) -> str:
    """
    Get prompt for COMPARE turn (observation vs predictions).
    """
    return f"""Your observation:
{observation_text}

{model_text}

TASK:
Compare your observation to your predictions.

1. Rate surprise: none / low / medium / high
2. Explain why
3. If uncertain, specify which stages to compare (max 2-3)

OUTPUT FORMAT (JSON):
{{
  "surprise_level": "none|low|medium|high",
  "surprise_reason": "explanation",
  "still_uncertain": false,
  "candidate_stages": []
}}"""


def get_verify_prompt(observation_text: str, candidate_stages: List[str]) -> str:
    """
    Get prompt for VERIFY turn (targeted comparison with references).
    """
    stages_upper = [s.upper() for s in candidate_stages]
    stages_list = ", ".join(stages_upper)

    return f"""Your observation:
{observation_text}

You are comparing to reference images for: {stages_list}

TASK:
Compare the embryo image to the reference images provided.
Which stage does your observation match?

For each candidate stage, cite specific features that:
- MATCH the references
- DIFFER from the references

OUTPUT FORMAT (JSON):
{{
  "verified_stage": "stage_name",
  "confidence": 0.85,
  "evidence": {{
    "matches": ["feature that matches reference"],
    "differs": ["feature that differs from other candidates"]
  }},
  "reasoning": "brief explanation"
}}"""


def get_integrate_prompt(
    perceive_output: str,
    compare_output: str,
    verify_output: str,
    model_text: str,
    narrative_text: str,
) -> str:
    """
    Get prompt for INTEGRATE turn (final decision).
    """
    narrative_section = f"\n\nNARRATIVE:\n{narrative_text}" if narrative_text else ""

    return f"""OBSERVATION (Turn 1):
{perceive_output}

COMPARISON (Turn 2):
{compare_output}

VERIFICATION (Turn 3):
{verify_output}

{model_text}
{narrative_section}

TASK:
Integrate all evidence and update your world model.

1. What is your final stage classification?
2. If this is a stage transition, record the evidence.
3. Generate a brief narrative update for this observation.

OUTPUT FORMAT (JSON):
{{
  "updated_stage": "stage_name",
  "updated_confidence": 0.85,
  "transition": {{
    "detected": false,
    "from_stage": null,
    "to_stage": null,
    "evidence": []
  }},
  "key_features": ["feature1", "feature2"],
  "narrative_update": "1-2 sentence summary of this observation",
  "reasoning": "explanation of your final decision"
}}"""


def get_backward_rejection_prompt(
    observation_text: str,
    suggested_stage: str,
    current_stage: str,
    trajectory: str,
    model_text: str,
    narrative_text: str,
) -> str:
    """
    Get prompt for backward jump rejection.
    """
    narrative_section = f"\n\nFULL NARRATIVE:\n{narrative_text}" if narrative_text else ""

    return f"""OBSERVATION:
{observation_text}

This observation suggests: {suggested_stage.upper()}
But your model says: {current_stage.upper()}

Trajectory so far: {trajectory}

{model_text}
{narrative_section}

BIOLOGICAL CONSTRAINT:
Backward developmental jumps are IMPOSSIBLE in C. elegans.
Embryos cannot regress from {current_stage} back to {suggested_stage}.

ANALYSIS REQUIRED:
Either:
A) Your current observation is wrong (misperception, bad angle, artifact)
B) Your model was wrong (previous observations were misclassified)

Consider:
- Current observation features
- Previous observation history
- Confidence levels over recent timepoints
- Biological plausibility

Which is more likely? Explain your reasoning.

OUTPUT FORMAT (JSON):
{{
  "analysis": "detailed analysis of the discrepancy",
  "conclusion": "A|B",
  "reasoning": "why you chose this conclusion",
  "action": {{
    "maintain_stage": "stage to keep",
    "confidence_adjustment": -0.1,
    "flag_for_review": true,
    "backward_rejected": true,
    "notes": "any additional notes"
  }}
}}"""


def get_arrest_detection_prompt(
    observation_text: str,
    model_text: str,
    narrative_text: str,
    recent_observations: List[str],
) -> str:
    """
    Get prompt for detecting arrested/dead embryos.
    """
    recent_text = "\n".join([f"  - T{i}: {obs}" for i, obs in enumerate(recent_observations)])

    return f"""CURRENT OBSERVATION:
{observation_text}

RECENT OBSERVATIONS:
{recent_text}

{model_text}

{narrative_text if narrative_text else ""}

TASK:
Assess whether this embryo may be arrested or dead.

Signs of arrest:
- No morphological change over many timepoints
- Same appearance across consecutive observations
- Degradation, fragmentation, or unusual texture
- No twitching or movement (expected in later stages)
- Collapsed or disintegrating appearance

Signs of healthy development:
- Morphological progression between timepoints
- Normal appearance for the current stage
- Movement or twitching (in later stages)
- Recent stage transition

OUTPUT FORMAT (JSON):
{{
  "assessment": "healthy|possibly_arrested|likely_arrested",
  "confidence": 0.75,
  "evidence": {{
    "signs_of_arrest": ["evidence1"],
    "signs_of_health": ["evidence2"]
  }},
  "recommendation": "continue_monitoring|flag_for_review|mark_as_arrested",
  "reasoning": "explanation"
}}"""


# Stage-specific guidance (can be included in prompts when needed)
STAGE_GUIDANCE = {
    "early": """
EARLY stage characteristics:
- Round to slightly oval shape
- Grainy texture with many visible nuclei
- Uniform or slightly asymmetric cellular mass
- No clear axis of elongation yet
- Compact appearance

NOT early if:
- Elongated bean-like shape
- Clear axis of elongation
- Any hint of C-curve""",

    "bean": """
BEAN stage characteristics:
- Elongated oval, "bean-shaped" appearance
- Clear axis of elongation established
- Slightly asymmetric - one end may be narrower
- Pre-comma curvature - hint of bend but not C-shaped
- Smooth outline, no tight curvature yet

NOT bean if:
- Still round/spherical (that's early)
- Clear C-curve or comma shape (that's comma)
- Pronounced ventral bend""",

    "comma": """
COMMA stage characteristics:
- Clear C-curve or comma shape
- Pronounced ventral bend
- Head/tail distinctly different
- Body axis clearly established

NOT comma if:
- No clear bend, just elongated (that's bean)
- Elongation beyond eggshell width (that's 1.5fold)
- Still oval/round (that's early)""",

    "1.5fold": """
1.5FOLD stage characteristics:
- Elongated ~1.5x eggshell width
- Embryo clearly longer than egg width
- Body starting to fold back on itself
- One fold/bend visible, tail beginning to turn back

NOT 1.5fold if:
- Fits within egg diameter (that's comma)
- Two clear folds visible (that's 2fold)
- Tight coil with 3 segments (that's pretzel)""",

    "2fold": """
2FOLD stage characteristics:
- Body folded back on itself twice
- Two clear bends/folds visible
- ~2x eggshell length when straightened
- More compaction than 1.5fold, less than pretzel
- Head and tail both curving inward

NOT 2fold if:
- Only one fold visible (that's 1.5fold)
- Tight pretzel coil with 3+ segments (that's pretzel)
- Body extending beyond shell (that's hatching)""",

    "pretzel": """
PRETZEL stage characteristics:
- Tightly coiled pretzel-like shape
- 3 or more body segments visible
- Maximum compaction within shell
- May show occasional twitching
- Also called "3fold" - three folds/bends

NOT pretzel if:
- Only two folds visible (that's 2fold)
- Any part outside shell (that's hatching)
- Shell breach visible""",

    "hatching": """
HATCHING stage characteristics:
- Eggshell breach/tear VISIBLE
- Part of embryo OUTSIDE shell
- Part of embryo STILL INSIDE shell
- Active pushing/wriggling to escape

NOT hatching if:
- Fully inside shell (that's pretzel)
- Fully outside shell (that's hatched)
- No visible shell breach""",

    "hatched": """
HATCHED stage characteristics:
- Larva FULLY OUTSIDE eggshell
- Empty or nearly-empty eggshell visible
- Free-moving L1 larva
- Elongated worm body, no longer coiled in shell

NOT hatched if:
- Any part still inside shell (that's hatching)""",
}


def get_stage_guidance(stage: str) -> str:
    """Get detailed guidance for a specific stage."""
    return STAGE_GUIDANCE.get(stage, "")


def get_boundary_guidance(from_stage: str, to_stage: str) -> str:
    """Get guidance for distinguishing between two stages."""
    from_guidance = STAGE_GUIDANCE.get(from_stage, "")
    to_guidance = STAGE_GUIDANCE.get(to_stage, "")

    return f"""STAGE BOUNDARY: {from_stage.upper()} vs {to_stage.upper()}

{from_guidance}

{to_guidance}

KEY DIFFERENCES:
Compare features carefully. The transition from {from_stage} to {to_stage}
involves specific morphological changes. Focus on the distinguishing features."""
