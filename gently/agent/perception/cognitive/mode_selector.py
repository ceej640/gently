"""
Mode Selector for Cognitive Perception.

Decides whether to use Monitor (lightweight) or Deep (full cognitive) mode
for each observation based on world model state and configuration.
"""

from typing import List, Literal, Optional

from .world_model import WorldModel
from .trace import MonitorResult
from .config import get_stage_config, StageConfig


PerceptionMode = Literal["monitor", "deep"]


def select_perception_mode(
    world_model: Optional[WorldModel],
    monitor_history: Optional[List[MonitorResult]] = None,
    force_deep: bool = False,
) -> PerceptionMode:
    """
    Decide whether to use monitor or deep perception mode.

    Parameters
    ----------
    world_model : WorldModel or None
        Current world model. None for first observation.
    monitor_history : list of MonitorResult
        Recent monitor results (for escalation detection)
    force_deep : bool
        Force deep mode (e.g., user request, critical embryo)

    Returns
    -------
    PerceptionMode
        "monitor" or "deep"
    """
    if force_deep:
        return "deep"

    # ALWAYS DEEP: First observation (no model yet)
    if world_model is None:
        return "deep"

    stage = world_model.current_stage
    config = get_stage_config(stage)

    # ALWAYS DEEP: Fast-transitioning stages
    if _is_fast_stage(stage, config):
        return "deep"

    # ALWAYS DEEP: First few observations in a new stage
    if world_model.observations_in_stage < config.min_observations_before_monitor:
        return "deep"

    # DEEP: Approaching transition window
    if _in_transition_window(world_model, config):
        return "deep"

    # DEEP: Recent monitor returned uncertain or no
    if monitor_history and _monitor_flagged_issue(monitor_history):
        return "deep"

    # DEEP: Confidence has been dropping
    if _confidence_declining(world_model):
        return "deep"

    # DEEP: Too many monitors without deep check
    if _exceeded_deep_interval(world_model, config):
        return "deep"

    # DEFAULT: Monitor mode
    return "monitor"


def _is_fast_stage(stage: str, config: StageConfig) -> bool:
    """Check if this is a fast-transitioning stage that needs constant deep monitoring."""
    return config.default_mode == "deep"


def _in_transition_window(world_model: WorldModel, config: StageConfig) -> bool:
    """Check if we're in the expected transition window for this stage."""
    # Can only calculate if we observed stage entry
    if not world_model.stage_entry_observed:
        # Without timing, we can't know - use observation count as proxy
        # If we've had many observations, probably approaching transition
        return world_model.observations_in_stage > config.deep_check_interval * 2

    # Calculate progress through stage
    if world_model.entered_at_timepoint is None:
        return False

    # This requires knowing current timepoint - caller should provide via world_model update
    # For now, use observations_in_stage as a proxy
    # Typical observation interval is ~5 min, so observations * 5 / expected_duration
    # But we don't have expected_duration here - it's in the knowledge base

    # Simplified: deep check interval already accounts for transition window
    return False


def _monitor_flagged_issue(monitor_history: List[MonitorResult]) -> bool:
    """Check if recent monitors flagged uncertainty or stage change."""
    if not monitor_history:
        return False

    # Check last result
    last = monitor_history[-1]
    if last.confirmation in ("uncertain", "no"):
        return True

    # Check for red flags
    if last.red_flags:
        return True

    # Check for declining confidence trend in monitors
    if len(monitor_history) >= 3:
        recent = monitor_history[-3:]
        confidences = [m.confidence for m in recent]
        if all(confidences[i] > confidences[i+1] for i in range(len(confidences)-1)):
            # Monotonically decreasing
            return True

    return False


def _confidence_declining(world_model: WorldModel) -> bool:
    """Check if confidence has been monotonically decreasing."""
    trend = world_model.confidence_trend
    if len(trend) < 3:
        return False

    recent = trend[-3:]
    return all(recent[i] > recent[i+1] for i in range(len(recent)-1))


def _exceeded_deep_interval(world_model: WorldModel, config: StageConfig) -> bool:
    """Check if we've exceeded the configured deep check interval."""
    return world_model.monitors_since_last_deep >= config.deep_check_interval


def get_mode_reason(
    world_model: Optional[WorldModel],
    monitor_history: Optional[List[MonitorResult]] = None,
    force_deep: bool = False,
) -> str:
    """
    Get a human-readable reason for the mode selection.

    Useful for tracing and debugging.
    """
    if force_deep:
        return "Forced deep mode"

    if world_model is None:
        return "First observation - no world model"

    stage = world_model.current_stage
    config = get_stage_config(stage)

    if _is_fast_stage(stage, config):
        return f"Fast stage ({stage}) - always deep"

    if world_model.observations_in_stage < config.min_observations_before_monitor:
        return f"Early in stage ({world_model.observations_in_stage} < {config.min_observations_before_monitor})"

    if _in_transition_window(world_model, config):
        return "In transition window"

    if monitor_history and _monitor_flagged_issue(monitor_history):
        last = monitor_history[-1]
        if last.confirmation == "uncertain":
            return "Monitor returned uncertain"
        elif last.confirmation == "no":
            return "Monitor detected stage change"
        elif last.red_flags:
            return f"Monitor flagged: {last.red_flags[0]}"
        else:
            return "Monitor confidence declining"

    if _confidence_declining(world_model):
        return "Confidence trend declining"

    if _exceeded_deep_interval(world_model, config):
        return f"Exceeded deep interval ({world_model.monitors_since_last_deep} >= {config.deep_check_interval})"

    return "Stable stage - using monitor"


def should_use_multi_turn(
    surprise_level: str,
    needs_verification: bool,
    candidate_stages: List[str],
) -> bool:
    """
    Determine if multi-turn cognition is needed after single-turn assessment.

    Called after initial deep perception to decide if extended reasoning is needed.

    Parameters
    ----------
    surprise_level : str
        Surprise level from single-turn ("none", "low", "medium", "high")
    needs_verification : bool
        VLM's self-assessment of whether verification is needed
    candidate_stages : list of str
        Candidate stages to compare (if uncertain)

    Returns
    -------
    bool
        True if multi-turn processing should be used
    """
    # High surprise always needs verification
    if surprise_level == "high":
        return True

    # Medium surprise with uncertainty
    if surprise_level == "medium" and needs_verification:
        return True

    # Explicit request for verification
    if needs_verification and len(candidate_stages) >= 2:
        return True

    return False


def is_backward_jump(
    current_stage: str,
    detected_stage: str,
    stages_visited: List[str],
) -> bool:
    """
    Check if the detected stage would be a backward developmental jump.

    Backward jumps are biologically impossible in C. elegans development.

    Parameters
    ----------
    current_stage : str
        Current believed stage
    detected_stage : str
        Newly detected stage
    stages_visited : list of str
        History of stages visited

    Returns
    -------
    bool
        True if this would be a backward jump
    """
    from ..stages import DevelopmentalStage

    # Can't be backward if detected stage is same as current
    if detected_stage == current_stage:
        return False

    # Can't be backward if detected stage is arrested (special state)
    if detected_stage == "arrested":
        return False

    # Check if detected stage comes before current in developmental order
    try:
        current_order = DevelopmentalStage.get_order(current_stage)
        detected_order = DevelopmentalStage.get_order(detected_stage)
        return detected_order < current_order
    except ValueError:
        # Invalid stage - not a backward jump
        return False


def get_escalation_reason(monitor_result: MonitorResult) -> str:
    """
    Get reason for escalating from monitor to deep mode.

    Parameters
    ----------
    monitor_result : MonitorResult
        The monitor result that triggered escalation

    Returns
    -------
    str
        Human-readable escalation reason
    """
    if monitor_result.confirmation == "uncertain":
        return f"Monitor uncertain: {monitor_result.brief_observation[:100]}"
    elif monitor_result.confirmation == "no":
        return f"Monitor detected change: {monitor_result.brief_observation[:100]}"
    elif monitor_result.red_flags:
        return f"Monitor flagged: {', '.join(monitor_result.red_flags[:2])}"
    elif monitor_result.confidence < 0.6:
        return f"Low monitor confidence: {monitor_result.confidence:.0%}"
    else:
        return "Escalated from monitor"
