"""
Stage Perception Configuration.

Defines how each developmental stage should be monitored,
including mode selection thresholds and timing parameters.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from ..stages import DevelopmentalStage


@dataclass
class StageConfig:
    """Configuration for perception of a specific stage."""

    # Default perception mode for this stage
    default_mode: Literal["monitor", "deep"]

    # How often to force deep perception (every N observations)
    deep_check_interval: int

    # When to switch to deep mode based on expected duration progress
    # e.g., 0.7 means switch to deep after 70% of expected duration
    transition_window: float

    # Minimum observations in stage before allowing monitor mode
    min_observations_for_monitor: int = 3

    # Expected duration in minutes (from stages.py STAGE_CRITERIA)
    expected_duration_minutes: Optional[float] = None

    def should_use_deep(
        self,
        observations_in_stage: int,
        monitors_since_deep: int,
        progress_ratio: Optional[float],
    ) -> bool:
        """Determine if deep perception should be used."""
        # Always deep for fast stages
        if self.default_mode == "deep":
            return True

        # First few observations always deep
        if observations_in_stage < self.min_observations_for_monitor:
            return True

        # Periodic deep checks
        if monitors_since_deep >= self.deep_check_interval:
            return True

        # Approaching transition window (if timing available)
        if progress_ratio is not None and progress_ratio >= self.transition_window:
            return True

        return False


# Stage-specific perception configurations
STAGE_PERCEPTION_CONFIG: Dict[str, StageConfig] = {
    "early": StageConfig(
        default_mode="monitor",      # Stable, long duration
        deep_check_interval=10,      # Deep every 10 observations
        transition_window=0.8,       # Deep after 80% of expected duration
        min_observations_for_monitor=3,
        expected_duration_minutes=60,
    ),
    "bean": StageConfig(
        default_mode="deep",         # Fast stage, always deep
        deep_check_interval=1,       # Every observation
        transition_window=0.5,       # Earlier transition window
        min_observations_for_monitor=1,  # Never use monitor for bean
        expected_duration_minutes=30,
    ),
    "comma": StageConfig(
        default_mode="monitor",
        deep_check_interval=8,
        transition_window=0.7,
        min_observations_for_monitor=3,
        expected_duration_minutes=30,
    ),
    "1.5fold": StageConfig(
        default_mode="monitor",
        deep_check_interval=8,
        transition_window=0.7,
        min_observations_for_monitor=3,
        expected_duration_minutes=30,
    ),
    "2fold": StageConfig(
        default_mode="monitor",
        deep_check_interval=6,
        transition_window=0.6,
        min_observations_for_monitor=3,
        expected_duration_minutes=45,
    ),
    "pretzel": StageConfig(
        default_mode="monitor",
        deep_check_interval=5,
        transition_window=0.6,
        min_observations_for_monitor=3,
        expected_duration_minutes=60,
    ),
    "hatching": StageConfig(
        default_mode="deep",         # Critical stage, always deep
        deep_check_interval=1,
        transition_window=0.0,       # Always in transition window
        min_observations_for_monitor=1,
        expected_duration_minutes=15,
    ),
    "hatched": StageConfig(
        default_mode="monitor",      # Terminal, just confirm
        deep_check_interval=20,      # Rarely need deep
        transition_window=1.0,       # Never approaching transition
        min_observations_for_monitor=1,
        expected_duration_minutes=None,  # No expected duration
    ),
    "arrested": StageConfig(
        default_mode="monitor",      # Terminal, just confirm
        deep_check_interval=20,
        transition_window=1.0,
        min_observations_for_monitor=1,
        expected_duration_minutes=None,
    ),
}


def get_stage_config(stage: str) -> StageConfig:
    """Get configuration for a specific stage."""
    if stage in STAGE_PERCEPTION_CONFIG:
        return STAGE_PERCEPTION_CONFIG[stage]

    # Default config for unknown stages (shouldn't happen)
    return StageConfig(
        default_mode="deep",
        deep_check_interval=5,
        transition_window=0.7,
        min_observations_for_monitor=3,
        expected_duration_minutes=None,
    )


def is_fast_stage(stage: str) -> bool:
    """Check if a stage is considered fast-transitioning."""
    config = get_stage_config(stage)
    return config.default_mode == "deep"


def is_terminal_stage(stage: str) -> bool:
    """Check if a stage is terminal (hatched or arrested)."""
    return stage in ("hatched", "arrested")


def get_all_stage_configs() -> Dict[str, StageConfig]:
    """Get all stage configurations."""
    return STAGE_PERCEPTION_CONFIG.copy()


# Model configuration
@dataclass
class ModelConfig:
    """Configuration for VLM models used in perception."""

    # Monitor mode uses Haiku (fast, cheap)
    monitor_model: str = "claude-3-5-haiku-20241022"

    # Deep mode uses Sonnet (more capable)
    deep_model: str = "claude-sonnet-4-20250514"

    # Temperature settings
    monitor_temperature: float = 0.0  # Deterministic for consistency
    deep_temperature: float = 0.1     # Slight variation for reasoning

    # Token limits
    monitor_max_tokens: int = 256     # Short responses for monitor
    deep_max_tokens: int = 2048       # Longer for deep reasoning

    # Cost estimates (per 1M tokens)
    haiku_input_cost: float = 0.25
    haiku_output_cost: float = 1.25
    sonnet_input_cost: float = 3.0
    sonnet_output_cost: float = 15.0


# Default model configuration
DEFAULT_MODEL_CONFIG = ModelConfig()


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    mode: Literal["monitor", "deep"],
    config: ModelConfig = DEFAULT_MODEL_CONFIG,
) -> float:
    """Estimate cost in USD for a perception call."""
    if mode == "monitor":
        input_cost = (input_tokens / 1_000_000) * config.haiku_input_cost
        output_cost = (output_tokens / 1_000_000) * config.haiku_output_cost
    else:
        input_cost = (input_tokens / 1_000_000) * config.sonnet_input_cost
        output_cost = (output_tokens / 1_000_000) * config.sonnet_output_cost

    return input_cost + output_cost
