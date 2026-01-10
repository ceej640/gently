"""
Cognitive Architecture for Embryo Perception.

This package implements a VLM-based cognitive system for embryo stage classification.
The VLM (Claude) is the cognitive core - our code manages context loading and memory persistence.

Key Components:
- WorldModel: The VLM's beliefs about an embryo
- CognitiveEngine: Orchestrates perception (monitor vs deep modes)
- Monitor: Lightweight stage confirmation (Haiku)
- ContextManager: Assembles VLM context for each cognitive operation
- MemoryManager: Parses VLM output and persists state

Perception Modes:
- Monitor: Quick "still X stage?" checks (~100ms, Haiku)
- Deep (single-turn): Full cognitive assessment (~500ms, Sonnet)
- Deep (multi-turn): Extended reasoning with verification (~2000ms, Sonnet)
"""

from .world_model import (
    WorldModel,
    Predictions,
    TransitionRecord,
)
from .memory import (
    StageNarrative,
    TransitionNarrative,
    EmbryoNarrative,
    BatchNarrative,
    StageMetrics,
)
from .trace import (
    CognitiveTrace,
    TurnTrace,
    MonitorResult,
    WorldModelSnapshot,
)
from .config import (
    STAGE_PERCEPTION_CONFIG,
    StageConfig,
    get_stage_config,
)
from .engine import CognitiveEngine
from .monitor import Monitor
from .mode_selector import select_perception_mode
from .context_manager import ContextManager
from .memory_manager import MemoryManager

__all__ = [
    # World Model
    "WorldModel",
    "Predictions",
    "TransitionRecord",
    # Memory/Narratives
    "StageNarrative",
    "TransitionNarrative",
    "EmbryoNarrative",
    "BatchNarrative",
    "StageMetrics",
    # Traces
    "CognitiveTrace",
    "TurnTrace",
    "MonitorResult",
    "WorldModelSnapshot",
    # Config
    "STAGE_PERCEPTION_CONFIG",
    "StageConfig",
    "get_stage_config",
    # Engine
    "CognitiveEngine",
    "Monitor",
    "select_perception_mode",
    # Managers
    "ContextManager",
    "MemoryManager",
]
