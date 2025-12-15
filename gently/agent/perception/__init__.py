"""
Perception System for C. elegans Embryo Microscopy

A unified VLM-based perception system that replaces the discontinuous
detector + verifier pipeline. Provides:

1. Continuous multi-round reasoning with belief accumulation
2. Few-shot learning from example images
3. Anomaly detection (dead embryos, technical failures)
4. Tiered model selection for cost optimization
"""

from .session import (
    BeliefState,
    EvidencePoint,
    PerceptionSession,
    PerceptionRoundInput,
    PerceptionRoundOutput,
)
from .example_store import ExampleStore
from .engine import PerceptionEngine
from .belief_updater import BeliefUpdater
from .anomaly import AnomalyDetector
from .manager import PerceptionManager
from .scheduler import AdaptivePerceptionScheduler

__all__ = [
    "BeliefState",
    "EvidencePoint",
    "PerceptionSession",
    "PerceptionRoundInput",
    "PerceptionRoundOutput",
    "ExampleStore",
    "PerceptionEngine",
    "BeliefUpdater",
    "AnomalyDetector",
    "PerceptionManager",
    "AdaptivePerceptionScheduler",
]
