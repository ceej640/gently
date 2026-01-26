"""
Conversational Microscopy Copilot

AI agent that acts as a scientific collaborator for microscopy experiments.
Backend-agnostic — works with any hardware implementing MicroscopeBackend.
"""

from .copilot import MicroscopyCopilot
from .state import EmbryoState, ExperimentState, ImageRecord
from .plan_synthesis import PlanSynthesizer, PlanValidator
from .image_manager import ImageManager
from .perception import PerceptionManager, PerceptionResult, PerceptionSession
from .rich_cli import run_rich_cli, RichCopilotCLI
from .autocomplete import create_completer, CopilotCompleter
from .tool_registry import ToolRegistry, get_tool_registry, tool, ToolCategory

# Import tools package to register all tools
from . import tools

__all__ = [
    'MicroscopyCopilot',
    'EmbryoState',
    'ExperimentState',
    'ImageRecord',
    'PlanSynthesizer',
    'PlanValidator',
    'ImageManager',
    # Perception system
    'PerceptionManager',
    'PerceptionResult',
    'PerceptionSession',
    'run_rich_cli',
    'RichCopilotCLI',
    'create_completer',
    'CopilotCompleter',
    # Tool registry
    'ToolRegistry',
    'get_tool_registry',
    'tool',
    'ToolCategory',
]
