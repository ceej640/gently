"""
Gently
======

Backend-agnostic AI agent for microscopy.
Provides conversational AI-driven experiment orchestration that works with
any hardware backend implementing the MicroscopeBackend protocol.

Key Components:
    - interface: MicroscopeBackend protocol for hardware abstraction
    - agent: AI copilot for conversational microscope control
    - analysis: Focus scoring and curve fitting utilities
    - coordinates: Coordinate transformations and reference mapping
    - visualization: Web-based experiment visualization
"""

# Backend interface
from .interface import MicroscopeBackend

# Analysis utilities - device-agnostic focus analysis
try:
    from .analysis.core import (
        FocusAnalysisConfig,
        FocusResult,
        FocusAlgorithm,
        FitFunction,
        calculate_focus_score,
        analyze_focus_stack,
        fit_focus_curve
    )
    _ANALYSIS_AVAILABLE = True
except ImportError:
    _ANALYSIS_AVAILABLE = False

# Coordinate utilities - transformations for pixel/stage conversions
try:
    from .coordinates import (
        pixel_to_stage_position,
        stage_to_pixel_position,
        pixel_displacement_to_stage_movement,
        get_um_per_pixel,
        DEFAULT_PIXEL_SIZE_UM,
        DEFAULT_OBJECTIVE_MAG
    )
    _COORDINATES_AVAILABLE = True
except ImportError:
    _COORDINATES_AVAILABLE = False

# Visualization utilities - optional napari integration
try:
    from .visualization import (
        setup_napari_callback,
        create_napari_viewer,
        enable_focus_sweep_visualization,
        enable_embryo_detection_visualization,
        enable_full_visualization,
        NapariCallback,
        NAPARI_AVAILABLE
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False
    NAPARI_AVAILABLE = False

# Main entry point
from .gently import Gently, create_gently

# Core infrastructure
from .core import (
    TiledStore,
    DatabrokerStore,
    EventBus,
    EventType,
    get_event_bus,
    get_data_store,
)

__version__ = "0.3.0"
__all__ = [
    # Backend interface
    "MicroscopeBackend",

    # Main entry point
    "Gently",
    "create_gently",

    # Core infrastructure
    "TiledStore",
    "DatabrokerStore",
    "EventBus",
    "EventType",
    "get_event_bus",
    "get_data_store",

    # Analysis functions
    "calculate_focus_score",
    "fit_focus_curve",
    "analyze_focus_stack",
    "FocusAnalysisConfig",
    "FocusResult",
    "FocusAlgorithm",
    "FitFunction",

    # Coordinate functions
    "pixel_to_stage_position",
    "stage_to_pixel_position",
    "pixel_displacement_to_stage_movement",
    "get_um_per_pixel",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_OBJECTIVE_MAG",
]

# Add visualization functions if available
if _VISUALIZATION_AVAILABLE:
    __all__.extend([
        "setup_napari_callback",
        "create_napari_viewer",
        "enable_focus_sweep_visualization",
        "enable_embryo_detection_visualization",
        "enable_full_visualization",
        "NapariCallback",
        "NAPARI_AVAILABLE",
    ])
