"""Watchdog layer — background channels + observer agent.

See docs/watchdog_layer.md for architecture.
"""
from .channels import (
    Observation,
    Channel,
    ProgressChannel,
    PerceptionChannel,
    HardwareChannel,
    SystemHealthChannel,
    build_default_channels,
)
from .observer import Observer

__all__ = [
    "Observation",
    "Channel",
    "ProgressChannel",
    "PerceptionChannel",
    "HardwareChannel",
    "SystemHealthChannel",
    "build_default_channels",
    "Observer",
]
