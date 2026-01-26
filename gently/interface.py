"""
MicroscopeBackend Protocol

Defines the abstract interface for microscope hardware backends.
Any backend implementation must satisfy this protocol to work with
the gently agent system.

Implementations:
- dispim_control.DiSPIMBackend: DiSPIM with ASI Tiger controller

Usage:
    from gently.interface import MicroscopeBackend
    from dispim_control import DiSPIMBackend

    backend: MicroscopeBackend = DiSPIMBackend(http_url="http://127.0.0.1:60610")
    copilot = MicroscopyCopilot(backend=backend)
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class MicroscopeBackend(Protocol):
    """
    Protocol defining the interface for microscope hardware backends.

    Backends are passed to MicroscopyCopilot and accessed by agent tools
    through the execution context via context.get('backend').
    """

    # =========================================================================
    # Connection
    # =========================================================================

    async def connect(self) -> bool:
        """Connect to the hardware server. Returns True on success."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the hardware server."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the backend is currently connected."""
        ...

    @property
    def has_sam(self) -> bool:
        """Whether SAM detection server is available."""
        ...

    @property
    def has_databroker(self) -> bool:
        """Whether Databroker catalog is available."""
        ...

    # =========================================================================
    # Stage Movement
    # =========================================================================

    async def move_to_position(self, x: float, y: float) -> Dict:
        """Move XY stage to position in micrometers."""
        ...

    async def get_stage_position(self) -> Tuple[float, float]:
        """Get current XY stage position in micrometers."""
        ...

    async def get_piezo_position(self) -> float:
        """Get current piezo position in micrometers."""
        ...

    # =========================================================================
    # Acquisition
    # =========================================================================

    async def capture_lightsheet_image(
        self,
        piezo_position: float = 50.0,
        galvo_position: float = 0.0,
    ) -> Dict:
        """Capture a single lightsheet image at specified positions."""
        ...

    async def acquire_volume(
        self,
        num_slices: int = 50,
        exposure_ms: float = 10.0,
        **kwargs,
    ) -> Dict:
        """Acquire a 3D volume stack. Returns dict with 'volume' numpy array."""
        ...

    async def capture_bottom_image(self, **kwargs) -> np.ndarray:
        """Capture a single 2D image from the bottom camera."""
        ...

    # =========================================================================
    # Illumination
    # =========================================================================

    async def set_led(self, state: str) -> Dict:
        """Set LED state ('Open' or 'Closed')."""
        ...

    async def get_led_status(self) -> Dict:
        """Get current LED status."""
        ...

    # =========================================================================
    # Detection (SAM-dependent)
    # =========================================================================

    async def detect_embryos(self, **kwargs) -> Dict:
        """Run SAM-based embryo detection."""
        ...

    async def manual_mark_embryos(self, **kwargs) -> Dict:
        """Manually mark embryo positions."""
        ...

    async def edit_embryos(self, **kwargs) -> Dict:
        """Edit existing embryo positions."""
        ...

    async def view_embryos(self, **kwargs) -> Dict:
        """View current embryo detections."""
        ...

    # =========================================================================
    # Status
    # =========================================================================

    async def get_status(self) -> Dict:
        """Get hardware server status."""
        ...
