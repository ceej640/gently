#!/usr/bin/env python3
"""
Electron Desktop App Entry Point

Slim FastAPI server for the Electron wrapper. Launched as a subprocess
by electron/main.js with --port and --storage-path arguments.
The Anthropic API key is passed via the ANTHROPIC_API_KEY env var.

Protocol:
  - Prints "READY:{port}" to stdout when the server is listening.
  - Electron polls GET /health for liveness.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path when running from PyInstaller bundle
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    # PyInstaller one-dir: the executable lives in python-dist/electron_entry/
    _bundle_dir = Path(sys._MEIPASS)
    if str(_bundle_dir) not in sys.path:
        sys.path.insert(0, str(_bundle_dir))
else:
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

import yaml

from gently.organisms import load_organism
from gently.hardware import load_hardware
from gently.store import GentlyStore

logger = logging.getLogger(__name__)

HAS_API_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))


async def main(port: int, storage_path: Path) -> None:
    """Start the copilot server for Electron."""

    # Load organism/hardware config
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yml"
    if getattr(sys, "frozen", False):
        config_path = Path(sys._MEIPASS) / "config" / "config.yml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    load_organism(config.get("organism", "celegans"))
    load_hardware(config.get("hardware", "dispim"))

    # Ensure storage directory exists
    storage_path.mkdir(parents=True, exist_ok=True)

    # Create unified store
    store = GentlyStore(storage_path)

    viz_server = None

    if HAS_API_KEY:
        # Full mode: copilot + viz server
        from gently.agent import MicroscopyCopilot
        from gently.agent.copilot_bridge import CopilotBridge
        from gently.context import ContextStore

        copilot = MicroscopyCopilot(
            microscope_client=None,
            storage_path=storage_path,
            store=store,
        )

        await copilot.start_viz_server(port=port)
        viz_server = copilot.viz_server

        if viz_server is not None:
            # Attach copilot bridge
            bridge = CopilotBridge(copilot)
            bridge.set_launch_info({
                "device_connected": False,
                "sam_available": False,
                "offline": True,
                "store_path": str(storage_path),
                "viz_url": f"http://127.0.0.1:{port}",
                "resumed": False,
            })

            # Initialize context store (agent's mind)
            context_db = storage_path / "context" / "agent_mind.db"
            context_db.parent.mkdir(parents=True, exist_ok=True)
            context_store = ContextStore(context_db)
            copilot.set_context_store(context_store)
            bridge.init_wizard(context_store=context_store, claude_client=copilot.claude)

            viz_server.copilot_bridge = bridge
            viz_server.set_context_store(context_store)
    else:
        # UI-only mode: viz server without copilot (no API key)
        from gently.visualization.server import VisualizationServer
        from gently.context import ContextStore

        viz_server = VisualizationServer(
            port=port,
            gently_store=store,
        )

        # Context store is needed for campaigns/plans (no API key required)
        context_db = storage_path / "context" / "agent_mind.db"
        context_db.parent.mkdir(parents=True, exist_ok=True)
        viz_server.set_context_store(ContextStore(context_db))

        await viz_server.start()

    if viz_server is None:
        print("ERROR:Failed to start visualization server", flush=True)
        store.close()
        return

    # Add /health endpoint (reports whether copilot is available)
    @viz_server.app.get("/health")
    async def health():
        return {"status": "ok", "copilot": HAS_API_KEY}

    # Signal readiness to Electron — server is already listening
    print(f"READY:{port}", flush=True)

    # Block until interrupted (same pattern as launch_copilot.py).
    # Don't use run_forever() because it calls start() again.
    import signal

    stop_event = asyncio.Event()

    def _shutdown(*args):
        stop_event.set()

    if sys.platform == "win32":
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    else:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _shutdown)

    # Wait until signal or server task completes
    server_task = viz_server._server_task
    if server_task:
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            [server_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    # Cleanup
    await viz_server.stop()
    store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gently Electron Backend")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--storage-path",
        type=str,
        default=str(Path.home() / "Documents" / "Gently"),
        help="Data storage directory",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(port=args.port, storage_path=Path(args.storage_path)))
    except KeyboardInterrupt:
        pass
