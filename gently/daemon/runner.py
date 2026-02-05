"""
Daemon runner — entry point for starting the daemon.

Usage:
    from gently.daemon.runner import run_daemon
    await run_daemon(context_db="./context.db")

Or from command line:
    python -m gently.daemon.runner
"""

import asyncio
import logging
import signal
from pathlib import Path
from typing import Any, Optional

from ..context import ContextStore
from ..core.event_bus import get_event_bus
from ..capabilities import Capabilities
from ..agent_core.reasoning import create_think_function
from .core import Daemon

logger = logging.getLogger(__name__)


async def run_daemon(
    context_db: str = "./context.db",
    claude_api_key: Optional[str] = None,
    device_client: Optional[Any] = None,
    perception_manager: Optional[Any] = None,
    message_handler: Optional[Any] = None,
    notifier: Optional[Any] = None,
    log_level: str = "INFO",
):
    """
    Run the daemon.

    Parameters
    ----------
    context_db : str
        Path to context database file
    claude_api_key : str, optional
        API key for Claude. If not provided, thinking is simulated.
    device_client : Any, optional
        HTTP client for device layer
    perception_manager : Any, optional
        Perception system manager
    message_handler : callable, optional
        Function to handle agent messages
    notifier : Any, optional
        Notification system
    log_level : str
        Logging level
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Starting Gently Daemon")

    # Initialize components
    context_store = ContextStore(Path(context_db))
    event_bus = get_event_bus()

    # Create capabilities
    capabilities = Capabilities(
        device_client=device_client,
        perception_manager=perception_manager,
        message_handler=message_handler,
        notifier=notifier,
    )

    # Create Claude client if API key provided
    claude_client = None
    if claude_api_key:
        try:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=claude_api_key)
            logger.info("Claude client initialized")
        except ImportError:
            logger.warning("anthropic package not installed, using simulated thinking")
        except Exception as e:
            logger.warning(f"Failed to create Claude client: {e}, using simulated thinking")

    # Create think function
    think_fn = await create_think_function(claude_client)

    # Create daemon
    daemon = Daemon(
        context_store=context_store,
        event_bus=event_bus,
        think_fn=think_fn,
        capabilities=capabilities,
    )

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def handle_shutdown():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Run daemon with shutdown handling
    daemon_task = asyncio.create_task(daemon.start())

    # Wait for shutdown
    await shutdown_event.wait()

    # Stop daemon
    await daemon.stop()
    daemon_task.cancel()

    try:
        await daemon_task
    except asyncio.CancelledError:
        pass

    # Cleanup
    context_store.close()
    logger.info("Daemon stopped")


def main():
    """Command line entry point."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run the Gently Daemon")
    parser.add_argument(
        "--context-db",
        default="./context.db",
        help="Path to context database",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Claude API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    asyncio.run(run_daemon(
        context_db=args.context_db,
        claude_api_key=args.api_key,
        log_level=args.log_level,
    ))


if __name__ == "__main__":
    main()
