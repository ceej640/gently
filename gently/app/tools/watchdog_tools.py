"""Watchdog tools — orchestrator-facing pull API for the observer.

Exposes ``get_system_status``, which returns the current observer narrative,
open concerns, and channel summary. This is the canonical "ask the watchdog
what's going on" tool for the orchestrator; the push path goes through
:meth:`gently.harness.conversation.ConversationManager.inject_critical_handoff`.
"""
from typing import Dict

from gently.harness.tools.registry import tool, ToolCategory
from gently.harness.tools.helpers import require_agent


@tool(
    name="get_system_status",
    description=(
        "Pull the watchdog observer's current narrative and open concerns. "
        "Use this when you want a quick synthesized view of the experiment "
        "(progress, per-embryo concerns, hardware health, system health) "
        "without having to dig through logs or event history. Returns a "
        "short narrative string plus a list of open concerns."
    ),
    category=ToolCategory.EXPERIMENT,
)
async def get_system_status(context: Dict = None) -> str:
    """Return a human-readable status block from the watchdog observer."""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator = getattr(agent, "timelapse_orchestrator", None)
    observer = None
    if orchestrator is not None:
        observer = getattr(orchestrator, "observer", None)
    if observer is None:
        observer = getattr(agent, "observer", None)

    if observer is None:
        return (
            "Watchdog observer is not running. No background narrative is "
            "being maintained. (Start a timelapse with the watchdog enabled "
            "to populate this.)"
        )

    try:
        status = observer.get_status()
    except Exception as e:
        return f"Error pulling watchdog status: {e}"

    lines = []
    session_id = status.get("session_id") or "(no session)"
    lines.append(f"Watchdog observer — session {session_id}")
    lines.append(
        f"Tick {status.get('tick_count', 0)}, last updated "
        f"{status.get('last_updated', '')}"
    )
    if status.get("channels"):
        lines.append("Channels: " + ", ".join(status["channels"]))
    lines.append("")
    lines.append("NARRATIVE:")
    lines.append(status.get("narrative") or "(empty)")

    concerns = status.get("open_concerns") or []
    lines.append("")
    lines.append(f"OPEN CONCERNS ({len(concerns)}):")
    if concerns:
        for c in concerns:
            lines.append(f"  - {c}")
    else:
        lines.append("  (none)")

    pending = status.get("pending_handoffs", 0)
    if pending:
        lines.append("")
        lines.append(
            f"NOTE: {pending} critical handoff(s) queued — you will see them "
            "injected into this conversation on the next turn."
        )

    return "\n".join(lines)
