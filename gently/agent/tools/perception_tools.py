"""
Perception System Tools

Tools for managing the VLM-based perception system:
- Adding few-shot examples for stages and anomalies
- Querying perception beliefs
- Managing perception sessions
"""

from typing import Dict, Optional
from pathlib import Path

from ..tool_registry import tool, ToolCategory, ToolExample


@tool(
    name="add_stage_example",
    description="""Add an image as a few-shot example for a developmental stage.
Use when you have a good representative image of an embryo at a specific stage.
The image will be used by the perception system to improve stage classification.
Valid stages: early, comma, pretzel, 3fold, hatching, hatched""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Save this as a comma stage example", {"stage": "comma"}),
        ToolExample("Add this image as pretzel example", {"stage": "pretzel"}),
    ],
)
async def add_stage_example(
    stage: str,
    embryo_id: Optional[str] = None,
    timepoint: Optional[int] = None,
    annotation: Optional[str] = None,
    context: Dict = None,
) -> str:
    """Add current image as example for a developmental stage"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    valid_stages = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]
    if stage not in valid_stages:
        return f"Invalid stage '{stage}'. Valid stages: {', '.join(valid_stages)}"

    # Get perception manager
    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    # Get image - either from specified embryo/timepoint or current
    try:
        if embryo_id and timepoint is not None:
            # Get from image manager
            image_manager = getattr(copilot, "image_manager", None)
            if image_manager:
                image_b64 = await image_manager.get_image_b64(embryo_id, timepoint)
            else:
                return f"Error: Cannot retrieve image for {embryo_id} T{timepoint}"
        else:
            # Get current image from microscope
            client = context.get("client")
            if not client:
                return "Error: No microscope connected and no embryo_id/timepoint specified"

            import base64
            import io
            from PIL import Image

            image = await client.capture_bottom_image()
            if image is None:
                return "Error: Failed to capture image"

            # Convert to base64
            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add to example store
        example_store = perception_manager.example_store
        filename = example_store.add_example(
            image_b64=image_b64,
            category="stages",
            subcategory=stage,
            annotation=annotation,
        )

        return f"Added stage example: {filename}"

    except Exception as e:
        return f"Error adding example: {str(e)}"


@tool(
    name="add_anomaly_example",
    description="""Add an image as a few-shot example for an anomaly type.
Use when you have a good representative image of an anomaly condition.
Valid anomaly types: dead_embryo, blank_technical, blank_biological""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Save this as dead embryo example", {"anomaly_type": "dead_embryo"}),
        ToolExample("Add this as technical blank example", {"anomaly_type": "blank_technical"}),
    ],
)
async def add_anomaly_example(
    anomaly_type: str,
    embryo_id: Optional[str] = None,
    timepoint: Optional[int] = None,
    annotation: Optional[str] = None,
    context: Dict = None,
) -> str:
    """Add current image as example for an anomaly type"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    valid_types = ["dead_embryo", "blank_technical", "blank_biological"]
    if anomaly_type not in valid_types:
        return f"Invalid anomaly type '{anomaly_type}'. Valid types: {', '.join(valid_types)}"

    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    try:
        if embryo_id and timepoint is not None:
            image_manager = getattr(copilot, "image_manager", None)
            if image_manager:
                image_b64 = await image_manager.get_image_b64(embryo_id, timepoint)
            else:
                return f"Error: Cannot retrieve image for {embryo_id} T{timepoint}"
        else:
            client = context.get("client")
            if not client:
                return "Error: No microscope connected and no embryo_id/timepoint specified"

            import base64
            import io
            from PIL import Image

            image = await client.capture_bottom_image()
            if image is None:
                return "Error: Failed to capture image"

            pil_image = Image.fromarray(image)
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        example_store = perception_manager.example_store
        filename = example_store.add_example(
            image_b64=image_b64,
            category="anomalies",
            subcategory=anomaly_type,
            annotation=annotation,
        )

        return f"Added anomaly example: {filename}"

    except Exception as e:
        return f"Error adding example: {str(e)}"


@tool(
    name="list_stage_examples",
    description="""List all available few-shot examples for the perception system.
Shows counts per stage and anomaly type, plus total examples available.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("How many examples do we have?", {}),
        ToolExample("Show available stage examples", {}),
    ],
)
async def list_stage_examples(
    context: Dict = None,
) -> str:
    """Show available examples per stage and anomaly type"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    try:
        counts = perception_manager.example_store.get_example_counts()

        lines = ["**Stage Examples:**"]
        stages = counts.get("stages", {})
        for stage in ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]:
            count = stages.get(stage, 0)
            lines.append(f"  {stage}: {count}")

        lines.append("\n**Anomaly Examples:**")
        anomalies = counts.get("anomalies", {})
        for atype in ["dead_embryo", "blank_technical", "blank_biological"]:
            count = anomalies.get(atype, 0)
            lines.append(f"  {atype}: {count}")

        total = counts.get("total", 0)
        lines.append(f"\n**Total examples:** {total}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing examples: {str(e)}"


@tool(
    name="get_perception_beliefs",
    description="""Get the current perception beliefs for an embryo.
Shows the probabilistic stage distribution, hatching state, anomaly flags,
and recent evidence. Use to understand what the perception system thinks about an embryo.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("What does perception think about embryo_1?", {"embryo_id": "embryo_1"}),
        ToolExample("Show perception beliefs for this embryo", {"embryo_id": "embryo_3"}),
    ],
)
async def get_perception_beliefs(
    embryo_id: str,
    context: Dict = None,
) -> str:
    """Get current perception beliefs for an embryo"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    session = perception_manager.get_session(embryo_id)
    if not session:
        return f"No perception session for {embryo_id}"

    beliefs = session.beliefs
    lines = [f"**Perception Beliefs for {embryo_id}:**\n"]

    # Stage distribution
    lines.append("**Stage Distribution:**")
    for stage, prob in sorted(
        beliefs.stage_distribution.items(), key=lambda x: -x[1]
    ):
        bar = "#" * int(prob * 20)
        lines.append(f"  {stage}: {prob:.0%} {bar}")

    lines.append(f"\n**Most Likely:** {beliefs.most_likely_stage} ({beliefs.stage_confidence:.0%})")

    # Hatching state
    lines.append("\n**Hatching State:**")
    lines.append(f"  In progress: {beliefs.hatching_in_progress}")
    lines.append(f"  Breach detected: {beliefs.breach_detected}")
    lines.append(f"  Worm exiting: {beliefs.worm_exiting}")
    lines.append(f"  Complete: {beliefs.hatching_complete}")
    if beliefs.hatching_timepoint:
        lines.append(f"  Hatching timepoint: T{beliefs.hatching_timepoint}")

    # Anomaly flags
    lines.append("\n**Anomaly Flags:**")
    lines.append(f"  Possibly dead: {beliefs.possibly_dead} ({beliefs.dead_confidence:.0%})")
    if beliefs.dead_evidence:
        for ev in beliefs.dead_evidence[:3]:
            lines.append(f"    - {ev}")
    lines.append(f"  Technical issue: {beliefs.technical_issue_suspected}")
    if beliefs.technical_issue_type:
        lines.append(f"    Type: {beliefs.technical_issue_type}")

    # Temporal state
    lines.append("\n**Temporal State:**")
    lines.append(f"  Hours since change: {beliefs.hours_since_change:.1f}")
    lines.append(f"  Movement detected: {beliefs.movement_detected}")
    lines.append(f"  Rounds processed: {session.rounds_processed}")
    lines.append(f"  Unchanged count: {session.unchanged_count}")

    # Recent evidence
    recent = session.get_recent_evidence(3)
    if recent:
        lines.append("\n**Recent Evidence:**")
        for ev in recent:
            lines.append(f"  T{ev.timepoint}: {ev.observation[:80]}...")
            if ev.supports:
                lines.append(f"    Supports: {', '.join(ev.supports)}")

    return "\n".join(lines)


@tool(
    name="get_perception_summary",
    description="""Get a summary of perception status across all embryos.
Shows stage distribution, anomaly alerts, and system statistics.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Show perception summary", {}),
        ToolExample("What's the perception system status?", {}),
    ],
)
async def get_perception_summary(
    context: Dict = None,
) -> str:
    """Get summary of perception system status"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    sessions = perception_manager.get_all_sessions()
    if not sessions:
        return "No active perception sessions"

    lines = [f"**Perception System Summary ({len(sessions)} embryos):**\n"]

    # Group by stage
    stage_counts = {}
    alerts = []

    for embryo_id, session in sessions.items():
        beliefs = session.beliefs
        stage = beliefs.most_likely_stage
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

        # Check for alerts
        if beliefs.hatching_complete:
            alerts.append(f"  {embryo_id}: HATCHED")
        elif beliefs.hatching_in_progress:
            alerts.append(f"  {embryo_id}: Hatching in progress")
        elif beliefs.possibly_dead and beliefs.dead_confidence > 0.5:
            alerts.append(f"  {embryo_id}: Possibly dead ({beliefs.dead_confidence:.0%})")
        elif beliefs.technical_issue_suspected:
            alerts.append(f"  {embryo_id}: Technical issue ({beliefs.technical_issue_type})")

    # Stage distribution
    lines.append("**Embryos by Stage:**")
    for stage in ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]:
        count = stage_counts.get(stage, 0)
        if count > 0:
            lines.append(f"  {stage}: {count}")

    # Alerts
    if alerts:
        lines.append("\n**Alerts:**")
        lines.extend(alerts)
    else:
        lines.append("\n**Alerts:** None")

    # Scheduler stats
    scheduler_stats = perception_manager.scheduler.get_stats()
    lines.append("\n**Scheduler:**")
    lines.append(f"  Tracked embryos: {scheduler_stats['tracked_embryos']}")
    lines.append(f"  Min interval: {scheduler_stats['min_interval_minutes']} min")
    lines.append(f"  Max interval: {scheduler_stats['max_interval_minutes']} min")

    # Example counts
    example_counts = perception_manager.example_store.get_example_counts()
    lines.append(f"\n**Examples loaded:** {example_counts.get('total', 0)}")

    return "\n".join(lines)


@tool(
    name="reset_perception_session",
    description="""Reset the perception session for an embryo.
Clears all accumulated beliefs and evidence, starting fresh.
Use when perception state has become corrupted or needs to be reset.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Reset perception for embryo_1", {"embryo_id": "embryo_1"}),
    ],
)
async def reset_perception_session(
    embryo_id: str,
    context: Dict = None,
) -> str:
    """Reset perception session for an embryo"""
    copilot = context.get("copilot")

    if not copilot:
        return "Error: No copilot context"

    perception_manager = getattr(copilot, "perception_manager", None)
    if not perception_manager:
        return "Error: Perception system not initialized"

    if perception_manager.clear_session(embryo_id):
        perception_manager.scheduler.reset(embryo_id)
        return f"Reset perception session for {embryo_id}"
    else:
        return f"No perception session found for {embryo_id}"
