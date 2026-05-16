"""
Embryo Detection Tools

Tools for detecting and marking embryos in microscope images.
"""

import uuid
from typing import Dict
from datetime import datetime
from pathlib import Path

from gently.harness.tools.registry import tool, ToolCategory, ToolExample
from gently.harness.tools.helpers import require_agent
from gently.core.coordinates import (
    stage_to_pixel_position,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)


@tool(
    name="detect_embryos",
    description="""Automatically detect embryos in the current field of view using brightness detection and SAM segmentation.
Use when user says "find embryos", "detect embryos", or at the start of an experiment to locate samples.
Captures a bottom camera image and identifies bright spots as potential embryos.
When open_editor=True (default), the bottom-camera image plus SAM-detected positions
appear on the web Marking canvas so the operator can add, move, or remove markers
before pressing Done. Confirmed embryos are added to the experiment and appear on
the Devices > Map view as coarse waypoints.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Find all embryos", {}),
        ToolExample("Detect embryos automatically", {}),
    ],
)
async def detect_embryos(
    auto_calibrate: bool = False,
    min_confidence: float = 0.7,
    use_claude_review: bool = False,
    exposure_ms: float = None,
    brightness_percentile: float = 99.0,
    min_area: int = 5000,
    max_area: int = 150000,
    open_editor: bool = True,
    editor_timeout_sec: float = 600.0,
    context: Dict = None
) -> str:
    """Detect embryos automatically"""
    agent = context.get('agent')
    client = context.get('client')

    if not agent:
        return "Error: No agent context"

    if not client:
        return "Error: Microscope not connected. Cannot detect embryos in offline mode."

    if not client.has_sam:
        return "Error: SAM server not connected. Embryo detection requires the SAM segmentation server."

    try:
        # Run SAM detection only — napari editor is intentionally bypassed.
        # When open_editor=True we hand off to the web Marking canvas via
        # viz_server below; SAM positions are pre-populated so the operator
        # adjusts rather than redoes detection.
        result = await client.detect_embryos(
            min_confidence=min_confidence,
            use_claude_review=use_claude_review,
            exposure_ms=exposure_ms,
            brightness_percentile=brightness_percentile,
            min_area=min_area,
            max_area=max_area,
            open_editor=False,
        )

        if not result.get('success'):
            return f"Detection failed: {result.get('error', 'Unknown error')}"

        embryos = result.get('embryos', [])
        editor_note = ""

        viz_server = getattr(agent, 'viz_server', None)
        operator_marked = False
        if open_editor and embryos and viz_server is not None:
            edited, editor_note = await _run_web_editor(
                client=client,
                viz_server=viz_server,
                detection_result=result,
                embryos=embryos,
                exposure_ms=exposure_ms,
                timeout_sec=editor_timeout_sec,
            )
            if edited is not None:
                embryos = edited
                operator_marked = True
        elif open_editor and viz_server is None:
            editor_note = " (web editor unavailable — viz server not running)"

        # Add to experiment (Phase 1 schema: position is treated as coarse)
        for emb in embryos:
            position = {
                'x': emb.get('stage_x_um', emb.get('stage_x', 0)),
                'y': emb.get('stage_y_um', emb.get('stage_y', 0))
            }
            agent.experiment.add_embryo(
                embryo_id=emb['embryo_id'],
                position=position,
                confidence=emb.get('confidence', 0.0),
                uid=emb.get('uid'),  # Preserve UID from detection
            )

        # OPERATOR_MARKED_EMBRYOS fires only when the human actually
        # confirmed via the web canvas — that's the intent signal. If the
        # editor was skipped (no viz_server) the SAM list still landed in
        # experiment.embryos, but it wasn't operator-confirmed so we
        # don't emit the operator event.
        if operator_marked:
            bus = getattr(agent, '_event_bus', None)
            if bus is not None:
                from gently.core.event_bus import EventType
                try:
                    bus.publish(
                        event_type=EventType.OPERATOR_MARKED_EMBRYOS,
                        data={
                            'embryo_ids': [e.get('embryo_id') for e in embryos],
                            'count': len(embryos),
                            'stage_origin': list(result.get('stage_position', (0.0, 0.0))),
                            'pre_edit_count': len(result.get('embryos', [])),
                        },
                        source='detect_embryos:web-editor',
                    )
                except Exception:
                    pass

        if auto_calibrate and embryos:
            return f"Detected {len(embryos)} embryos{editor_note}. Starting calibration..."
        if open_editor:
            return f"Detection complete: {len(embryos)} embryos confirmed via web editor{editor_note}."
        return (f"Detected {len(embryos)} embryos. Use show_detected_embryos "
                f"to visualize or edit_embryos to modify.")

    except Exception as e:
        return f"Error detecting embryos: {str(e)}"


async def _run_web_editor(client, viz_server, detection_result, embryos,
                          exposure_ms, timeout_sec):
    """Run the web Marking canvas editor for a SAM detection result.

    Returns (edited_embryos_or_None, note_string). On failure or timeout
    falls back to the original SAM list so detection never blocks on a
    closed browser tab.
    """
    image = await client._get_detection_image(detection_result, exposure_ms)
    if image is None:
        return None, " (editor skipped — no image available)"

    # SAM dicts carry pixel_x / pixel_y; the canvas uses pixelX / pixelY.
    initial_markers = []
    for e in embryos:
        px, py = e.get('pixel_x'), e.get('pixel_y')
        if px is None or py is None:
            continue
        initial_markers.append({'pixelX': float(px), 'pixelY': float(py)})

    stage_pos = tuple(detection_result.get('stage_position', (0.0, 0.0)))
    um_per_pixel = get_um_per_pixel(DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG)

    session_id = await viz_server.start_marking_session(
        image=image,
        initial_stage_position=stage_pos,
        pixel_size_um=um_per_pixel,
        initial_markers=initial_markers,
    )
    edited = await viz_server.wait_for_marking(session_id, timeout=timeout_sec)

    if not edited:
        return None, " (editor returned no markers — keeping SAM result)"
    # wait_for_marking already converted pixels -> stage coords and assigned
    # sequential embryo_ids; the rest of the detect_embryos path only needs
    # stage_x_um / stage_y_um / confidence / embryo_id which are present.
    return edited, ""


# ============================================================================
# Shared helpers for the web marking-canvas flow.
# Used by manual_mark_embryos and edit_embryos (and could absorb the duplicate
# detect_embryos logic later — keeping that path separate for now because it
# layers SAM detection on top).
# ============================================================================

async def _capture_for_marking(agent, client, exposure_ms):
    """Capture a fresh bottom-camera image and archive it.

    Returns (image_ndarray, stage_pos_tuple) or (None, None) on failure.
    """
    try:
        snap = await client.capture_bottom_image(exposure_ms=exposure_ms)
    except Exception:
        return None, None
    image = snap.get('image') if isinstance(snap, dict) else None
    if image is None:
        return None, None
    try:
        stage_pos = await client.get_stage_position()
    except Exception:
        stage_pos = (0.0, 0.0)
    # Best-effort archive — failure here mustn't block marking.
    try:
        if snap.get('image_path') and agent.store and agent.session_id:
            from gently.harness.tools.helpers import build_snapshot_metadata
            meta = build_snapshot_metadata(stage_pos, image.shape, agent.experiment)
            agent.store.register_snapshot(
                agent.session_id, "bottom_camera", snap['image_path'],
                metadata=meta,
            )
    except Exception:
        pass
    return image, tuple(stage_pos)


def _embryos_to_pixel_markers(embryo_items, image, stage_pos):
    """Convert existing embryo stage positions to pixel markers for the canvas.

    Markers within the image bounds become seed entries; anything off-image
    (because the stage moved since the embryo was sighted) is skipped — the
    operator can't see it so we shouldn't pretend otherwise.
    """
    if image is None:
        return []
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    um_per_pixel = get_um_per_pixel(DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG)
    sx0, sy0 = stage_pos
    out = []
    for _, st in embryo_items:
        pos = getattr(st, 'stage_position', None) or {}
        sx, sy = pos.get('x'), pos.get('y')
        if sx is None or sy is None:
            continue
        # Inverse of pixel_to_stage_position: pixel = center + (stage - origin) / um_per_pixel.
        # Note Y sign convention: pixel_to_stage flips Y, so the inverse flips it back.
        px = cx + (sx - sx0) / um_per_pixel
        py = cy - (sy - sy0) / um_per_pixel
        if not (0 <= px <= w and 0 <= py <= h):
            continue
        out.append({'pixelX': float(px), 'pixelY': float(py)})
    return out


async def _run_marking_canvas(viz_server, image, stage_pos, initial_markers,
                              *, timeout_sec):
    """Open the web Marking canvas and await the operator's Done press.

    Returns (markers_or_None, note_string). ``markers`` is the list as
    returned by ``wait_for_marking`` — each entry carries stage_x_um /
    stage_y_um already computed.
    """
    um_per_pixel = get_um_per_pixel(DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG)
    session_id = await viz_server.start_marking_session(
        image=image,
        initial_stage_position=stage_pos,
        pixel_size_um=um_per_pixel,
        initial_markers=initial_markers,
    )
    marked = await viz_server.wait_for_marking(session_id, timeout=timeout_sec)
    if marked is None:
        # wait_for_marking returns [] on timeout; None reserved for hard errors.
        return None, " (no response from canvas)"
    if not marked:
        return [], ""
    return marked, ""


def _next_embryo_number(existing_ids):
    """Find the next available 'embryo_N' suffix that doesn't collide."""
    max_num = 0
    for eid in existing_ids:
        if not eid.startswith('embryo_'):
            continue
        try:
            n = int(eid.replace('embryo_', ''))
        except ValueError:
            continue
        max_num = max(max_num, n)
    return max_num + 1


def _publish_operator_marked(agent, *, embryo_ids, count,
                              stage_origin, pre_edit_count, source):
    """Emit OPERATOR_MARKED_EMBRYOS so candidate orchestrators / Map see the
    fact that the operator just did this. Best-effort; never raises.
    """
    bus = getattr(agent, '_event_bus', None)
    if bus is None:
        return
    try:
        from gently.core.event_bus import EventType
        bus.publish(
            event_type=EventType.OPERATOR_MARKED_EMBRYOS,
            data={
                'embryo_ids': list(embryo_ids),
                'count': count,
                'stage_origin': list(stage_origin),
                'pre_edit_count': pre_edit_count,
            },
            source=source,
        )
    except Exception:
        pass


@tool(
    name="manual_mark_embryos",
    description="""Open the web Marking canvas to manually mark embryos by clicking on them. Existing embryos are pre-placed as reference markers.
Use when automatic detection missed embryos, or user wants to add embryos manually (e.g., "let me mark embryos", "I'll click on them").
The web canvas shows the current bottom-camera image with existing embryos numbered 1..N as reference; the operator adds new markers by clicking, then presses Done. Only the newly-added markers become new embryos with fresh unique IDs; existing embryos are untouched regardless of whether the operator left them in place or removed them in the canvas.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Let me mark embryos manually", {}),
        ToolExample("I want to click on embryos", {}),
    ],
)
async def manual_mark_embryos(
    exposure_ms: float = None,
    editor_timeout_sec: float = 600.0,
    context: Dict = None
) -> str:
    """Manual embryo marking via the web Marking canvas — additive only."""
    agent = context.get('agent')
    client = context.get('client')

    if not agent:
        return "Error: No agent context"
    if not client:
        return "Error: Microscope not connected. Cannot mark embryos in offline mode."

    viz_server = getattr(agent, 'viz_server', None)
    if viz_server is None:
        return "Error: Web visualization server not running. Marking requires the web UI."

    try:
        snap, stage_pos = await _capture_for_marking(agent, client, exposure_ms)
        if snap is None:
            return "Failed to capture image for marking."

        # Build initial markers from existing embryos so the operator has
        # context. Seed count is recorded so the post-Done logic can
        # treat anything beyond it as a new addition.
        initial_markers = _embryos_to_pixel_markers(
            agent.experiment.embryos.items(), snap, stage_pos,
        )
        seed_count = len(initial_markers)

        marked, note = await _run_marking_canvas(
            viz_server, snap, stage_pos, initial_markers,
            timeout_sec=editor_timeout_sec,
        )
        if marked is None:
            return f"Marking canceled or timed out{note}."

        # ADD-ONLY semantic: keep existing embryos as-is. Anything beyond
        # the seed count is a fresh sighting.
        new_markers = marked[seed_count:]
        if not new_markers:
            return f"No new embryos marked{note}."

        next_num = _next_embryo_number(agent.experiment.embryos.keys())
        added_ids = []
        for emb in new_markers:
            new_id = f'embryo_{next_num}'
            next_num += 1
            agent.experiment.add_embryo(
                embryo_id=new_id,
                position={
                    'x': emb.get('stage_x_um', 0.0),
                    'y': emb.get('stage_y_um', 0.0),
                },
                confidence=emb.get('confidence', 1.0),
                uid=str(uuid.uuid4()),
            )
            added_ids.append(new_id)

        _publish_operator_marked(
            agent,
            embryo_ids=added_ids,
            count=len(added_ids),
            stage_origin=list(stage_pos),
            pre_edit_count=seed_count,
            source='manual_mark_embryos:web-editor',
        )

        return f"Added {len(added_ids)} embryo(s): {', '.join(added_ids)}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="edit_embryos",
    description="""Open the web Marking canvas to modify embryo positions.
Allows adding new embryos, removing existing ones, and moving embryos to correct positions.
Use when user wants to adjust detection results (e.g., "edit embryos", "remove embryo_3", "adjust embryo positions", "fix detection").
The web canvas opens with the current bottom-camera image and existing embryos pre-placed; drag, add, or remove markers, then press Done. The experiment's embryo list is REPLACED with the marker set the operator confirms (positions go to coarse; any prior fine calibration is invalidated, matching the Map's edit semantics).""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Edit embryos", {}),
        ToolExample("Let me adjust embryo positions", {}),
        ToolExample("Fix the detection", {}),
    ],
)
async def edit_embryos(
    exposure_ms: float = None,
    editor_timeout_sec: float = 600.0,
    context: Dict = None
) -> str:
    """Interactive embryo editor via the web Marking canvas — replace-semantic."""
    agent = context.get('agent')
    client = context.get('client')

    if not agent:
        return "Error: No agent context"
    if not client:
        return "Error: Microscope not connected. Cannot edit embryos in offline mode."
    if not agent.experiment.embryos:
        return "No embryos to edit. Run detect_embryos or manual_mark_embryos first."

    viz_server = getattr(agent, 'viz_server', None)
    if viz_server is None:
        return "Error: Web visualization server not running. Editing requires the web UI."

    try:
        snap, stage_pos = await _capture_for_marking(agent, client, exposure_ms)
        if snap is None:
            return "Failed to capture image for editing."

        # Seed with every non-skipped existing embryo so the operator can
        # drag / delete / supplement against the live image.
        items = [(eid, st) for eid, st in agent.experiment.embryos.items()
                 if not getattr(st, 'should_skip', False)]
        initial_markers = _embryos_to_pixel_markers(items, snap, stage_pos)
        if not initial_markers:
            return "No editable embryos at this stage position (all are off-image or skipped)."

        marked, note = await _run_marking_canvas(
            viz_server, snap, stage_pos, initial_markers,
            timeout_sec=editor_timeout_sec,
        )
        if marked is None:
            return f"Edit canceled or timed out{note}; experiment unchanged."

        # REPLACE semantic: clear existing embryos and rebuild from the
        # operator's confirmed set. IDs are reassigned sequentially —
        # matching the napari path's effective behaviour. Position-based
        # ID preservation can come later if it's missed.
        old_ids = set(agent.experiment.embryos.keys())
        agent.experiment.embryos.clear()

        next_num = 1
        new_ids = []
        for emb in marked:
            new_id = f'embryo_{next_num}'
            next_num += 1
            agent.experiment.add_embryo(
                embryo_id=new_id,
                position={
                    'x': emb.get('stage_x_um', 0.0),
                    'y': emb.get('stage_y_um', 0.0),
                },
                confidence=emb.get('confidence', 1.0),
                uid=str(uuid.uuid4()),
            )
            new_ids.append(new_id)

        # Single consolidated broadcast so EMBRYOS_UPDATE arrives with the
        # final set (the clear()/add_embryo() pair would otherwise fan out
        # multiple intermediate updates).
        agent.experiment.notify_embryos_changed()

        _publish_operator_marked(
            agent,
            embryo_ids=new_ids,
            count=len(new_ids),
            stage_origin=list(stage_pos),
            pre_edit_count=len(old_ids),
            source='edit_embryos:web-editor',
        )

        added = max(0, len(new_ids) - len(old_ids))
        removed = max(0, len(old_ids) - len(new_ids))
        summary = f"Edit complete: {len(new_ids)} embryos"
        if added: summary += f", +{added}"
        if removed: summary += f", -{removed}"
        return summary

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


@tool(
    name="show_detected_embryos",
    description="""Capture a fresh image and display all tracked embryos with labeled bounding boxes. Shows embryo IDs at their positions.
Use when user wants to see where embryos are visually (e.g., "show me the embryos", "display embryo positions").
Captures a new bottom camera image and overlays all active (non-skipped) embryo positions. Image is saved to detection_results/.""",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the embryos", {}),
        ToolExample("Display embryo positions", {}),
    ],
)
async def show_detected_embryos(
    save_to_file: bool = True,
    context: Dict = None
) -> str:
    """Show detected embryos visualization using experiment.embryos as source of truth"""
    agent = context.get('agent')
    client = context.get('client')

    if not agent:
        return "Error: No agent context"

    if not client:
        return "Error: Microscope not connected. Cannot show embryos in offline mode."

    if not agent.experiment.embryos:
        return "No embryos in experiment. Run detect_embryos first."

    try:
        snap = await client.capture_bottom_image()
        image = snap['image']
        if image is None or image.shape == (100, 100):
            return "Failed to capture image for visualization."

        current_stage = await client.get_stage_position()

        # Archive the bottom camera image with metadata
        if snap.get('image_path') and agent.store and agent.session_id:
            try:
                from gently.harness.tools.helpers import build_snapshot_metadata
                meta = build_snapshot_metadata(
                    current_stage, image.shape, agent.experiment)
                agent.store.register_snapshot(
                    agent.session_id, "bottom_camera", snap['image_path'],
                    metadata=meta)
            except Exception:
                pass

        # Calculate pixel positions from experiment embryo positions
        um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py

        image_center_x = image.shape[1] / 2
        image_center_y = image.shape[0] / 2

        embryos = []
        for embryo_id, embryo_state in agent.experiment.embryos.items():
            pos = embryo_state.stage_position or {}

            stage_x = pos.get('x', current_stage[0])
            stage_y = pos.get('y', current_stage[1])

            # Convert stage to pixel using centralized function
            pixel_x, pixel_y = stage_to_pixel_position(
                stage_x=stage_x,
                stage_y=stage_y,
                current_stage_x=current_stage[0],
                current_stage_y=current_stage[1],
                image_center_x=image_center_x,
                image_center_y=image_center_y,
                um_per_pixel=um_per_pixel
            )

            embryos.append({
                'embryo_id': embryo_id,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y,
                'stage_x_um': stage_x,
                'stage_y_um': stage_y,
                'confidence': embryo_state.detection_confidence,
            })

        if not embryos:
            return "No embryos to display."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"detection_results/detected_embryos_{timestamp}.jpg"
        Path("detection_results").mkdir(exist_ok=True)

        view_result = await client.view_embryos(
            image=image,
            embryos=embryos,
            title=f"Embryos ({len(embryos)})",
            save_path=save_path,
            show=True
        )

        if view_result.get('success'):
            embryo_ids = [e.get('embryo_id', '?') for e in embryos]
            return f"Showing {len(embryos)} embryos: {', '.join(embryo_ids)}\nSaved to: {save_path}"
        elif view_result.get('error'):
            return f"Display error: {view_result.get('error')}"
        else:
            return f"Visualization complete. Check {save_path}"

    except Exception as e:
        return f"Error showing detections: {str(e)}"
