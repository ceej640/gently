"""
Belief Updater - Updates belief state based on perception round output.

Handles:
- Stage probability updates with recency weighting
- One-way hatching state machine
- Dead embryo detection through temporal reasoning
- Evidence history accumulation
"""

import logging
from datetime import datetime
from typing import Optional

from .session import (
    BeliefState,
    EvidencePoint,
    PerceptionSession,
    PerceptionRoundOutput,
)

logger = logging.getLogger(__name__)


class BeliefUpdater:
    """
    Updates PerceptionSession beliefs based on new perception round output.

    Key dynamics:
    - Stage probabilities use weighted average (0.7 new / 0.3 prior)
    - Hatching state machine is one-way (can't un-hatch)
    - Dead embryo detection based on temporal stasis + stage context
    - Evidence accumulates for reasoning trace
    """

    def __init__(
        self,
        recency_weight: float = 0.7,
        dead_threshold_hours: float = 2.0,
        confidence_decay_rate: float = 0.95,
    ):
        """
        Parameters
        ----------
        recency_weight : float
            Weight for new observations (0.7 = 70% new, 30% prior)
        dead_threshold_hours : float
            Hours without change before suspecting dead embryo
        confidence_decay_rate : float
            Per-round decay for confidence without new evidence
        """
        self.recency_weight = recency_weight
        self.dead_threshold_hours = dead_threshold_hours
        self.confidence_decay_rate = confidence_decay_rate

    def update_beliefs(
        self,
        session: PerceptionSession,
        round_output: PerceptionRoundOutput,
        timepoint: int,
        time_delta_hours: float = 0.0,
    ) -> None:
        """
        Update session beliefs based on perception round output.

        Parameters
        ----------
        session : PerceptionSession
            Active session to update
        round_output : PerceptionRoundOutput
            Output from perception engine
        timepoint : int
            Current timepoint number
        time_delta_hours : float
            Time since last perception round in hours
        """
        old_beliefs = session.beliefs
        new_beliefs = round_output.updated_beliefs

        # 1. Update stage distribution (weighted average)
        self._update_stage_distribution(session, new_beliefs)

        # 2. Update temporal tracking
        self._update_temporal_state(session, new_beliefs, timepoint, time_delta_hours)

        # 3. Update hatching state machine (one-way)
        self._update_hatching_state(session, new_beliefs, timepoint)

        # 4. Update anomaly flags
        self._update_anomaly_flags(session, new_beliefs)

        # 5. Add evidence to history
        self._add_evidence(session, round_output, timepoint)

        # 6. Update session metadata
        session.rounds_processed += 1
        session.last_processed_timepoint = timepoint

        logger.debug(
            f"[{session.embryo_id}] Beliefs updated: "
            f"stage={session.beliefs.most_likely_stage} "
            f"({session.beliefs.stage_confidence:.0%}), "
            f"hatching_complete={session.beliefs.hatching_complete}, "
            f"dead={session.beliefs.possibly_dead}"
        )

    def _update_stage_distribution(
        self,
        session: PerceptionSession,
        new_beliefs: BeliefState,
    ) -> None:
        """Update stage probabilities with weighted average"""
        old_dist = session.beliefs.stage_distribution
        new_dist = new_beliefs.stage_distribution

        # Get all stages from both distributions
        all_stages = set(old_dist.keys()) | set(new_dist.keys())

        # Weighted average
        updated_dist = {}
        for stage in all_stages:
            old_prob = old_dist.get(stage, 0.0)
            new_prob = new_dist.get(stage, 0.0)
            updated_dist[stage] = (
                self.recency_weight * new_prob +
                (1 - self.recency_weight) * old_prob
            )

        # Normalize to sum to 1
        total = sum(updated_dist.values())
        if total > 0:
            updated_dist = {k: v / total for k, v in updated_dist.items()}

        session.beliefs.stage_distribution = updated_dist

        # Update most likely stage
        if updated_dist:
            session.beliefs.most_likely_stage = max(
                updated_dist, key=updated_dist.get
            )
            session.beliefs.stage_confidence = updated_dist[
                session.beliefs.most_likely_stage
            ]
        else:
            session.beliefs.most_likely_stage = "early"
            session.beliefs.stage_confidence = 0.3

    def _update_temporal_state(
        self,
        session: PerceptionSession,
        new_beliefs: BeliefState,
        timepoint: int,
        time_delta_hours: float,
    ) -> None:
        """Update temporal tracking for change detection"""
        session.beliefs.movement_detected = new_beliefs.movement_detected

        # Check if there was significant change
        stage_changed = (
            new_beliefs.most_likely_stage != session.beliefs.most_likely_stage
        )
        confidence_changed = abs(
            new_beliefs.stage_confidence - session.beliefs.stage_confidence
        ) > 0.1

        has_significant_change = (
            stage_changed or
            confidence_changed or
            new_beliefs.movement_detected
        )

        if has_significant_change:
            # Reset unchanged counter
            session.unchanged_count = 0
            session.beliefs.hours_since_change = 0.0
            session.beliefs.last_significant_change_timepoint = timepoint
        else:
            # Increment unchanged counter
            session.unchanged_count += 1
            session.beliefs.hours_since_change += time_delta_hours

    def _update_hatching_state(
        self,
        session: PerceptionSession,
        new_beliefs: BeliefState,
        timepoint: int,
    ) -> None:
        """
        Update hatching state machine.

        States: not_hatching -> breach_detected -> worm_exiting -> hatching_complete

        This is a ONE-WAY progression. Once hatching starts, we track it
        through completion even if intermediate frames are ambiguous.
        """
        old = session.beliefs

        # Hatching complete is terminal state
        if old.hatching_complete:
            return

        # Check for state transitions (one-way only)
        if new_beliefs.hatching_complete and not old.hatching_complete:
            session.beliefs.hatching_complete = True
            session.beliefs.hatching_in_progress = False
            session.beliefs.hatching_timepoint = timepoint
            logger.info(f"[{session.embryo_id}] Hatching COMPLETE at T{timepoint}")

        elif new_beliefs.worm_exiting and not old.worm_exiting:
            session.beliefs.worm_exiting = True
            session.beliefs.hatching_in_progress = True
            logger.info(f"[{session.embryo_id}] Worm EXITING detected at T{timepoint}")

        elif new_beliefs.breach_detected and not old.breach_detected:
            session.beliefs.breach_detected = True
            session.beliefs.hatching_in_progress = True
            logger.info(f"[{session.embryo_id}] Breach DETECTED at T{timepoint}")

        elif new_beliefs.hatching_in_progress and not old.hatching_in_progress:
            session.beliefs.hatching_in_progress = True
            logger.info(f"[{session.embryo_id}] Hatching IN PROGRESS at T{timepoint}")

    def _update_anomaly_flags(
        self,
        session: PerceptionSession,
        new_beliefs: BeliefState,
    ) -> None:
        """Update anomaly detection flags"""
        # Dead embryo detection - combine VLM output with temporal reasoning
        if new_beliefs.possibly_dead:
            session.beliefs.possibly_dead = True
            session.beliefs.dead_confidence = max(
                session.beliefs.dead_confidence,
                new_beliefs.dead_confidence
            )
            if new_beliefs.dead_evidence:
                for ev in new_beliefs.dead_evidence:
                    if ev not in session.beliefs.dead_evidence:
                        session.beliefs.dead_evidence.append(ev)

        # Also check temporal-based dead detection
        if (session.beliefs.hours_since_change > self.dead_threshold_hours and
            session.beliefs.most_likely_stage in ["pretzel", "3fold"] and
            not session.beliefs.movement_detected):
            # Late stage + no change + no movement = possibly dead
            session.beliefs.possibly_dead = True
            session.beliefs.dead_confidence = min(
                0.9,
                session.beliefs.hours_since_change / 4  # Max out at 4 hours
            )
            evidence = f"No change for {session.beliefs.hours_since_change:.1f} hours in late stage"
            if evidence not in session.beliefs.dead_evidence:
                session.beliefs.dead_evidence.append(evidence)

        # If movement is detected, clear dead suspicion
        if new_beliefs.movement_detected and session.beliefs.possibly_dead:
            session.beliefs.possibly_dead = False
            session.beliefs.dead_confidence = 0.0
            session.beliefs.dead_evidence = []
            logger.info(f"[{session.embryo_id}] Movement detected, clearing dead suspicion")

        # Technical issue flags
        session.beliefs.technical_issue_suspected = new_beliefs.technical_issue_suspected
        session.beliefs.technical_issue_type = new_beliefs.technical_issue_type

    def _add_evidence(
        self,
        session: PerceptionSession,
        round_output: PerceptionRoundOutput,
        timepoint: int,
    ) -> None:
        """Add new evidence to session history"""
        for ev_dict in round_output.new_evidence:
            evidence = EvidencePoint(
                timepoint=timepoint,
                timestamp=datetime.now(),
                observation=ev_dict.get("observation", ""),
                supports=ev_dict.get("supports", []),
                contradicts=ev_dict.get("contradicts", []),
                confidence=ev_dict.get("confidence", 0.5),
                image_uid=ev_dict.get("image_uid"),
            )
            session.add_evidence(evidence)

        # Add reasoning to trace (truncated)
        if round_output.reasoning:
            session.add_reasoning(
                round_output.reasoning[:500],
                timepoint=timepoint
            )
