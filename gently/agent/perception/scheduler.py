"""
Adaptive Perception Scheduler - Optimizes perception round frequency for cost.

Runs full analysis less frequently when embryos are stable,
more frequently during transitions and critical states.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

from .session import BeliefState, PerceptionSession

logger = logging.getLogger(__name__)


class AdaptivePerceptionScheduler:
    """
    Schedule perception rounds adaptively based on embryo state.

    Cost optimization strategy:
    - Run full analysis always during critical states (near hatching, anomalies)
    - Reduce frequency for stable embryos (use fast checks instead)
    - Adapt interval based on hours since last change
    """

    def __init__(
        self,
        min_interval_minutes: float = 2.0,
        max_interval_minutes: float = 10.0,
        critical_interval_minutes: float = 2.0,
    ):
        """
        Parameters
        ----------
        min_interval_minutes : float
            Minimum interval between full analyses for stable embryos
        max_interval_minutes : float
            Maximum interval for very stable embryos
        critical_interval_minutes : float
            Interval during critical states (always full analysis)
        """
        self.min_interval = min_interval_minutes
        self.max_interval = max_interval_minutes
        self.critical_interval = critical_interval_minutes

        # Track when full analysis was last run per embryo
        self._last_full_analysis: Dict[str, datetime] = {}

    def should_run_full_analysis(
        self,
        embryo_id: str,
        session: PerceptionSession,
        current_timepoint: int,
    ) -> bool:
        """
        Determine if full analysis is needed this round.

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        session : PerceptionSession
            Current perception session
        current_timepoint : int
            Current timepoint number

        Returns
        -------
        bool
            True if full analysis should be run, False for fast check
        """
        beliefs = session.beliefs

        # Always run full analysis in critical states
        if self._is_critical_state(beliefs):
            logger.debug(f"[{embryo_id}] Critical state - running full analysis")
            return True

        # First round always gets full analysis
        if session.rounds_processed == 0:
            return True

        # Check time since last full analysis
        last_full = self._last_full_analysis.get(embryo_id)
        if last_full is None:
            return True

        minutes_since = (datetime.now() - last_full).total_seconds() / 60

        # Calculate adaptive interval based on stability
        interval = self._get_adaptive_interval(session)

        should_run = minutes_since >= interval

        if should_run:
            logger.debug(
                f"[{embryo_id}] {minutes_since:.1f}min since last full analysis "
                f"(threshold: {interval:.1f}min) - running full"
            )
        else:
            logger.debug(
                f"[{embryo_id}] {minutes_since:.1f}min since last full analysis "
                f"(threshold: {interval:.1f}min) - using fast check"
            )

        return should_run

    def _is_critical_state(self, beliefs: BeliefState) -> bool:
        """Check if embryo is in a critical state requiring full analysis"""
        # Near hatching
        if beliefs.hatching_in_progress or beliefs.worm_exiting:
            return True

        # Late stage with high confidence (approaching hatching)
        if beliefs.most_likely_stage in ["pretzel", "3fold"]:
            if beliefs.stage_confidence > 0.7:
                return True

        # Anomaly suspected
        if beliefs.possibly_dead and beliefs.dead_confidence > 0.5:
            return True

        # Technical issue suspected
        if beliefs.technical_issue_suspected:
            return True

        return False

    def _get_adaptive_interval(self, session: PerceptionSession) -> float:
        """
        Get adaptive interval based on embryo stability.

        More stable = longer interval (save cost)
        Less stable = shorter interval (catch changes)
        """
        beliefs = session.beliefs

        # Very stable (5+ unchanged rounds)
        if session.unchanged_count > 5:
            return self.max_interval

        # Moderately stable (2-5 unchanged rounds)
        if session.unchanged_count > 2:
            return (self.min_interval + self.max_interval) / 2

        # Uncertain about stage
        if beliefs.stage_confidence < 0.6:
            return self.min_interval

        # Significant time since last change
        if beliefs.hours_since_change > 1:
            return (self.min_interval + self.max_interval) / 2

        # Default to minimum interval
        return self.min_interval

    def mark_full_analysis_run(self, embryo_id: str) -> None:
        """Mark that full analysis was run for an embryo"""
        self._last_full_analysis[embryo_id] = datetime.now()

    def get_next_full_analysis_time(
        self,
        embryo_id: str,
        session: PerceptionSession,
    ) -> Optional[datetime]:
        """
        Estimate when next full analysis will be needed.

        Returns
        -------
        datetime or None
            Estimated time for next full analysis
        """
        last_full = self._last_full_analysis.get(embryo_id)
        if last_full is None:
            return datetime.now()

        interval = self._get_adaptive_interval(session)

        from datetime import timedelta
        return last_full + timedelta(minutes=interval)

    def reset(self, embryo_id: Optional[str] = None) -> None:
        """
        Reset scheduler state.

        Parameters
        ----------
        embryo_id : str, optional
            Reset specific embryo, or all if None
        """
        if embryo_id:
            self._last_full_analysis.pop(embryo_id, None)
        else:
            self._last_full_analysis.clear()

    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        return {
            "tracked_embryos": len(self._last_full_analysis),
            "min_interval_minutes": self.min_interval,
            "max_interval_minutes": self.max_interval,
            "critical_interval_minutes": self.critical_interval,
        }
