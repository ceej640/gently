"""
Global Error Log for cross-embryo hardware error correlation.

Provides a unified log of errors across all embryos during timelapse,
enabling detection of systematic hardware issues vs embryo-specific problems.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ErrorEntry:
    """Single error log entry."""
    timestamp: datetime
    round_number: int
    embryo_id: str
    timepoint: int
    error_type: str
    message: str
    exception: Optional[Exception] = None

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "round_number": self.round_number,
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "error_type": self.error_type,
            "message": self.message,
        }


class GlobalErrorLog:
    """
    Global error log for tracking hardware and acquisition errors.

    Enables cross-embryo error correlation to distinguish:
    - Systematic hardware issues (errors across multiple embryos)
    - Embryo-specific issues (errors on single embryo)
    """

    def __init__(self, max_entries: int = 1000):
        """
        Parameters
        ----------
        max_entries : int
            Maximum number of entries to retain (FIFO when exceeded)
        """
        self._entries: List[ErrorEntry] = []
        self._max_entries = max_entries
        self._by_round: Dict[int, List[ErrorEntry]] = defaultdict(list)
        self._by_embryo: Dict[str, List[ErrorEntry]] = defaultdict(list)

    def log_error(
        self,
        round_number: int,
        embryo_id: str,
        timepoint: int,
        error_type: str,
        message: str,
        exception: Optional[Exception] = None,
    ) -> None:
        """
        Log an error entry.

        Parameters
        ----------
        round_number : int
            Current timelapse round
        embryo_id : str
            Embryo that encountered the error
        timepoint : int
            Timepoint during error
        error_type : str
            Category of error (e.g., 'acquisition', 'stage_move', 'timeout')
        message : str
            Human-readable error message
        exception : Exception, optional
            Original exception if available
        """
        entry = ErrorEntry(
            timestamp=datetime.now(),
            round_number=round_number,
            embryo_id=embryo_id,
            timepoint=timepoint,
            error_type=error_type,
            message=message,
            exception=exception,
        )

        self._entries.append(entry)
        self._by_round[round_number].append(entry)
        self._by_embryo[embryo_id].append(entry)

        # Trim if over max
        if len(self._entries) > self._max_entries:
            removed = self._entries.pop(0)
            # Clean up indexes (simple approach)
            if removed.round_number in self._by_round:
                round_entries = self._by_round[removed.round_number]
                if removed in round_entries:
                    round_entries.remove(removed)
            if removed.embryo_id in self._by_embryo:
                embryo_entries = self._by_embryo[removed.embryo_id]
                if removed in embryo_entries:
                    embryo_entries.remove(removed)

        logger.warning(
            f"[ErrorLog] Round {round_number}, {embryo_id}, T{timepoint}: "
            f"{error_type} - {message}"
        )

    def compile_for_verification(self, current_round: int) -> str:
        """
        Compile error summary for perception/verification context.

        Returns a human-readable summary of recent errors that can help
        the perception system distinguish technical vs biological issues.

        Parameters
        ----------
        current_round : int
            Current timelapse round

        Returns
        -------
        str
            Summary of recent hardware errors
        """
        # Look at errors from last 3 rounds
        recent_rounds = range(max(0, current_round - 2), current_round + 1)
        recent_errors = []
        for r in recent_rounds:
            recent_errors.extend(self._by_round.get(r, []))

        if not recent_errors:
            return "No hardware errors in recent rounds."

        # Summarize
        lines = [f"Hardware errors (last 3 rounds):"]

        # Group by error type
        by_type: Dict[str, List[ErrorEntry]] = defaultdict(list)
        for e in recent_errors:
            by_type[e.error_type].append(e)

        for error_type, entries in by_type.items():
            embryos_affected = set(e.embryo_id for e in entries)
            if len(embryos_affected) > 1:
                lines.append(
                    f"  - {error_type}: {len(entries)} errors across "
                    f"{len(embryos_affected)} embryos (systematic issue?)"
                )
            else:
                lines.append(
                    f"  - {error_type}: {len(entries)} errors on {list(embryos_affected)[0]}"
                )

        return "\n".join(lines)

    def get_errors_for_embryo(self, embryo_id: str, limit: int = 10) -> List[Dict]:
        """Get recent errors for a specific embryo."""
        entries = self._by_embryo.get(embryo_id, [])
        return [e.to_dict() for e in entries[-limit:]]

    def get_errors_for_round(self, round_number: int) -> List[Dict]:
        """Get all errors for a specific round."""
        entries = self._by_round.get(round_number, [])
        return [e.to_dict() for e in entries]

    def clear(self) -> None:
        """Clear all error entries."""
        self._entries.clear()
        self._by_round.clear()
        self._by_embryo.clear()

    def __len__(self) -> int:
        return len(self._entries)
