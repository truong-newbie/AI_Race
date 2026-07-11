"""
Conflict Logger

Logs entity conflicts and merges for analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging


class ConflictType(str, Enum):
    """Types of entity conflicts."""
    SAME_SPAN_DIFFERENT_TYPE = "same_span_different_type"
    OVERLAP_SAME_TYPE = "overlap_same_type"
    OVERLAP_DIFFERENT_TYPE = "overlap_different_type"
    TYPE_PAIR_CONFLICT = "type_pair_conflict"  # e.g., TÊN_XÉT_NGHIỆM vs KẾT_QUẢ_XÉT_NGHIỆM


@dataclass
class ConflictRecord:
    """Record of a single conflict."""
    timestamp: str
    conflict_type: ConflictType
    text: str
    span: tuple  # (start, end)
    entities: List[Dict[str, Any]]
    resolution: str
    winner: Optional[str] = None
    section: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "conflict_type": self.conflict_type.value,
            "text": self.text,
            "span": self.span,
            "entities": self.entities,
            "resolution": self.resolution,
            "winner": self.winner,
            "section": self.section,
            "notes": self.notes,
        }


@dataclass
class ConflictReport:
    """Report of all conflicts."""
    conflicts: List[ConflictRecord] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def add_conflict(self, conflict: ConflictRecord):
        """Add a conflict to the report."""
        self.conflicts.append(conflict)
        ctype = conflict.conflict_type.value
        self.stats[ctype] = self.stats.get(ctype, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_conflicts": len(self.conflicts),
            "stats": self.stats,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }

    def save(self, path: str):
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class ConflictLogger:
    """Logger for entity conflicts during ensemble resolution."""

    def __init__(self, enable_logging: bool = True):
        """Initialize conflict logger.

        Args:
            enable_logging: Whether to also log to Python logger
        """
        self.conflicts: List[ConflictRecord] = []
        self.enable_logging = enable_logging

        if enable_logging:
            self.logger = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_same_span_diff_type(
        self,
        text: str,
        start: int,
        end: int,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        resolution: str,
        winner: Optional[str] = None,
        section: Optional[str] = None,
    ):
        """Log conflict where same span has different types."""
        conflict = ConflictRecord(
            timestamp=datetime.now().isoformat(),
            conflict_type=ConflictType.SAME_SPAN_DIFFERENT_TYPE,
            text=text[start:end],
            span=(start, end),
            entities=[entity1, entity2],
            resolution=resolution,
            winner=winner,
            section=section,
        )
        self._add_and_log(conflict)

    def log_overlap_same_type(
        self,
        text: str,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        resolution: str,
        winner: Optional[str] = None,
        section: Optional[str] = None,
    ):
        """Log conflict where overlapping entities have same type."""
        conflict = ConflictRecord(
            timestamp=datetime.now().isoformat(),
            conflict_type=ConflictType.OVERLAP_SAME_TYPE,
            text=text,
            span=(entity1.get("start", 0), entity2.get("end", 0)),
            entities=[entity1, entity2],
            resolution=resolution,
            winner=winner,
            section=section,
        )
        self._add_and_log(conflict)

    def log_overlap_diff_type(
        self,
        text: str,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        resolution: str,
        winner: Optional[str] = None,
        section: Optional[str] = None,
    ):
        """Log conflict where overlapping entities have different types."""
        conflict = ConflictRecord(
            timestamp=datetime.now().isoformat(),
            conflict_type=ConflictType.OVERLAP_DIFFERENT_TYPE,
            text=text,
            span=(min(entity1.get("start", 0), entity2.get("start", 0)),
                  max(entity1.get("end", 0), entity2.get("end", 0))),
            entities=[entity1, entity2],
            resolution=resolution,
            winner=winner,
            section=section,
        )
        self._add_and_log(conflict)

    def log_type_pair_conflict(
        self,
        text: str,
        start: int,
        end: int,
        entity1: Dict[str, Any],
        entity2: Dict[str,
        Any],
        resolution: str,
        section: Optional[str] = None,
    ):
        """Log conflict for forbidden type pairs (e.g., TÊN_XÉT_NGHIỆM + KẾT_QUẢ)."""
        conflict = ConflictRecord(
            timestamp=datetime.now().isoformat(),
            conflict_type=ConflictType.TYPE_PAIR_CONFLICT,
            text=text[start:end],
            span=(start, end),
            entities=[entity1, entity2],
            resolution=resolution,
            section=section,
        )
        self._add_and_log(conflict)

    def _add_and_log(self, conflict: ConflictRecord):
        """Add conflict and optionally log."""
        self.conflicts.append(conflict)

        if self.enable_logging:
            self.logger.info(
                f"Conflict [{conflict.conflict_type.value}]: "
                f"'{conflict.text}' ({conflict.span}) - "
                f"Resolution: {conflict.resolution}"
            )

    def get_report(self) -> ConflictReport:
        """Generate conflict report."""
        report = ConflictReport()

        for conflict in self.conflicts:
            report.add_conflict(conflict)

        return report

    def save_report(self, path: str):
        """Save conflict report to file."""
        report = self.get_report()
        report.save(path)

    def print_summary(self):
        """Print summary of conflicts."""
        report = self.get_report()
        print("=" * 60)
        print("CONFLICT SUMMARY")
        print("=" * 60)
        print(f"Total conflicts: {len(self.conflicts)}")
        print("\nBy type:")
        for ctype, count in report.stats.items():
            print(f"  {ctype}: {count}")
        print("=" * 60)
