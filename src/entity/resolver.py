"""
Entity Resolver

Core logic for resolving entity conflicts and merging.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass

from src.entity.confidence import ConfidenceConfig, EntitySource, should_never_merge
from src.entity.conflict_logger import ConflictLogger, ConflictType


@dataclass
class ResolvedEntity:
    """A resolved entity after merging."""
    text: str
    start: int
    end: int
    type: str
    confidence: float
    source: str  # Comma-separated sources
    source_scores: Dict[str, float]
    original_entities: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to entity dict."""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "confidence": self.confidence,
            "source": self.source,
            "source_scores": self.source_scores,
        }


class EntityResolver:
    """Resolves entity conflicts and merges overlapping entities."""

    def __init__(
        self,
        config: Optional[ConfidenceConfig] = None,
        conflict_logger: Optional[ConflictLogger] = None,
    ):
        """Initialize resolver.

        Args:
            config: Confidence configuration
            conflict_logger: Optional conflict logger
        """
        self.config = config or ConfidenceConfig()
        self.logger = conflict_logger or ConflictLogger(enable_logging=False)

    def compute_iou(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
        """Compute Intersection over Union for two spans.

        Args:
            span1: First span (start, end)
            span2: Second span (start, end)

        Returns:
            IoU score (0-1)
        """
        start1, end1 = span1
        start2, end2 = span2

        # Compute intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        if intersection_start >= intersection_end:
            return 0.0

        intersection = intersection_end - intersection_start

        # Compute union
        union = max(end1, end2) - min(start1, start2)

        return intersection / union if union > 0 else 0.0

    def spans_overlap(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans overlap."""
        return self.compute_iou(span1, span2) > 0

    def spans_equal(self, span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
        """Check if two spans are equal."""
        return span1[0] == span2[0] and span1[1] == span2[1]

    def resolve_entities(
        self,
        entities: List[Dict[str, Any]],
        original_text: str,
        section: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve and merge entities.

        Args:
            entities: List of entities from all sources
            original_text: Original text for extracting entity text
            section: Current section context

        Returns:
            List of resolved entities
        """
        if not entities:
            return []

        # Step 1: Group entities by span
        span_groups = self._group_by_span(entities)

        # Step 2: Resolve each group
        resolved = []
        for span, group in span_groups.items():
            resolved_entity = self._resolve_group(
                group, original_text, section
            )
            if resolved_entity:
                resolved.append(resolved_entity)

        # Step 3: Handle overlaps between resolved entities
        resolved = self._resolve_overlaps(resolved, original_text, section)

        return [e.to_dict() for e in resolved]

    def _group_by_span(self, entities: List[Dict[str, Any]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
        """Group entities by exact span."""
        groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}

        for entity in entities:
            span = (entity.get("start", 0), entity.get("end", 0))
            if span not in groups:
                groups[span] = []
            groups[span].append(entity)

        return groups

    def _resolve_group(
        self,
        group: List[Dict[str, Any]],
        original_text: str,
        section: Optional[str],
    ) -> Optional[ResolvedEntity]:
        """Resolve a group of entities with the same span."""
        if not group:
            return None

        # Get span info
        span = (group[0].get("start", 0), group[0].get("end", 0))

        # Get unique types in this group
        types = set(e.get("type") for e in group)

        # Rule 5: TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM must not be merged
        if should_never_merge_any_pair(types, self.config):
            # Return all entities separately
            if len(group) == 1:
                return self._entity_to_resolved(group[0], original_text)
            else:
                # Keep the highest confidence one
                best = max(group, key=lambda e: e.get("confidence", 0))
                self.logger.log_type_pair_conflict(
                    text=original_text,
                    start=span[0],
                    end=span[1],
                    entity1=group[0],
                    entity2=group[1],
                    resolution="kept_highest_confidence",
                    section=section,
                )
                return self._entity_to_resolved(best, original_text)

        # Check for same span, different types (Rule 2)
        if len(types) > 1:
            return self._resolve_type_conflict(group, original_text, section)

        # Same span, same type - merge (Rule 1)
        return self._merge_same_type(group, original_text)

    def _resolve_type_conflict(
        self,
        group: List[Dict[str, Any]],
        original_text: str,
        section: Optional[str],
    ) -> ResolvedEntity:
        """Resolve entities with same span but different types."""
        span = (group[0].get("start", 0), group[0].get("end", 0))
        text = original_text[span[0]:span[1]]

        # Get confidence for each type with section context
        scored = []
        for entity in group:
            conf = self._compute_context_confidence(entity, section)
            scored.append((entity, conf))

        # Sort by confidence (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        best_entity, best_conf = scored[0]

        # Get all sources
        sources = set(e.get("source", "unknown") for e in group)
        source_scores = {}
        for e in group:
            src = e.get("source", "unknown")
            source_scores[src] = max(source_scores.get(src, 0), e.get("confidence", 0))

        # Log conflict
        self.logger.log_same_span_diff_type(
            text=original_text,
            start=span[0],
            end=span[1],
            entity1=group[0],
            entity2=group[1],
            resolution=f"selected_{best_entity.get('type')}_with_conf_{best_conf:.2f}",
            winner=best_entity.get("type"),
            section=section,
        )

        return ResolvedEntity(
            text=text,
            start=span[0],
            end=span[1],
            type=best_entity.get("type"),
            confidence=best_conf,
            source=",".join(sorted(sources)),
            source_scores=source_scores,
            original_entities=group,
        )

    def _merge_same_type(
        self,
        group: List[Dict[str, Any]],
        original_text: str,
    ) -> ResolvedEntity:
        """Merge entities with same span and same type."""
        span = (group[0].get("start", 0), group[0].get("end", 0))
        text = original_text[span[0]:span[1]]

        entity_type = group[0].get("type")

        # Compute merged confidence
        confidences = [e.get("confidence", 0) for e in group]
        merged_conf = sum(confidences) / len(confidences)

        # Add agreement bonus
        if len(group) > 1:
            merged_conf = min(merged_conf + self.config.agreement_bonus, 1.0)

        # Get all sources
        sources = set(e.get("source", "unknown") for e in group)
        source_scores = {}
        for e in group:
            src = e.get("source", "unknown")
            source_scores[src] = max(source_scores.get(src, 0), e.get("confidence", 0))

        return ResolvedEntity(
            text=text,
            start=span[0],
            end=span[1],
            type=entity_type,
            confidence=merged_conf,
            source=",".join(sorted(sources)),
            source_scores=source_scores,
            original_entities=group,
        )

    def _resolve_overlaps(
        self,
        entities: List[ResolvedEntity],
        original_text: str,
        section: Optional[str],
    ) -> List[ResolvedEntity]:
        """Resolve overlapping entities."""
        if len(entities) <= 1:
            return entities

        resolved = []
        remaining = list(entities)

        while remaining:
            current = remaining.pop(0)
            overlaps = []

            for other in remaining:
                if self.spans_overlap(
                    (current.start, current.end),
                    (other.start, other.end)
                ):
                    overlaps.append(other)

            if not overlaps:
                resolved.append(current)
            else:
                # Remove overlapping entities from remaining
                for o in overlaps:
                    remaining.remove(o)

                # Resolve overlap
                winner = self._resolve_overlap(current, overlaps, original_text, section)
                resolved.append(winner)

        return resolved

    def _resolve_overlap(
        self,
        entity: ResolvedEntity,
        others: List[ResolvedEntity],
        original_text: str,
        section: Optional[str],
    ) -> ResolvedEntity:
        """Resolve overlap between entities."""
        # Rule 3: Prefer higher confidence
        all_entities = [entity] + others
        all_entities.sort(key=lambda e: e.confidence, reverse=True)

        winner = all_entities[0]

        # Log overlap conflict
        self.logger.log_overlap_same_type(
            text=original_text,
            entity1=entity.to_dict(),
            entity2=others[0].to_dict(),
            resolution=f"selected_{winner.type}_with_conf_{winner.confidence:.2f}",
            winner=winner.type,
            section=section,
        )

        return winner

    def _compute_context_confidence(
        self,
        entity: Dict[str, Any],
        section: Optional[str],
    ) -> float:
        """Compute confidence with context adjustment."""
        base_conf = entity.get("confidence", 0.5)

        if not section:
            return base_conf

        # Apply section weights
        entity_type = entity.get("type")
        section_lower = section.lower()

        for section_key, type_weights in self.config.section_weights.items():
            if section_key in section_lower:
                weight = type_weights.get(entity_type, 1.0)
                return min(base_conf * weight, 1.0)

        return base_conf

    def _entity_to_resolved(
        self,
        entity: Dict[str, Any],
        original_text: str,
    ) -> ResolvedEntity:
        """Convert entity dict to ResolvedEntity."""
        start = entity.get("start", 0)
        end = entity.get("end", 0)

        # Rule 7 & 8: Always get text from original
        text = original_text[start:end] if 0 <= start < end else entity.get("text", "")

        return ResolvedEntity(
            text=text,
            start=start,
            end=end,
            type=entity.get("type", "UNKNOWN"),
            confidence=entity.get("confidence", 0.5),
            source=entity.get("source", "unknown"),
            source_scores=entity.get("source_scores", {}),
            original_entities=[entity],
        )


def should_never_merge_any_pair(types: Set[str], config: ConfidenceConfig) -> bool:
    """Check if any forbidden pair exists in types."""
    for t1, t2 in config.never_merge_pairs:
        if t1 in types and t2 in types:
            return True
    return False
