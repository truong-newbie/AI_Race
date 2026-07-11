"""
Span Resolver - Overlap Resolution

Module để resolve overlapping entities và chọn entity tốt nhất.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Một span với position và metadata."""
    start: int
    end: int
    text: str
    entity_type: str
    confidence: float = 1.0
    source: str = "unknown"  # rule, model, etc.

    def __post_init__(self):
        """Ensure end > start."""
        if self.end <= self.start:
            raise ValueError(f"Invalid span: [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """Length của span."""
        return self.end - self.start

    def overlaps_with(self, other: "Span") -> bool:
        """Check if overlaps với another span."""
        return self.start < other.end and other.start < self.end

    def contains(self, other: "Span") -> bool:
        """Check if contains another span."""
        return self.start <= other.start and self.end >= other.end

    def intersection(self, other: "Span") -> Optional["Span"]:
        """Get intersection với another span."""
        if not self.overlaps_with(other):
            return None

        new_start = max(self.start, other.start)
        new_end = min(self.end, other.end)
        return Span(
            start=new_start,
            end=new_end,
            text="",  # Will be filled later
            entity_type="OVERLAP"
        )


@dataclass
class ResolutionResult:
    """Kết quả của span resolution."""
    resolved_spans: list[Span]
    removed_spans: list[Span]
    merged_spans: list[Span]


class SpanResolver:
    """
    Resolver để xử lý overlapping entities.

    Resolution strategies:
    1. Longest span wins
    2. Highest confidence wins
    3. Specific type wins over general type
    4. Merge adjacent spans of same type
    """

    # Type priority (higher = more specific, wins in overlap)
    TYPE_PRIORITY = {
        "THUỐC": 5,
        "CHẨN_ĐOÁN": 4,
        "TRIỆU_CHỨNG": 3,
        "KẾT_QUẢ_XÉT_NGHIỆM": 2,
        "TÊN_XÉT_NGHIỆM": 1,
    }

    def __init__(self, strategy: str = "confidence"):
        """
        Initialize resolver.

        Args:
            strategy: Resolution strategy
                - "longest": Longest span wins
                - "confidence": Highest confidence wins
                - "type_priority": More specific type wins
                - "hybrid": Combination of above
        """
        self.strategy = strategy

    def resolve(self, spans: list[Span]) -> ResolutionResult:
        """
        Resolve overlapping spans.

        Args:
            spans: List of spans to resolve

        Returns:
            ResolutionResult với resolved spans
        """
        if not spans:
            return ResolutionResult(resolved_spans=[], removed_spans=[], merged_spans=[])

        # Sort spans
        sorted_spans = sorted(spans, key=lambda x: (x.start, -x.end))

        resolved = []
        removed = []
        merged = []

        i = 0
        while i < len(sorted_spans):
            current = sorted_spans[i]

            # Find all overlapping spans
            overlapping = [current]
            j = i + 1
            while j < len(sorted_spans):
                next_span = sorted_spans[j]
                if current.overlaps_with(next_span):
                    overlapping.append(next_span)
                    j += 1
                else:
                    # No overlap - since spans are sorted by start, no subsequent span will overlap either
                    break

            # Resolve overlapping group
            if len(overlapping) == 1:
                resolved.append(current)
            else:
                winner = self._resolve_overlap(overlapping)
                resolved.append(winner)

                # Mark others as removed
                for span in overlapping:
                    if span != winner:
                        removed.append(span)

            i = j

        # Try to merge adjacent spans of same type
        resolved = self._merge_adjacent(resolved, merged)

        return ResolutionResult(
            resolved_spans=resolved,
            removed_spans=removed,
            merged_spans=merged
        )

    def _resolve_overlap(self, overlapping: list[Span]) -> Span:
        """
        Resolve a group of overlapping spans.

        Returns the winning span.
        """
        if len(overlapping) == 1:
            return overlapping[0]

        if self.strategy == "longest":
            return max(overlapping, key=lambda x: x.length)
        elif self.strategy == "confidence":
            return max(overlapping, key=lambda x: x.confidence)
        elif self.strategy == "type_priority":
            return max(
                overlapping,
                key=lambda x: self.TYPE_PRIORITY.get(x.entity_type, 0)
            )
        elif self.strategy == "hybrid":
            # Combination: longer + higher confidence + more specific type
            def hybrid_score(span: Span) -> tuple:
                return (
                    span.length,
                    span.confidence,
                    self.TYPE_PRIORITY.get(span.entity_type, 0)
                )
            return max(overlapping, key=hybrid_score)
        else:
            return overlapping[0]

    def _merge_adjacent(self, spans: list[Span], merged_list: list) -> list[Span]:
        """
        Merge adjacent spans of the same type.

        E.g., ["ho"] + ["đờm"] -> ["ho đờm"] if both are symptoms
        """
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x.start)
        if not sorted_spans:
            return []

        merged = [sorted_spans[0]]

        for span in sorted_spans[1:]:
            if not merged:
                merged.append(span)
                continue
            last = merged[-1]

            # Check if adjacent (end of last = start of current)
            if last.end == span.start and last.entity_type == span.entity_type:
                # Merge them
                merged_text = f"{last.text} {span.text}"
                merged_span = Span(
                    start=last.start,
                    end=span.end,
                    text=merged_text,
                    entity_type=last.entity_type,
                    confidence=min(last.confidence, span.confidence),
                    source="merged"
                )
                merged[-1] = merged_span
                merged_list.append(merged_span)
            else:
                merged.append(span)

        return merged


class EntitySpanMerger:
    """
    Merger để combine entities từ different extractors.

    Priority: model > rule-based
    """

    def merge(
        self,
        rule_spans: list[Span],
        model_spans: list[Span],
        prefer_model: bool = True
    ) -> list[Span]:
        """
        Merge spans từ different sources.

        Args:
            rule_spans: Spans từ rule-based extraction
            model_spans: Spans từ ML model
            prefer_model: If True, prefer model spans in conflicts

        Returns:
            Merged list of spans
        """
        all_spans = rule_spans + model_spans

        if prefer_model:
            # Sort: model first, then by confidence
            all_spans.sort(
                key=lambda x: (
                    0 if x.source == "model" else 1,
                    -x.confidence
                )
            )
        else:
            # Sort by confidence only
            all_spans.sort(key=lambda x: -x.confidence)

        resolver = SpanResolver(strategy="confidence")
        result = resolver.resolve(all_spans)

        return result.resolved_spans


# =============================================================================
# Utility Functions
# =============================================================================

def resolve_spans(spans: list[Span], strategy: str = "hybrid") -> list[Span]:
    """
    Convenience function để resolve spans.

    Args:
        spans: List of spans
        strategy: Resolution strategy

    Returns:
        List of resolved spans
    """
    resolver = SpanResolver(strategy=strategy)
    result = resolver.resolve(spans)
    return result.resolved_spans


def create_span(
    text: str,
    position: list[int],
    entity_type: str,
    confidence: float = 1.0,
    source: str = "rule"
) -> Span:
    """
    Create a Span from text and position.

    Args:
        text: Entity text
        position: [start, end] position
        entity_type: Entity type
        confidence: Confidence score
        source: Source of extraction

    Returns:
        Span object
    """
    return Span(
        start=position[0],
        end=position[1],
        text=text,
        entity_type=entity_type,
        confidence=confidence,
        source=source
    )


# =============================================================================
# Tests
# =============================================================================

def test_span_resolver():
    """Test span resolver."""
    resolver = SpanResolver(strategy="hybrid")

    # Test case 1: Overlapping spans
    spans = [
        Span(12, 19, "ho đờm", "TRIỆU_CHỨNG", 0.9),
        Span(16, 24, "đờm xanh", "TRIỆU_CHỨNG", 0.8),
        Span(12, 17, "ho", "TRIỆU_CHỨNG", 0.7),
    ]

    result = resolver.resolve(spans)
    print("=== Overlap Resolution ===")
    print(f"Input spans: {len(spans)}")
    print(f"Resolved: {len(result.resolved_spans)}")
    print(f"Removed: {len(result.removed_spans)}")
    for span in result.resolved_spans:
        print(f"  [{span.start}:{span.end}] {span.text} ({span.entity_type})")
    print()

    # Test case 2: Non-overlapping spans
    spans2 = [
        Span(0, 5, "BN ho", "TRIỆU_CHỨNG", 0.9),
        Span(10, 15, "tức", "TRIỆU_CHỨNG", 0.8),
    ]

    result2 = resolver.resolve(spans2)
    print("=== Non-overlapping ===")
    print(f"Resolved: {len(result2.resolved_spans)}")
    for span in result2.resolved_spans:
        print(f"  [{span.start}:{span.end}] {span.text}")
    print()

    # Test case 3: Type priority
    spans3 = [
        Span(0, 10, "trào ngược", "TRIỆU_CHỨNG", 0.9),
        Span(0, 10, "trào ngược", "CHẨN_ĐOÁN", 0.7),  # Should win due to type priority
    ]

    resolver3 = SpanResolver(strategy="type_priority")
    result3 = resolver3.resolve(spans3)
    print("=== Type Priority ===")
    for span in result3.resolved_spans:
        print(f"  [{span.start}:{span.end}] {span.text} ({span.entity_type})")


if __name__ == "__main__":
    test_span_resolver()
