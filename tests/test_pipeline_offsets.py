"""
Tests for Unicode offset correctness in pipeline output.

Requirement: entity.text must be exactly text[start:end] where text is original.
Tests verify character-level slicing (not byte-level) works correctly for UTF-8.
"""

import pytest
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import MedicalOntologyPipeline


def make_pipeline(**kw) -> MedicalOntologyPipeline:
    cfg = PipelineConfig(
        extract_labs=True, extract_drugs=True,
        extract_diseases=True, extract_symptoms=True,
        detect_assertions=True, link_icd=True, link_rxnorm=True,
        validate_output=True, deterministic=True, **kw
    )
    return MedicalOntologyPipeline(cfg)


class TestUnicodeOffsets:
    """Verify that positions are Unicode character offsets, not byte offsets."""

    def test_vietnamese_positions_match_character_boundaries(self):
        """Vietnamese text entity positions match character boundaries."""
        pipeline = make_pipeline()
        text = "BN bị đau đầu dữ dội."

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            # Slicing by character position must match entity.text exactly
            extracted = text[start:end]
            assert extracted == entity.text, (
                f"Mismatch at [{start}:{end}]: "
                f"extracted={extracted!r} entity.text={entity.text!r}"
            )

    def test_ascii_equivalent(self):
        """ASCII text serves as baseline for offset correctness."""
        pipeline = make_pipeline()
        text = "Patient has fever and cough."

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            extracted = text[start:end]
            assert extracted == entity.text

    def test_mixed_vietnamese_ascii(self):
        """Mixed Vietnamese/ASCII text offsets are correct."""
        pipeline = make_pipeline()
        text = "BN tăng huyết áp (hypertension)"

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            extracted = text[start:end]
            assert extracted == entity.text

    def test_positions_within_text_bounds(self):
        """All entity positions are within original text bounds."""
        pipeline = make_pipeline()
        text = "BN bị sốt 39 độ, ho khan."

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            assert 0 <= start < len(text), f"start={start} out of bounds for len={len(text)}"
            assert start < end <= len(text), f"end={end} out of bounds"
            assert end > start

    def test_duplicate_positions_not_generated(self):
        """No two entities share the same [start, end) position."""
        pipeline = make_pipeline()
        text = "BN ho, sốt, đau bụng, chóng mặt."

        result = pipeline.process(text)

        positions = [tuple(e.position) for e in result.entities]
        assert len(positions) == len(set(positions)), \
            f"Duplicate positions found: {[p for p in positions if positions.count(p) > 1]}"

    def test_sorted_by_start(self):
        """Output is sorted by start position, not arbitrary order."""
        pipeline = make_pipeline()
        text = "BN bị chóng mặt, đau đầu, buồn nôn."

        result = pipeline.process(text)

        starts = [e.position[0] for e in result.entities]
        assert starts == sorted(starts), \
            f"Entities not sorted by start: {list(zip([e.text for e in result.entities], starts))}"

    def test_entity_text_immutable(self):
        """entity.text is always sliced from original, never re-computed."""
        pipeline = make_pipeline()
        text = "BN viêm phổi cộng đồng."

        result = pipeline.process(text)

        for entity in result.entities:
            # Re-slice from original
            re_sliced = text[entity.position[0]:entity.position[1]]
            assert re_sliced == entity.text, (
                f"Entity text != original slice: "
                f"text[{entity.position[0]}:{entity.position[1]}]={re_sliced!r} "
                f"!= entity.text={entity.text!r}"
            )

    def test_multibyte_utf8_positions(self):
        """Positions work correctly with multi-byte UTF-8 characters."""
        # Characters: ă (2 bytes), ệ (2 bytes), ường (3 bytes per char)
        pipeline = make_pipeline()
        text = "BN bị viêm phổi nặng."

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            # These should be character positions, not byte positions
            assert 0 <= start < len(text)
            assert start < end <= len(text)
            extracted = text[start:end]
            assert extracted == entity.text

    def test_zero_width_entities_not_generated(self):
        """Entities with zero width (start==end) are not generated."""
        pipeline = make_pipeline()
        text = "BN bình thường."

        result = pipeline.process(text)

        for entity in result.entities:
            start, end = entity.position
            assert end > start, f"Zero-width entity at [{start}, {end})"

    def test_overlapping_entities_resolved(self):
        """Overlapping entities are resolved, not both emitted."""
        pipeline = make_pipeline(resolve_overlaps=True)
        text = "viêm phổi nặng"

        result = pipeline.process(text)

        # Overlapping spans should be resolved to one
        positions = [tuple(e.position) for e in result.entities]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i == j:
                    continue
                s1, e1 = pos1
                s2, e2 = pos2
                # After resolution, non-identical spans should not overlap
                if pos1 != pos2:
                    assert not (s1 < e2 and s2 < e1), \
                        f"Overlapping entities found: {pos1} and {pos2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
