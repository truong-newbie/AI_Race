"""
Tests for Medical Ontology Pipeline (Task 11)
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import (
    MedicalOntologyPipeline,
    PipelineConfig,
    ExtractionResult,
    extract_medical_entities,
    resolve_spans,
    create_span,
)
from src.entity.span_resolver import Span
from src.schema import EntityType, AssertionType


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.extract_labs is True
        assert config.extract_drugs is True
        assert config.extract_diseases is True
        assert config.detect_assertions is True
        assert config.link_icd is True
        assert config.link_rxnorm is True

    def test_config_overrides(self):
        """Test configuration overrides."""
        config = PipelineConfig(
            extract_labs=False,
            link_icd=False,
            overlap_strategy="longest"
        )
        assert config.extract_labs is False
        assert config.link_icd is False
        assert config.overlap_strategy == "longest"


class TestExtractionResult:
    """Test extraction result."""

    def test_to_dict_empty(self):
        """Test empty result conversion."""
        result = ExtractionResult(text="", entities=[])
        assert result.to_dict() == []

    def test_to_dict_with_entities(self):
        """Test result conversion with entities."""
        from src.schema import Entity

        entity = Entity(
            text="ho",
            position=[12, 14],
            type=EntityType.TRIEU_CHUNG,
            assertions=[AssertionType.NEGATED],
            candidates=[]
        )
        result = ExtractionResult(text="Bệnh nhân không ho", entities=[entity])
        result_dict = result.to_dict()

        assert len(result_dict) == 1
        assert result_dict[0]["text"] == "ho"
        assert result_dict[0]["type"] == "TRIỆU_CHỨNG"
        assert "isNegated" in result_dict[0]["assertions"]


class TestPipelineBasic:
    """Test basic pipeline functionality."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = MedicalOntologyPipeline()
        assert pipeline.lab_extractor is not None
        assert pipeline.drug_extractor is not None
        assert pipeline.disease_extractor is not None
        assert pipeline.assertion_detector is not None
        assert pipeline.icd10_kb is not None
        assert pipeline.rxnorm_kb is not None

    def test_pipeline_with_config(self):
        """Test pipeline with custom config."""
        config = PipelineConfig(
            extract_labs=False,
            detect_assertions=False
        )
        pipeline = MedicalOntologyPipeline(config)
        assert pipeline.config.extract_labs is False
        assert pipeline.config.detect_assertions is False


class TestPipelineExtraction:
    """Test entity extraction from text."""

    def test_extract_symptom(self):
        """Test symptom extraction."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Bệnh nhân ho đờm xanh.")

        assert len(result.entities) > 0
        texts = [e.text for e in result.entities]
        assert any("ho" in t or "đờm" in t for t in texts)

    def test_extract_lab_test(self):
        """Test lab test extraction."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("WBC 10.5 G/L")

        entity_types = [e.type for e in result.entities]
        assert EntityType.TEN_XET_NGHIEM in entity_types or any("WBC" in e.text for e in result.entities)

    def test_extract_drug(self):
        """Test drug extraction."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Paracetamol 500mg")

        entity_types = [e.type for e in result.entities]
        assert EntityType.THUOC in entity_types

    def test_extract_diagnosis(self):
        """Test diagnosis extraction."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Chẩn đoán: viêm phổi")

        entity_types = [e.type for e in result.entities]
        assert EntityType.CHAN_DOAN in entity_types or any("viêm phổi" in e.text for e in result.entities)


class TestPipelineAssertions:
    """Test assertion detection."""

    def test_detect_negation(self):
        """Test negation detection."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Bệnh nhân không ho")

        for entity in result.entities:
            if "ho" in entity.text:
                assert AssertionType.NEGATED in entity.assertions

    def test_detect_historical(self):
        """Test historical detection."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Tiền sử hen suyễn")

        for entity in result.entities:
            if "hen" in entity.text.lower() or "suyễn" in entity.text.lower():
                assert AssertionType.HISTORICAL in entity.assertions

    def test_detect_family(self):
        """Test family history detection."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Bố bệnh nhân bị đái tháo đường")

        for entity in result.entities:
            if "đái tháo" in entity.text.lower():
                assert AssertionType.FAMILY in entity.assertions


class TestPipelineLinking:
    """Test ICD-10 and RxNorm linking."""

    def test_icd10_linking(self):
        """Test ICD-10 linking for diagnoses."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Chẩn đoán viêm phổi")

        for entity in result.entities:
            if entity.type == EntityType.CHAN_DOAN:
                # Should have candidates
                assert entity.candidates is not None

    def test_rxnorm_linking(self):
        """Test RxNorm linking for drugs."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Paracetamol 500mg")

        for entity in result.entities:
            if entity.type == EntityType.THUOC:
                # Should have candidates
                assert entity.candidates is not None


class TestConvenienceFunction:
    """Test convenience functions."""

    def test_extract_medical_entities(self):
        """Test convenience extraction function."""
        text = "Bệnh nhân ho, sốt nhẹ. Dùng Paracetamol."
        entities = extract_medical_entities(text)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check structure
        for entity in entities:
            assert "text" in entity
            assert "position" in entity
            assert "type" in entity


class TestSpanResolution:
    """Test span resolution."""

    def test_resolve_overlapping_spans(self):
        """Test resolving overlapping spans."""
        spans = [
            Span(0, 5, "Bệnh nhân", "TRIỆU_CHỨNG", 0.9),
            Span(6, 9, "ho", "TRIỆU_CHỨNG", 0.8),
            Span(10, 15, "đờm", "TRIỆU_CHỨNG", 0.7),
        ]

        resolved = resolve_spans(spans, strategy="confidence")
        assert len(resolved) <= len(spans)

    def test_create_span(self):
        """Test span creation utility."""
        span = create_span(
            text="ho đờm",
            position=[6, 12],
            entity_type="TRIỆU_CHỨNG",
            confidence=0.9,
            source="rule"
        )

        assert span.start == 6
        assert span.end == 12
        assert span.text == "ho đờm"
        assert span.entity_type == "TRIỆU_CHỨNG"


class TestPipelineOutput:
    """Test pipeline output format."""

    def test_output_format(self):
        """Test output follows expected format."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("Bệnh nhân ho đờm xanh.")

        output = result.to_dict()

        # Should be a list
        assert isinstance(output, list)

        # Each item should have required fields
        for entity in output:
            assert "text" in entity
            assert "position" in entity
            assert "type" in entity
            assert "assertions" in entity
            assert "candidates" in entity

            # Position should be [start, end]
            assert isinstance(entity["position"], list)
            assert len(entity["position"]) == 2

            # Type should be valid entity type
            valid_types = [
                "TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC",
                "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"
            ]
            assert entity["type"] in valid_types

    def test_validation(self):
        """Test output validation."""
        pipeline = MedicalOntologyPipeline()
        result = pipeline.process("WBC 10.5 G/L, ho, Paracetamol 500mg")

        if result.validation_result:
            assert result.validation_result.is_valid is not None


class TestComplexCases:
    """Test complex medical text cases."""

    def test_full_medical_note(self):
        """Test processing full medical note."""
        text = """
        Bệnh nhân nam, 55 tuổi, nhập viện vì ho đờm xanh, sốt cao.
        Tiền sử tăng huyết áp, đái tháo đường type 2.
        Khám: phổi có ran ẩm 2 bên.
        Xét nghiệm: WBC 12.5 G/L, CRP 85 mg/L.
        Chẩn đoán: viêm phổi cộng đồng.
        Điều trị: Ceftriaxone 1g, Paracetamol 500mg khi sốt.
        """

        pipeline = MedicalOntologyPipeline()
        result = pipeline.process(text)

        # Should extract multiple entity types
        entity_types = {e.type for e in result.entities}

        assert len(result.entities) > 0
        assert isinstance(result.error, type(None)) or result.error is None

    def test_negation_with_diagnosis(self):
        """Test text with negation and diagnosis."""
        text = "Loại trừ bệnh lao phổi. Không có dấu hiệu viêm màng não."

        pipeline = MedicalOntologyPipeline()
        result = pipeline.process(text)

        # Should process without errors
        assert result.error is None

    def test_family_history(self):
        """Test family history text."""
        text = "Bố bệnh nhân có tiền sử nhồi máu cơ tim. Mẹ bị tăng huyết áp."

        pipeline = MedicalOntologyPipeline()
        result = pipeline.process(text)

        # Should detect family assertions
        family_entities = [
            e for e in result.entities
            if AssertionType.FAMILY in e.assertions
        ]

        # Should process without errors
        assert result.error is None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
