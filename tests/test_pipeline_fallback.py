"""
Tests for model unavailable fallback behavior.

Requirement: when a model/component fails to load, the pipeline must:
  - Log a clear error message
  - Continue with partial results (graceful degradation)
  - Not raise exceptions to the caller
"""

import pytest
import importlib
from unittest.mock import patch, MagicMock
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import MedicalOntologyPipeline, ExtractionResult, ModuleError


def make_pipeline(**overrides) -> MedicalOntologyPipeline:
    cfg = PipelineConfig(
        deterministic=True,
        **overrides,
    )
    return MedicalOntologyPipeline(cfg)


class TestModelUnavailableFallback:
    """Test graceful degradation when models fail to load."""

    def test_pipeline_runs_when_lab_extractor_fails(self):
        """Pipeline continues when LabTestExtractor fails."""
        pipeline = make_pipeline(extract_labs=True)

        with patch(
            "src.entity.lab_extractor.LabTestExtractor.__init__",
            side_effect=RuntimeError("Model file not found"),
        ):
            text = "BN ho và sốt."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            assert isinstance(result.errors, list)
            # Should still extract other entities (drugs/diseases)
            assert isinstance(result.entities, list)

    def test_pipeline_runs_when_drug_extractor_fails(self):
        """Pipeline continues when DrugExtractor fails."""
        pipeline = make_pipeline(extract_drugs=True)

        with patch(
            "src.entity.drug_extractor.DrugExtractor.__init__",
            side_effect=RuntimeError("Drug dictionary not found"),
        ):
            text = "BN bị tăng huyết áp."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            # Should still extract diagnosis
            assert isinstance(result.entities, list)

    def test_pipeline_runs_when_disease_extractor_fails(self):
        """Pipeline continues when DiseaseExtractor fails."""
        pipeline = make_pipeline(extract_diseases=True)

        with patch(
            "src.entity.disease_extractor.DiseaseExtractor.__init__",
            side_effect=RuntimeError("Disease dictionary not found"),
        ):
            text = "BN được kê Metformin."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            assert isinstance(result.entities, list)

    def test_pipeline_runs_when_assertion_detector_fails(self):
        """Pipeline continues when AssertionDetector fails."""
        pipeline = make_pipeline(detect_assertions=True)

        with patch(
            "src.assertion.rules.AssertionDetector.__init__",
            side_effect=RuntimeError("Cue patterns not loaded"),
        ):
            text = "BN ho."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            # Should still extract entities
            assert isinstance(result.entities, list)

    def test_pipeline_runs_when_icd_retriever_fails(self):
        """Pipeline continues when ICD retriever fails."""
        pipeline = make_pipeline(link_icd=True)

        with patch(
            "src.linking.icd.hybrid_retriever.HybridRetriever.__init__",
            side_effect=RuntimeError("ICD KB not found"),
        ):
            text = "BN bị tăng huyết áp."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            # Should still extract entity, just without candidates
            assert isinstance(result.entities, list)
            # Candidates may be empty but should not crash
            for entity in result.entities:
                assert isinstance(entity.candidates, list)

    def test_pipeline_runs_when_rxnorm_retriever_fails(self):
        """Pipeline continues when RxNorm retriever fails."""
        pipeline = make_pipeline(link_rxnorm=True)

        with patch(
            "src.linking.rxnorm.hybrid_retriever.DrugHybridRetriever.__init__",
            side_effect=RuntimeError("RxNorm KB not found"),
        ):
            text = "BN được kê Amlodipine 5mg."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            assert isinstance(result.entities, list)

    def test_pipeline_runs_when_span_resolver_fails(self):
        """Pipeline continues when SpanResolver fails."""
        pipeline = make_pipeline(resolve_overlaps=True)

        with patch(
            "src.entity.span_resolver.SpanResolver.resolve",
            side_effect=RuntimeError("Span resolver error"),
        ):
            text = "BN ho và sốt."
            result = pipeline.process(text)

            assert isinstance(result, ExtractionResult)
            assert isinstance(result.entities, list)


class TestModuleErrors:
    """Test that module failures produce clear error messages."""

    def test_error_message_contains_stage_name(self):
        """Module error messages contain the stage name."""
        pipeline = make_pipeline(extract_labs=True)

        with patch(
            "src.entity.lab_extractor.LabTestExtractor.__init__",
            side_effect=RuntimeError("Lab extractor broken"),
        ):
            result = pipeline.process("BN ho.")

            stage_errors = [e for e in result.errors if e.stage == "lab_extraction"]
            assert len(stage_errors) >= 1
            assert "lab_extraction" in stage_errors[0].stage.lower()
            assert "Lab extractor broken" in stage_errors[0].message

    def test_error_message_contains_exception_details(self):
        """Error messages contain the exception details."""
        pipeline = make_pipeline(link_icd=True)

        with patch(
            "src.linking.icd.hybrid_retriever.HybridRetriever.__init__",
            side_effect=FileNotFoundError("/path/to/kb.json not found"),
        ):
            result = pipeline.process("BN bị bệnh.")

            icd_errors = [e for e in result.errors if "icd" in e.stage.lower()]
            assert len(icd_errors) >= 1
            # Error message should contain details
            assert len(icd_errors[0].message) > 0

    def test_multiple_errors_collected(self):
        """Multiple module failures are all collected."""
        pipeline = make_pipeline(
            extract_labs=True, extract_drugs=True, link_icd=True,
        )

        with patch(
            "src.entity.lab_extractor.LabTestExtractor.__init__",
            side_effect=RuntimeError("Lab fail"),
        ), patch(
            "src.entity.drug_extractor.DrugExtractor.__init__",
            side_effect=RuntimeError("Drug fail"),
        ):
            result = pipeline.process("BN ho.")

            assert len(result.errors) >= 2
            stage_names = {e.stage for e in result.errors}
            assert "lab_extraction" in stage_names
            assert "drug_extraction" in stage_names


class TestGracefulDegradation:
    """Test that partial results are still usable after component failures."""

    def test_extraction_result_still_valid_after_failures(self):
        """Result is valid even when some modules fail."""
        pipeline = make_pipeline(
            extract_labs=True, extract_diseases=True,
            link_icd=True, link_rxnorm=True,
        )

        with patch(
            "src.entity.lab_extractor.LabTestExtractor.__init__",
            side_effect=RuntimeError("Lab model missing"),
        ), patch(
            "src.linking.icd.hybrid_retriever.HybridRetriever.__init__",
            side_effect=RuntimeError("ICD KB missing"),
        ):
            result = pipeline.process("BN được kê Metformin.")

            # Result structure should still be valid
            assert hasattr(result, "text")
            assert hasattr(result, "entities")
            assert hasattr(result, "errors")
            assert result.text == "BN được kê Metformin."

    def test_to_dict_works_after_failures(self):
        """to_dict() works even when some modules fail."""
        pipeline = make_pipeline(link_icd=True)

        with patch(
            "src.linking.icd.hybrid_retriever.HybridRetriever.__init__",
            side_effect=RuntimeError("ICD KB missing"),
        ):
            result = pipeline.process("BN bị bệnh.")

            # Should not raise
            output = result.to_dict()
            assert isinstance(output, list)

    def test_empty_text_produces_empty_result(self):
        """Empty input produces empty result, not an error."""
        pipeline = make_pipeline()
        result = pipeline.process("")

        assert result.text == ""
        assert result.entities == []
        assert result.errors == []

    def test_very_long_text_still_processes(self):
        """Very long text is processed without hanging or crashing."""
        pipeline = make_pipeline()
        text = "BN bị ho. " * 500  # Long text
        result = pipeline.process(text)

        assert isinstance(result, ExtractionResult)
        assert isinstance(result.entities, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
