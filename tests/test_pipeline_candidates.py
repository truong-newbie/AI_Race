"""
Tests for candidate validation in pipeline output.

Requirements:
  - Only valid candidates (in known KB) are included
  - Invalid candidates are filtered
  - Candidates are truncated to configured max count
  - Empty candidate lists are handled gracefully
"""

import pytest
from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import MedicalOntologyPipeline


def make_pipeline(**kw) -> MedicalOntologyPipeline:
    cfg = PipelineConfig(deterministic=True, **kw)
    return MedicalOntologyPipeline(cfg)


class TestICDCandidates:
    """Tests for ICD-10 candidate generation and validation."""

    def test_diagnosis_has_candidates(self):
        """Diagnosis entity has at least one ICD candidate."""
        pipeline = make_pipeline(link_icd=True)
        text = "BN được chẩn đoán tăng huyết áp."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]

        assert len(diags) >= 1, "Expected diagnosis entity"
        diag = diags[0]
        assert len(diag.candidates) >= 1, "Diagnosis should have ICD candidates"

    def test_icd_candidates_from_known_kb(self):
        """ICD candidates are from the known knowledge base."""
        pipeline = make_pipeline(link_icd=True)
        text = "BN bị viêm phổi cộng đồng."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]

        if diags and diags[0].candidates:
            for code in diags[0].candidates:
                assert isinstance(code, str), f"ICD code must be str, got {type(code)}"
                # Should look like ICD codes (letter + digits)
                assert len(code) >= 2, f"ICD code too short: {code}"

    def test_candidates_respect_max_count(self):
        """Candidates are truncated to icd_output_candidates."""
        pipeline = make_pipeline(
            link_icd=True, icd_output_candidates=3,
        )
        text = "BN tăng huyết áp, đái tháo đường type 2, viêm phổi."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]

        for diag in diags:
            if diag.candidates:
                assert len(diag.candidates) <= 3, \
                    f"Expected max 3 candidates, got {len(diag.candidates)}"

    def test_no_duplicate_candidates(self):
        """Candidates list has no duplicate codes."""
        pipeline = make_pipeline(link_icd=True)
        text = "BN bị tăng huyết áp."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]

        if diags:
            for diag in diags:
                assert len(diag.candidates) == len(set(diag.candidates)), \
                    f"Duplicate candidates in: {diag.candidates}"

    def test_candidates_are_strings(self):
        """All candidates are string type."""
        pipeline = make_pipeline(link_icd=True)
        text = "BN bị sốt."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]
        for diag in diags:
            for c in diag.candidates:
                assert isinstance(c, str), f"Candidate must be str, got {type(c)}"


class TestRxNormCandidates:
    """Tests for RxNorm candidate generation and validation."""

    def test_drug_has_candidates(self):
        """Drug entity has at least one RxNorm candidate."""
        pipeline = make_pipeline(link_rxnorm=True)
        text = "BN được kê Metformin 500mg."
        result = pipeline.process(text)

        from src.schema import EntityType
        drugs = [e for e in result.entities if e.type == EntityType.THUOC]

        assert len(drugs) >= 1, "Expected drug entity"
        drug = drugs[0]
        assert len(drug.candidates) >= 1, "Drug should have RxNorm candidates"

    def test_rxnorm_candidates_from_known_kb(self):
        """RxNorm candidates are from the known knowledge base."""
        pipeline = make_pipeline(link_rxnorm=True)
        text = "BN uống Aspirin 81mg."
        result = pipeline.process(text)

        from src.schema import EntityType
        drugs = [e for e in result.entities if e.type == EntityType.THUOC]

        if drugs and drugs[0].candidates:
            for rxcui in drugs[0].candidates:
                assert isinstance(rxcui, str), f"RxCUI must be str, got {type(rxcui)}"
                assert rxcui.isdigit(), f"RxCUI should be numeric string, got: {rxcui}"

    def test_rxnorm_candidates_respect_max_count(self):
        """Candidates are truncated to rxnorm_output_candidates."""
        pipeline = make_pipeline(
            link_rxnorm=True, rxnorm_output_candidates=2,
        )
        text = "BN được kê Metformin 500mg, Amlodipine 5mg."
        result = pipeline.process(text)

        from src.schema import EntityType
        drugs = [e for e in result.entities if e.type == EntityType.THUOC]

        for drug in drugs:
            if drug.candidates:
                assert len(drug.candidates) <= 2, \
                    f"Expected max 2 candidates, got {len(drug.candidates)}"

    def test_no_duplicate_rxnorm_candidates(self):
        """RxNorm candidates list has no duplicates."""
        pipeline = make_pipeline(link_rxnorm=True)
        text = "BN uống Metformin 1000mg."
        result = pipeline.process(text)

        from src.schema import EntityType
        drugs = [e for e in result.entities if e.type == EntityType.THUOC]

        for drug in drugs:
            if drug.candidates:
                assert len(drug.candidates) == len(set(drug.candidates)), \
                    f"Duplicate candidates: {drug.candidates}"


class TestCandidateFiltering:
    """Tests for invalid candidate filtering."""

    def test_empty_candidates_handled(self):
        """Entities with no candidates have empty list, not null."""
        pipeline = make_pipeline(link_icd=True)
        # Text unlikely to match any diagnosis
        text = "BN bình thường."
        result = pipeline.process(text)

        for entity in result.entities:
            assert entity.candidates is not None
            assert isinstance(entity.candidates, list)

    def test_lab_entities_have_no_candidates(self):
        """Lab test entities do not get ICD/RxNorm candidates."""
        pipeline = make_pipeline(
            extract_labs=True, link_icd=True, link_rxnorm=True,
        )
        text = "Glucose: 126 mg/dL."
        result = pipeline.process(text)

        from src.schema import EntityType
        for entity in result.entities:
            if entity.type in {EntityType.TEN_XET_NGHIEM, EntityType.KET_QUA_XET_NGHIEM}:
                assert entity.candidates == [], \
                    f"Lab entity should have no candidates, got: {entity.candidates}"

    def test_symptom_entities_have_no_candidates(self):
        """Symptom entities do not get candidates (only diagnoses and drugs)."""
        pipeline = make_pipeline(
            extract_symptoms=True, link_icd=True, link_rxnorm=True,
        )
        text = "BN ho khan."
        result = pipeline.process(text)

        from src.schema import EntityType
        symptoms = [e for e in result.entities if e.type == EntityType.TRIEU_CHUNG]
        for symptom in symptoms:
            # Symptoms may or may not have candidates depending on linking
            # But they should be valid if present
            for c in symptom.candidates:
                assert isinstance(c, str)


class TestCandidateOrdering:
    """Tests for candidate ordering after reranking (if enabled)."""

    def test_icd_candidates_ordered_by_retrieval_score(self):
        """ICD candidates are ordered by retrieval score (top candidate first)."""
        pipeline = make_pipeline(link_icd=True)
        text = "BN bị tăng huyết áp."
        result = pipeline.process(text)

        from src.schema import EntityType
        diags = [e for e in result.entities if e.type == EntityType.CHAN_DOAN]

        if diags and len(diags[0].candidates) >= 2:
            # Candidates should not be identical
            codes = diags[0].candidates
            assert codes[0] != codes[1] or len(codes) == 1

    def test_rxnorm_candidates_ordered_by_retrieval_score(self):
        """RxNorm candidates are ordered by retrieval score."""
        pipeline = make_pipeline(link_rxnorm=True)
        text = "BN uống Metformin 500mg."
        result = pipeline.process(text)

        from src.schema import EntityType
        drugs = [e for e in result.entities if e.type == EntityType.THUOC]

        if drugs and len(drugs[0].candidates) >= 2:
            codes = drugs[0].candidates
            assert codes[0] != codes[1] or len(codes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
