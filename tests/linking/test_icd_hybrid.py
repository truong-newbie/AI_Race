"""Tests for Hybrid Retriever (combined retrieval)."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.icd.hybrid_retriever import HybridRetriever, MergeConfig, CandidateResult


class TestHybridRetrieverBuild:
    """Hybrid retriever initialization and build."""

    def test_build_from_entries(self):
        """Retriever builds successfully from KB entries."""
        entries = get_knowledge_base()
        retriever = HybridRetriever(entries=entries, top_k=10)
        assert retriever._built is True

    def test_default_config(self):
        """Default merge config is RRF."""
        entries = get_knowledge_base()
        retriever = HybridRetriever(entries=entries)
        assert retriever.merge_config.method == "rrf"
        assert retriever.merge_config.rrf_k == 60

    def test_weighted_config(self):
        """Can switch to weighted merge."""
        entries = get_knowledge_base()
        cfg = MergeConfig(method="weighted")
        retriever = HybridRetriever(entries=entries, merge_config=cfg)
        assert retriever.merge_config.method == "weighted"


class TestHybridRetrieval:
    """Hybrid retrieval integration tests."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        cfg = MergeConfig(method="rrf", rrf_k=60)
        return HybridRetriever(entries=entries, merge_config=cfg, top_k=20)

    def test_retrieve_returns_candidates(self, retriever):
        """retrieve() returns list of CandidateResult."""
        results = retriever.retrieve("Bệnh nhân bị suy tim", mention="suy tim")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, CandidateResult) for r in results)

    def test_retrieve_codes(self, retriever):
        """retrieve_codes() returns code list."""
        codes = retriever.retrieve_codes(
            "Bệnh nhân bị suy tim", mention="suy tim"
        )
        assert isinstance(codes, list)
        assert all(isinstance(c, str) for c in codes)

    def test_gold_code_in_top_k(self, retriever):
        """Gold code is in top-k for each sample."""
        samples = [
            ("suy tim", "I50.9"),
            ("viêm phổi", "J18.9"),
            ("đái tháo đường", "E11.9"),
            ("tăng huyết áp", "I10"),
            ("động kinh", "G40.909"),
            ("viêm phế quản cấp", "J20.9"),
            ("hen", "J45.9"),
            ("nhiễm trùng tiết niệu", "N39.0"),
            ("sỏi thận", "N20.0"),
            ("đau nửa đầu", "G43.909"),
        ]
        for mention, gold_code in samples:
            results = retriever.retrieve("bệnh nhân " + mention, mention=mention)
            codes = [r.code for r in results]
            assert gold_code in codes, f"{gold_code} not found for '{mention}' — got {codes[:5]}"

    def test_top_k_sorted(self, retriever):
        """Results are sorted by score descending."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_sources_field_populated(self, retriever):
        """CandidateResult includes sources list."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        for r in results:
            assert isinstance(r.sources, list)

    def test_to_dict(self, retriever):
        """CandidateResult.to_dict() works."""
        results = retriever.retrieve("suy tim", mention="suy tim")
        d = results[0].to_dict()
        assert "code" in d
        assert "score" in d
        assert "sources" in d

    def test_top_k_override(self, retriever):
        """top_k parameter overrides default."""
        results = retriever.retrieve("suy tim", mention="suy tim", top_k=5)
        assert len(results) == 5

    def test_empty_mention_uses_query(self, retriever):
        """Empty mention falls back to query."""
        results = retriever.retrieve("suy tim")
        assert len(results) > 0

    def test_retrieve_unbuilt_raises(self):
        """Calling retrieve before build raises."""
        retriever = HybridRetriever()
        with pytest.raises(RuntimeError):
            retriever.retrieve("suy tim")


class TestHybridEdgeCases:
    """Edge cases for hybrid retrieval."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        return HybridRetriever(entries=entries, top_k=20)

    def test_gerd(self, retriever):
        """GERD abbreviation retrieves K21.9."""
        results = retriever.retrieve("bệnh trào ngược dạ dày", mention="trào ngược")
        codes = [r.code for r in results]
        assert "K21.9" in codes

    def test_urinary_tract_infection(self, retriever):
        """Nhiễm trùng tiết niệu retrieves N39.0."""
        results = retriever.retrieve("bệnh nhân nhiễm trùng tiết niệu", mention="nhiễm trùng tiết niệu")
        codes = [r.code for r in results]
        assert "N39.0" in codes

    def test_copd(self, retriever):
        """COPD retrieves J44.9."""
        results = retriever.retrieve("bệnh COPD", mention="COPD")
        codes = [r.code for r in results]
        assert "J44.9" in codes

    def test_acute_bronchitis(self, retriever):
        """viêm phế quản cấp retrieves J20.9."""
        results = retriever.retrieve("viêm phế quản cấp", mention="viêm phế quản cấp")
        codes = [r.code for r in results]
        assert "J20.9" in codes


class TestMergeConfig:
    """Merge configuration tests."""

    def test_rrf_default_weights(self):
        cfg = MergeConfig(method="rrf")
        assert cfg.get_weight("exact") == 1.0
        assert cfg.get_weight("bm25") == 0.7
        assert cfg.get_weight("dense") == 0.75

    def test_custom_weights(self):
        cfg = MergeConfig(
            method="weighted",
            weights={"exact": 1.0, "fuzzy": 0.9, "bm25": 0.5},
        )
        assert cfg.get_weight("exact") == 1.0
        assert cfg.get_weight("fuzzy") == 0.9
        assert cfg.get_weight("unknown") == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
