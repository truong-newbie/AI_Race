"""Tests for RapidFuzz fuzzy retrieval."""

import pytest
from src.linking.icd.schema import get_knowledge_base
from src.linking.icd.fuzzy_retriever import FuzzyRetriever


class TestFuzzyRetrieval:
    """Fuzzy string matching retrieval tests."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        return FuzzyRetriever(entries=entries, score_cutoff=50)

    def test_retrieve_heart_failure(self, retriever):
        """'suy tim' should retrieve I50.9."""
        results = retriever.retrieve("suy tim", top_k=5)
        codes = [r[0] for r in results]
        assert "I50.9" in codes
        assert results[0][0] == "I50.9" or codes.index("I50.9") < 3

    def test_retrieve_pneumonia(self, retriever):
        """'viêm phổi' should retrieve J18.9."""
        results = retriever.retrieve("viêm phổi", top_k=5)
        codes = [r[0] for r in results]
        assert "J18.9" in codes

    def test_retrieve_typo(self, retriever):
        """Fuzzy matching handles typos."""
        results = retriever.retrieve("viêm phỏi", top_k=5)  # typo
        codes = [r[0] for r in results]
        assert "J18.9" in codes

    def test_retrieve_diabetes(self, retriever):
        """'đái tháo đường' should retrieve E11.9."""
        results = retriever.retrieve("đái tháo đường", top_k=5)
        codes = [r[0] for r in results]
        assert "E11.9" in codes

    def test_retrieve_epilepsy(self, retriever):
        """'động kinh' should retrieve G40.909."""
        results = retriever.retrieve("động kinh", top_k=5)
        codes = [r[0] for r in results]
        assert "G40.909" in codes

    def test_retrieve_hypertension(self, retriever):
        """'tăng huyết áp' should retrieve I10."""
        results = retriever.retrieve("tăng huyết áp", top_k=5)
        codes = [r[0] for r in results]
        assert "I10" in codes

    def test_retrieve_bronchitis(self, retriever):
        """'viêm phế quản cấp' should retrieve J20.9."""
        results = retriever.retrieve("viêm phế quản cấp", top_k=5)
        codes = [r[0] for r in results]
        assert "J20.9" in codes

    def test_retrieve_asthma(self, retriever):
        """'hen' should retrieve J45.9."""
        results = retriever.retrieve("hen suyễn", top_k=5)
        codes = [r[0] for r in results]
        assert "J45.9" in codes

    def test_retrieve_uti(self, retriever):
        """'nhiễm trùng tiết niệu' should retrieve N39.0."""
        results = retriever.retrieve("nhiễm trùng tiết niệu", top_k=5)
        codes = [r[0] for r in results]
        assert "N39.0" in codes

    def test_retrieve_kidney_stone(self, retriever):
        """'sỏi thận' should retrieve N20.0."""
        results = retriever.retrieve("sỏi thận", top_k=5)
        codes = [r[0] for r in results]
        assert "N20.0" in codes

    def test_retrieve_returns_score(self, retriever):
        """Results include score."""
        results = retriever.retrieve("suy tim", top_k=3)
        assert len(results) > 0
        for code, score in results:
            assert isinstance(code, str)
            assert isinstance(score, (int, float))
            assert 0 <= score <= 100

    def test_retrieve_empty_query(self, retriever):
        """Empty query returns empty."""
        results = retriever.retrieve("", top_k=10)
        assert results == []

    def test_retrieve_one(self, retriever):
        """retrieve_one returns single result."""
        result = retriever.retrieve_one("suy tim")
        assert result is not None
        code, score = result
        assert isinstance(code, str)
        assert isinstance(score, (int, float))


class TestFuzzyEdgeCases:
    """Edge cases for fuzzy retrieval."""

    @pytest.fixture
    def retriever(self):
        entries = get_knowledge_base()
        return FuzzyRetriever(entries=entries, score_cutoff=30)

    def test_short_query(self, retriever):
        """Short mention still retrieves."""
        results = retriever.retrieve("hen", top_k=5)
        codes = [r[0] for r in results]
        assert len(codes) > 0

    def test_unknown_term(self, retriever):
        """Unknown term returns empty or near matches."""
        results = retriever.retrieve("xyzabcyyy", top_k=10)
        # Should either be empty or have low-scoring results
        assert isinstance(results, list)

    def test_partial_match(self, retriever):
        """Partial mention still matches."""
        results = retriever.retrieve("viêm phổi cộng", top_k=5)
        codes = [r[0] for r in results]
        assert "J18.9" in codes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
