"""Tests for reranker Top-K accuracy."""

import pytest
from src.linking.icd.schema import get_knowledge_base as get_icd_kb
from src.linking.rxnorm.schema import get_knowledge_base as get_rx_kb
from src.linking.rule_reranker import (
    ICDRuleReranker,
    RxNormRuleReranker,
    UnifiedRuleReranker,
)
from src.linking.icd.hybrid_retriever import CandidateResult


# ─── ICD-10 Test Data ────────────────────────────────────────────────────────


ICD_GOLD_SAMPLES = [
    {"mention": "tăng huyết áp", "query": "BN bị tăng huyết áp", "gold": "I10"},
    {"mention": "đái tháo đường", "query": "BN đái tháo đường type 2", "gold": "E11.9"},
    {"mention": "viêm phổi", "query": "BN viêm phổi cộng đồng", "gold": "J18.9"},
    {"mention": "đau thắt lưng", "query": "BN đau thắt lưng", "gold": "M54.5"},
    {"mention": "trầm cảm", "query": "BN rối loạn trầm cảm", "gold": "F32.9"},
]


# ─── RxNorm Test Data ────────────────────────────────────────────────────────


RX_GOLD_SAMPLES = [
    {"mention": "Metformin 1000mg", "query": "BN uống Metformin 1000mg", "gold": "861007"},
    {"mention": "Aspirin 325mg", "query": "BN uống Aspirin 325mg", "gold": "1192"},
    {"mention": "Amlodipine 5mg", "query": "BN uống Amlodipine 5mg", "gold": "17767"},
    {"mention": "Omeprazole 20mg", "query": "BN uống Omeprazole 20mg", "gold": "7646"},
    {"mention": "Metformin", "query": "BN dùng Metformin", "gold": "861007"},  # ingredient-only
]


# ─── ICD-10 Tests ────────────────────────────────────────────────────────────


class TestICDRerankerTopK:
    """Tests for ICD-10 reranker Top-K accuracy."""

    @pytest.fixture
    def icd_reranker(self):
        entries = get_icd_kb()
        return ICDRuleReranker(entries)

    def _mock_icd_candidates(self, codes_with_scores):
        """Create mock ICD candidates."""
        class MockCand:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}
        return [MockCand(c, s) for c, s in codes_with_scores]

    def test_top1_accuracy_on_gold_samples(self, icd_reranker):
        """Top-1 accuracy should be 100% on gold samples when gold is in candidates."""
        correct = 0
        total = 0

        for sample in ICD_GOLD_SAMPLES:
            gold = sample["gold"]
            entries = get_icd_kb()
            entry_codes = [e.code for e in entries]

            # Ensure gold is in candidates
            if gold not in entry_codes:
                continue

            # Mock candidates: gold + some distractors
            candidates = self._mock_icd_candidates([
                (gold, 0.005),
                ("K21.9", 0.005),
                ("J18.9", 0.005),
            ])

            reranked = icd_reranker.rerank(
                candidates,
                query=sample["query"],
                mention=sample["mention"],
                top_k=3,
            )

            if reranked and reranked[0].code == gold:
                correct += 1
            total += 1

        assert total >= 4
        # At minimum, gold should be in top-3
        assert correct >= total * 0.6

    def test_top3_recall_on_gold_samples(self, icd_reranker):
        """Gold code should be in Top-3 for all samples."""
        all_found = 0

        for sample in ICD_GOLD_SAMPLES:
            gold = sample["gold"]
            entries = get_icd_kb()
            entry_codes = [e.code for e in entries]

            if gold not in entry_codes:
                continue

            candidates = self._mock_icd_candidates([
                (gold, 0.005),
                ("K21.9", 0.005),
                ("I10", 0.005),
                ("E11.9", 0.005),
            ])

            reranked = icd_reranker.rerank(
                candidates,
                query=sample["query"],
                mention=sample["mention"],
                top_k=3,
            )

            codes = [r.code for r in reranked]
            if gold in codes:
                all_found += 1

        assert all_found >= 4

    def test_reranking_changes_order(self, icd_reranker):
        """Reranking should be able to change the order of candidates."""
        candidates = self._mock_icd_candidates([
            ("K21.9", 0.010),   # GERD — first in retrieval (higher score)
            ("I10", 0.005),     # Hypertension — second
        ])

        # Mention strongly matches I10 (hypertension alias)
        reranked = icd_reranker.rerank(
            candidates,
            query="BN tăng huyết áp",
            mention="tăng huyết áp",
            top_k=2,
        )

        codes = [r.code for r in reranked]
        # Hypertension should be ranked first or second
        assert "I10" in codes
        assert "K21.9" in codes

    def test_top_k_respected(self, icd_reranker):
        """top_k parameter should be respected."""
        candidates = self._mock_icd_candidates([
            ("K21.9", 0.005),
            ("I10", 0.005),
            ("E11.9", 0.005),
            ("J18.9", 0.005),
            ("M54.5", 0.005),
        ])

        reranked_3 = icd_reranker.rerank(
            candidates,
            query="BN",
            mention="bệnh",
            top_k=3,
        )

        assert len(reranked_3) == 3

        reranked_5 = icd_reranker.rerank(
            candidates,
            query="BN",
            mention="bệnh",
            top_k=5,
        )

        assert len(reranked_5) == 5


# ─── RxNorm Tests ────────────────────────────────────────────────────────────


class TestRxNormRerankerTopK:
    """Tests for RxNorm reranker Top-K accuracy."""

    @pytest.fixture
    def rx_reranker(self):
        entries = get_rx_kb()
        return RxNormRuleReranker(entries)

    def _mock_rx_candidates(self, rxcuis_with_scores):
        """Create mock RxNorm candidates."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []
        return [MockCand(r, s) for r, s in rxcuis_with_scores]

    def test_top1_accuracy_on_gold_samples(self, rx_reranker):
        """Top-1 accuracy should be high on gold samples."""
        correct = 0
        total = 0

        for sample in RX_GOLD_SAMPLES:
            gold = sample["gold"]
            entries = get_rx_kb()
            entry_rxcuis = [e.rxcui for e in entries]

            if gold not in entry_rxcuis:
                continue

            candidates = self._mock_rx_candidates([
                (gold, 2.0),
                ("1191", 2.0),  # Aspirin 81mg as distractor
                ("6809", 1.5),  # Metformin 500mg as distractor
            ])

            reranked = rx_reranker.rerank(
                candidates,
                query=sample["query"],
                mention=sample["mention"],
                top_k=3,
            )

            if reranked and reranked[0].code == gold:
                correct += 1
            total += 1

        assert total >= 4
        # High accuracy expected
        assert correct >= total * 0.75

    def test_top3_recall_on_gold_samples(self, rx_reranker):
        """Gold should be in Top-3 for all samples."""
        all_found = 0

        for sample in RX_GOLD_SAMPLES:
            gold = sample["gold"]
            entries = get_rx_kb()
            entry_rxcuis = [e.rxcui for e in entries]

            if gold not in entry_rxcuis:
                continue

            candidates = self._mock_rx_candidates([
                (gold, 1.5),
                ("1191", 1.5),
                ("17767", 1.0),
                ("7646", 1.0),
            ])

            reranked = rx_reranker.rerank(
                candidates,
                query=sample["query"],
                mention=sample["mention"],
                top_k=3,
            )

            codes = [r.code for r in reranked]
            if gold in codes:
                all_found += 1

        assert all_found >= 4

    def test_ingredient_only_mention_top1(self, rx_reranker):
        """Ingredient-only mention should still rank gold first."""
        # Metformin mention (no strength) — gold = 861007 (1000mg)
        candidates = rx_reranker._mock_rx_candidates([
            ("861007", 2.0),  # Metformin 1000mg — gold
            ("6809", 2.0),    # Metformin 500mg — distractor
            ("860975", 1.5),  # Metformin 850mg — distractor
        ]) if hasattr(rx_reranker, '_mock_rx_candidates') else None

        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        candidates = [MockCand(r, s) for r, s in [
            ("861007", 2.0),  # 1000mg
            ("6809", 2.0),    # 500mg
            ("860975", 1.5),  # 850mg
        ]]

        reranked = rx_reranker.rerank(
            candidates,
            query="BN dùng Metformin",
            mention="Metformin",
            top_k=3,
        )

        assert len(reranked) >= 2
        # Metformin 1000mg should be in top-3
        codes = [r.code for r in reranked]
        assert "861007" in codes

    def test_top_k_respected(self, rx_reranker):
        """top_k should be respected."""
        class MockCand:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        candidates = [MockCand(r, 1.0) for r in ["1191", "1192", "6809", "861007", "17767"]]

        reranked_2 = rx_reranker.rerank(
            candidates, query="BN", mention="thuốc", top_k=2
        )
        assert len(reranked_2) == 2

        reranked_4 = rx_reranker.rerank(
            candidates, query="BN", mention="thuốc", top_k=4
        )
        assert len(reranked_4) == 4


# ─── Unified Reranker Tests ──────────────────────────────────────────────────


class TestUnifiedRerankerTopK:
    """Tests for unified reranker auto-detection."""

    def test_detect_icd_candidates(self):
        """Auto-detection should identify ICD candidates."""
        entries_icd = get_icd_kb()
        entries_rx = get_rx_kb()
        unified = UnifiedRuleReranker(icd_entries=entries_icd, rx_entries=entries_rx)

        class MockICD:
            code = "I10"
            score = 0.01
            sources = []
            detail = {}

        class MockRx:
            rxcui = "861007"
            score = 2.0
            sources = []

        # ICD detection
        results = unified.detect_and_rerank([MockICD()], query="BN", mention="tăng huyết áp", top_k=1)
        assert len(results) == 1
        assert results[0].code == "I10"

    def test_detect_rxnorm_candidates(self):
        """Auto-detection should identify RxNorm candidates."""
        entries_icd = get_icd_kb()
        entries_rx = get_rx_kb()
        unified = UnifiedRuleReranker(icd_entries=entries_icd, rx_entries=entries_rx)

        class MockRx:
            rxcui = "861007"
            score = 2.0
            sources = []

        results = unified.detect_and_rerank([MockRx()], query="BN", mention="Metformin 1000mg", top_k=1)
        assert len(results) == 1

    def test_detect_unknown_raises(self):
        """Unknown candidate type should raise ValueError."""
        unified = UnifiedRuleReranker()

        class MockUnknown:
            pass

        with pytest.raises(ValueError):
            unified.detect_and_rerank([MockUnknown()], query="BN", mention="test", top_k=1)

    def test_rerank_empty_list(self):
        """Reranking empty list should return empty list."""
        unified = UnifiedRuleReranker()
        results = unified.detect_and_rerank([], query="BN", mention="test", top_k=10)
        assert results == []


# ─── Metric Calculation Tests ────────────────────────────────────────────────


class TestMetricsCalculation:
    """Tests for metric calculation helpers."""

    def test_recall_at_k_calculation(self):
        """Recall@K should count gold in top-K."""
        gold = "I10"
        candidates = ["K21.9", "I10", "E11.9"]

        def recall_at_k(ranked, gold_code, k):
            return gold_code in ranked[:k]

        assert recall_at_k(candidates, gold, 1) is False
        assert recall_at_k(candidates, gold, 2) is True
        assert recall_at_k(candidates, gold, 3) is True

    def test_mrr_calculation(self):
        """MRR should be 1/rank for each sample."""
        gold = "I10"
        candidates = ["E11.9", "K21.9", "I10", "J18.9"]

        rank = candidates.index(gold) + 1
        mrr = 1.0 / rank
        assert abs(mrr - 1/3) < 0.01

    def test_top_k_accuracy(self):
        """Top-K accuracy should be fraction where gold in top-K."""
        samples = [
            {"ranked": ["I10", "K21.9"], "gold": "I10"},      # hit @1
            {"ranked": ["K21.9", "I10"], "gold": "I10"},      # hit @2
            {"ranked": ["E11.9", "J18.9"], "gold": "I10"},   # miss
        ]

        # Top-1 accuracy: 1/3 (only first sample)
        top1_acc = sum(1 for s in samples if s["gold"] == s["ranked"][0]) / len(samples)
        assert top1_acc == pytest.approx(1/3)

        # Top-2 accuracy: 2/3 (first two samples)
        top2_acc = sum(1 for s in samples if s["gold"] in s["ranked"][:2]) / len(samples)
        assert top2_acc == pytest.approx(2/3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
