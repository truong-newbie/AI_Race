"""
Drug Hybrid Retriever

Ket hop structured matcher, fuzzy retriever, dense retriever
voi weighted scoring.
"""

from dataclasses import dataclass
from typing import Optional

from src.linking.rxnorm.schema import RxNormEntry, ParsedDrug
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.normalizer import DrugTextNormalizer
from src.linking.rxnorm.structured_matcher import StructuredMatcher
from src.linking.rxnorm.fuzzy_retriever import FuzzyDrugRetriever
from src.linking.rxnorm.dense_retriever import DenseDrugRetriever


@dataclass
class DrugCandidateResult:
    """Mot candidate tra ve tu retrieval."""
    rxcui: str
    score: float
    sources: list[str]
    ingredient_score: float = 0.0
    strength_score: float = 0.0
    match_details: str = ""


class DrugHybridRetriever:
    """
    Hybrid drug retriever combining multiple sources:

    1. Structured matcher (exact, ingredient+strength+form, ingredient-only)
    2. Fuzzy retriever (RapidFuzz WRatio)
    3. Dense retriever (sentence-transformer embeddings)

    Merge via weighted normalized score or RRF.
    """

    def __init__(
        self,
        entries: list[RxNormEntry],
        top_k: int = 20,
        use_dense: bool = True,
        use_fuzzy: bool = True,
        use_structured: bool = True,
    ):
        self.entries = entries
        self.top_k = top_k
        self.parser = DrugMentionParser()
        self.normalizer = DrugTextNormalizer()

        # Build retrievers
        self.structured = None
        self.fuzzy = None
        self.dense = None

        if use_structured:
            self.structured = StructuredMatcher(
                entries=entries,
                parser=self.parser,
                normalizer=self.normalizer,
            )

        if use_fuzzy:
            self.fuzzy = FuzzyDrugRetriever(
                entries=entries,
                score_cutoff=40,
                normalizer=self.normalizer,
            )

        if use_dense:
            try:
                self.dense = DenseDrugRetriever(
                    entries=entries,
                    model_name="intfloat/multilingual-e5-small",
                    cache_dir=".cache/rxnorm_dense",
                    normalizer=self.normalizer,
                )
            except Exception:
                self.dense = None

    def retrieve(
        self,
        query: str,
        mention: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[DrugCandidateResult]:
        """
        Retrieve top-k drug candidates.

        Args:
            query: Full query text
            mention: Extracted drug mention (if None, uses query)
            top_k: Override default top_k

        Returns:
            List of DrugCandidateResult sorted by score descending
        """
        if top_k is None:
            top_k = self.top_k

        # Use mention if provided, otherwise use query
        text = mention if mention else query
        if not text:
            return []

        # Parse drug mention
        parsed = self.parser.parse(text)
        if parsed is None:
            return []

        # Collect results from each source
        source_results: dict[str, list[tuple[str, float]]] = {}

        # 1. Structured matcher
        if self.structured:
            struct_results = self.structured.match(parsed, top_k=top_k * 2)
            source_results["structured"] = [
                (rxcui, ms.total_score) for rxcui, ms in struct_results
            ]

        # 2. Fuzzy retriever
        if self.fuzzy:
            fuzzy_results = self.fuzzy.retrieve(text, top_k=top_k * 2)
            source_results["fuzzy"] = fuzzy_results

        # 3. Dense retriever
        if self.dense:
            try:
                dense_results = self.dense.retrieve(text, top_k=top_k * 2)
                source_results["dense"] = dense_results
            except Exception:
                pass

        # Merge results
        merged = self._merge_sources(source_results)

        return merged[:top_k]

    def _merge_sources(
        self,
        source_results: dict[str, list[tuple[str, float]]],
    ) -> list[DrugCandidateResult]:
        """Merge results from multiple sources using RRF."""
        if not source_results:
            return []

        rxcui_scores: dict[str, dict[str, float]] = {}

        for source, results in source_results.items():
            if not results:
                continue

            # Normalize scores to [0, 1]
            max_score = max(s for _, s in results) if results else 1.0
            if max_score == 0:
                max_score = 1.0

            # Assign ranks and add RRF score
            for rank, (rxcui, score) in enumerate(results):
                norm = score / max_score
                rrf_score = 1.0 / (60 + rank + 1)  # k=60

                if rxcui not in rxcui_scores:
                    rxcui_scores[rxcui] = {}

                if source not in rxcui_scores[rxcui]:
                    rxcui_scores[rxcui][source] = 0.0

                rxcui_scores[rxcui][source] = max(rxcui_scores[rxcui][source], norm)

        # Calculate final scores
        final_scores: dict[str, float] = {}
        code_sources: dict[str, list[str]] = {}

        for rxcui, source_scores in rxcui_scores.items():
            total = 0.0
            for source, norm_score in source_scores.items():
                # Weight by source importance
                if source == "structured":
                    weight = 1.0
                elif source == "fuzzy":
                    weight = 0.8
                else:
                    weight = 0.7

                total += (0.5 + 0.5 * norm_score) * weight

            final_scores[rxcui] = total

            # Track sources
            code_sources[rxcui] = list(source_scores.keys())

        # Sort by final score
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rxcui, score in ranked:
            results.append(DrugCandidateResult(
                rxcui=rxcui,
                score=score,
                sources=code_sources[rxcui],
            ))

        return results

    def retrieve_one(self, query: str, mention: Optional[str] = None) -> Optional[DrugCandidateResult]:
        """Retrieve single best candidate."""
        results = self.retrieve(query, mention=mention, top_k=1)
        return results[0] if results else None
