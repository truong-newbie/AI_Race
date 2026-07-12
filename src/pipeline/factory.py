"""
Pipeline Component Factory.

Lazy-loads all components with clear error messages and fallback support.
Each factory method catches exceptions and returns (component, error_msg).
"""

from __future__ import annotations

import logging
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── Fallback result ──────────────────────────────────────────────────────────


@dataclass
class ComponentResult:
    """Result of a factory method — either component or error."""
    component: Any
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.component is not None

    def unwrap(self) -> Any:
        if not self.ok:
            raise RuntimeError(f"Component unavailable: {self.error}")
        return self.component


# ─── Seed helpers ─────────────────────────────────────────────────────────────


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)


# ─── Extractors ───────────────────────────────────────────────────────────────


def build_lab_extractor() -> ComponentResult:
    """Build LabTestExtractor with fallback."""
    try:
        from src.entity.lab_extractor import LabTestExtractor
        comp = LabTestExtractor()
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"LabTestExtractor: {e}")


def build_drug_extractor() -> ComponentResult:
    """Build DrugExtractor with fallback."""
    try:
        from src.entity.drug_extractor import DrugExtractor
        comp = DrugExtractor()
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"DrugExtractor: {e}")


def build_disease_extractor() -> ComponentResult:
    """Build DiseaseExtractor with fallback."""
    try:
        from src.entity.disease_extractor import DiseaseExtractor
        comp = DiseaseExtractor()
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"DiseaseExtractor: {e}")


def build_span_resolver(strategy: str = "hybrid") -> ComponentResult:
    """Build SpanResolver with fallback."""
    try:
        from src.entity.span_resolver import SpanResolver
        comp = SpanResolver(strategy=strategy)
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"SpanResolver: {e}")


# ─── Assertions ───────────────────────────────────────────────────────────────


def build_assertion_detector() -> ComponentResult:
    """Build AssertionDetector with fallback."""
    try:
        from src.assertion.rules import AssertionDetector
        comp = AssertionDetector()
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"AssertionDetector: {e}")


# ─── ICD-10 Retrieval ─────────────────────────────────────────────────────────


def build_icd_retriever(
    kb_path: Optional[str] = None,
    merge_method: str = "rrf",
    rrf_k: int = 60,
    dense_model: str = "intfloat/multilingual-e5-small",
    dense_enabled: bool = True,
    top_k: int = 10,
) -> ComponentResult:
    """Build HybridRetriever with fallback to sample KB."""
    try:
        from src.linking.icd.schema import get_knowledge_base, ICD10Entry
        from src.linking.icd.hybrid_retriever import (
            HybridRetriever,
            MergeConfig,
        )

        if kb_path:
            # Try to load from path
            try:
                import json
                with open(kb_path, encoding="utf-8") as f:
                    raw = json.load(f)
                entries = [ICD10Entry(**r) for r in raw]
                logger.info(f"Loaded {len(entries)} ICD entries from {kb_path}")
            except Exception:
                logger.warning(f"Failed to load ICD KB from {kb_path}, using sample KB")
                entries = get_knowledge_base()
        else:
            entries = get_knowledge_base()

        merge_cfg = MergeConfig(
            method=merge_method,
            rrf_k=rrf_k,
        )

        comp = HybridRetriever(
            entries=entries,
            merge_config=merge_cfg,
            top_k=top_k,
            dense_model=dense_model if dense_enabled else "",
        )
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"ICD HybridRetriever: {e}")


def build_icd_reranker() -> ComponentResult:
    """Build ICDRuleReranker with fallback."""
    try:
        from src.linking.icd.schema import get_knowledge_base
        from src.linking.rule_reranker import ICDRuleReranker

        entries = get_knowledge_base()
        comp = ICDRuleReranker(entries)
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"ICDRuleReranker: {e}")


# ─── RxNorm Retrieval ─────────────────────────────────────────────────────────


def build_rxnorm_retriever(
    kb_path: Optional[str] = None,
    top_k: int = 10,
    dense_model: str = "intfloat/multilingual-e5-small",
    dense_enabled: bool = True,
    use_structured: bool = True,
    use_fuzzy: bool = True,
) -> ComponentResult:
    """Build DrugHybridRetriever with fallback to sample KB."""
    try:
        from src.linking.rxnorm.schema import get_knowledge_base, RxNormEntry
        from src.linking.rxnorm.hybrid_retriever import DrugHybridRetriever

        if kb_path:
            try:
                import json
                with open(kb_path, encoding="utf-8") as f:
                    raw = json.load(f)
                entries = [RxNormEntry(**r) for r in raw]
                logger.info(f"Loaded {len(entries)} RxNorm entries from {kb_path}")
            except Exception:
                logger.warning(f"Failed to load RxNorm KB from {kb_path}, using sample KB")
                entries = get_knowledge_base()
        else:
            entries = get_knowledge_base()

        comp = DrugHybridRetriever(
            entries=entries,
            top_k=top_k,
            use_dense=dense_enabled,
            use_fuzzy=use_fuzzy,
            use_structured=use_structured,
        )
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"RxNorm DrugHybridRetriever: {e}")


def build_rxnorm_reranker() -> ComponentResult:
    """Build RxNormRuleReranker with fallback."""
    try:
        from src.linking.rxnorm.schema import get_knowledge_base
        from src.linking.rule_reranker import RxNormRuleReranker

        entries = get_knowledge_base()
        comp = RxNormRuleReranker(entries)
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"RxNormRuleReranker: {e}")


# ─── Cross-Encoder ────────────────────────────────────────────────────────────


def build_cross_encoder_reranker(
    model_name: str = "dmis-lab/biobert-v1.1",
) -> ComponentResult:
    """Build CrossEncoderReranker with fallback to None (skip reranking)."""
    try:
        from src.linking.cross_encoder_reranker import CrossEncoderReranker

        comp = CrossEncoderReranker(model_name=model_name)
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"CrossEncoderReranker ({model_name}): {e}")


def build_hybrid_cross_encoder_reranker(
    model_name: str = "dmis-lab/biobert-v1.1",
    alpha: float = 0.3,
) -> ComponentResult:
    """Build HybridCrossEncoderReranker with fallback."""
    try:
        from src.linking.cross_encoder_reranker import (
            HybridCrossEncoderReranker,
        )

        comp = HybridCrossEncoderReranker(
            model_name=model_name,
            alpha=alpha,
        )
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"HybridCrossEncoderReranker: {e}")


# ─── Validator ───────────────────────────────────────────────────────────────


def build_entity_validator(
    original_text: str,
    known_icd_codes: Optional[set[str]] = None,
    known_rxnorm_codes: Optional[set[str]] = None,
) -> ComponentResult:
    """Build EntityValidator for a given text."""
    try:
        from src.validation.validator import OutputValidator

        # Build known code sets from sample KBs if not provided
        if known_icd_codes is None:
            try:
                from src.linking.icd.schema import get_knowledge_base
                known_icd_codes = {e.code for e in get_knowledge_base()}
            except Exception:
                known_icd_codes = set()

        if known_rxnorm_codes is None:
            try:
                from src.linking.rxnorm.schema import get_knowledge_base
                known_rxnorm_codes = {e.rxcui for e in get_knowledge_base()}
            except Exception:
                known_rxnorm_codes = set()

        comp = OutputValidator(
            original_text=original_text,
            known_icd_codes=known_icd_codes,
            known_rxnorm_codes=known_rxnorm_codes,
        )
        return ComponentResult(comp)
    except Exception as e:
        return ComponentResult(None, f"EntityValidator: {e}")
