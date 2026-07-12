"""
RxNorm Candidate Retrieval Module

Với entity THUOC, tra ve top-k RxCUI phu hop nhat.
"""

# Re-export baseline classes from rxnorm_legacy (backward compatibility)
from src.linking.rxnorm_legacy import (
    RxNormKnowledgeBase,
    DrugParser,
    RxNormLinker,
    create_sample_rxnorm_kb,
)

# New module classes
from src.linking.rxnorm.schema import RxNormEntry, ParsedDrug, get_knowledge_base
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.normalizer import DrugTextNormalizer
from src.linking.rxnorm.structured_matcher import StructuredMatcher
from src.linking.rxnorm.fuzzy_retriever import FuzzyDrugRetriever
from src.linking.rxnorm.dense_retriever import DenseDrugRetriever
from src.linking.rxnorm.hybrid_retriever import DrugHybridRetriever, DrugCandidateResult
from src.linking.rxnorm.evaluator import DrugRetrievalEvaluator
from src.linking.rxnorm.reranker import DrugReranker, RerankScore

__all__ = [
    # Re-exported baseline classes
    "RxNormKnowledgeBase",
    "DrugParser",
    "RxNormLinker",
    "create_sample_rxnorm_kb",
    # New module classes
    "RxNormEntry",
    "ParsedDrug",
    "get_knowledge_base",
    "DrugMentionParser",
    "DrugTextNormalizer",
    "StructuredMatcher",
    "FuzzyDrugRetriever",
    "DenseDrugRetriever",
    "DrugHybridRetriever",
    "DrugCandidateResult",
    "DrugRetrievalEvaluator",
    "DrugReranker",
    "RerankScore",
]
