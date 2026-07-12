"""Linking module - ICD-10 và RxNorm entity linking."""

from src.linking.icd10 import (
    ICD10Entry,
    ICD10KnowledgeBase,
    create_sample_icd10_kb,
)
# Baseline RxNorm classes from rxnorm_legacy
from src.linking.rxnorm_legacy import (
    RxNormKnowledgeBase,
    DrugParser,
    RxNormLinker,
    create_sample_rxnorm_kb,
)
# New RxNormEntry from rxnorm schema (extended fields)
from src.linking.rxnorm.schema import RxNormEntry

# Reranking modules
from src.linking.base_reranker import BaseReranker, RerankResult
from src.linking.ontology_constraints import OntologyValidator
from src.linking.rule_reranker import (
    ICDRuleReranker,
    RxNormRuleReranker,
    UnifiedRuleReranker,
)
from src.linking.cross_encoder_dataset import CrossEncoderDatasetBuilder
from src.linking.cross_encoder_reranker import (
    CrossEncoderReranker,
    HybridCrossEncoderReranker,
)

__all__ = [
    # ICD-10
    'ICD10Entry',
    'ICD10KnowledgeBase',
    'create_sample_icd10_kb',
    # RxNorm (baseline)
    'RxNormKnowledgeBase',
    'DrugParser',
    'RxNormLinker',
    'create_sample_rxnorm_kb',
    # RxNorm (new schema)
    'RxNormEntry',
    # Reranking
    'BaseReranker',
    'RerankResult',
    'OntologyValidator',
    'ICDRuleReranker',
    'RxNormRuleReranker',
    'UnifiedRuleReranker',
    'CrossEncoderDatasetBuilder',
    'CrossEncoderReranker',
    'HybridCrossEncoderReranker',
]
