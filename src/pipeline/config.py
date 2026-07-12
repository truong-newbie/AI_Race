"""
Pipeline Configuration.

Defines all configurable parameters for the MedicalOntologyPipeline.
Read from YAML or dict — no hardcoded values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Logging ──────────────────────────────────────────────────────────────────

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# ─── Candidate Count ─────────────────────────────────────────────────────────

DEFAULT_MAX_CANDIDATES = 5


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """
    Full configuration for MedicalOntologyPipeline.

    All fields have sensible defaults so the pipeline can run with zero config.
    """

    # ── Extraction ──────────────────────────────────────────────────────────

    extract_labs: bool = True
    extract_drugs: bool = True
    extract_diseases: bool = True
    extract_symptoms: bool = True

    # Confidence thresholds (0.0–1.0). Entity below threshold is dropped.
    lab_confidence_threshold: float = 0.7
    drug_confidence_threshold: float = 0.6
    disease_confidence_threshold: float = 0.6
    symptom_confidence_threshold: float = 0.6

    # ── Span resolution ─────────────────────────────────────────────────────

    resolve_overlaps: bool = True
    overlap_strategy: str = "hybrid"  # longest | confidence | type_priority | hybrid

    # ── Assertions ──────────────────────────────────────────────────────────

    detect_assertions: bool = True

    # Minimum confidence for assertion detection (0.0–1.0)
    assertion_min_confidence: float = 0.5

    # Assertion whitelist — only these assertion types are emitted.
    # Leave empty to allow all valid assertion types.
    allowed_assertions: set[str] = field(default_factory=lambda: {
        "isNegated", "isFamily", "isHistorical"
    })

    # ── ICD-10 Retrieval ────────────────────────────────────────────────────

    link_icd: bool = True
    icd_top_k: int = 10          # Candidates returned per entity
    icd_output_candidates: int = DEFAULT_MAX_CANDIDATES
    icd_merge_method: str = "rrf"  # rrf | weighted
    icd_rrf_k: int = 60

    # Dense model
    icd_dense_model: str = "intfloat/multilingual-e5-small"
    icd_dense_enabled: bool = True

    # KB path — None means use built-in sample KB
    icd_kb_path: Optional[str] = None

    # ── RxNorm Retrieval ────────────────────────────────────────────────────

    link_rxnorm: bool = True
    rxnorm_top_k: int = 10
    rxnorm_output_candidates: int = DEFAULT_MAX_CANDIDATES

    # Dense model
    rxnorm_dense_model: str = "intfloat/multilingual-e5-small"
    rxnorm_dense_enabled: bool = True

    # Structured / fuzzy matching
    rxnorm_use_structured: bool = True
    rxnorm_use_fuzzy: bool = True

    # KB path — None means use built-in sample KB
    rxnorm_kb_path: Optional[str] = None

    # ── Reranking ──────────────────────────────────────────────────────────

    reranker_enabled: bool = False          # Global toggle
    icd_reranker_enabled: bool = False
    rxnorm_reranker_enabled: bool = False

    # Cross-encoder model (lazy-loaded)
    cross_encoder_model: str = "dmis-lab/biobert-v1.1"
    cross_encoder_enabled: bool = False
    cross_encoder_alpha: float = 0.3       # Blend weight for cross-encoder score

    # ── Validation ───────────────────────────────────────────────────────────

    validate_output: bool = True

    # KB for candidate existence check — paths or None for skip
    validation_icd_kb_path: Optional[str] = None
    validation_rxnorm_kb_path: Optional[str] = None

    # ── Runtime ─────────────────────────────────────────────────────────────

    deterministic: bool = False  # If True, set seeds for reproducibility
    log_level: str = "INFO"
    error_on_module_failure: bool = False  # If True, raise; if False, return partial result

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def to_logging_level(self) -> int:
        """Convert log_level string to Python logging constant."""
        return getattr(logging, self.log_level.upper(), logging.INFO)

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        """Build config from a flat dict (e.g. loaded from YAML)."""
        # accepted aliases for snake_case fields
        known = {
            "extract_labs", "extract_drugs", "extract_diseases", "extract_symptoms",
            "lab_confidence_threshold", "drug_confidence_threshold",
            "disease_confidence_threshold", "symptom_confidence_threshold",
            "resolve_overlaps", "overlap_strategy",
            "detect_assertions", "assertion_min_confidence", "allowed_assertions",
            "link_icd", "icd_top_k", "icd_output_candidates", "icd_merge_method",
            "icd_rrf_k", "icd_dense_model", "icd_dense_enabled", "icd_kb_path",
            "link_rxnorm", "rxnorm_top_k", "rxnorm_output_candidates",
            "rxnorm_dense_model", "rxnorm_dense_enabled",
            "rxnorm_use_structured", "rxnorm_use_fuzzy", "rxnorm_kb_path",
            "reranker_enabled", "icd_reranker_enabled", "rxnorm_reranker_enabled",
            "cross_encoder_model", "cross_encoder_enabled", "cross_encoder_alpha",
            "validate_output", "validation_icd_kb_path", "validation_rxnorm_kb_path",
            "deterministic", "log_level", "error_on_module_failure",
        }
        return cls(**{k: v for k, v in data.items() if k in known and v is not None})

    def effective_icd_output_candidates(self) -> int:
        return min(self.icd_output_candidates, self.icd_top_k)

    def effective_rxnorm_output_candidates(self) -> int:
        return min(self.rxnorm_output_candidates, self.rxnorm_top_k)

    def use_icd_reranker(self) -> bool:
        return self.reranker_enabled and self.icd_reranker_enabled

    def use_rxnorm_reranker(self) -> bool:
        return self.reranker_enabled and self.rxnorm_reranker_enabled
