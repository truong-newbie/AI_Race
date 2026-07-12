"""
Medical Ontology Pipeline — End-to-End System

Data flow:
  original_text → preprocessing → rule/NER extraction → span resolver
  → assertion detection → ICD/RxNorm retrieval → reranking → validation
  → JSON serialization

Design principles:
  - original_text is immutable; all entity.text is sliced from it
  - Every module failure is caught and logged; pipeline continues with
    partial results (graceful degradation)
  - Canonical Entity schema from src.schema; no duplicate entity models
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from src.schema import Entity, EntityType, AssertionType
from src.preprocessing.loader import load_text, load_texts_from_directory
from src.entity.lab_extractor import LabTestExtractor
from src.entity.drug_extractor import DrugExtractor
from src.entity.disease_extractor import DiseaseExtractor
from src.entity.span_resolver import Span, SpanResolver, resolve_spans, create_span
from src.assertion.rules import AssertionDetector

from src.pipeline.config import PipelineConfig
from src.pipeline.factory import (
    ComponentResult,
    build_icd_retriever,
    build_icd_reranker,
    build_rxnorm_retriever,
    build_rxnorm_reranker,
    build_entity_validator,
    set_seeds,
)

logger = logging.getLogger(__name__)


# ─── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ModuleError:
    """Error from a pipeline stage."""
    stage: str
    message: str
    recoverable: bool = True   # if True, pipeline continues with partial results


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    entities: list[Entity] = field(default_factory=list)
    errors: list[ModuleError] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # stages that were disabled


@dataclass
class ExtractionResult:
    """
    Complete output of a pipeline run.

    Attributes:
        text: Original input text (immutable).
        entities: Sorted by start position, deduped, valid candidates.
        errors: Module-level errors with clear messages.
        validation_errors: Validation failures from final check.
    """
    text: str
    entities: list[Entity]
    errors: list[ModuleError] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> list[dict]:
        """
        Convert to competition output format.

        Output is sorted by start position, deduplicated, assertions filtered
        to allowed list, candidates validated.
        """
        return [
            {
                "text": e.text,
                "position": e.position,
                "type": e.type.value,
                "assertions": [a.value for a in e.assertions],
                "candidates": e.candidates,
            }
            for e in self.entities
        ]

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class MedicalOntologyPipeline:
    """
    End-to-end pipeline for medical entity extraction and linking.

    Pipeline stages (each is individually wrapped for graceful degradation):
      1. Extract spans (labs, drugs, diseases, symptoms)
      2. Resolve overlapping spans
      3. Detect assertions (negation, historical, family)
      4. Link to ICD-10 and RxNorm knowledge bases
      5. Optionally rerank candidates
      6. Validate output
      7. Return JSON-serializable list

    Usage:
        config = PipelineConfig(link_icd=True, reranker_enabled=True)
        pipeline = MedicalOntologyPipeline(config)
        result = pipeline.process("BN tăng huyết áp, được kê Amlodipine 5mg.")
        print(result.to_dict())
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # ── Logging ──────────────────────────────────────────────────────────
        logging.basicConfig(level=self.config.to_logging_level())
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

        # ── Deterministic mode ──────────────────────────────────────────────
        if self.config.deterministic:
            set_seeds(42)

        # ── Extractor cache (built lazily on first process() call) ───────────
        self._lab_extractor: Optional[LabTestExtractor] = None
        self._drug_extractor: Optional[DrugExtractor] = None
        self._disease_extractor: Optional[DiseaseExtractor] = None
        self._assertion_detector: Optional[AssertionDetector] = None
        self._span_resolver: Optional[SpanResolver] = None

        # ── Linking components (built lazily) ────────────────────────────────
        self._icd_retriever: Any = None
        self._icd_reranker: Any = None
        self._rxnorm_retriever: Any = None
        self._rxnorm_reranker: Any = None

        # ── Known code sets for validation ─────────────────────────────────
        self._known_icd_codes: set[str] = set()
        self._known_rxnorm_codes: set[str] = set()
        self._kb_loaded: bool = False

        # ── Module-level error log ──────────────────────────────────────────
        self._module_errors: list[ModuleError] = []

        self._logger.info("MedicalOntologyPipeline initialized (lazy-load)")

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> ExtractionResult:
        """
        Process a single text. Returns ExtractionResult with sorted, deduped entities.

        Args:
            text: Original input text (must be the canonical, immutable text).

        Returns:
            ExtractionResult — never raises; all errors are captured in result.errors.
        """
        self._module_errors = []

        if not text:
            return ExtractionResult(text="", entities=[], errors=[])

        # Canonicalize: store only the original
        original = text

        try:
            # ── Stage 1: Extract ──────────────────────────────────────────────
            spans = self._extract_spans(original)

            # ── Stage 2: Resolve overlaps ───────────────────────────────────
            if self.config.resolve_overlaps:
                spans = self._resolve_overlaps(spans)
            else:
                self._module_errors.append(
                    ModuleError("overlap_resolution", "disabled by config", recoverable=True)
                )

            # ── Stage 3: Spans → Entities ───────────────────────────────────
            entities = self._spans_to_entities(original, spans)

            # ── Stage 4: Assertions ─────────────────────────────────────────
            if self.config.detect_assertions:
                entities = self._detect_assertions(original, entities)
            else:
                self._module_errors.append(
                    ModuleError("assertion_detection", "disabled by config", recoverable=True)
                )

            # ── Stage 5: Linking ───────────────────────────────────────────
            self._ensure_kb_loaded()
            if self.config.link_icd or self.config.link_rxnorm:
                entities = self._link_entities(original, entities)
            else:
                self._module_errors.append(
                    ModuleError("linking", "disabled by config", recoverable=True)
                )

            # ── Stage 6: Reranking ─────────────────────────────────────────
            if self.config.reranker_enabled:
                entities = self._rerank_entities(original, entities)
            else:
                self._module_errors.append(
                    ModuleError("reranking", "disabled by config", recoverable=True)
                )

            # ── Stage 7: Post-process ───────────────────────────────────────
            entities = self._post_process(entities, original)

            # ── Stage 8: Validate ────────────────────────────────────────────
            val_errors = []
            if self.config.validate_output:
                val_errors = self._validate(original, entities)

            return ExtractionResult(
                text=original,
                entities=entities,
                errors=self._module_errors.copy(),
                validation_errors=val_errors,
            )

        except Exception as e:
            self._logger.exception("Unhandled pipeline error")
            self._module_errors.append(
                ModuleError("pipeline", f"Unhandled exception: {e}", recoverable=False)
            )
            return ExtractionResult(
                text=original,
                entities=[],
                errors=self._module_errors.copy(),
            )

    def process_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """
        Process multiple texts in sequence.

        Args:
            texts: List of input texts.

        Returns:
            List of ExtractionResult, one per input text.
        """
        return [self.process(t) for t in texts]

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> ExtractionResult:
        """
        Process a single text file.

        Args:
            input_path: Path to .txt file.
            output_path: Optional path for .json output.

        Returns:
            ExtractionResult.
        """
        text = load_text(input_path)
        result = self.process(text)

        if output_path:
            result.to_json(output_path)

        return result

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> dict[str, ExtractionResult]:
        """
        Process all .txt files in a directory.

        Args:
            input_dir: Directory containing input files.
            output_dir: Directory for output JSON files.

        Returns:
            Dict mapping filename → ExtractionResult.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        texts = load_texts_from_directory(input_dir)

        for filename, text in texts.items():
            result = self.process(text)
            out_path = output_dir / f"{filename}.json"
            result.to_json(out_path)
            results[filename] = result

        return results

    # ── Stage 1: Extraction ───────────────────────────────────────────────────

    def _extract_spans(self, text: str) -> list[Span]:
        """Extract all entity spans. Failures per-extractor are logged, not raised."""
        spans: list[Span] = []

        # ── Labs ──────────────────────────────────────────────────────────
        if self.config.extract_labs:
            result = self._try_lab_extraction(text)
            if result.error:
                self._module_errors.append(result.error)
            else:
                spans.extend(result.component)

        # ── Drugs ─────────────────────────────────────────────────────────
        if self.config.extract_drugs:
            result = self._try_drug_extraction(text)
            if result.error:
                self._module_errors.append(result.error)
            else:
                spans.extend(result.component)

        # ── Diseases / Symptoms ────────────────────────────────────────────
        if self.config.extract_diseases or self.config.extract_symptoms:
            result = self._try_disease_extraction(text)
            if result.error:
                self._module_errors.append(result.error)
            else:
                spans.extend(result.component)

        return spans

    def _try_lab_extraction(self, text: str) -> ComponentResult[list[Span]]:
        if self._lab_extractor is None:
            try:
                self._lab_extractor = LabTestExtractor()
            except Exception as e:
                return ComponentResult(None, ModuleError("lab_extraction", f"{e}"))

        try:
            tests, results = self._lab_extractor.extract_all(text)
            spans: list[Span] = []

            for t in tests:
                conf = getattr(t, "confidence", 1.0)
                if conf < self.config.lab_confidence_threshold:
                    continue
                spans.append(create_span(
                    text=t.text,
                    position=[t.start, t.end],
                    entity_type="TEN_XET_NGHIEM",
                    confidence=conf,
                    source="lab_rule",
                ))

            for r in results:
                conf = getattr(r, "confidence", 1.0)
                if conf < self.config.lab_confidence_threshold:
                    continue
                spans.append(create_span(
                    text=r.text,
                    position=[r.start, r.end],
                    entity_type="KET_QUA_XET_NGHIEM",
                    confidence=conf,
                    source="lab_rule",
                ))

            return ComponentResult(spans)
        except Exception as e:
            return ComponentResult(None, ModuleError("lab_extraction", f"{e}"))

    def _try_drug_extraction(self, text: str) -> ComponentResult[list[Span]]:
        if self._drug_extractor is None:
            try:
                self._drug_extractor = DrugExtractor()
            except Exception as e:
                return ComponentResult(None, ModuleError("drug_extraction", f"{e}"))

        try:
            matches = self._drug_extractor.extract(text)
            spans: list[Span] = []

            for m in matches:
                if m.confidence < self.config.drug_confidence_threshold:
                    continue
                spans.append(create_span(
                    text=m.text,
                    position=[m.start, m.end],
                    entity_type="THUOC",
                    confidence=m.confidence,
                    source="drug_rule",
                ))

            return ComponentResult(spans)
        except Exception as e:
            return ComponentResult(None, ModuleError("drug_extraction", f"{e}"))

    def _try_disease_extraction(self, text: str) -> ComponentResult[list[Span]]:
        if self._disease_extractor is None:
            try:
                self._disease_extractor = DiseaseExtractor()
            except Exception as e:
                return ComponentResult(None, ModuleError("disease_extraction", f"{e}"))

        try:
            matches = self._disease_extractor.extract(text)
            spans: list[Span] = []

            for m in matches:
                threshold = (
                    self.config.disease_confidence_threshold
                    if m.is_diagnosed
                    else self.config.symptom_confidence_threshold
                )
                if m.confidence < threshold:
                    continue

                entity_type = (
                    "CHAN_DOAN" if m.is_diagnosed
                    else "TRIEU_CHUNG"
                )
                spans.append(create_span(
                    text=m.text,
                    position=[m.start, m.end],
                    entity_type=entity_type,
                    confidence=m.confidence,
                    source="disease_rule",
                ))

            return ComponentResult(spans)
        except Exception as e:
            return ComponentResult(None, ModuleError("disease_extraction", f"{e}"))

    # ── Stage 2: Overlap Resolution ───────────────────────────────────────────

    def _resolve_overlaps(self, spans: list[Span]) -> list[Span]:
        if not spans:
            return []

        try:
            if self._span_resolver is None:
                self._span_resolver = SpanResolver(strategy=self.config.overlap_strategy)
            result = self._span_resolver.resolve(spans)
            self._logger.debug(
                f"Overlap resolved: {len(spans)} → {len(result.resolved_spans)} spans, "
                f"removed={len(result.removed_spans)}"
            )
            return result.resolved_spans
        except Exception as e:
            self._module_errors.append(
                ModuleError("overlap_resolution", f"{e}")
            )
            return spans  # fallback: return original spans

    # ── Stage 3: Spans → Entities ─────────────────────────────────────────────

    # Mapping from raw (non-diacritic) entity type strings to enum values.
    # DiseaseExtractor / lab extractor return ASCII-safe strings, but
    # EntityType enum values contain Vietnamese diacritics.
    _ENTITY_TYPE_MAP: dict[str, EntityType] = {
        "TRIEU_CHUNG": EntityType.TRIEU_CHUNG,
        "CHAN_DOAN": EntityType.CHAN_DOAN,
        "THUOC": EntityType.THUOC,
        "TEN_XET_NGHIEM": EntityType.TEN_XET_NGHIEM,
        "KET_QUA_XET_NGHIEM": EntityType.KET_QUA_XET_NGHIEM,
        # Also accept the diacritic forms directly (some callers may use them)
        EntityType.TRIEU_CHUNG.value: EntityType.TRIEU_CHUNG,
        EntityType.CHAN_DOAN.value: EntityType.CHAN_DOAN,
        EntityType.THUOC.value: EntityType.THUOC,
        EntityType.TEN_XET_NGHIEM.value: EntityType.TEN_XET_NGHIEM,
        EntityType.KET_QUA_XET_NGHIEM.value: EntityType.KET_QUA_XET_NGHIEM,
    }

    def _spans_to_entities(self, text: str, spans: list[Span]) -> list[Entity]:
        """
        Convert spans to Entity objects.

        REQUIREMENT: entity.text is always sliced from original text (immutability).
        """
        entities: list[Entity] = []

        for span in spans:
            try:
                # Normalize: handle both ASCII-safe and diacritic forms
                raw_type = span.entity_type
                entity_type = self._ENTITY_TYPE_MAP.get(raw_type)
                if entity_type is None:
                    entity_type = EntityType(raw_type)
            except ValueError:
                self._logger.warning(
                    f"Unknown entity type '{span.entity_type}' at [{span.start}, {span.end}]"
                )
                continue

            # Extract text from original (REQUIREMENT: immutable original text)
            span_text = text[span.start:span.end]

            entity = Entity(
                text=span_text,
                position=[span.start, span.end],
                type=entity_type,
                assertions=[],
                candidates=[],
            )
            entities.append(entity)

        return entities

    # ── Stage 4: Assertions ───────────────────────────────────────────────────

    def _detect_assertions(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Detect negation / historical / family assertions."""
        allowed_types = {EntityType.TRIEU_CHUNG, EntityType.CHAN_DOAN, EntityType.THUOC}

        if self._assertion_detector is None:
            try:
                self._assertion_detector = AssertionDetector()
            except Exception as e:
                self._module_errors.append(
                    ModuleError("assertion_detection", f"AssertionDetector init: {e}")
                )
                return entities

        for entity in entities:
            if entity.type not in allowed_types:
                continue

            try:
                result = self._assertion_detector.detect(
                    text,
                    entity.position[0],
                    entity.position[1],
                )

                assertions: list[AssertionType] = []
                if result.status.is_negated:
                    assertions.append(AssertionType.NEGATED)
                if result.status.is_historical:
                    assertions.append(AssertionType.HISTORICAL)
                if result.status.is_family:
                    assertions.append(AssertionType.FAMILY)

                # Filter to allowed assertions list (None = no filter, empty set = filter all)
                if self.config.allowed_assertions is not None:
                    if not self.config.allowed_assertions:
                        assertions = []  # empty set → strip all
                    else:
                        assertions = [
                            a for a in assertions
                            if a.value in self.config.allowed_assertions
                        ]

                # Filter by minimum confidence
                if result.status.confidence >= self.config.assertion_min_confidence:
                    entity.assertions = assertions
                else:
                    entity.assertions = []

            except Exception as e:
                self._module_errors.append(
                    ModuleError(
                        "assertion_detection",
                        f"Entity [{entity.position[0]},{entity.position[1]}] '{entity.text}': {e}",
                        recoverable=True,
                    )
                )

        return entities

    # ── Stage 5: Linking ─────────────────────────────────────────────────────

    def _ensure_kb_loaded(self) -> None:
        """Lazy-load knowledge bases and linking components."""
        if self._kb_loaded:
            return

        # ── ICD-10 ─────────────────────────────────────────────────────────
        if self.config.link_icd:
            icd_res = build_icd_retriever(
                kb_path=self.config.icd_kb_path,
                merge_method=self.config.icd_merge_method,
                rrf_k=self.config.icd_rrf_k,
                dense_model=self.config.icd_dense_model,
                dense_enabled=self.config.icd_dense_enabled,
                top_k=self.config.icd_top_k,
            )
            if icd_res.ok:
                self._icd_retriever = icd_res.component
                # Cache known codes
                try:
                    from src.linking.icd.schema import get_knowledge_base
                    self._known_icd_codes = {e.code for e in get_knowledge_base()}
                except Exception:
                    pass
            else:
                self._module_errors.append(
                    ModuleError("icd_retriever", icd_res.error)
                )

            # Reranker (lazy)
            if self.config.use_icd_reranker():
                rerank_res = build_icd_reranker()
                if rerank_res.ok:
                    self._icd_reranker = rerank_res.component
                else:
                    self._module_errors.append(
                        ModuleError("icd_reranker", rerank_res.error)
                    )

        # ── RxNorm ────────────────────────────────────────────────────────
        if self.config.link_rxnorm:
            rx_res = build_rxnorm_retriever(
                kb_path=self.config.rxnorm_kb_path,
                top_k=self.config.rxnorm_top_k,
                dense_model=self.config.rxnorm_dense_model,
                dense_enabled=self.config.rxnorm_dense_enabled,
                use_structured=self.config.rxnorm_use_structured,
                use_fuzzy=self.config.rxnorm_use_fuzzy,
            )
            if rx_res.ok:
                self._rxnorm_retriever = rx_res.component
                # Cache known codes
                try:
                    from src.linking.rxnorm.schema import get_knowledge_base
                    self._known_rxnorm_codes = {e.rxcui for e in get_knowledge_base()}
                except Exception:
                    pass
            else:
                self._module_errors.append(
                    ModuleError("rxnorm_retriever", rx_res.error)
                )

            # Reranker (lazy)
            if self.config.use_rxnorm_reranker():
                rerank_res = build_rxnorm_reranker()
                if rerank_res.ok:
                    self._rxnorm_reranker = rerank_res.component
                else:
                    self._module_errors.append(
                        ModuleError("rxnorm_reranker", rerank_res.error)
                    )

        self._kb_loaded = True

    def _link_entities(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Run ICD-10 and RxNorm retrieval for each entity."""
        for entity in entities:
            try:
                if entity.type == EntityType.CHAN_DOAN and self.config.link_icd:
                    self._link_icd(text, entity)
                elif entity.type == EntityType.THUOC and self.config.link_rxnorm:
                    self._link_rxnorm(text, entity)
            except Exception as e:
                self._module_errors.append(
                    ModuleError(
                        "linking",
                        f"Entity '{entity.text}' [{entity.position}]: {e}",
                        recoverable=True,
                    )
                )

        return entities

    def _link_icd(self, text: str, entity: Entity) -> None:
        """Link a CHẨN_ĐOÁN entity to ICD-10."""
        if self._icd_retriever is None:
            return

        mention = entity.text
        query = text  # full context

        try:
            candidates = self._icd_retriever.retrieve(
                query, mention=mention, top_k=self.config.icd_top_k
            )

            if not candidates:
                return

            # Take top-k and limit to configured output count
            top_n = self.config.effective_icd_output_candidates()
            codes = [c.code for c in candidates[:top_n]]

            # Validate: keep only codes that exist in known KB
            if self._known_icd_codes:
                codes = [c for c in codes if c in self._known_icd_codes]

            entity.candidates = codes

        except Exception as e:
            self._module_errors.append(
                ModuleError(
                    "icd_retrieval",
                    f"'{entity.text}': {e}",
                    recoverable=True,
                )
            )

    def _link_rxnorm(self, text: str, entity: Entity) -> None:
        """Link a THUỐC entity to RxNorm."""
        if self._rxnorm_retriever is None:
            return

        mention = entity.text
        query = text

        try:
            candidates = self._rxnorm_retriever.retrieve(
                query, mention=mention, top_k=self.config.rxnorm_top_k
            )

            if not candidates:
                return

            top_n = self.config.effective_rxnorm_output_candidates()
            codes = [c.rxcui for c in candidates[:top_n]]

            # Validate: keep only codes that exist in known KB
            if self._known_rxnorm_codes:
                codes = [c for c in codes if c in self._known_rxnorm_codes]

            entity.candidates = codes

        except Exception as e:
            self._module_errors.append(
                ModuleError(
                    "rxnorm_retrieval",
                    f"'{entity.text}': {e}",
                    recoverable=True,
                )
            )

    # ── Stage 6: Reranking ───────────────────────────────────────────────────

    def _rerank_entities(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Apply reranking to linked entities."""
        for entity in entities:
            try:
                if entity.type == EntityType.CHAN_DOAN and self.config.use_icd_reranker():
                    if self._icd_reranker is not None and entity.candidates:
                        entity.candidates = self._rerank_icd(text, entity)
                elif entity.type == EntityType.THUOC and self.config.use_rxnorm_reranker():
                    if self._rxnorm_reranker is not None and entity.candidates:
                        entity.candidates = self._rerank_rxnorm(text, entity)
            except Exception as e:
                self._module_errors.append(
                    ModuleError(
                        "reranking",
                        f"Entity '{entity.text}': {e}",
                        recoverable=True,
                    )
                )

        return entities

    def _rerank_icd(self, text: str, entity: Entity) -> list[str]:
        """Rerank ICD candidates for one entity."""
        # Build mock candidate objects for the reranker
        class _MockICD:
            def __init__(self, code, score):
                self.code = code
                self.score = score
                self.sources = []
                self.detail = {}

        # Retrieve fresh candidates for reranking
        raw = self._icd_retriever.retrieve(
            text, mention=entity.text, top_k=self.config.icd_top_k
        )

        # Intersect with already-linked codes (preserve linked set)
        linked_set = set(entity.candidates)
        candidates = [c for c in raw if c.code in linked_set]
        if not candidates:
            return entity.candidates  # fallback to original order

        try:
            reranked = self._icd_reranker.rerank(
                candidates, text, mention=entity.text, top_k=len(candidates)
            )
            return [r.code for r in reranked]
        except Exception:
            return entity.candidates  # fallback

    def _rerank_rxnorm(self, text: str, entity: Entity) -> list[str]:
        """Rerank RxNorm candidates for one entity."""
        class _MockRx:
            def __init__(self, rxcui, score):
                self.rxcui = rxcui
                self.score = score
                self.sources = []

        raw = self._rxnorm_retriever.retrieve(
            text, mention=entity.text, top_k=self.config.rxnorm_top_k
        )

        linked_set = set(entity.candidates)
        candidates = [c for c in raw if c.rxcui in linked_set]
        if not candidates:
            return entity.candidates

        try:
            reranked = self._rxnorm_reranker.rerank(
                candidates, text, mention=entity.text, top_k=len(candidates)
            )
            return [r.code for r in reranked]
        except Exception:
            return entity.candidates

    # ── Stage 7: Post-process ─────────────────────────────────────────────────

    def _post_process(self, entities: list[Entity], original: str) -> list[Entity]:
        """
        Post-processing steps:
          1. Sort by start position (REQUIREMENT: output sorted by start)
          2. Remove duplicates (overlapping positions) (REQUIREMENT: no duplicates)
          3. Verify entity.text matches original slice (REQUIREMENT: immutable original)
          4. Strip empty candidates
          5. Truncate candidates to max count
        """
        # 1. Sort
        entities.sort(key=lambda e: e.position[0])

        # 2. Deduplicate: keep first entity for each [start, end) position
        seen_positions: set[tuple[int, int]] = set()
        deduped: list[Entity] = []

        for entity in entities:
            pos = tuple(entity.position)
            if pos in seen_positions:
                continue
            seen_positions.add(pos)

            # 3. Re-verify text against original (re-slice)
            start, end = entity.position
            if start >= len(original) or end > len(original) or start < 0:
                self._logger.warning(
                    f"Invalid position [{start}, {end}] for text len {len(original)}"
                )
                continue

            canonical_text = original[start:end]
            if canonical_text != entity.text:
                # Fix: re-slice from original
                entity.text = canonical_text

            # 4. Strip empty/invalid candidates
            if entity.candidates:
                # Truncate to max
                max_c = (
                    self.config.effective_icd_output_candidates()
                    if entity.type == EntityType.CHAN_DOAN
                    else self.config.effective_rxnorm_output_candidates()
                )
                entity.candidates = entity.candidates[:max_c]

            deduped.append(entity)

        return deduped

    # ── Stage 8: Validation ───────────────────────────────────────────────────

    def _validate(self, text: str, entities: list[Entity]) -> list[str]:
        """Run EntityValidator. Returns list of error strings."""
        errors: list[str] = []

        try:
            val_res = build_entity_validator(
                original_text=text,
                known_icd_codes=self._known_icd_codes,
                known_rxnorm_codes=self._known_rxnorm_codes,
            )
            if not val_res.ok:
                errors.append(f"EntityValidator init: {val_res.error}")
                return errors

            validator = val_res.component
            val_result = validator.validate(entities)

            for err in val_result.errors:
                errors.append(f"[entity {err.entity_index}] {err.field}: {err.message}")
            for warn in val_result.warnings:
                errors.append(f"[entity {warn.entity_index}] WARNING {warn.field}: {warn.message}")

        except Exception as e:
            errors.append(f"Validation exception: {e}")

        return errors


# =============================================================================
# Convenience functions
# =============================================================================


def extract_medical_entities(text: str, **kwargs) -> list[dict]:
    """
    One-liner: extract and link entities from text.

    Args:
        text: Input text.
        **kwargs: Passed to PipelineConfig (e.g. link_icd=True, reranker_enabled=True).

    Returns:
        List of entity dicts in competition output format.
    """
    config = PipelineConfig(**kwargs)
    pipeline = MedicalOntologyPipeline(config)
    result = pipeline.process(text)
    return result.to_dict()


def process_file(input_path: str, output_path: str, **kwargs) -> list[dict]:
    """Process a file and save JSON output."""
    config = PipelineConfig(**kwargs)
    pipeline = MedicalOntologyPipeline(config)
    result = pipeline.process_file(input_path, output_path)
    return result.to_dict()
