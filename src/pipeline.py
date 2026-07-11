"""
Medical Ontology Pipeline - End-to-End Baseline

Pipeline kết hợp tất cả modules để extract entities từ văn bản y khoa.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from src.schema import Entity, EntityType, AssertionType
from src.preprocessing.loader import load_text, load_texts_from_directory, save_text
from src.validation.validator import OutputValidator, ValidationResult
from src.entity.lab_extractor import LabTestExtractor
from src.entity.drug_extractor import DrugExtractor
from src.entity.disease_extractor import DiseaseExtractor
from src.entity.span_resolver import Span, SpanResolver, resolve_spans, create_span
from src.assertion.rules import AssertionDetector
from src.linking.icd10 import ICD10KnowledgeBase, create_sample_icd10_kb
from src.linking.rxnorm import RxNormKnowledgeBase, DrugParser, RxNormLinker, create_sample_rxnorm_kb

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration cho pipeline."""
    # KB settings
    icd10_kb_path: Optional[str] = None
    rxnorm_kb_path: Optional[str] = None

    # Extraction settings
    extract_labs: bool = True
    extract_drugs: bool = True
    extract_diseases: bool = True
    extract_symptoms: bool = True

    # Assertion settings
    detect_assertions: bool = True

    # Linking settings
    link_icd: bool = True
    link_rxnorm: bool = True

    # Resolution settings
    resolve_overlaps: bool = True
    overlap_strategy: str = "hybrid"


@dataclass
class ExtractionResult:
    """Kết quả của một extraction run."""
    text: str
    entities: list[Entity]
    validation_result: Optional[ValidationResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert sang dict for JSON output."""
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


class MedicalOntologyPipeline:
    """
    End-to-end pipeline để extract entities từ văn bản y khoa.

    Pipeline stages:
    1. Load text
    2. Extract entities (labs, drugs, diseases, symptoms)
    3. Resolve overlaps
    4. Detect assertions
    5. Link to ICD/RxNorm
    6. Validate output
    7. Return JSON
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize extractors
        self.lab_extractor = LabTestExtractor()
        self.drug_extractor = DrugExtractor()
        self.disease_extractor = DiseaseExtractor()
        self.assertion_detector = AssertionDetector()
        self.span_resolver = SpanResolver(strategy=self.config.overlap_strategy)

        # Initialize KBs
        self.icd10_kb: Optional[ICD10KnowledgeBase] = None
        self.rxnorm_kb: Optional[RxNormKnowledgeBase] = None
        self.drug_linker: Optional[RxNormLinker] = None

        # Load KBs if paths provided
        self._load_knowledge_bases()

    def _load_knowledge_bases(self):
        """Load knowledge bases."""
        # ICD-10 KB
        if self.config.icd10_kb_path:
            try:
                self.icd10_kb = ICD10KnowledgeBase.load(self.config.icd10_kb_path)
                logger.info(f"Loaded ICD-10 KB from {self.config.icd10_kb_path}")
            except Exception as e:
                logger.warning(f"Failed to load ICD-10 KB: {e}. Using sample KB.")
                self.icd10_kb = create_sample_icd10_kb()
        else:
            self.icd10_kb = create_sample_icd10_kb()
            logger.info("Using sample ICD-10 KB")

        # RxNorm KB
        if self.config.rxnorm_kb_path:
            try:
                self.rxnorm_kb = RxNormKnowledgeBase.load(self.config.rxnorm_kb_path)
                self.drug_linker = RxNormLinker(self.rxnorm_kb)
                logger.info(f"Loaded RxNorm KB from {self.config.rxnorm_kb_path}")
            except Exception as e:
                logger.warning(f"Failed to load RxNorm KB: {e}. Using sample KB.")
                self.rxnorm_kb = create_sample_rxnorm_kb()
                self.drug_linker = RxNormLinker(self.rxnorm_kb)
        else:
            self.rxnorm_kb = create_sample_rxnorm_kb()
            self.drug_linker = RxNormLinker(self.rxnorm_kb)
            logger.info("Using sample RxNorm KB")

    def process(self, text: str) -> ExtractionResult:
        """
        Process một văn bản y khoa.

        Args:
            text: Input text

        Returns:
            ExtractionResult
        """
        try:
            # Step 1: Extract entities
            spans = self._extract_entities(text)

            # Step 2: Resolve overlaps
            if self.config.resolve_overlaps:
                spans = resolve_spans(spans, strategy=self.config.overlap_strategy)

            # Step 3: Convert spans to entities
            entities = self._spans_to_entities(text, spans)

            # Step 4: Detect assertions
            if self.config.detect_assertions:
                entities = self._detect_assertions(text, entities)

            # Step 5: Link to ICD/RxNorm
            if self.config.link_icd or self.config.link_rxnorm:
                entities = self._link_entities(entities, text)

            # Step 6: Validate
            validator = OutputValidator(
                text,
                known_icd_codes=self.icd10_kb.get_all_codes() if self.icd10_kb else set(),
                known_rxnorm_codes=self.rxnorm_kb.get_all_rxcuis() if self.rxnorm_kb else set()
            )
            validation_result = validator.validate(entities)

            return ExtractionResult(
                text=text,
                entities=entities,
                validation_result=validation_result
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return ExtractionResult(
                text=text,
                entities=[],
                error=str(e)
            )

    def _extract_entities(self, text: str) -> list[Span]:
        """Extract entities using all extractors."""
        spans = []

        # Extract lab tests
        if self.config.extract_labs:
            tests, results = self.lab_extractor.extract_all(text)
            for test in tests:
                spans.append(create_span(
                    text=test.text,
                    position=[test.start, test.end],
                    entity_type="TÊN_XÉT_NGHIỆM",
                    confidence=0.9,
                    source="lab_rule"
                ))
            for result in results:
                spans.append(create_span(
                    text=result.text,
                    position=[result.start, result.end],
                    entity_type="KẾT_QUẢ_XÉT_NGHIỆM",
                    confidence=0.85,
                    source="lab_rule"
                ))

        # Extract drugs
        if self.config.extract_drugs:
            drug_matches = self.drug_extractor.extract(text)
            for drug in drug_matches:
                spans.append(create_span(
                    text=drug.text,
                    position=[drug.start, drug.end],
                    entity_type="THUỐC",
                    confidence=drug.confidence,
                    source="drug_rule"
                ))

        # Extract diseases
        if self.config.extract_diseases or self.config.extract_symptoms:
            disease_matches = self.disease_extractor.extract(text)
            for disease in disease_matches:
                # Use the determined context type
                entity_type = disease.context
                if disease.is_diagnosed:
                    entity_type = "CHẨN_ĐOÁN"

                spans.append(create_span(
                    text=disease.text,
                    position=[disease.start, disease.end],
                    entity_type=entity_type,
                    confidence=disease.confidence,
                    source="disease_rule"
                ))

        return spans

    def _spans_to_entities(self, text: str, spans: list[Span]) -> list[Entity]:
        """Convert spans to Entity objects."""
        entities = []

        for span in spans:
            # Get text from original
            span_text = text[span.start:span.end]

            # Map string type to EntityType
            try:
                entity_type = EntityType(span.entity_type)
            except ValueError:
                entity_type = EntityType.TRIEU_CHUNG

            entity = Entity(
                text=span_text,
                position=[span.start, span.end],
                type=entity_type,
                assertions=[],
                candidates=[]
            )
            entities.append(entity)

        return entities

    def _detect_assertions(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Detect assertions for entities."""
        allowed_types = {EntityType.TRIEU_CHUNG, EntityType.CHAN_DOAN, EntityType.THUOC}

        for entity in entities:
            if entity.type not in allowed_types:
                continue

            assertion_result = self.assertion_detector.detect(
                text, entity.position[0], entity.position[1]
            )

            assertions = []
            if assertion_result.is_negated:
                assertions.append(AssertionType.NEGATED)
            if assertion_result.is_historical:
                assertions.append(AssertionType.HISTORICAL)
            if assertion_result.is_family:
                assertions.append(AssertionType.FAMILY)

            entity.assertions = assertions

        return entities

    def _link_entities(self, entities: list[Entity], text: str) -> list[Entity]:
        """Link entities to ICD/RxNorm."""
        for entity in entities:
            # Link CHẨN_ĐOÁN to ICD-10
            if entity.type == EntityType.CHAN_DOAN and self.config.link_icd:
                if self.icd10_kb:
                    # Search ICD-10 KB
                    results = self.icd10_kb.search(entity.text, limit=5)
                    if results:
                        entity.candidates = [r.code for r in results]

            # Link THUỐC to RxNorm
            elif entity.type == EntityType.THUOC and self.config.link_rxnorm:
                if self.drug_linker:
                    # Link to RxNorm
                    results = self.drug_linker.link(entity.text, limit=5)
                    if results:
                        entity.candidates = [r.rxcui for r, _ in results]

        return entities

    def process_file(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> ExtractionResult:
        """
        Process một file.

        Args:
            input_path: Path to input .txt file
            output_path: Optional path for output .json file

        Returns:
            ExtractionResult
        """
        text = load_text(input_path)
        result = self.process(text)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return result

    def process_directory(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> dict[str, ExtractionResult]:
        """
        Process tất cả files trong directory.

        Args:
            input_dir: Directory chứa input .txt files
            output_dir: Directory cho output .json files

        Returns:
            Dict mapping filename to ExtractionResult
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        texts = load_texts_from_directory(input_path)

        for filename, text in texts.items():
            result = self.process(text)

            # Save output
            output_file = output_path / f"{filename}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

            results[filename] = result

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_medical_entities(text: str) -> list[dict]:
    """
    Convenience function để extract entities từ text.

    Args:
        text: Input text

    Returns:
        List of entity dicts
    """
    pipeline = MedicalOntologyPipeline()
    result = pipeline.process(text)
    return result.to_dict()


def process_medical_file(input_file: str, output_file: str) -> dict:
    """
    Convenience function để process một file.

    Args:
        input_file: Input .txt file
        output_file: Output .json file

    Returns:
        Extraction result as dict
    """
    pipeline = MedicalOntologyPipeline()
    result = pipeline.process_file(input_file, output_file)
    return result.to_dict()


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Medical Ontology Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--icd10", help="Path to ICD-10 KB JSON")
    parser.add_argument("--rxnorm", help="Path to RxNorm KB JSON")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create pipeline
    config = PipelineConfig(
        icd10_kb_path=args.icd10,
        rxnorm_kb_path=args.rxnorm
    )
    pipeline = MedicalOntologyPipeline(config)

    input_path = Path(args.input)

    if input_path.is_file():
        result = pipeline.process_file(args.input, args.output)
        print(f"Processed {args.input}")
        print(f"Found {len(result.entities)} entities")
        if result.validation_result:
            print(f"Validation: {result.validation_result.summary()}")
    elif input_path.is_dir():
        results = pipeline.process_directory(args.input, args.output)
        print(f"Processed {len(results)} files")
    else:
        print(f"Error: {args.input} not found")


if __name__ == "__main__":
    main()
