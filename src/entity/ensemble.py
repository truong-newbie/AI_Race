"""
Entity Ensemble

Combines entities from regex, dictionary, and NER model extractors.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from src.entity.confidence import ConfidenceConfig, EntitySource, get_source_base_confidence
from src.entity.conflict_logger import ConflictLogger
from src.entity.resolver import EntityResolver


@dataclass
class ExtractionSource:
    """A single extraction source."""
    name: str
    source_type: EntitySource
    extractor: Optional[Callable] = None

    def extract(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Extract entities from text."""
        if self.extractor:
            return self.extractor(text, **kwargs)
        return []


class EntityEnsemble:
    """Ensemble entity extractor combining multiple sources."""

    def __init__(
        self,
        config: Optional[ConfidenceConfig] = None,
        conflict_logger: Optional[ConflictLogger] = None,
        enable_regex: bool = True,
        enable_dictionary: bool = True,
        enable_ner: bool = True,
    ):
        """Initialize ensemble.

        Args:
            config: Confidence configuration
            conflict_logger: Conflict logger
            enable_regex: Enable regex extraction
            enable_dictionary: Enable dictionary extraction
            enable_ner: Enable NER model extraction
        """
        self.config = config or ConfidenceConfig()
        self.conflict_logger = conflict_logger or ConflictLogger(enable_logging=False)
        self.resolver = EntityResolver(
            config=self.config,
            conflict_logger=self.conflict_logger,
        )

        self.enable_regex = enable_regex
        self.enable_dictionary = enable_dictionary
        self.enable_ner = enable_ner

        # Source extractors (will be set by user)
        self.regex_extractor: Optional[Callable] = None
        self.dictionary_extractor: Optional[Callable] = None
        self.ner_extractor: Optional[Callable] = None

    def set_extractors(
        self,
        regex_extractor: Optional[Callable] = None,
        dictionary_extractor: Optional[Callable] = None,
        ner_extractor: Optional[Callable] = None,
    ):
        """Set extractor functions.

        Each extractor should be a callable that takes (text, **kwargs)
        and returns a list of entity dicts with: text, start, end, type, confidence
        """
        self.regex_extractor = regex_extractor
        self.dictionary_extractor = dictionary_extractor
        self.ner_extractor = ner_extractor

    def extract(
        self,
        text: str,
        section: Optional[str] = None,
        apply_threshold: bool = True,
    ) -> List[Dict[str, Any]]:
        """Extract entities using all enabled sources.

        Args:
            text: Input text
            section: Optional section context
            apply_threshold: Whether to apply confidence thresholds

        Returns:
            List of merged and resolved entities
        """
        all_entities: List[Dict[str, Any]] = []

        # Extract from regex
        if self.enable_regex and self.regex_extractor:
            regex_entities = self._extract_with_source(
                self.regex_extractor,
                text,
                EntitySource.REGEX,
                apply_threshold,
            )
            all_entities.extend(regex_entities)

        # Extract from dictionary
        if self.enable_dictionary and self.dictionary_extractor:
            dict_entities = self._extract_with_source(
                self.dictionary_extractor,
                text,
                EntitySource.DICTIONARY,
                apply_threshold,
            )
            all_entities.extend(dict_entities)

        # Extract from NER model
        if self.enable_ner and self.ner_extractor:
            ner_entities = self._extract_with_source(
                self.ner_extractor,
                text,
                EntitySource.NER_MODEL,
                apply_threshold,
            )
            all_entities.extend(ner_entities)

        # Resolve and merge
        resolved = self.resolver.resolve_entities(
            all_entities,
            original_text=text,
            section=section,
        )

        return resolved

    def _extract_with_source(
        self,
        extractor: Callable,
        text: str,
        source_type: EntitySource,
        apply_threshold: bool,
    ) -> List[Dict[str, Any]]:
        """Extract entities and add source info."""
        try:
            entities = extractor(text)
        except Exception as e:
            print(f"Extractor error ({source_type}): {e}")
            return []

        base_conf = get_source_base_confidence(source_type, self.config)

        for entity in entities:
            # Add source info
            entity["source"] = source_type.value

            # Set base confidence if not set
            if "confidence" not in entity:
                entity["confidence"] = base_conf
            elif apply_threshold:
                # Apply threshold
                if entity["confidence"] < self.config.ner_threshold:
                    # Adjust based on source
                    threshold = self.config.regex_confidence if source_type == EntitySource.REGEX else \
                               self.config.dictionary_confidence if source_type == EntitySource.DICTIONARY else \
                               self.config.ner_threshold
                    if entity["confidence"] < threshold:
                        entity["confidence"] = threshold

            # Initialize source_scores
            entity["source_scores"] = {source_type.value: entity["confidence"]}

        return entities

    def extract_single_source(
        self,
        text: str,
        source: EntitySource,
    ) -> List[Dict[str, Any]]:
        """Extract using only a single source.

        Args:
            text: Input text
            source: Source type to use

        Returns:
            List of entities from that source only
        """
        if source == EntitySource.REGEX and self.regex_extractor:
            return self._extract_with_source(
                self.regex_extractor, text, source, True
            )
        elif source == EntitySource.DICTIONARY and self.dictionary_extractor:
            return self._extract_with_source(
                self.dictionary_extractor, text, source, True
            )
        elif source == EntitySource.NER_MODEL and self.ner_extractor:
            return self._extract_with_source(
                self.ner_extractor, text, source, True
            )
        return []

    def get_conflict_report(self) -> Dict[str, Any]:
        """Get conflict report."""
        return self.conflict_logger.get_report().to_dict()


class SimpleEnsemble:
    """Simplified ensemble without complex resolver (union-based)."""

    def __init__(self, config: Optional[ConfidenceConfig] = None):
        self.config = config or ConfidenceConfig()

    def merge_entities(
        self,
        entities_list: List[List[Dict[str, Any]]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Simple union merge.

        Keep all unique entities, merge exact duplicates.
        """
        all_entities = []
        for entities in entities_list:
            all_entities.extend(entities)

        # Group by span
        span_map: Dict[tuple, List[Dict[str, Any]]] = {}
        for entity in all_entities:
            span = (entity.get("start", 0), entity.get("end", 0))
            if span not in span_map:
                span_map[span] = []
            span_map[span].append(entity)

        # Merge each group
        merged = []
        for span, group in span_map.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_group(group, original_text))

        return merged

    def _merge_group(
        self,
        group: List[Dict[str, Any]],
        original_text: str,
    ) -> Dict[str, Any]:
        """Merge a group of entities at the same span."""
        span = (group[0].get("start", 0), group[0].get("end", 0))

        # Use text from original
        text = original_text[span[0]:span[1]]

        # Use highest confidence
        best = max(group, key=lambda e: e.get("confidence", 0))

        # Collect all sources
        sources = []
        for e in group:
            src = e.get("source", "unknown")
            if src not in sources:
                sources.append(src)

        # Average confidence with bonus
        avg_conf = sum(e.get("confidence", 0) for e in group) / len(group)
        if len(group) > 1:
            avg_conf = min(avg_conf + self.config.agreement_bonus, 1.0)

        return {
            "text": text,
            "start": span[0],
            "end": span[1],
            "type": best.get("type"),
            "confidence": avg_conf,
            "source": ",".join(sources),
            "source_scores": {e.get("source", "unknown"): e.get("confidence", 0) for e in group},
        }


def create_baseline_extractor() -> Callable:
    """Create a simple baseline extractor using existing rules."""
    from src.entity.lab_extractor import LabExtractor
    from src.entity.drug_extractor import DrugExtractor

    lab_ext = LabExtractor()
    drug_ext = DrugExtractor()

    def extract(text: str) -> List[Dict[str, Any]]:
        entities = []

        # Lab tests and results
        for match in lab_ext.extract_lab_tests(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "TÊN_XÉT_NGHIỆM",
                "confidence": 0.85,
            })

        for match in lab_ext.extract_lab_results(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                "confidence": 0.85,
            })

        # Drugs
        for match in drug_ext.extract_drugs(text):
            entities.append({
                "text": match.get("text", ""),
                "start": match.get("start", 0),
                "end": match.get("end", 0),
                "type": "THUỐC",
                "confidence": 0.90,
            })

        return entities

    return extract


def create_ensemble_with_baseline() -> EntityEnsemble:
    """Create ensemble with baseline extractors."""
    ensemble = EntityEnsemble()
    ensemble.set_extractors(
        regex_extractor=create_baseline_extractor(),
    )
    return ensemble
