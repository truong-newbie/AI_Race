"""
Assertion Ensemble Module

Combines rule-based detection with optional classifier for improved assertion detection.
Ensemble Strategy:
    1. Apply rules (high precision)
    2. Optionally apply classifier (handles complex cases)
    3. Resolve disagreements with confidence-based voting
    4. Log discrepancies for analysis
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from src.assertion.rules import (
    RuleBasedDetector,
    EntityAssertion,
    AssertionStatus,
)

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Strategy for combining rule and classifier."""
    RULE_ONLY = "rule_only"
    CLASSIFIER_ONLY = "classifier_only"
    RULE_THEN_CLASSIFIER = "rule_then_classifier"
    CLASSIFIER_THEN_RULE = "classifier_then_rule"
    VOTING = "voting"
    RULE_OVERRIDE = "rule_override"


@dataclass
class AssertionConfig:
    """Configuration for assertion detection ensemble."""
    strategy: EnsembleStrategy = EnsembleStrategy.RULE_ONLY
    rule_precision_override: bool = True  # High-precision rules override classifier
    min_confidence_threshold: float = 0.5
    disagreement_logging: bool = True
    negation_threshold: float = 0.5
    historical_threshold: float = 0.5
    family_threshold: float = 0.5


@dataclass
class ClassifierPrediction:
    """Prediction from classifier."""
    p_negated: float
    p_historical: float
    p_family: float
    confidence: float


@dataclass
class AssertionResult:
    """Final assertion result after ensemble processing."""
    entity_text: str
    entity_start: int
    entity_end: int
    entity_type: Optional[str]
    is_negated: bool
    is_historical: bool
    is_family: bool
    confidence: float
    source: str  # "rule", "classifier", "ensemble"
    rule_confidence: Optional[float] = None
    classifier_confidence: Optional[ClassifierPrediction] = None
    disagreement: bool = False
    cues_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        result = {
            "entity_text": self.entity_text,
            "entity_start": self.entity_start,
            "entity_end": self.entity_end,
            "entity_type": self.entity_type,
            "is_negated": self.is_negated,
            "is_historical": self.is_historical,
            "is_family": self.is_family,
            "confidence": self.confidence,
            "source": self.source,
        }
        if self.classifier_confidence:
            result["p_negated"] = self.classifier_confidence.p_negated
            result["p_historical"] = self.classifier_confidence.p_historical
            result["p_family"] = self.classifier_confidence.p_family
        return result

    def to_list(self) -> List[str]:
        """Convert assertions to list format."""
        result = []
        if self.is_negated:
            result.append("isNegated")
        if self.is_family:
            result.append("isFamily")
        if self.is_historical:
            result.append("isHistorical")
        return result


class AssertionClassifier:
    """
    Placeholder for XLM-R multi-label classifier.

    In production, this would load a fine-tuned model.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self._loaded = False

    def load(self):
        """Load the classifier model."""
        if self.model_path is None:
            logger.warning("No model path provided, classifier not loaded")
            return

        # Placeholder for actual model loading
        # from transformers import AutoModelForSequenceClassification
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._loaded = True

    def predict(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        entity_type: Optional[str] = None
    ) -> ClassifierPrediction:
        """
        Predict assertion status using classifier.

        Args:
            text: Full text
            entity_start: Entity start position
            entity_end: Entity end position
            entity_type: Entity type (optional)

        Returns:
            ClassifierPrediction with probabilities
        """
        if not self._loaded:
            # Return default prediction if model not loaded
            return ClassifierPrediction(
                p_negated=0.5,
                p_historical=0.5,
                p_family=0.5,
                confidence=0.0
            )

        # Placeholder: implement actual prediction
        # Input: sentence with entity marker
        # Bệnh nhân [E]ho[/E] nhưng đau ngực.
        # Output: p_negated, p_historical, p_family

        return ClassifierPrediction(
            p_negated=0.5,
            p_historical=0.5,
            p_family=0.5,
            confidence=0.0
        )

    def predict_batch(
        self,
        texts: List[str],
        entities: List[dict]
    ) -> List[ClassifierPrediction]:
        """Batch prediction."""
        return [
            self.predict(
                texts[i],
                entities[i].get("start", 0),
                entities[i].get("end", 0),
                entities[i].get("type")
            )
            for i in range(len(entities))
        ]


class AssertionEnsemble:
    """
    Ensemble assertion detector combining rules and optional classifier.

    Architecture:
        1. Rule cue detector (high precision)
        2. Optional XLM-R multi-label classifier
        3. Ensemble resolution with configurable strategy
    """

    def __init__(
        self,
        config: Optional[AssertionConfig] = None,
        classifier: Optional[AssertionClassifier] = None
    ):
        self.config = config or AssertionConfig()
        self.rule_detector = RuleBasedDetector()
        self.classifier = classifier
        self._disagreement_log = []

    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
        entity_type: Optional[str] = None
    ) -> AssertionResult:
        """
        Detect assertions for a single entity.

        Args:
            text: Full text
            entity_start: Entity start position
            entity_end: Entity end position
            entity_type: Entity type

        Returns:
            AssertionResult with ensemble prediction
        """
        entity_text = text[entity_start:entity_end]

        # Get rule-based prediction
        rule_result = self.rule_detector.detect(text, entity_start, entity_end, entity_type)
        rule_status = rule_result.status

        # Initialize result
        if self.config.strategy == EnsembleStrategy.RULE_ONLY:
            return self._rule_only_result(rule_result, entity_text)

        # Get classifier prediction if available
        classifier_pred = None
        if self.classifier and self.classifier._loaded:
            classifier_pred = self.classifier.predict(text, entity_start, entity_end, entity_type)

        # Combine predictions based on strategy
        if self.config.strategy == EnsembleStrategy.RULE_THEN_CLASSIFIER:
            return self._rule_then_classifier(
                rule_result, classifier_pred, entity_text, entity_type
            )
        elif self.config.strategy == EnsembleStrategy.RULE_OVERRIDE:
            return self._rule_override(
                rule_result, classifier_pred, entity_text, entity_type
            )
        elif self.config.strategy == EnsembleStrategy.VOTING:
            return self._voting(
                rule_result, classifier_pred, entity_text, entity_type
            )
        else:
            return self._rule_only_result(rule_result, entity_text)

    def detect_all(
        self,
        text: str,
        entities: List[dict]
    ) -> List[AssertionResult]:
        """
        Detect assertions for multiple entities.

        Args:
            text: Full text
            entities: List of entity dicts

        Returns:
            List of AssertionResult
        """
        results = []
        for entity in entities:
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            entity_type = entity.get("type")

            result = self.detect(text, start, end, entity_type)
            results.append(result)

        return results

    def _rule_only_result(
        self,
        rule_result: EntityAssertion,
        entity_text: str
    ) -> AssertionResult:
        """Build result from rule only."""
        return AssertionResult(
            entity_text=entity_text,
            entity_start=rule_result.entity_start,
            entity_end=rule_result.entity_end,
            entity_type=rule_result.entity_type,
            is_negated=rule_result.status.is_negated,
            is_historical=rule_result.status.is_historical,
            is_family=rule_result.status.is_family,
            confidence=rule_result.status.confidence,
            source="rule",
            rule_confidence=rule_result.status.confidence,
            cues_used=rule_result.status.cues_used
        )

    def _rule_then_classifier(
        self,
        rule_result: EntityAssertion,
        classifier_pred: Optional[ClassifierPrediction],
        entity_text: str,
        entity_type: Optional[str]
    ) -> AssertionResult:
        """Rule first, classifier fills gaps."""
        if classifier_pred is None:
            return self._rule_only_result(rule_result, entity_text)

        # If rule has high confidence, use it
        if rule_result.status.confidence > self.config.min_confidence_threshold:
            return self._rule_only_result(rule_result, entity_text)

        # Otherwise, use classifier
        return self._classifier_to_result(classifier_pred, rule_result, entity_text)

    def _rule_override(
        self,
        rule_result: EntityAssertion,
        classifier_pred: Optional[ClassifierPrediction],
        entity_text: str,
        entity_type: Optional[str]
    ) -> AssertionResult:
        """
        Rule has high precision - override classifier when rule is confident.
        """
        if classifier_pred is None:
            return self._rule_only_result(rule_result, entity_text)

        # Check for disagreement
        disagreement = self._check_disagreement(
            rule_result.status, classifier_pred
        )

        if disagreement and self.config.disagreement_logging:
            self._log_disagreement(rule_result, classifier_pred, entity_text)

        # Rule precision override
        if self.config.rule_precision_override:
            if rule_result.status.is_negated and classifier_pred.p_negated < self.config.negation_threshold:
                return self._rule_only_result(rule_result, entity_text)

            if rule_result.status.is_historical and classifier_pred.p_historical < self.config.historical_threshold:
                return self._rule_only_result(rule_result, entity_text)

            if rule_result.status.is_family and classifier_pred.p_family < self.config.family_threshold:
                return self._rule_only_result(rule_result, entity_text)

        # Otherwise use voting
        return self._voting(rule_result, classifier_pred, entity_text, entity_type)

    def _voting(
        self,
        rule_result: EntityAssertion,
        classifier_pred: Optional[ClassifierPrediction],
        entity_text: str,
        entity_type: Optional[str]
    ) -> AssertionResult:
        """Combine rule and classifier via voting."""
        if classifier_pred is None:
            return self._rule_only_result(rule_result, entity_text)

        # Negation
        is_negated = (
            rule_result.status.is_negated or
            classifier_pred.p_negated >= self.config.negation_threshold
        )

        # Historical
        is_historical = (
            rule_result.status.is_historical or
            classifier_pred.p_historical >= self.config.historical_threshold
        )

        # Family
        is_family = (
            rule_result.status.is_family or
            classifier_pred.p_family >= self.config.family_threshold
        )

        # Confidence is average
        confidence = (rule_result.status.confidence + classifier_pred.confidence) / 2

        return AssertionResult(
            entity_text=entity_text,
            entity_start=rule_result.entity_start,
            entity_end=rule_result.entity_end,
            entity_type=entity_type,
            is_negated=is_negated,
            is_historical=is_historical,
            is_family=is_family,
            confidence=confidence,
            source="ensemble",
            rule_confidence=rule_result.status.confidence,
            classifier_confidence=classifier_pred,
            disagreement=self._check_disagreement(rule_result.status, classifier_pred),
            cues_used=rule_result.status.cues_used
        )

    def _classifier_to_result(
        self,
        classifier_pred: ClassifierPrediction,
        rule_result: EntityAssertion,
        entity_text: str
    ) -> AssertionResult:
        """Convert classifier prediction to result."""
        return AssertionResult(
            entity_text=entity_text,
            entity_start=rule_result.entity_start,
            entity_end=rule_result.entity_end,
            entity_type=rule_result.entity_type,
            is_negated=classifier_pred.p_negated >= self.config.negation_threshold,
            is_historical=classifier_pred.p_historical >= self.config.historical_threshold,
            is_family=classifier_pred.p_family >= self.config.family_threshold,
            confidence=classifier_pred.confidence,
            source="classifier",
            classifier_confidence=classifier_pred
        )

    def _check_disagreement(
        self,
        rule_status: AssertionStatus,
        classifier_pred: ClassifierPrediction
    ) -> bool:
        """Check if rule and classifier disagree."""
        if rule_status.is_negated and classifier_pred.p_negated < self.config.negation_threshold:
            return True
        if rule_status.is_historical and classifier_pred.p_historical < self.config.historical_threshold:
            return True
        if rule_status.is_family and classifier_pred.p_family < self.config.family_threshold:
            return True
        return False

    def _log_disagreement(
        self,
        rule_result: EntityAssertion,
        classifier_pred: ClassifierPrediction,
        entity_text: str
    ):
        """Log disagreement between rule and classifier."""
        entry = {
            "entity_text": entity_text,
            "rule_is_negated": rule_result.status.is_negated,
            "rule_is_historical": rule_result.status.is_historical,
            "rule_is_family": rule_result.status.is_family,
            "rule_confidence": rule_result.status.confidence,
            "classifier_p_negated": classifier_pred.p_negated,
            "classifier_p_historical": classifier_pred.p_historical,
            "classifier_p_family": classifier_pred.p_family,
        }
        self._disagreement_log.append(entry)
        logger.debug(f"Disagreement: {entry}")

    def get_disagreement_log(self) -> List[dict]:
        """Get disagreement log."""
        return self._disagreement_log

    def clear_disagreement_log(self):
        """Clear disagreement log."""
        self._disagreement_log = []


# Convenience functions
def create_ensemble(
    strategy: EnsembleStrategy = EnsembleStrategy.RULE_ONLY,
    classifier_path: Optional[str] = None
) -> AssertionEnsemble:
    """Create an assertion ensemble."""
    config = AssertionConfig(strategy=strategy)
    classifier = None

    if classifier_path:
        classifier = AssertionClassifier(classifier_path)
        classifier.load()

    return AssertionEnsemble(config=config, classifier=classifier)
