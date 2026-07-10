"""Medical Ontology AI - Core module."""

from src.schema import (
    EntityType,
    AssertionType,
    Entity,
    MedicalDocument,
    validate_span,
    extract_span,
    span_from_match,
)

__all__ = [
    'EntityType',
    'AssertionType',
    'Entity',
    'MedicalDocument',
    'validate_span',
    'extract_span',
    'span_from_match',
]
