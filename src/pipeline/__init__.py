"""Pipeline package — end-to-end medical entity extraction."""

from src.pipeline.config import PipelineConfig
from src.pipeline.factory import ComponentResult
from src.pipeline.pipeline import (
    MedicalOntologyPipeline,
    ExtractionResult,
    ModuleError,
    extract_medical_entities,
    process_file,
)
from src.entity.span_resolver import resolve_spans, create_span

__all__ = [
    "PipelineConfig",
    "ComponentResult",
    "MedicalOntologyPipeline",
    "ExtractionResult",
    "ModuleError",
    "extract_medical_entities",
    "process_file",
    "resolve_spans",
    "create_span",
]
