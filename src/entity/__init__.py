"""
NER Entity Module

Fine-tuning XLM-RoBERTa for Vietnamese Medical NER.
"""

from src.entity.labels import (
    NER_ENTITY_TYPES,
    LABEL_LIST,
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    get_entity_type_from_label,
    is_beginning_label,
    is_inside_label,
    is_entity_label,
)

from src.entity.token_alignment import (
    align_labels_to_tokens,
    align_character_spans_to_tokens,
    decode_token_labels_to_entities,
)

from src.entity.decoder import (
    NERDecoder,
    predictions_to_entities,
    extract_entities_from_model_output,
)

from src.entity.model_ner import (
    XLMRobertaForNER,
    create_ner_model,
    load_ner_model,
)

from src.entity.dataset import (
    NERDataset,
    NERCollator,
    load_ner_dataset,
    analyze_dataset,
)

from src.entity.metrics import (
    compute_entity_metrics,
    compute_per_class_f1,
    analyze_errors,
    detailed_error_analysis,
    print_metrics_report,
)

__all__ = [
    # Labels
    "NER_ENTITY_TYPES",
    "LABEL_LIST",
    "LABEL2ID",
    "ID2LABEL",
    "NUM_LABELS",
    "get_entity_type_from_label",
    "is_beginning_label",
    "is_inside_label",
    "is_entity_label",
    # Token alignment
    "align_labels_to_tokens",
    "align_character_spans_to_tokens",
    "decode_token_labels_to_entities",
    # Decoder
    "NERDecoder",
    "predictions_to_entities",
    "extract_entities_from_model_output",
    # Model
    "XLMRobertaForNER",
    "create_ner_model",
    "load_ner_model",
    # Dataset
    "NERDataset",
    "NERCollator",
    "load_ner_dataset",
    "analyze_dataset",
    # Metrics
    "compute_entity_metrics",
    "compute_per_class_f1",
    "analyze_errors",
    "detailed_error_analysis",
    "print_metrics_report",
]
