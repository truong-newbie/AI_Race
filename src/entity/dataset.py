"""
NER Dataset Module

PyTorch Dataset for NER training.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from src.entity.labels import LABEL2ID, ID2LABEL
from src.entity.token_alignment import align_labels_to_tokens


class NERDataset(Dataset):
    """Dataset for NER training/inference.

    Handles loading from JSONL files and tokenization.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        label2id: Dict[str, int] = None,
        transform: Optional[Callable] = None,
    ):
        """Initialize NER dataset.

        Args:
            data_path: Path to JSONL file with samples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label2id: Label to ID mapping (defaults to global LABEL2ID)
            transform: Optional transform function for augmentation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or LABEL2ID
        self.transform = transform

        # Load data
        self.samples = self._load_samples(data_path)

    def _load_samples(self, data_path: str) -> List[Dict[str, Any]]:
        """Load samples from JSONL file."""
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dict with input_ids, attention_mask, labels
        """
        sample = self.samples[idx]

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        # Get text and entities
        text = sample["text"]
        entities = sample.get("entities", [])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Align labels
        token_labels, offset_mapping = align_labels_to_tokens(
            text=text,
            entities=entities,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label2id=self.label2id,
        )

        # Squeeze tensors (remove batch dimension)
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Convert labels to tensor
        labels = torch.tensor(token_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "offset_mapping": encoding["offset_mapping"].squeeze(0).tolist(),
            "text": text,
            "sample_id": sample.get("id", f"sample_{idx}"),
        }

    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Get raw sample without tokenization."""
        return self.samples[idx]


class NERCollator:
    """Collator for NER batches.

    Handles padding and prepares batch data.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, ignore_index: int = -100):
        """Initialize collator.

        Args:
            tokenizer: HuggingFace tokenizer
            ignore_index: Index to ignore in loss computation
        """
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch.

        Args:
            batch: List of samples from NERDataset

        Returns:
            Batched data
        """
        # Stack tensors
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])

        # Collect metadata
        texts = [x["text"] for x in batch]
        sample_ids = [x["sample_id"] for x in batch]
        offset_mappings = [x["offset_mapping"] for x in batch]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "texts": texts,
            "sample_ids": sample_ids,
            "offset_mappings": offset_mappings,
        }


def load_ner_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    batch_size: int = 8,
) -> tuple[NERDataset, NERCollator]:
    """Load NER dataset and return collator.

    Args:
        data_path: Path to JSONL file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size (used for info only)

    Returns:
        Tuple of (dataset, collator)
    """
    dataset = NERDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    collator = NERCollator(tokenizer=tokenizer)

    return dataset, collator


def analyze_dataset(data_path: str) -> Dict[str, Any]:
    """Analyze dataset statistics.

    Args:
        data_path: Path to JSONL file

    Returns:
        Dictionary with dataset statistics
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Count entities by type
    entity_counts = {}
    total_entities = 0
    text_lengths = []
    entity_lengths = []

    for sample in samples:
        text = sample["text"]
        text_lengths.append(len(text))

        entities = sample.get("entities", [])
        for entity in entities:
            entity_type = entity["type"]
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            total_entities += 1

            # Entity text length
            entity_text = entity["text"]
            entity_lengths.append(len(entity_text))

    return {
        "num_samples": len(samples),
        "num_entities": total_entities,
        "entity_counts": entity_counts,
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
        "avg_entity_length": sum(entity_lengths) / len(entity_lengths) if entity_lengths else 0,
        "max_entity_length": max(entity_lengths) if entity_lengths else 0,
    }
