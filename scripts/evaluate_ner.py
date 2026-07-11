"""
NER Evaluation Script

Evaluate trained NER model on test set.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity.labels import NUM_LABELS, ID2LABEL, LABEL2ID
from src.entity.dataset import NERDataset, NERCollator
from src.entity.model_ner import XLMRobertaForNER, load_ner_model
from src.entity.decoder import NERDecoder
from src.entity.metrics import (
    compute_entity_metrics,
    detailed_error_analysis,
    print_metrics_report,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NEREvaluator:
    """NER Model Evaluator."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        data_path: str,
        config: Dict[str, Any],
        batch_size: int = 16,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to model checkpoint
            model_name: Base model name for loading
            data_path: Path to evaluation data
            config: Configuration dict
            batch_size: Evaluation batch size
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load dataset
        self.dataset = NERDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=config.get("max_length", 256),
        )

        # Load model
        self.model = load_ner_model(
            model_path=model_path,
            model_name=model_name,
            num_labels=NUM_LABELS,
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize decoder
        self.decoder = NERDecoder(id2label=ID2LABEL)

        # Collator
        self.collator = NERCollator(tokenizer=self.tokenizer)
        self.batch_size = batch_size

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation.

        Returns:
            Evaluation metrics
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        all_true_entities = []
        all_pred_entities = []

        logger.info("Running evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )

                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)

                # Decode each sample
                for idx in range(len(batch["texts"])):
                    text = batch["texts"][idx]
                    offset_mapping = batch["offset_mappings"][idx]
                    pred = predictions[idx].cpu().tolist()

                    # Decode predictions to entities
                    pred_entities = self.decoder.decode(
                        text=text,
                        predictions=pred,
                        offset_mapping=offset_mapping,
                    )

                    # Get true entities from labels
                    labels = batch["labels"][idx].tolist()
                    true_entities = self._labels_to_entities(
                        text=text,
                        labels=labels,
                        offset_mapping=offset_mapping,
                    )

                    all_pred_entities.extend(pred_entities)
                    all_true_entities.extend(true_entities)

        # Compute metrics
        metrics = compute_entity_metrics(all_true_entities, all_pred_entities)

        # Detailed error analysis
        if metrics["num_pred"] > 0 or metrics["num_true"] > 0:
            error_analysis = detailed_error_analysis(
                all_true_entities,
                all_pred_entities,
            )
            metrics["error_analysis"] = error_analysis

        return metrics

    def _labels_to_entities(
        self,
        text: str,
        labels: List[int],
        offset_mapping: List,
    ) -> List[Dict[str, Any]]:
        """Convert token labels to entities.

        Args:
            text: Original text
            labels: Token labels
            offset_mapping: Token offset mapping

        Returns:
            List of entity dicts
        """
        from src.entity.token_alignment import decode_token_labels_to_entities

        return decode_token_labels_to_entities(
            text=text,
            token_labels=labels,
            offset_mapping=offset_mapping,
            id2label=ID2LABEL,
        )

    def predict_file(
        self,
        output_path: str,
    ):
        """Predict entities for all samples and save to file.

        Args:
            output_path: Path to save predictions
        """
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        results = []

        logger.info("Predicting...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )

                predictions = torch.argmax(outputs.logits, dim=-1)

                for idx in range(len(batch["texts"])):
                    text = batch["texts"][idx]
                    offset_mapping = batch["offset_mappings"][idx]
                    pred = predictions[idx].cpu().tolist()
                    sample_id = batch["sample_ids"][idx]

                    # Decode
                    entities = self.decoder.decode(
                        text=text,
                        predictions=pred,
                        offset_mapping=offset_mapping,
                    )

                    results.append({
                        "id": sample_id,
                        "text": text,
                        "entities": entities,
                    })

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Predictions saved to {output_path}")

        return results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER model")
    parser.add_argument("--config", type=str, default="configs/ner_xlmr_base.yaml")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to test data (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override data path if provided
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = config.get("test_path", config.get("dev_path"))

    # Initialize evaluator
    evaluator = NEREvaluator(
        model_path=args.model_path,
        model_name=config["model_name"],
        data_path=data_path,
        config=config,
        batch_size=args.batch_size,
    )

    # Run evaluation
    logger.info(f"Evaluating on: {data_path}")
    metrics = evaluator.evaluate()

    # Print report
    report = print_metrics_report(metrics)
    print("\n" + report)

    # Save metrics
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Add predictions if requested
        if args.output.endswith(".json"):
            evaluator.predict_file(args.output)
        else:
            # Save just metrics
            metrics_path = args.output if args.output.endswith(".json") else f"{args.output}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
    else:
        # Save metrics to default location
        output_dir = config.get("output_dir", "outputs/ner_xlmr_base")
        os.makedirs(output_dir, exist_ok=True)

        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
