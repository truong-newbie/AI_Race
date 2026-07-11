"""
NER Training Script

Train XLM-RoBERTa for Named Entity Recognition.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.entity.labels import NUM_LABELS, LABEL2ID, ID2LABEL
from src.entity.dataset import NERDataset, NERCollator
from src.entity.model_ner import XLMRobertaForNER
from src.entity.decoder import NERDecoder
from src.entity.token_alignment import decode_token_labels_to_entities


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NERTrainer:
    """NER Trainer class."""

    def __init__(
        self,
        model_name: str,
        train_path: str,
        dev_path: str,
        config: Dict[str, Any],
    ):
        """Initialize trainer.

        Args:
            model_name: Pretrained model name
            train_path: Path to training data
            dev_path: Path to dev data
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Initialize datasets
        self.train_dataset = NERDataset(
            data_path=train_path,
            tokenizer=self.tokenizer,
            max_length=config.get("max_length", 256),
        )

        self.dev_dataset = NERDataset(
            data_path=dev_path,
            tokenizer=self.tokenizer,
            max_length=config.get("max_length", 256),
        )

        # Initialize collator
        self.collator = NERCollator(tokenizer=self.tokenizer)

        # Initialize model
        self.model = XLMRobertaForNER(
            model_name=model_name,
            num_labels=NUM_LABELS,
        )
        self.model.to(self.device)

        # Initialize decoder
        self.decoder = NERDecoder(id2label=ID2LABEL)

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Best metric tracking
        self.best_metric = 0.0
        self.patience_counter = 0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
            eps=self.config.get("adam_epsilon", 1e-8),
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        total_steps = len(self.train_dataset) * self.config.get("epochs", 10)
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))

        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train(self) -> Dict[str, Any]:
        """Train the model."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("train_batch_size", 16),
            shuffle=True,
            collate_fn=self.collator,
        )

        epochs = self.config.get("epochs", 10)
        gradient_accumulation = self.config.get("gradient_accumulation_steps", 1)

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training
            self.model.train()
            total_loss = 0

            progress = tqdm(train_loader, desc="Training")
            for step, batch in enumerate(progress):
                # Forward
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )

                loss = outputs.loss / gradient_accumulation
                loss.backward()
                total_loss += loss.item() * gradient_accumulation

                # Update weights
                if (step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                progress.set_postfix(loss=f"{loss.item() * gradient_accumulation:.4f}")

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Training loss: {avg_loss:.4f}")

            # Evaluation
            metrics = self.evaluate()

            # Log metrics
            logger.info(f"Dev metrics: {metrics}")

            # Check for improvement
            current_metric = metrics.get("entity_f1", 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint("best_model")
                logger.info(f"New best model! F1: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}")

            # Early stopping
            patience = self.config.get("early_stopping_patience", 3)
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return {"best_metric": self.best_metric}

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate on dev set."""
        self.model.eval()

        eval_loader = DataLoader(
            self.dev_dataset,
            batch_size=self.config.get("eval_batch_size", 16),
            shuffle=False,
            collate_fn=self.collator,
        )

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                )

                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)

                all_predictions.append({
                    "predictions": predictions.cpu(),
                    "labels": batch["labels"],
                    "texts": batch["texts"],
                    "offset_mappings": batch["offset_mappings"],
                })

        # Compute metrics
        metrics = self._compute_metrics(all_predictions)
        return metrics

    def _compute_metrics(self, predictions_data: list) -> Dict[str, Any]:
        """Compute entity-level metrics."""
        from src.entity.metrics import compute_entity_metrics

        # Collect all true and pred entities
        all_true_entities = []
        all_pred_entities = []

        for batch_data in predictions_data:
            preds = batch_data["predictions"]
            labels = batch_data["labels"]
            texts = batch_data["texts"]
            offset_mappings = batch_data["offset_mappings"]

            for idx in range(len(texts)):
                text = texts[idx]
                offset = offset_mappings[idx]

                # Decode predictions
                pred_entities = self.decoder.decode(
                    text=text,
                    predictions=preds[idx].tolist(),
                    offset_mapping=offset,
                )

                # Get true entities from labels
                true_entities = decode_token_labels_to_entities(
                    text=text,
                    token_labels=labels[idx].tolist(),
                    offset_mapping=offset,
                    id2label=ID2LABEL,
                )

                all_pred_entities.extend(pred_entities)
                all_true_entities.extend(true_entities)

        # Compute metrics
        metrics = compute_entity_metrics(all_true_entities, all_pred_entities)
        return metrics

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        output_dir = self.config.get("output_dir", "outputs/ner_xlmr_base")
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = os.path.join(output_dir, f"{name}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "best_metric": self.best_metric,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model loaded from {path}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--config", type=str, default="configs/ner_xlmr_base.yaml")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.model_name:
        config["model_name"] = args.model_name
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch_size:
        config["train_batch_size"] = args.batch_size
        config["eval_batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.seed:
        config["seed"] = args.seed

    # Set seed
    set_seed(config.get("seed", 42))

    # Create output directory
    os.makedirs(config.get("output_dir", "outputs/ner_xlmr_base"), exist_ok=True)

    # Initialize trainer
    trainer = NERTrainer(
        model_name=config["model_name"],
        train_path=config["train_path"],
        dev_path=config["dev_path"],
        config=config,
    )

    # Train
    logger.info("Starting training...")
    results = trainer.train()

    logger.info(f"Training complete! Best F1: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()
