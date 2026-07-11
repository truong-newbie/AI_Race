"""
NER Prediction Script

Run NER prediction on new text or files.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.entity.labels import NUM_LABELS, ID2LABEL
from src.entity.model_ner import load_ner_model
from src.entity.decoder import NERDecoder
from src.entity.token_alignment import align_labels_to_tokens


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NERPredictor:
    """NER Model Predictor."""

    def __init__(
        self,
        model_path: str,
        model_name: str = "xlm-roberta-base",
        max_length: int = 256,
        confidence_threshold: float = 0.0,
    ):
        """Initialize predictor.

        Args:
            model_path: Path to model checkpoint
            model_name: Base model name
            max_length: Maximum sequence length
            confidence_threshold: Minimum confidence for entities
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Load model
        self.model = load_ner_model(
            model_path=model_path,
            model_name=model_name,
            num_labels=NUM_LABELS,
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize decoder
        self.decoder = NERDecoder(
            id2label=ID2LABEL,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        """Predict entities for a single text.

        Args:
            text: Input text

        Returns:
            List of predicted entities
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].tolist()[0]

        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions = predictions.squeeze(0).cpu().tolist()

        # Decode
        entities = self.decoder.decode(
            text=text,
            predictions=predictions,
            offset_mapping=offset_mapping,
        )

        return entities

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """Predict entities for batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size

        Returns:
            List of entity lists
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            offset_mappings = encodings["offset_mapping"].tolist()

            # Predict
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().tolist()

            # Decode each
            for idx, text in enumerate(batch_texts):
                entities = self.decoder.decode(
                    text=text,
                    predictions=predictions[idx],
                    offset_mapping=offset_mappings[idx],
                )
                results.append(entities)

        return results

    def predict_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Predict entities for texts in a file.

        Args:
            input_path: Path to input file (JSON or JSONL)
            output_path: Optional path to save results

        Returns:
            List of results with text and entities
        """
        # Load input
        with open(input_path, "r", encoding="utf-8") as f:
            if input_path.endswith(".jsonl"):
                samples = [json.loads(line) for line in f]
            else:
                samples = json.load(f)

        results = []

        for sample in tqdm(samples, desc="Processing"):
            text = sample.get("text", sample.get("content", ""))
            sample_id = sample.get("id", sample.get("sample_id", ""))

            entities = self.predict(text)

            results.append({
                "id": sample_id,
                "text": text,
                "entities": entities,
            })

        # Save results
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                if output_path.endswith(".jsonl"):
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results


def format_entities(entities: List[Dict[str, Any]]) -> str:
    """Format entities for display.

    Args:
        entities: List of entity dicts

    Returns:
        Formatted string
    """
    if not entities:
        return "No entities found."

    lines = []
    for i, entity in enumerate(entities, 1):
        text = entity.get("text", "")
        entity_type = entity.get("type", "")
        confidence = entity.get("confidence", 1.0)
        lines.append(f"{i}. [{entity_type}] \"{text}\" (conf={confidence:.2f})")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run NER prediction")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
    parser.add_argument("--text", type=str, default=None,
                        help="Text to predict")
    parser.add_argument("--texts", type=str, nargs="+", default=None,
                        help="Multiple texts to predict")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input file (JSON or JSONL)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Confidence threshold")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Initialize predictor
    predictor = NERPredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        max_length=args.max_length,
        confidence_threshold=args.threshold,
    )

    # Predict
    if args.text:
        # Single text
        entities = predictor.predict(args.text)
        print(f"\nText: {args.text}")
        print(f"\nEntities:\n{format_entities(entities)}")

    elif args.texts:
        # Multiple texts
        results = predictor.predict_batch(args.texts, batch_size=args.batch_size)
        for text, entities in zip(args.texts, results):
            print(f"\nText: {text}")
            print(f"Entities:\n{format_entities(entities)}")
            print("-" * 50)

    elif args.input_file:
        # File input
        results = predictor.predict_file(args.input_file, args.output)

        # Print summary
        for result in results[:5]:  # Show first 5
            print(f"\nText: {result['text'][:100]}...")
            print(f"Entities:\n{format_entities(result['entities'])}")
            print("-" * 50)

        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more samples")

    else:
        # Interactive mode
        print("Interactive NER Prediction")
        print("Enter text to predict (Ctrl+C to exit)\n")

        while True:
            try:
                text = input("Text: ").strip()
                if not text:
                    continue

                entities = predictor.predict(text)
                print(f"\nEntities:\n{format_entities(entities)}\n")

            except KeyboardInterrupt:
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
