"""
Drug Retrieval Evaluator

Evaluation metrics:
- Recall@1/3/5/10
- Top-1 accuracy
- Ingredient accuracy
- Strength accuracy
- Dose-form accuracy
- Combination accuracy
- Error classification
"""

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.linking.rxnorm.schema import get_knowledge_base, RxNormEntry
from src.linking.rxnorm.parser import DrugMentionParser
from src.linking.rxnorm.hybrid_retriever import DrugHybridRetriever
from src.linking.rxnorm.reranker import DrugReranker


@dataclass
class RetrievalError:
    """Mot retrieval error."""
    sample_id: str
    mention: str
    query: str
    gold_rxcui: str
    retrieved_rxcuis: list[str]
    error_type: str
    details: str


@dataclass
class EvaluationMetrics:
    """Evaluation metrics."""
    n_samples: int
    covered: int
    coverage: float
    top1_accuracy: float
    mrr: float
    ingredient_accuracy: float
    strength_accuracy: float
    dose_form_accuracy: float
    recall: dict[str, float]
    error_types: dict[str, int]
    errors: list[RetrievalError] = field(default_factory=list)


class DrugRetrievalEvaluator:
    """
    Evaluator for drug candidate retrieval.

    Loads evaluation samples, runs retrieval, computes metrics,
    classifies errors, and saves reports.
    """

    def __init__(
        self,
        data_path: str = "data/synthetic/rxnorm_linking_samples.jsonl",
        output_dir: str = "outputs",
        top_k: int = 10,
        use_reranker: bool = True,
        use_cross_encoder: bool = False,
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.use_reranker = use_reranker

        self.entries = get_knowledge_base()
        self.parser = DrugMentionParser()
        self.retriever = DrugHybridRetriever(
            entries=self.entries,
            top_k=top_k,
        )
        if use_reranker:
            self.reranker = DrugReranker(
                entries=self.entries,
                use_cross_encoder=use_cross_encoder,
            )
        else:
            self.reranker = None

    def load_samples(self) -> list[dict]:
        """Load evaluation samples from JSONL."""
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def evaluate(self) -> EvaluationMetrics:
        """Run full evaluation."""
        samples = self.load_samples()
        n = len(samples)

        covered = 0
        top1_correct = 0
        ingredient_correct = 0
        strength_correct = 0
        dose_form_correct = 0

        recall_at_k: dict[str, int] = {f"recall@{k}": 0 for k in [1, 3, 5, 10]}
        error_types: dict[str, int] = {}
        errors: list[RetrievalError] = []

        for sample in samples:
            sample_id = sample.get("id", "")
            mention = sample.get("mention", "")
            query = sample.get("query_text", "")
            gold_rxcui = sample.get("positive_rxcui", "")

            results = self.retriever.retrieve(query, mention=mention, top_k=self.top_k)
            if self.reranker and results:
                reranked = self.reranker.rerank(results, query, mention=mention, top_k=self.top_k)
                retrieved = [r.rxcui for r in reranked]
            else:
                retrieved = [r.rxcui for r in results]

            # Coverage (gold in top-k)
            in_top_k = gold_rxcui in retrieved
            if in_top_k:
                covered += 1

            # Top-1 accuracy
            if retrieved and retrieved[0] == gold_rxcui:
                top1_correct += 1

            # Recall@K
            rank = self._get_rank(retrieved, gold_rxcui)
            for k in [1, 3, 5, 10]:
                if rank is not None and rank < k:
                    recall_at_k[f"recall@{k}"] += 1

            # MRR
            if rank is not None and rank < self.top_k:
                pass  # handled separately below

            # Component accuracy
            parsed = self.parser.parse(mention)
            if parsed:
                # Ingredient accuracy: gold entry has matching ingredient
                gold_entry = self._get_entry(gold_rxcui)
                if gold_entry:
                    if parsed.main_ingredient() and gold_entry.ingredient:
                        if parsed.main_ingredient().lower() == gold_entry.ingredient.lower():
                            ingredient_correct += 1

                    # Strength accuracy
                    if parsed.has_strength():
                        strength_val, _ = parsed.main_strength()
                        if gold_entry.strength_value is not None and strength_val is not None:
                            if abs(strength_val - gold_entry.strength_value) < 0.01:
                                strength_correct += 1
                        elif gold_entry.strength_value is None:
                            strength_correct += 1

                    # Dose form accuracy
                    if parsed.dose_form and gold_entry.dose_form:
                        if parsed.dose_form == gold_entry.dose_form:
                            dose_form_correct += 1

            # Error classification
            if not in_top_k:
                error_type = self._classify_error(mention, gold_rxcui, parsed, results)
                error_types[error_type] = error_types.get(error_type, 0) + 1
                errors.append(RetrievalError(
                    sample_id=sample_id,
                    mention=mention,
                    query=query,
                    gold_rxcui=gold_rxcui,
                    retrieved_rxcuis=retrieved[:5],
                    error_type=error_type,
                    details=f"rank={rank}",
                ))

        # Calculate MRR
        mrr = self._calculate_mrr(samples)

        # Build metrics
        metrics = EvaluationMetrics(
            n_samples=n,
            covered=covered,
            coverage=covered / n if n > 0 else 0.0,
            top1_accuracy=top1_correct / n if n > 0 else 0.0,
            mrr=mrr,
            ingredient_accuracy=ingredient_correct / n if n > 0 else 0.0,
            strength_accuracy=strength_correct / n if n > 0 else 0.0,
            dose_form_accuracy=dose_form_correct / n if n > 0 else 0.0,
            recall={
                k: v / n for k, v in recall_at_k.items()
            },
            error_types=error_types,
            errors=errors,
        )

        return metrics

    def _get_rank(self, retrieved: list[str], gold: str) -> Optional[int]:
        """Get rank of gold rxcui in retrieved list."""
        for i, rxcui in enumerate(retrieved):
            if rxcui == gold:
                return i
        return None

    def _get_entry(self, rxcui: str) -> Optional[RxNormEntry]:
        """Get entry by rxcui."""
        for e in self.entries:
            if e.rxcui == rxcui:
                return e
        return None

    def _calculate_mrr(self, samples: list[dict]) -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        for sample in samples:
            mention = sample.get("mention", "")
            query = sample.get("query_text", "")
            gold_rxcui = sample.get("positive_rxcui", "")

            results = self.retriever.retrieve(query, mention=mention, top_k=self.top_k)
            if self.reranker and results:
                reranked = self.reranker.rerank(results, query, mention=mention, top_k=self.top_k)
                retrieved = [r.rxcui for r in reranked]
            else:
                retrieved = [r.rxcui for r in results]

            rank = self._get_rank(retrieved, gold_rxcui)
            if rank is not None:
                reciprocal_ranks.append(1.0 / (rank + 1))
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    def _classify_error(
        self,
        mention: str,
        gold_rxcui: str,
        parsed: Optional,
        results: list,
    ) -> str:
        """Classify the type of retrieval error."""
        gold_entry = self._get_entry(gold_rxcui)

        if not gold_entry:
            return "gold_not_in_kb"

        if not parsed:
            return "parse_failed"

        # Check ingredient match
        if parsed.main_ingredient():
            ing_norm = parsed.main_ingredient().lower().strip()
            if gold_entry.ingredient:
                if ing_norm != gold_entry.ingredient.lower().strip():
                    return "ingredient_mismatch"

        # Check strength match
        if parsed.has_strength():
            strength_val, _ = parsed.main_strength()
            if gold_entry.strength_value is not None and strength_val is not None:
                if abs(strength_val - gold_entry.strength_value) >= 0.01:
                    return "strength_mismatch"
            elif gold_entry.strength_value is not None and strength_val is None:
                return "strength_missing_in_mention"

        # If results exist but gold is not in them
        if results:
            retrieved_rxcuis = [r.rxcui for r in results]
            # Check if similar drugs are retrieved (lexical mismatch)
            return "lexical_mismatch"

        return "no_candidates"

    def print_report(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation report."""
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

        print(f"\n{'='*60}")
        print(f"  Drug Retrieval Evaluation Report")
        print(f"{'='*60}")
        print(f"Dataset: {self.data_path}")
        print(f"Samples: {metrics.n_samples}")
        print(f"Top-K: {self.top_k}")
        print()
        print(f"Coverage:     {metrics.covered}/{metrics.n_samples} ({metrics.coverage:.1%})")
        print(f"Top-1 Acc:   {metrics.top1_accuracy:.4f} ({int(metrics.top1_accuracy * metrics.n_samples)}/{metrics.n_samples})")
        print(f"MRR:          {metrics.mrr:.4f}")
        print()
        print(f"Ingredient Acc:  {metrics.ingredient_accuracy:.4f}")
        print(f"Strength Acc:    {metrics.strength_accuracy:.4f}")
        print(f"Dose Form Acc:   {metrics.dose_form_accuracy:.4f}")
        print()
        print("Recall@K:")
        for k in [1, 3, 5, 10]:
            key = f"recall@{k}"
            val = metrics.recall.get(key, 0.0)
            print(f"  R@{k}: {val:.4f}")
        print()

        if metrics.errors:
            print(f"Errors: {len(metrics.errors)}")
            for err_type, count in sorted(metrics.error_types.items(), key=lambda x: -x[1]):
                print(f"  {err_type}: {count}")
        else:
            print("Errors: 0 (perfect retrieval)")
        print(f"{'='*60}\n")

    def save_report(self, metrics: EvaluationMetrics) -> None:
        """Save metrics as JSON."""
        path = self.output_dir / "rxnorm_retrieval_report.json"
        report = {
            "n_samples": metrics.n_samples,
            "covered": metrics.covered,
            "coverage": round(metrics.coverage, 4),
            "top1_accuracy": round(metrics.top1_accuracy, 4),
            "mrr": round(metrics.mrr, 4),
            "ingredient_accuracy": round(metrics.ingredient_accuracy, 4),
            "strength_accuracy": round(metrics.strength_accuracy, 4),
            "dose_form_accuracy": round(metrics.dose_form_accuracy, 4),
            "recall": {k: round(v, 4) for k, v in metrics.recall.items()},
            "error_types": metrics.error_types,
            "reranking": {
                "enabled": self.use_reranker,
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to {path}")

    def save_errors_csv(self, metrics: EvaluationMetrics) -> None:
        """Save errors as CSV."""
        path = self.output_dir / "rxnorm_retrieval_errors.csv"
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "mention", "query", "gold_rxcui",
                             "retrieved_rxcuis", "error_type", "details"])
            for err in metrics.errors:
                writer.writerow([
                    err.sample_id,
                    err.mention,
                    err.query,
                    err.gold_rxcui,
                    "|".join(err.retrieved_rxcuis),
                    err.error_type,
                    err.details,
                ])
        print(f"Errors saved to {path}")
