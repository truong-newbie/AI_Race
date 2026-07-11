"""
ICD-10 Retrieval Evaluator

Metrics:
  - Recall@1, @3, @5, @10, @20
  - MRR (Mean Reciprocal Rank)
  - Candidate coverage
  - Error analysis classification

Output:
  - Recall@K report
  - retrieval_errors.csv
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.linking.icd.schema import get_knowledge_base, ICD10Entry
from src.linking.icd.hybrid_retriever import HybridRetriever, MergeConfig, CandidateResult


# --- Error Classification ---

ERROR_LABELS = [
    "exact_alias_missing",    # gold alias not in alias index
    "synonym_missing",         # gold synonym not in index
    "translation_issue",      # EN/VI mismatch
    "abbreviation_issue",     # abbreviation not expanded
    "wrong_granularity",      # parent/child code mismatch
    "dense_mismatch",         # dense embedding failed to retrieve
    "lexical_mismatch",       # mention too different from KB text
]


def classify_retrieval_error(
    gold_code: str,
    retrieved_codes: list[str],
    entry: ICD10Entry,
    mention: str,
) -> str:
    """
    Classify why the gold code was missed.

    Returns one of ERROR_LABELS or "no_error" if found.
    """
    norm_mention = mention.lower().strip()

    # 1. exact_alias_missing: mention text exactly matches an alias
    for alias in entry.aliases:
        if alias.lower().strip() == norm_mention:
            if gold_code not in retrieved_codes:
                return "exact_alias_missing"

    # 2. synonym_missing: mention matches a synonym
    for syn in entry.synonyms:
        if syn.lower().strip() == norm_mention:
            if gold_code not in retrieved_codes:
                return "synonym_missing"

    # 3. translation_issue: mention is in one language, KB name in another
    if entry.name_vi and entry.name_en:
        # Both exist but mention matches neither
        if gold_code not in retrieved_codes:
            return "translation_issue"

    # 4. abbreviation_issue: mention is an abbreviation
    abbrev_pattern = re.match(r"^[A-Z]{2,}$", mention)
    if abbrev_pattern and gold_code not in retrieved_codes:
        return "abbreviation_issue"

    # 5. wrong_granularity: parent/child mismatch
    if gold_code not in retrieved_codes:
        # Check if a parent or child code was retrieved
        for retrieved in retrieved_codes[:5]:
            if retrieved.startswith(gold_code[:3]) or gold_code.startswith(retrieved[:3]):
                return "wrong_granularity"

    # 6. dense_mismatch: mention was likely semantic (not lexical)
    if gold_code not in retrieved_codes:
        return "dense_mismatch"

    return "lexical_mismatch"


# --- Main Evaluator ---

class ICDRetrievalEvaluator:
    """
    Evaluate ICD-10 candidate retrieval.

    Loads data from `data/synthetic/icd_linking_samples.jsonl`
    and computes Recall@K, MRR, coverage, and error analysis.
    """

    def __init__(
        self,
        data_path: str = "data/synthetic/icd_linking_samples.jsonl",
        output_dir: str = "outputs",
        top_k: int = 20,
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k

        self.retriever = self._build_retriever()
        self.entries_by_code: dict[str, ICD10Entry] = {
            e.code: e for e in get_knowledge_base()
        }

    def _build_retriever(self) -> HybridRetriever:
        """Build the hybrid retriever with knowledge base."""
        entries = get_knowledge_base()
        cfg = MergeConfig(method="rrf", rrf_k=60)
        retriever = HybridRetriever(
            entries=entries,
            merge_config=cfg,
            top_k=self.top_k,
        )
        return retriever

    def _load_samples(self) -> list[dict]:
        """Load evaluation samples."""
        samples = []
        with open(self.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def evaluate(
        self,
        samples: Optional[list[dict]] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Run full evaluation.

        Returns metrics dict and populates self.errors.
        """
        if samples is None:
            samples = self._load_samples()

        k = top_k or self.top_k

        recall_at_k: dict[int, int] = {1: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0}
        mrr_total = 0.0
        covered = 0
        errors: list[dict] = []

        for sample in samples:
            gold_code = sample.get("positive_code", "")
            mention = sample.get("mention", "")
            query = sample.get("query_text", mention)

            results = self.retriever.retrieve(query, mention=mention, top_k=k)
            retrieved_codes = [r.code for r in results]
            ranks = {code: rank + 1 for rank, code in enumerate(retrieved_codes)}

            gold_found = gold_code in retrieved_codes
            gold_rank = ranks.get(gold_code, None)

            if gold_found:
                covered += 1
                for tk in [1, 3, 5, 10, 20, 50]:
                    if gold_rank <= tk:
                        recall_at_k[tk] += 1
                mrr_total += 1.0 / gold_rank
            else:
                entry = self.entries_by_code.get(gold_code)
                if entry is None:
                    error_type = "code_not_in_kb"
                else:
                    error_type = classify_retrieval_error(
                        gold_code, retrieved_codes, entry, mention
                    )

                errors.append({
                    "sample_id": sample.get("id", ""),
                    "mention": mention,
                    "query": query,
                    "gold_code": gold_code,
                    "retrieved_codes": retrieved_codes[:5],
                    "error_type": error_type,
                    "coverage": "miss",
                })

        n = len(samples)
        mrr = mrr_total / n if n > 0 else 0.0

        metrics = {
            "n_samples": n,
            "covered": covered,
            "coverage": round(covered / n, 4) if n > 0 else 0.0,
            "mrr": round(mrr, 4),
            "recall": {
                f"recall@{tk}": round(recall_at_k[tk] / n, 4) if n > 0 else 0.0
                for tk in [1, 3, 5, 10, 20]
            },
            "recall_raw": {f"recall@{tk}": recall_at_k[tk] for tk in [1, 3, 5, 10, 20]},
            "error_types": self._summarize_errors(errors),
        }

        self._errors = errors
        self._metrics = metrics
        return metrics

    def _summarize_errors(self, errors: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for e in errors:
            counts[e.get("error_type", "unknown")] += 1
        return dict(counts)

    def save_errors_csv(self, path: Optional[str] = None) -> str:
        """Save errors to CSV."""
        if not hasattr(self, "_errors"):
            raise RuntimeError("Run evaluate() first.")
        path = path or str(self.output_dir / "retrieval_errors.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "mention", "query", "gold_code",
                            "retrieved_codes", "error_type", "coverage"],
            )
            writer.writeheader()
            writer.writerows(self._errors)

        return path

    def save_report(self, path: Optional[str] = None) -> str:
        """Save metrics report to JSON."""
        if not hasattr(self, "_metrics"):
            raise RuntimeError("Run evaluate() first.")
        path = path or str(self.output_dir / "icd_retrieval_report.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._metrics, f, indent=2, ensure_ascii=False)

        return path

    def print_report(self) -> None:
        """Print human-readable report."""
        if not hasattr(self, "_metrics"):
            raise RuntimeError("Run evaluate() first.")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

        m = self._metrics
        print("\n" + "=" * 55)
        print("  ICD-10 Candidate Retrieval Evaluation Report")
        print("=" * 55)
        print(f"  Dataset: {self.data_path}")
        print(f"  Samples: {m['n_samples']}")
        print(f"  Coverage: {m['covered']}/{m['n_samples']} ({m['coverage']*100:.1f}%)")
        print()
        print(f"  MRR:     {m['mrr']:.4f}")
        for tk in [1, 3, 5, 10, 20]:
            raw = m["recall_raw"].get(f"recall@{tk}", 0)
            pct = m["recall"].get(f"recall@{tk}", 0)
            bar = "=" * int(pct * 20)
            print(f"  R@{tk:2d}:   {pct:.4f} ({raw:3d}/{m['n_samples']:3d}) {bar}")
        print()
        if m["error_types"]:
            print("  Error types:")
            for et, count in sorted(m["error_types"].items(), key=lambda x: -x[1]):
                print(f"    {et:30s}: {count:3d}")
        print("=" * 55)
