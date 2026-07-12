"""
Cross-Encoder Dataset Builder for ICD-10 and RxNorm Reranking

GIAI ĐOẠN 2: Build training dataset for cross-encoder reranking.

Training data:
  - Positive: gold (mention, candidate) pairs
  - Hard negatives: same ICD branch, different specificity level
  - Hard negatives: RxNorm same ingredient, different strength/dose form
  - Cross-lingual: query in Vietnamese, candidate in English

Format: sentence-transformers compatible (query, candidate, label)
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from itertools import combinations

from src.linking.icd.schema import ICD10Entry, get_knowledge_base as get_icd_kb
from src.linking.rxnorm.schema import RxNormEntry, get_knowledge_base as get_rx_kb


@dataclass
class CrossEncoderSample:
    """A single training sample for cross-encoder."""
    query: str          # Full sentence/query text
    mention: str        # Entity mention text
    candidate_code: str # ICD-10 code or RxNorm RxCUI
    candidate_text: str # Concatenated: code + name + description
    label: float        # 1.0 = positive, 0.0 = negative
    negative_type: str  # "hard_same_branch", "hard_same_ingredient", "random"

    def to_sbert_format(self) -> tuple[str, str, float]:
        """Convert to sentence-transformers format: (text_a, text_b, label)."""
        combined = f"{self.query} {self.mention}".strip()
        return combined, self.candidate_text, self.label

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mention": self.mention,
            "candidate_code": self.candidate_code,
            "candidate_text": self.candidate_text,
            "label": self.label,
            "negative_type": self.negative_type,
        }


class CrossEncoderDatasetBuilder:
    """
    Build cross-encoder training datasets for ICD-10 and RxNorm reranking.

    Strategy:
    1. Load all entries from knowledge base
    2. Load evaluation samples (gold mentions)
    3. Generate hard negatives:
       - ICD: same chapter/branch, different specificity
       - RxNorm: same ingredient, different strength/dose form
    4. Output sentence-transformers format

    Usage:
        builder = CrossEncoderDatasetBuilder(icd_entries, rx_entries)
        samples = builder.build(
            gold_samples=gold_data,
            negatives_per_positive=4,
        )
        builder.save(samples, "outputs/cross_encoder_train.json")
    """

    def __init__(
        self,
        icd_entries: Optional[list[ICD10Entry]] = None,
        rx_entries: Optional[list[RxNormEntry]] = None,
        random_seed: int = 42,
    ):
        self.icd_entries = {e.code: e for e in (icd_entries or [])}
        self.rx_entries = {e.rxcui: e for e in (rx_entries or [])}
        self.icd_by_chapter: dict[str, list[ICD10Entry]] = {}
        self.icd_by_parent: dict[str, list[ICD10Entry]] = {}
        self.icd_by_alias: dict[str, list[ICD10Entry]] = {}
        self.rx_by_ingredient: dict[str, list[RxNormEntry]] = {}
        self._build_indices()
        random.seed(random_seed)

    def _build_indices(self) -> None:
        """Build lookup indices for efficient negative mining."""
        for e in self.icd_entries.values():
            # By chapter
            ch = e.chapter or "unknown"
            if ch not in self.icd_by_chapter:
                self.icd_by_chapter[ch] = []
            self.icd_by_chapter[ch].append(e)

            # By parent
            parent = e.parent_code or "root"
            if parent not in self.icd_by_parent:
                self.icd_by_parent[parent] = []
            self.icd_by_parent[parent].append(e)

            # By alias
            for alias in e.aliases + e.synonyms:
                key = alias.lower().strip()
                if key not in self.icd_by_alias:
                    self.icd_by_alias[key] = []
                self.icd_by_alias[key].append(e)

        for e in self.rx_entries.values():
            ing = (e.ingredient or "").lower().strip()
            if ing:
                if ing not in self.rx_by_ingredient:
                    self.rx_by_ingredient[ing] = []
                self.rx_by_ingredient[ing].append(e)

    def _build_candidate_text(self, entry: ICD10Entry) -> str:
        """Build candidate text for ICD-10 entry."""
        parts = [entry.code]
        if entry.name_en:
            parts.append(entry.name_en)
        if entry.name_vi:
            parts.append(entry.name_vi)
        if entry.description:
            parts.append(entry.description)
        return " | ".join(parts)

    def _build_candidate_text_rx(self, entry: RxNormEntry) -> str:
        """Build candidate text for RxNorm entry."""
        parts = [f"RxCUI: {entry.rxcui}"]
        if entry.name_short:
            parts.append(entry.name_short)
        if entry.ingredient:
            parts.append(f"ingredient: {entry.ingredient}")
        if entry.strength_value is not None:
            unit = entry.strength_unit or "mg"
            parts.append(f"strength: {entry.strength_value} {unit}")
        if entry.dose_form:
            parts.append(f"dose form: {entry.dose_form}")
        if entry.brand_name:
            parts.append(f"brand: {entry.brand_name}")
        if entry.description:
            parts.append(entry.description)
        return " | ".join(parts)

    # ─── ICD-10 Dataset Building ──────────────────────────────────────────────

    def build_icd_dataset(
        self,
        gold_samples: list[dict],
        negatives_per_positive: int = 4,
        include_random_negatives: bool = True,
    ) -> list[CrossEncoderSample]:
        """
        Build ICD-10 cross-encoder dataset.

        Args:
            gold_samples: List of dicts with keys: mention, query_text, positive_code
            negatives_per_positive: Number of hard negatives per positive
            include_random_negatives: Include random negatives from different chapters

        Returns:
            List of CrossEncoderSample
        """
        samples = []

        for sample in gold_samples:
            mention = sample.get("mention", "")
            query = sample.get("query_text", mention)
            gold_code = sample.get("positive_code", "")

            gold_entry = self.icd_entries.get(gold_code)
            if gold_entry is None:
                continue

            gold_text = self._build_candidate_text(gold_entry)

            # Positive sample
            samples.append(CrossEncoderSample(
                query=query,
                mention=mention,
                candidate_code=gold_code,
                candidate_text=gold_text,
                label=1.0,
                negative_type="positive",
            ))

            # Hard negatives: same chapter, different code
            chapter = gold_entry.chapter or "unknown"
            same_chapter = [
                e for e in self.icd_by_chapter.get(chapter, [])
                if e.code != gold_code
            ]
            hard_negatives = self._sample_negatives(
                same_chapter, negatives_per_positive, gold_code, gold_entry
            )
            for neg in hard_negatives:
                samples.append(CrossEncoderSample(
                    query=query,
                    mention=mention,
                    candidate_code=neg.code,
                    candidate_text=self._build_candidate_text(neg),
                    label=0.0,
                    negative_type="hard_same_chapter",
                ))

            # Hard negatives: same alias keyword, different code
            alias_negatives = self._find_alias_negatives(
                mention, gold_code, gold_entry, negatives_per_positive
            )
            for neg in alias_negatives:
                samples.append(CrossEncoderSample(
                    query=query,
                    mention=mention,
                    candidate_code=neg.code,
                    candidate_text=self._build_candidate_text(neg),
                    label=0.0,
                    negative_type="hard_same_alias",
                ))

            # Random negatives: different chapter
            if include_random_negatives:
                diff_chapter = [
                    e for e in self.icd_entries.values()
                    if e.code != gold_code and e.chapter != chapter
                ]
                random.shuffle(diff_chapter)
                for neg in diff_chapter[:max(1, negatives_per_positive // 2)]:
                    samples.append(CrossEncoderSample(
                        query=query,
                        mention=mention,
                        candidate_code=neg.code,
                        candidate_text=self._build_candidate_text(neg),
                        label=0.0,
                        negative_type="random",
                    ))

        return samples

    def _find_alias_negatives(
        self,
        mention: str,
        gold_code: str,
        gold_entry: ICD10Entry,
        max_negatives: int,
    ) -> list[ICD10Entry]:
        """Find hard negatives by shared alias keywords."""
        results = []
        mention_lower = mention.lower()

        for alias in gold_entry.aliases + gold_entry.synonyms:
            alias_key = alias.lower().strip()
            # Check if mention shares this alias keyword
            if alias_key and alias_key in mention_lower:
                for entry in self.icd_by_alias.get(alias_key, []):
                    if entry.code != gold_code and entry not in results:
                        results.append(entry)
                        if len(results) >= max_negatives:
                            return results

        return results

    def _sample_negatives(
        self,
        candidates: list[ICD10Entry],
        count: int,
        gold_code: str,
        gold_entry: ICD10Entry,
    ) -> list[ICD10Entry]:
        """Sample hard negatives, preferring those most similar to gold."""
        valid = [e for e in candidates if e.code != gold_code]
        if not valid:
            return []

        # Score by name similarity
        gold_text = (gold_entry.name_en or "").lower()

        def score(e: ICD10Entry) -> float:
            e_text = (e.name_en or "").lower()
            if not gold_text or not e_text:
                return 0.0
            # Simple word overlap
            gold_words = set(gold_text.split())
            e_words = set(e_text.split())
            if not gold_words or not e_words:
                return 0.0
            return len(gold_words & e_words) / max(len(gold_words), len(e_words))

        scored = [(score(e), e) for e in valid]
        scored.sort(key=lambda x: -x[0])

        # Mix top-scored (hardest) with random
        top = scored[:count]
        random.shuffle(valid)
        rest = [e for _, e in scored[count:count * 2]]
        mixed = top + rest

        random.shuffle(mixed)
        return mixed[:count]

    # ─── RxNorm Dataset Building ───────────────────────────────────────────────

    def build_rxnorm_dataset(
        self,
        gold_samples: list[dict],
        negatives_per_positive: int = 4,
    ) -> list[CrossEncoderSample]:
        """
        Build RxNorm cross-encoder dataset.

        Hard negatives: same ingredient, different strength or dose form.
        """
        samples = []

        for sample in gold_samples:
            mention = sample.get("mention", "")
            query = sample.get("query_text", mention)
            gold_rxcui = sample.get("positive_rxcui", "")

            gold_entry = self.rx_entries.get(gold_rxcui)
            if gold_entry is None:
                continue

            gold_text = self._build_candidate_text_rx(gold_entry)
            gold_ing = (gold_entry.ingredient or "").lower().strip()

            # Positive sample
            samples.append(CrossEncoderSample(
                query=query,
                mention=mention,
                candidate_code=gold_rxcui,
                candidate_text=gold_text,
                label=1.0,
                negative_type="positive",
            ))

            # Hard negatives: same ingredient, different strength
            same_ing = [
                e for e in self.rx_by_ingredient.get(gold_ing, [])
                if e.rxcui != gold_rxcui
            ]

            for neg in same_ing[:negatives_per_positive]:
                samples.append(CrossEncoderSample(
                    query=query,
                    mention=mention,
                    candidate_code=neg.rxcui,
                    candidate_text=self._build_candidate_text_rx(neg),
                    label=0.0,
                    negative_type="hard_same_ingredient",
                ))

            # Random negatives: different ingredient
            diff_ing = [
                e for e in self.rx_entries.values()
                if e.rxcui != gold_rxcui and (e.ingredient or "").lower().strip() != gold_ing
            ]
            random.shuffle(diff_ing)
            for neg in diff_ing[:max(1, negatives_per_positive // 2)]:
                samples.append(CrossEncoderSample(
                    query=query,
                    mention=mention,
                    candidate_code=neg.rxcui,
                    candidate_text=self._build_candidate_text_rx(neg),
                    label=0.0,
                    negative_type="random",
                ))

        return samples

    # ─── Export ────────────────────────────────────────────────────────────────

    def save(
        self,
        samples: list[CrossEncoderSample],
        output_path: str,
        format: str = "sbert",
    ) -> None:
        """
        Save samples to file.

        Formats:
          - "sbert": sentence-transformers format (query, candidate, label)
          - "json": Full CrossEncoderSample format
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "sbert":
            with open(path, "w", encoding="utf-8") as f:
                for s in samples:
                    a, b, label = s.to_sbert_format()
                    f.write(json.dumps({"text_a": a, "text_b": b, "label": label}, ensure_ascii=False) + "\n")
        elif format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in samples], f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"Saved {len(samples)} samples to {path}")

    def print_stats(self, samples: list[CrossEncoderSample]) -> None:
        """Print dataset statistics."""
        total = len(samples)
        pos = sum(1 for s in samples if s.label > 0.5)
        neg = total - pos
        by_type: dict[str, int] = {}
        for s in samples:
            by_type[s.negative_type] = by_type.get(s.negative_type, 0) + 1

        print(f"\nCross-Encoder Dataset Stats:")
        print(f"  Total:   {total}")
        print(f"  Positive: {pos} ({pos/total*100:.1f}%)")
        print(f"  Negative: {neg} ({neg/total*100:.1f}%)")
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {t}: {c}")
