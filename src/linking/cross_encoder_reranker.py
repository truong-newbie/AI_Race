"""
Cross-Encoder Reranker for ICD-10 and RxNorm

GIAI ĐOẠN 2: Fine-tune or use a pre-trained biomedical cross-encoder
to score (query, candidate) pairs.

Input pair:
  - Query: mention + context (sentence)
  - Candidate: code + name + description

Training:
  - Positive: gold (mention, candidate) pairs
  - Hard negatives: same ICD branch, different specificity
  - Hard negatives: RxNorm same ingredient, different strength/dose form

Models tried (in order):
  - drbert/DRBERT (German clinical BERT — best for medical codes)
  - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  - dmis-lab/biobert-v1.1
  - cross-encoder/ms-marco-MiniLM-L6-en-decoder
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from src.linking.base_reranker import BaseReranker, RerankResult
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.hybrid_retriever import CandidateResult
from src.linking.rxnorm.schema import RxNormEntry
from src.linking.cross_encoder_dataset import CrossEncoderSample


@dataclass
class CrossEncoderRerankResult(RerankResult):
    """Cross-encoder rerank result with score breakdown."""
    raw_score: Optional[float] = None
    normalized_score: Optional[float] = None


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker for ICD-10 and RxNorm candidates.

    Uses a biomedical cross-encoder to score (query, candidate) pairs.
    Falls back to None if no model is available.

    Usage:
        reranker = CrossEncoderReranker(
            model_name="drbert/DRBERT",
            cache_dir=".cache/cross_encoder",
        )
        results = reranker.rerank(candidates, query, mention, top_k=10)
    """

    def __init__(
        self,
        model_name: str = "drbert/DRBERT",
        cache_dir: str = ".cache/cross_encoder",
        max_length: int = 128,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.device = device
        self._model = None
        self._model_loaded = False

    def name(self) -> str:
        return f"cross_encoder_{self.model_name.split('/')[-1]}"

    def _load_model(self) -> bool:
        """Lazy-load cross-encoder model with fallback."""
        if self._model_loaded:
            return self._model is not None

        self._model_loaded = True

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            print("[CrossEncoderReranker] sentence-transformers not installed")
            return False

        os.makedirs(self.cache_dir, exist_ok=True)

        # Try models in order of preference
        models_to_try = [
            self.model_name,
            "drbert/DRBERT",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "dmis-lab/biobert-v1.1",
        ]

        # Remove duplicates
        seen = set()
        unique_models = []
        for m in models_to_try:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        for model in unique_models:
            try:
                self._model = CrossEncoder(
                    model,
                    max_length=self.max_length,
                    cache_dir=self.cache_dir,
                    device=self.device,
                )
                self.model_name = model
                print(f"[CrossEncoderReranker] Loaded model: {model}")
                return True
            except Exception as e:
                print(f"[CrossEncoderReranker] Failed to load {model}: {e}")
                continue

        self._model = None
        return False

    def _build_icd_text(self, entry: ICD10Entry) -> str:
        """Build candidate text for ICD-10 entry."""
        parts = [entry.code]
        if entry.name_en:
            parts.append(entry.name_en)
        if entry.name_vi:
            parts.append(entry.name_vi)
        if entry.description:
            parts.append(entry.description)
        return " | ".join(parts)

    def _build_rxnorm_text(self, entry: RxNormEntry) -> str:
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
        return " | ".join(parts)

    def rerank(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[CrossEncoderRerankResult]:
        if not candidates:
            return []

        if not self._load_model():
            print("[CrossEncoderReranker] No model available, returning retrieval order")
            return [
                CrossEncoderRerankResult(
                    code=getattr(c, "code", None) or getattr(c, "rxcui", ""),
                    rerank_score=getattr(c, "score", 0.0),
                    source="retrieval_fallback",
                    rank_before=idx + 1,
                )
                for idx, c in enumerate(candidates[:top_k])
            ]

        text = mention if mention else query
        combined = f"{query} {text}".strip()

        results = []
        for idx, c in enumerate(candidates):
            code = getattr(c, "code", None) or getattr(c, "rxcui", "")
            entry = None

            # Determine entry type and build text
            if hasattr(c, "rxcui"):
                from src.linking.rxnorm.schema import get_knowledge_base as get_rx
                all_rx = get_rx()
                entry_map = {e.rxcui: e for e in all_rx}
                entry = entry_map.get(code)
                candidate_text = self._build_rxnorm_text(entry) if entry else code
            else:
                from src.linking.icd.schema import get_knowledge_base as get_icd
                all_icd = get_icd()
                entry_map = {e.code: e for e in all_icd}
                entry = entry_map.get(code)
                candidate_text = self._build_icd_text(entry) if entry else code

            raw_score = self._score(combined, candidate_text)
            norm_score = self._normalize(raw_score)

            results.append(CrossEncoderRerankResult(
                code=code,
                rerank_score=norm_score,
                features={"cross_encoder_raw": raw_score, "cross_encoder_norm": norm_score},
                source="cross_encoder",
                rank_before=idx + 1,
                raw_score=raw_score,
                normalized_score=norm_score,
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    def _score(self, query: str, candidate: str) -> float:
        """Score a single (query, candidate) pair."""
        try:
            scores = self._model.predict([(query, candidate)])
            raw = float(scores[0]) if hasattr(scores, "__iter__") else float(scores)
            return raw
        except Exception:
            return 0.0

    def _normalize(self, raw: float) -> float:
        """
        Normalize raw cross-encoder score to [0, 1] range.

        MS-MARCO models: raw in [-5, 5] for relevant pairs
        DRBERT/BioBERT: raw can be logits or cosine-like

        Strategy: detect range from first few scores and normalize accordingly.
        """
        # Simple sigmoid-like normalization
        # For relevance models, 3.0 is a typical "relevant" threshold
        # Convert to confidence: 1.0 when raw=3.0, 0.0 when raw=0.0
        norm = 1.0 / (1.0 + abs(raw - 3.0) / 3.0)
        return max(0.0, min(1.0, norm))


class HybridCrossEncoderReranker(BaseReranker):
    """
    Hybrid reranker combining cross-encoder with retrieval scores.

    Blend cross-encoder scores with retrieval scores for robust ranking.
    """

    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        blend_weight_ce: float = 0.5,
    ):
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.blend_weight_ce = blend_weight_ce

    def name(self) -> str:
        return f"hybrid_cross_encoder_{self.cross_encoder.name()}"

    def rerank(
        self,
        candidates: list,
        query: str,
        mention: Optional[str] = None,
        top_k: int = 10,
    ) -> list[RerankResult]:
        if not candidates:
            return []

        # Get cross-encoder scores
        ce_results = self.cross_encoder.rerank(candidates, query, mention, top_k=top_k * 2)
        ce_scores: dict[str, float] = {r.code: r.rerank_score for r in ce_results}

        # Get retrieval scores
        retrieval_scores: dict[str, float] = {}
        for c in candidates:
            code = getattr(c, "code", None) or getattr(c, "rxcui", "")
            retrieval_scores[code] = getattr(c, "score", 0.0)

        # Normalize retrieval scores
        max_ret = max(retrieval_scores.values()) if retrieval_scores else 1.0
        if max_ret <= 0:
            max_ret = 1.0

        results = []
        for idx, c in enumerate(candidates):
            code = getattr(c, "code", None) or getattr(c, "rxcui", "")
            ret_norm = retrieval_scores.get(code, 0.0) / max_ret
            ce_score = ce_scores.get(code, 0.0)

            # Blend
            blended = (1.0 - self.blend_weight_ce) * ret_norm + self.blend_weight_ce * ce_score

            results.append(RerankResult(
                code=code,
                rerank_score=blended,
                features={
                    "retrieval_norm": ret_norm,
                    "cross_encoder": ce_score,
                },
                source="hybrid",
                rank_before=idx + 1,
            ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]


# ─── Fine-tuning helper ───────────────────────────────────────────────────────


def fine_tune_cross_encoder(
    train_data_path: str,
    model_name: str = "drbert/DRBERT",
    output_dir: str = ".cache/cross_encoder_finetuned",
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 100,
    training_steps: Optional[int] = None,
) -> CrossEncoderReranker:
    """
    Fine-tune a cross-encoder on the ICD-10 + RxNorm dataset.

    Args:
        train_data_path: Path to training JSON with CrossEncoderSample format
        model_name: Base model to fine-tune
        output_dir: Where to save the fine-tuned model
        epochs: Number of training epochs
        batch_size: Training batch size
        warmup_steps: Learning rate warmup steps
        training_steps: Override total training steps (auto-computed if None)

    Returns:
        CrossEncoderReranker with fine-tuned model
    """
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformerTrainer
        from sentence_transformers import InputExample
        from sentence_transformers.training_params import TrainingParams
    except ImportError:
        raise ImportError("sentence-transformers >= 2.0 required for fine-tuning")

    import json

    # Load training data
    with open(train_data_path, encoding="utf-8") as f:
        raw_samples = json.load(f)

    # Convert to InputExample
    train_examples = []
    for item in raw_samples:
        if isinstance(item, dict):
            text_a = item.get("query", "")
            text_b = item.get("candidate_text", "")
            label = float(item.get("label", 0))
        else:
            text_a, text_b, label = item

        example = InputExample(
            texts=[text_a, text_b],
            label=label,
        )
        train_examples.append(example)

    print(f"[FineTune] Loaded {len(train_examples)} training examples")

    # Initialize cross-encoder
    model = CrossEncoder(
        model_name,
        max_length=128,
        cache_dir=".cache",
    )

    # Configure training
    params = TrainingParams(
        num_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_steps=warmup_steps,
        output_dir=output_dir,
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        train_examples=train_examples,
        args=params,
    )
    trainer.train()

    # Save
    model.save(output_dir)
    print(f"[FineTune] Model saved to {output_dir}")

    # Return reranker using fine-tuned model
    return CrossEncoderReranker(
        model_name=output_dir,
        cache_dir=output_dir,
    )
