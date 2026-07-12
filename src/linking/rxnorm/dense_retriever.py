"""
Dense Drug Retriever

Sentence-transformer embeddings cho drug retrieval.
"""

import hashlib
import os
from pathlib import Path
from typing import Optional

from src.linking.rxnorm.schema import RxNormEntry
from src.linking.rxnorm.normalizer import DrugTextNormalizer


class DenseDrugRetriever:
    """
    Dense embedding retrieval cho drug mentions.
    Lazy-loads sentence-transformer model.
    Caches embeddings to disk.
    """

    _model: Optional["SentenceTransformer"] = None

    def __init__(
        self,
        entries: list[RxNormEntry],
        model_name: str = "intfloat/multilingual-e5-small",
        cache_dir: str = ".cache/rxnorm_dense",
        normalizer: Optional[DrugTextNormalizer] = None,
    ):
        self.entries = entries
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.normalizer = normalizer or DrugTextNormalizer()
        self._built = False

        self._texts: list[str] = []
        self._codes: list[str] = []
        self._corpus_embeddings: Optional["np.ndarray"] = None

    def _load_model(self) -> "SentenceTransformer":
        """Lazy-load the sentence-transformer model."""
        if DenseDrugRetriever._model is None:
            from sentence_transformers import SentenceTransformer
            DenseDrugRetriever._model = SentenceTransformer(self.model_name)
        return DenseDrugRetriever._model

    def _get_cache_path(self) -> Path:
        """Get path to embeddings cache file."""
        entries_str = "|".join(sorted(f"{e.rxcui}:{e.name}" for e in self.entries))
        hash_val = hashlib.md5(entries_str.encode()).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"embeddings_{hash_val}.npy"

    def build(self) -> None:
        """Build embeddings and indices."""
        if self._built:
            return

        cache_path = self._get_cache_path()

        if cache_path.exists():
            self._corpus_embeddings = self._load_from_cache(cache_path)
        else:
            self._corpus_embeddings = self._build_embeddings()
            self._save_to_cache(cache_path)

        # Build text/code lists
        self._texts = []
        self._codes = []
        for entry in self.entries:
            for text in entry.get_all_searchable_texts():
                self._texts.append(text)
                self._codes.append(entry.rxcui)

        self._built = True

    def _build_embeddings(self) -> "np.ndarray":
        """Build corpus embeddings."""
        import numpy as np
        model = self._load_model()

        texts = []
        for entry in self.entries:
            texts.extend(entry.get_all_searchable_texts())

        if not texts:
            return np.array([])

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings

    def _load_from_cache(self, cache_path: Path) -> "np.ndarray":
        """Load embeddings from cache."""
        import numpy as np
        return np.load(cache_path)

    def _save_to_cache(self, cache_path: Path) -> None:
        """Save embeddings to cache."""
        if self._corpus_embeddings is not None:
            import numpy as np
            np.save(cache_path, self._corpus_embeddings)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Retrieve drugs by dense embedding similarity."""
        if not self._built:
            self.build()

        if self._corpus_embeddings is None or len(self._corpus_embeddings) == 0:
            return []

        import numpy as np

        model = self._load_model()
        query_emb = model.encode([query], normalize_embeddings=True)

        # Cosine similarity
        similarities = (self._corpus_embeddings @ query_emb.T).flatten()

        # Aggregate by rxcui: max similarity
        code_scores: dict[str, float] = {}
        for i, code in enumerate(self._codes):
            if code not in code_scores or similarities[i] > code_scores[code]:
                code_scores[code] = float(similarities[i])

        ranked = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
