"""
Dense Embedding Retriever for ICD-10

Uses sentence-transformers (multilingual-e5-small or BGE-multilingual).
Builds embedding index once and caches it.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense embedding retrieval using sentence-transformers.

    Prefers:
      - intfloat/multilingual-e5-small (fastest, good quality)
      - Then falls back to BAAI/bge-m3

    Caches embeddings to disk after first build.
    """

    def __init__(
        self,
        entries: Optional[list] = None,
        model_name: str = "intfloat/multilingual-e5-small",
        cache_dir: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or ".cache/icd_dense"
        self.normalize = normalize
        self.batch_size = batch_size

        self._model: Optional[object] = None
        self._entries: list = []
        self._embeddings: Optional[object] = None
        self._entry_codes: list[str] = []
        self._initialized: bool = False

        if entries:
            self.build(entries)

    def _get_cache_path(self) -> Path:
        """Path for cached embeddings."""
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        name_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return p / f"embeddings_{name_hash}.npy"

    def _get_meta_path(self) -> Path:
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        name_hash = hashlib.md5(self.model_name.encode()).hexdigest()[:8]
        return p / f"embeddings_{name_hash}_meta.json"

    def _load_model(self):
        """Load sentence-transformers model lazily."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading dense model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Dense retrieval will use fallback."
            )
            self._initialized = False

    def build(self, entries: list) -> None:
        """Build embedding index for all entries."""
        self._entries = entries
        self._entry_codes = [e.code for e in entries]

        cache_path = self._get_cache_path()
        meta_path = self._get_meta_path()

        if cache_path.exists() and meta_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            try:
                import numpy as np
                self._embeddings = np.load(cache_path)
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("codes") == self._entry_codes:
                    logger.info("Embeddings cache valid.")
                    return
            except Exception:
                pass

        self._load_model()
        if not self._initialized or self._model is None:
            logger.warning("Dense retriever unavailable — model not loaded.")
            return

        import numpy as np

        # Build corpus: concatenate all searchable texts per entry
        texts = []
        for entry in entries:
            all_texts = entry.get_all_searchable_texts()
            if all_texts:
                texts.append(" | ".join(all_texts))
            else:
                texts.append(entry.code)

        logger.info(f"Encoding {len(texts)} entry texts...")
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._embeddings = emb

        # Cache
        np.save(cache_path, emb)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"codes": self._entry_codes, "model": self.model_name}, f)
        logger.info(f"Cached embeddings to {cache_path}")

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-k ICD codes by cosine similarity.

        Returns list of (code, similarity_score).
        """
        if self._embeddings is None or self._model is None:
            return []

        import numpy as np

        emb = self._model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        # Cosine similarity = dot product (normalized)
        scores = np.dot(self._embeddings, emb[0])

        # Handle case where model doesn't normalize
        if not self.normalize:
            norm_q = emb[0] / (np.linalg.norm(emb[0]) + 1e-9)
            norm_db = self._embeddings / (
                np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-9
            )
            scores = np.dot(norm_db, norm_q)

        scored = list(zip(self._entry_codes, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve_one(self, query: str) -> Optional[tuple[str, float]]:
        """Retrieve top-1 match."""
        results = self.retrieve(query, top_k=1)
        if results:
            return results[0]
        return None
