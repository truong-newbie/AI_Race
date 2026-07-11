"""
Alias Index for ICD-10 Retrieval

Fast lookup by normalized alias/synonym name.
"""

from typing import Optional
from src.linking.icd.schema import ICD10Entry
from src.linking.icd.preprocess import TextNormalizer


class AliasIndex:
    """
    Bidirectional index: canonical form -> code and alias -> code.

    Supports exact match and normalized exact match lookups.
    """

    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        self.normalizer = normalizer or TextNormalizer()
        # code -> entry
        self.entries_by_code: dict[str, ICD10Entry] = {}
        # normalized canonical form -> code (built from name_vi + name_en)
        self.canonical_index: dict[str, str] = {}
        # normalized alias -> code
        self.alias_index: dict[str, str] = {}
        # normalized synonym -> code
        self.synonym_index: dict[str, str] = {}
        # normalized include_term -> code
        self.include_index: dict[str, str] = {}

    def add_entry(self, entry: ICD10Entry) -> None:
        """Index a single ICD-10 entry."""
        self.entries_by_code[entry.code] = entry

        # Index name_vi as canonical
        if entry.name_vi:
            norm = self.normalizer.normalize_for_alias(entry.name_vi)
            if norm and norm not in self.canonical_index:
                self.canonical_index[norm] = entry.code

        # Index name_en as canonical
        if entry.name_en:
            norm = self.normalizer.normalize_for_alias(entry.name_en)
            if norm and norm not in self.canonical_index:
                self.canonical_index[norm] = entry.code

        # Index aliases
        for alias in entry.aliases:
            norm = self.normalizer.normalize_for_alias(alias)
            if norm and norm not in self.alias_index:
                self.alias_index[norm] = entry.code

        # Index synonyms
        for syn in entry.synonyms:
            norm = self.normalizer.normalize_for_alias(syn)
            if norm and norm not in self.synonym_index:
                self.synonym_index[norm] = entry.code

        # Index include_terms
        for term in entry.include_terms:
            norm = self.normalizer.normalize_for_alias(term)
            if norm and norm not in self.include_index:
                self.include_index[norm] = entry.code

    def build(self, entries: list[ICD10Entry]) -> None:
        """Build index from list of entries."""
        for entry in entries:
            self.add_entry(entry)

    def lookup_exact(self, text: str) -> list[str]:
        """
        Exact match on original text with the same normalization used for indexing.
        Case-insensitive, punctuation-stripped, whitespace-normalized.

        Returns list of matching codes.
        """
        # Apply the same normalization as normalize_for_alias so lookup matches index
        norm = self.normalizer.normalize_for_alias(text)
        if not norm:
            return []

        results = []
        if norm in self.canonical_index:
            results.append(self.canonical_index[norm])
        if norm in self.alias_index:
            c = self.alias_index[norm]
            if c not in results:
                results.append(c)
        if norm in self.synonym_index:
            c = self.synonym_index[norm]
            if c not in results:
                results.append(c)

        return results

    def lookup_normalized(self, text: str) -> list[str]:
        """
        Normalized exact match (applies full normalization pipeline).

        Returns list of matching codes.
        """
        norm = self.normalizer.normalize_for_alias(text)
        if not norm:
            return []
        results = []

        if norm in self.canonical_index:
            results.append(self.canonical_index[norm])
        if norm in self.alias_index:
            c = self.alias_index[norm]
            if c not in results:
                results.append(c)
        if norm in self.synonym_index:
            c = self.synonym_index[norm]
            if c not in results:
                results.append(c)

        return results

    def get_entry(self, code: str) -> Optional[ICD10Entry]:
        """Get entry by code."""
        return self.entries_by_code.get(code)

    def all_codes(self) -> list[str]:
        """All indexed codes."""
        return list(self.entries_by_code.keys())
