"""
RxNorm Knowledge Base Loader và Preprocessing

Module để load và preprocess RxNorm data phục vụ cho drug entity linking.
Hỗ trợ:
- Ingredient lookup
- Strength/unit parsing
- Dose form matching
- Brand name search
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class RxNormEntry:
    """Một entry trong RxNorm knowledge base."""
    rxcui: str                          # RxNorm Concept Unique Identifier
    name: str                           # Tên chính
    ingredient: Optional[str] = None    # Hoạt chất
    strength: Optional[str] = None      # Nồng độ (VD: "10 MG")
    unit: Optional[str] = None          # Đơn vị (VD: "MG", "ML")
    dose_form: Optional[str] = None     # Dạng bào chế (VD: "TABLET", "CAPSULE")
    brand: Optional[str] = None         # Tên thương mại
    tty: Optional[str] = None          # Term Type (SCD, IN, BN, etc.)

    def to_dict(self) -> dict:
        """Convert sang dict."""
        return {
            "rxcui": self.rxcui,
            "name": self.name,
            "ingredient": self.ingredient,
            "strength": self.strength,
            "unit": self.unit,
            "dose_form": self.dose_form,
            "brand": self.brand,
            "tty": self.tty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RxNormEntry":
        """Tạo từ dict."""
        return cls(
            rxcui=data["rxcui"],
            name=data.get("name", ""),
            ingredient=data.get("ingredient"),
            strength=data.get("strength"),
            unit=data.get("unit"),
            dose_form=data.get("dose_form"),
            brand=data.get("brand"),
            tty=data.get("tty"),
        )


class RxNormKnowledgeBase:
    """
    RxNorm Knowledge Base để tra cứu và matching drugs.

    Supported operations:
    - Exact ingredient + strength + dose form lookup
    - Ingredient-only lookup
    - Brand name search
    - Fuzzy matching với RapidFuzz
    """

    def __init__(self):
        self.entries: dict[str, RxNormEntry] = {}
        # Index by different keys
        self.ingredient_index: dict[str, list[str]] = {}  # ingredient -> list of rxcuis
        self.brand_index: dict[str, list[str]] = {}  # brand -> list of rxcuis
        self.full_name_index: dict[str, str] = {}  # normalized full name -> rxcui

    def add_entry(self, entry: RxNormEntry) -> None:
        """Thêm một entry vào KB."""
        self.entries[entry.rxcui] = entry

        # Index by ingredient
        if entry.ingredient:
            norm_ing = self._normalize(entry.ingredient)
            if norm_ing not in self.ingredient_index:
                self.ingredient_index[norm_ing] = []
            if entry.rxcui not in self.ingredient_index[norm_ing]:
                self.ingredient_index[norm_ing].append(entry.rxcui)

        # Index by brand
        if entry.brand:
            norm_brand = self._normalize(entry.brand)
            if norm_brand not in self.brand_index:
                self.brand_index[norm_brand] = []
            if entry.rxcui not in self.brand_index[norm_brand]:
                self.brand_index[norm_brand].append(entry.rxcui)

        # Index by full name
        norm_name = self._normalize(entry.name)
        if norm_name and norm_name not in self.full_name_index:
            self.full_name_index[norm_name] = entry.rxcui

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text cho việc indexing."""
        if not text:
            return ""
        return " ".join(text.lower().strip().split())

    def get_by_rxcui(self, rxcui: str) -> Optional[RxNormEntry]:
        """Lấy entry theo RxCUI."""
        return self.entries.get(rxcui)

    def get_by_name(self, name: str) -> Optional[RxNormEntry]:
        """Lấy entry theo exact name match."""
        normalized = self._normalize(name)
        rxcui = self.full_name_index.get(normalized)
        if rxcui:
            return self.entries.get(rxcui)
        return None

    def get_by_ingredient(self, ingredient: str) -> list[RxNormEntry]:
        """Lấy entries theo ingredient."""
        normalized = self._normalize(ingredient)
        rxcuis = self.ingredient_index.get(normalized, [])
        return [self.entries[r] for r in rxcuis if r in self.entries]

    def get_by_brand(self, brand: str) -> list[RxNormEntry]:
        """Lấy entries theo brand name."""
        normalized = self._normalize(brand)
        rxcuis = self.brand_index.get(normalized, [])
        return [self.entries[r] for r in rxcuis if r in self.entries]

    def search_by_ingredient(self, ingredient: str) -> list[RxNormEntry]:
        """Search entries by ingredient (partial match)."""
        normalized = self._normalize(ingredient)
        if not normalized:
            return []

        results = []
        for norm_ing, rxcuis in self.ingredient_index.items():
            if normalized in norm_ing or norm_ing in normalized:
                for rxcui in rxcuis:
                    if rxcui in self.entries:
                        results.append(self.entries[rxcui])
        return results

    def get_all_rxcuis(self) -> set[str]:
        """Lấy tất cả RxCUI trong KB."""
        return set(self.entries.keys())

    def size(self) -> int:
        """Số lượng entries trong KB."""
        return len(self.entries)

    @classmethod
    def from_dict_list(cls, data: list[dict]) -> "RxNormKnowledgeBase":
        """Tạo KB từ list of dicts."""
        kb = cls()
        for item in data:
            entry = RxNormEntry.from_dict(item)
            kb.add_entry(entry)
        return kb

    def to_dict_list(self) -> list[dict]:
        """Convert KB sang list of dicts."""
        return [entry.to_dict() for entry in self.entries.values()]

    def save(self, path: Union[str, Path]) -> None:
        """Save KB ra file JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict_list(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved RxNorm KB to {path} ({self.size()} entries)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RxNormKnowledgeBase":
        """Load KB từ file JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kb = cls.from_dict_list(data)
        logger.info(f"Loaded RxNorm KB from {path} ({kb.size()} entries)")
        return kb


class DrugParser:
    """
    Parser để parse drug mention text thành structured components.

    Supported formats:
    - "Aspirin 81 mg"
    - "metoprolol succinate xl 50 mg"
    - "Amoxicillin 500mg capsule"
    - "amlodipine 10 mg po daily"
    """

    # Regex patterns
    STRENGTH_PATTERN = re.compile(
        r'(\d+(?:[.,]\d+)?)\s*(mg|g|ml|mcg|iu|%)',
        re.IGNORECASE
    )

    DOSE_FORM_PATTERN = re.compile(
        r'\b(tablet|capsule|injection|solution|cream|ointment|'
        r'drops|syrup|spray|patch|inhaler|powder|granules|'
        r'tab|cap|inj|sol|crem|ung)\b',
        re.IGNORECASE
    )

    ROUTE_PATTERN = re.compile(
        r'\b(po|oral|iv|im|sc|topical|inh|inhale|rectal|sublingual|sl|'
        r'intranasal|transdermal|ophthalmic|otic|vaginal)\b',
        re.IGNORECASE
    )

    FREQUENCY_PATTERN = re.compile(
        r'\b(q\d+h?|qd|bid|tid|qid|daily|weekly|monthly|'
        r'once|twice|thrice|prn|qhs|qam|qpm|qod|ac|pc|stat)\b',
        re.IGNORECASE
    )

    @dataclass
    class ParsedDrug:
        """Kết quả parse của một drug mention."""
        original: str
        ingredient: Optional[str] = None
        strength: Optional[str] = None
        strength_value: Optional[float] = None
        unit: Optional[str] = None
        dose_form: Optional[str] = None
        route: Optional[str] = None
        frequency: Optional[str] = None

    def parse(self, text: str) -> list["DrugParser.ParsedDrug"]:
        """
        Parse drug mention text.

        Args:
            text: Drug mention text (VD: "Aspirin 81 mg po daily")

        Returns:
            List of ParsedDrug objects
        """
        if not text:
            return []

        results = []

        # Extract strength
        strength_match = self.STRENGTH_PATTERN.search(text)
        strength = None
        strength_value = None
        unit = None
        if strength_match:
            strength = strength_match.group(0)
            strength_value = float(strength_match.group(1).replace(',', '.'))
            unit = strength_match.group(2).upper()

        # Extract dose form
        dose_form_match = self.DOSE_FORM_PATTERN.search(text)
        dose_form = dose_form_match.group(1).lower() if dose_form_match else None

        # Extract route
        route_match = self.ROUTE_PATTERN.search(text)
        route = route_match.group(1).lower() if route_match else None

        # Extract frequency
        freq_match = self.FREQUENCY_PATTERN.search(text)
        frequency = freq_match.group(1).lower() if freq_match else None

        # Extract ingredient (text before strength or dose form)
        ingredient = text
        if strength_match:
            ingredient = text[:strength_match.start()].strip()
        elif dose_form_match:
            ingredient = text[:dose_form_match.start()].strip()

        # Clean up ingredient
        ingredient = re.sub(r'\d+\s*$', '', ingredient).strip()

        result = self.ParsedDrug(
            original=text,
            ingredient=ingredient,
            strength=strength,
            strength_value=strength_value,
            unit=unit,
            dose_form=dose_form,
            route=route,
            frequency=frequency
        )
        results.append(result)

        return results

    def extract_ingredient(self, text: str) -> Optional[str]:
        """Extract just the ingredient name."""
        parsed = self.parse(text)
        if parsed:
            return parsed[0].ingredient
        return None


class RxNormLinker:
    """
    Link drug mentions với RxNorm entries.

    Priority:
    1. Exact full string match
    2. Ingredient + strength + dose form
    3. Ingredient + strength
    4. Ingredient + dose form
    5. Ingredient only
    6. Fuzzy fallback
    """

    def __init__(self, kb: RxNormKnowledgeBase):
        self.kb = kb
        self.parser = DrugParser()

    def link(self, drug_text: str, limit: int = 5) -> list[tuple[RxNormEntry, float]]:
        """
        Link drug text với RxNorm entries.

        Args:
            drug_text: Drug mention text
            limit: Max number of results

        Returns:
            List of (RxNormEntry, score) tuples sorted by score
        """
        # Parse drug text
        parsed = self.parser.parse(drug_text)
        if not parsed:
            return []

        parsed_drug = parsed[0]
        results = []

        # 1. Try exact full name match
        exact = self.kb.get_by_name(drug_text)
        if exact:
            results.append((exact, 1.0))

        # 2. Try ingredient + strength + dose form
        if parsed_drug.ingredient and parsed_drug.strength_value and parsed_drug.unit:
            by_ing = self.kb.get_by_ingredient(parsed_drug.ingredient)
            for entry in by_ing:
                score = self._calculate_score(parsed_drug, entry)
                if score > 0:
                    results.append((entry, score))

        # 3. Try ingredient only
        if parsed_drug.ingredient and not results:
            by_ing = self.kb.search_by_ingredient(parsed_drug.ingredient)
            for entry in by_ing:
                results.append((entry, 0.6))

        # Sort by score and dedupe
        seen = set()
        unique_results = []
        for entry, score in results:
            if entry.rxcui not in seen:
                seen.add(entry.rxcui)
                unique_results.append((entry, score))

        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:limit]

    def _calculate_score(self, parsed: DrugParser.ParsedDrug, entry: RxNormEntry) -> float:
        """Calculate matching score."""
        score = 0.0

        # Ingredient match (high weight)
        if parsed.ingredient and entry.ingredient:
            if parsed.ingredient.lower() == entry.ingredient.lower():
                score += 0.5
            elif parsed.ingredient.lower() in entry.ingredient.lower():
                score += 0.3

        # Strength match (medium weight)
        if parsed.strength_value and entry.strength:
            try:
                entry_strength_match = self.parser.STRENGTH_PATTERN.search(entry.strength)
                if entry_strength_match:
                    entry_value = float(entry_strength_match.group(1).replace(',', '.'))
                    if abs(parsed.strength_value - entry_value) < 0.01:
                        score += 0.3
                    elif abs(parsed.strength_value - entry_value) / max(parsed.strength_value, 1) < 0.1:
                        score += 0.2
            except (ValueError, AttributeError):
                pass

        # Dose form match (low weight)
        if parsed.dose_form and entry.dose_form:
            if parsed.dose_form.lower() == entry.dose_form.lower():
                score += 0.2

        return score


def create_sample_rxnorm_kb() -> RxNormKnowledgeBase:
    """
    Tạo sample RxNorm KB với các drugs phổ biến.

    Đây là sample data, trong thực tế cần load từ official RxNorm source.
    """
    sample_data = [
        # Amlodipine
        {"rxcui": "308135", "name": "amlodipine 10 MG Oral Tablet", "ingredient": "Amlodipine", "strength": "10 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "308136", "name": "amlodipine 5 MG Oral Tablet", "ingredient": "Amlodipine", "strength": "5 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Aspirin
        {"rxcui": "243670", "name": "aspirin 81 MG Oral Tablet", "ingredient": "Aspirin", "strength": "81 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "212033", "name": "aspirin 325 MG Oral Tablet", "ingredient": "Aspirin", "strength": "325 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Metoprolol
        {"rxcui": "866436", "name": "metoprolol succinate 50 MG Extended Release Tablet", "ingredient": "Metoprolol Succinate", "strength": "50 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "866924", "name": "metoprolol tartrate 50 MG Oral Tablet", "ingredient": "Metoprolol Tartrate", "strength": "50 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Guaifenesin
        {"rxcui": "392085", "name": "guaifenesin 200 MG Oral Tablet", "ingredient": "Guaifenesin", "strength": "200 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "198240", "name": "Guaifenesin 100 MG/5ML Oral Solution", "ingredient": "Guaifenesin", "strength": "100 MG/5ML", "unit": "MG/ML", "dose_form": "solution", "tty": "SCD"},

        # Nystatin
        {"rxcui": "7597", "name": "nystatin 100000 UNT/ML Oral Suspension", "ingredient": "Nystatin", "strength": "100000 UNT/ML", "unit": "UNT/ML", "dose_form": "suspension", "tty": "SCD"},

        # Acetaminophen
        {"rxcui": "313782", "name": "acetaminophen 325 MG Oral Tablet", "ingredient": "Acetaminophen", "strength": "325 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "313783", "name": "acetaminophen 500 MG Oral Tablet", "ingredient": "Acetaminophen", "strength": "500 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "313797", "name": "acetaminophen 650 MG Oral Tablet", "ingredient": "Acetaminophen", "strength": "650 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Pravastatin
        {"rxcui": "904475", "name": "pravastatin sodium 40 MG Oral Tablet", "ingredient": "Pravastatin", "strength": "40 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Docusate
        {"rxcui": "1099279", "name": "docusate sodium 100 MG Oral Capsule", "ingredient": "Docusate Sodium", "strength": "100 MG", "unit": "MG", "dose_form": "capsule", "tty": "SCD"},

        # Senna
        {"rxcui": "312935", "name": "senna 8.6 MG Oral Tablet", "ingredient": "Senna", "strength": "8.6 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Clonazepam
        {"rxcui": "197527", "name": "clonazepam 0.5 MG Oral Tablet", "ingredient": "Clonazepam", "strength": "0.5 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},
        {"rxcui": "197528", "name": "clonazepam 1.5 MG Oral Tablet", "ingredient": "Clonazepam", "strength": "1.5 MG", "unit": "MG", "dose_form": "tablet", "tty": "SCD"},

        # Chlorpheniramine
        {"rxcui": "360047", "name": "chlorpheniramine 0.4 MG/ML Oral Solution", "ingredient": "Chlorpheniramine", "strength": "0.4 MG/ML", "unit": "MG/ML", "dose_form": "solution", "tty": "SCD"},

        # Capsaicin
        {"rxcui": "1660761", "name": "capsaicin 0.38 MG/ML Topical Cream", "ingredient": "Capsaicin", "strength": "0.38 MG/ML", "unit": "MG/ML", "dose_form": "cream", "tty": "SCD"},
    ]

    return RxNormKnowledgeBase.from_dict_list(sample_data)
