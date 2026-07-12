"""
Drug Text Normalizer

Chi normalizes search view (khong sua original text):
- lowercase
- whitespace
- unit normalization
- synonym / alias expansion
"""

import re
import unicodedata
from typing import Optional

from src.linking.rxnorm.schema import RxNormEntry


# Ingredient synonyms (common drug name variations)
INGREDIENT_SYNONYMS: dict[str, str] = {
    "paracetamol": "paracetamol",
    "acetaminophen": "paracetamol",
    "tylenol": "paracetamol",
    "panadol": "paracetamol",
    "efferalgan": "paracetamol",
}

# Brand name aliases
BRAND_ALIASES: dict[str, str] = {
    "aspirin": "aspirin",
    "asp": "aspirin",
    "metformin": "metformin",
    "met": "metformin",
    "omeprazole": "omeprazole",
    "pantoprazole": "pantoprazole",
    "amlodipine": "amlodipine",
    "norvasc": "amlodipine",
    "atorvastatin": "atorvastatin",
    "lipitor": "atorvastatin",
    "simvastatin": "simvastatin",
    "zocor": "simvastatin",
    "clopidogrel": "clopidogrel",
    "plavix": "clopidogrel",
    "bisoprolol": "bisoprolol",
    "concor": "bisoprolol",
    "losartan": "losartan",
    "cozaar": "losartan",
    "metronidazole": "metronidazole",
    "flagyl": "metronidazole",
    "ciprofloxacin": "ciprofloxacin",
    "cipro": "ciprofloxacin",
    "azithromycin": "azithromycin",
    "zithromax": "azithromycin",
    "amoxicillin": "amoxicillin",
    "amoxil": "amoxicillin",
    "cefuroxime": "cefuroxime",
    "ceftriaxone": "ceftriaxone",
    "rocephin": "ceftriaxone",
    "prednisolone": "prednisolone",
    "prednison": "prednisolone",
    "dexamethasone": "dexamethasone",
    "decadron": "dexamethasone",
    "alprazolam": "alprazolam",
    "xanax": "alprazolam",
    "diazepam": "diazepam",
    "valium": "diazepam",
    "zopiclone": "zopiclone",
    "imovane": "zopiclone",
    "sertraline": "sertraline",
    "zoloft": "sertraline",
    "sitagliptin": "sitagliptin",
    "januvia": "sitagliptin",
    "glibenclamide": "glibenclamide",
    "gliburide": "glibenclamide",
    "spironolactone": "spironolactone",
    "aldactone": "spironolactone",
}


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs/newlines."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_dashes(text: str) -> str:
    """Normalize various dash characters to space."""
    result = re.sub(r"[‐‑‒–—―–—−]", "-", text)
    result = result.replace("-", " ")
    return result


def expand_ingredient_synonyms(text: str) -> str:
    """Expand ingredient synonyms to canonical forms."""
    result = text.lower()
    for term, canonical in INGREDIENT_SYNONYMS.items():
        result = re.sub(rf'\b{re.escape(term)}\b', canonical, result)
    return result


def expand_brand_aliases(text: str) -> str:
    """Expand brand name aliases."""
    result = text.lower()
    for term, canonical in BRAND_ALIASES.items():
        result = re.sub(rf'\b{re.escape(term)}\b', canonical, result)
    return result


class DrugTextNormalizer:
    """
    Configurable text normalizer for drug retrieval.
    Only normalizes search views, never modifies original text.
    """

    def __init__(
        self,
        lowercase: bool = True,
        normalize_whitespace: bool = True,
        normalize_dashes: bool = True,
        normalize_units: bool = True,
        expand_synonyms: bool = True,
        expand_brands: bool = True,
    ):
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.normalize_dashes = normalize_dashes
        self.normalize_units = normalize_units
        self.expand_synonyms = expand_synonyms
        self.expand_brands = expand_brands

    def normalize(self, text: str) -> str:
        """Apply all enabled normalization steps."""
        if not text:
            return ""

        result = text

        if self.lowercase:
            result = result.lower()

        if self.normalize_dashes:
            result = normalize_dashes(result)

        if self.normalize_whitespace:
            result = normalize_whitespace(result)

        if self.normalize_units:
            result = self._normalize_units(result)

        if self.expand_brands:
            result = expand_brand_aliases(result)

        if self.expand_synonyms:
            result = expand_ingredient_synonyms(result)

        return result

    def normalize_for_matching(self, text: str) -> str:
        """Normalize for exact/structured matching (lighter normalization)."""
        if not text:
            return ""
        result = text.lower()
        result = normalize_dashes(result)
        result = normalize_whitespace(result)
        result = expand_brand_aliases(result)
        return result

    def _normalize_units(self, text: str) -> str:
        """Normalize unit representations (e.g., '0,4' -> '0.4')."""
        # Normalize decimal comma to period
        result = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        return result
