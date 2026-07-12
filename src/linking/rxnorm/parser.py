"""
Drug Mention Parser

Parse drug mention text thanh cac thanh phan cau truc:
- ingredient(s)
- strength value(s)
- strength unit(s)
- dose form
- brand
"""

import re
from typing import Optional

from src.linking.rxnorm.schema import ParsedDrug


class DrugMentionParser:
    """
    Parser de parse drug mention text thanh cac thanh phan cau truc.
    """

    # Strength: so + don vi (VD: 25mg, 0.5 MG, 100 MG/ML, 1g)
    STRENGTH_PATTERN = re.compile(
        r'(\d+(?:[.,]\d+)?)\s*(mg|g|ml|mcg|iu|%)',
        re.IGNORECASE
    )

    # Dose form
    DOSE_FORM_TERMS = {
        "tablet": "tablet",
        "tab": "tablet",
        "capsule": "capsule",
        "cap": "capsule",
        "injection": "injection",
        "inj": "injection",
        "solution": "solution",
        "sol": "solution",
        "cream": "cream",
        "ointment": "ointment",
        "drops": "drops",
        "syrup": "syrup",
        "spray": "spray",
        "patch": "patch",
        "inhaler": "inhaler",
        "powder": "powder",
        "granules": "granules",
        "suspension": "suspension",
    }

    # Brand name indicators
    BRAND_INDICATORS = ["thuốc", "bản", "hãng", "pfizer", "novartis", "gsK", "merck"]

    def parse(self, text: str) -> Optional[ParsedDrug]:
        """
        Parse drug mention text.

        Args:
            text: Drug mention (VD: "Aspirin 81mg", "Metformin 500mg tablet")

        Returns:
            ParsedDrug hoac None
        """
        if not text:
            return None

        text = text.strip()
        original = text

        # Extract all strengths
        strengths: list[tuple[float, str]] = []
        for m in self.STRENGTH_PATTERN.finditer(text):
            val = float(m.group(1).replace(',', '.'))
            unit = m.group(2).upper()
            strengths.append((val, unit))

        # Normalize unit
        normalized_units: list[str] = []
        for _, unit in strengths:
            normalized_units.append(self._normalize_unit(unit))

        # Extract dose form
        dose_form = self._extract_dose_form(text)

        # Extract ingredient: text before first strength
        ingredient_text = text
        strength_match = self.STRENGTH_PATTERN.search(text)
        if strength_match:
            ingredient_text = text[:strength_match.start()].strip()

        # Remove dose form from ingredient
        if dose_form:
            for form_key in self.DOSE_FORM_TERMS:
                idx = ingredient_text.lower().find(form_key)
                if idx >= 0:
                    ingredient_text = ingredient_text[:idx].strip()

        # Split into multiple ingredients (for combinations like "Aspirin + Ibuprofen")
        # Also handle "/" separated ingredients
        ingredient_parts: list[str] = []
        for sep in ['+', '/', '&']:
            if sep in ingredient_text:
                parts = [p.strip() for p in ingredient_text.split(sep)]
                ingredient_parts.extend(parts)
                break
        else:
            ingredient_parts = [ingredient_text.strip()]

        # Clean up each ingredient
        cleaned_ingredients: list[str] = []
        for ing in ingredient_parts:
            ing = re.sub(r'^\d+\s*$', '', ing).strip()
            ing = re.sub(r'^(viên|tablet|capsule|mg|g|ml)\s+', '', ing, flags=re.IGNORECASE).strip()
            if ing:
                cleaned_ingredients.append(ing)

        # Extract brand (simple heuristic)
        brand = self._extract_brand(text)

        return ParsedDrug(
            original=original,
            ingredients=cleaned_ingredients,
            strength_values=[v for v, _ in strengths],
            strength_units=normalized_units,
            dose_form=dose_form,
            brand=brand,
        )

    def _normalize_unit(self, unit: str) -> str:
        """Normalize unit string."""
        unit = unit.upper()
        mapping = {
            "G": "G",
            "GR": "G",
            "MG": "MG",
            "ML": "ML",
            "MCG": "MCG",
            "%": "%",
            "IU": "IU",
        }
        return mapping.get(unit, unit)

    def _extract_dose_form(self, text: str) -> Optional[str]:
        """Extract dose form from text."""
        text_lower = text.lower()
        for term, form in self.DOSE_FORM_TERMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                return form
        return None

    def _extract_brand(self, text: str) -> Optional[str]:
        """Extract brand name from text (simple heuristic)."""
        text_lower = text.lower()
        for indicator in self.BRAND_INDICATORS:
            idx = text_lower.find(indicator)
            if idx >= 0:
                brand = text[idx:].strip()
                return brand
        return None

    def parse_ingredient_only(self, text: str) -> Optional[str]:
        """Extract just the ingredient name."""
        parsed = self.parse(text)
        if parsed:
            return parsed.main_ingredient()
        return None
