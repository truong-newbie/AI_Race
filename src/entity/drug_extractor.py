"""
Drug Entity Extractor - Rule-based drug extraction

Module để extract drug entities từ văn bản y khoa sử dụng regex và dictionary.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Patterns
# =============================================================================

# Common drug name patterns - using specific word boundaries
DRUG_PATTERNS = [
    # Drug with dosage - must have word boundary before drug name
    r"(?<![a-zA-Z])\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mcg|iu|%))\b",
    # Specific drug suffixes with positive lookbehind for common drug contexts
    r"(?<![a-zA-Z])(?:paracetamol|aspirin|metformin|amlodipine|losartan|ceftriaxone|paracetamol|acetaminophen|ibuprofen|omeprazole|esomeprazole|lansoprazole|pantoprazole|ranitidine|metoprolol|atenolol|bisoprolol|carvedilol|hydrochlorothiazide|furosemide|spironolactone|simvastatin|atorvastatin|rosuvastatin|lisinopril|enalapril|ramipril|vildagliptin|sitagliptin|glimepiride|glipizide|metformin|gliclazide|acarbose|dutasteride|finasteride|tamsulosin|doxazosin|prazosin|ciprofloxacin|levofloxacin|moxifloxacin|azithromycin|clarithromycin|erythromycin|amoxicillin|clavulanate|cephalexin|ceftazidime|ceftriaxone|cefotaxime|metronidazole|tinidazole|ciprofloxacin|norfloxacin|ofloxacin)\b",
]

# Vietnamese drug-related terms
VIETNAMESE_DRUG_TERMS = [
    r"thuốc\s+\w+",
    r"dùng\s+\w+",
    r"uống\s+\w+",
    r"tiêm\s+\w+",
    r"điều trị\s+\w+",
    r"đã\s+(?:sử dụng|dùng|uống)\s+\w+",
    r"đang\s+(?:dùng|sử dụng)\s+\w+",
    r"kê\s+\w+",
]

# Dosage patterns
DOSAGE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(mg|g|ml|mcg|iu|%)',
    re.IGNORECASE
)

# Route patterns
ROUTE_PATTERN = re.compile(
    r'\b(po|oral|ngậm|tiêm|ở|tại)\b',
    re.IGNORECASE
)

# Frequency patterns
FREQUENCY_PATTERN = re.compile(
    r'\b(q\d+h?|qd|bid|tid|qid|daily|ngày|ngày\s*\d+|lần|tối|sáng|chiều|prn|tùy\s*tình\s*trạng)\b',
    re.IGNORECASE
)


@dataclass
class DrugMatch:
    """Kết quả match của một drug mention."""
    text: str
    start: int
    end: int
    ingredient: Optional[str] = None
    strength: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    confidence: float = 1.0


class DrugExtractor:
    """
    Rule-based drug extractor.

    Extracts drug mentions và parse thành structured components.
    """

    def __init__(self, drug_kb: Optional = None):
        self.drug_kb = drug_kb
        self.dosage_pattern = DOSAGE_PATTERN
        self.route_pattern = ROUTE_PATTERN
        self.freq_pattern = FREQUENCY_PATTERN

    def extract(self, text: str) -> list[DrugMatch]:
        """
        Extract drug mentions từ text.

        Args:
            text: Input text

        Returns:
            List of DrugMatch
        """
        matches = []

        # Pattern 1: Specific drug name + dosage (e.g., "Aspirin 81 mg", "metoprolol 50mg")
        drug_with_dosage_pattern = re.compile(
            r'\b([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mcg|iu|%))\b',
            re.IGNORECASE
        )
        for match in drug_with_dosage_pattern.finditer(text):
            drug_name = match.group(1).strip()
            dosage = match.group(2)
            # Only include if drug name is reasonably long (not random letters)
            if len(drug_name) >= 4:
                matches.append(DrugMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    ingredient=drug_name,
                    strength=dosage,
                    confidence=0.9
                ))

        # Pattern 2: Common drug suffixes (e.g., "paracetamol", "ceftriaxone")
        common_drugs = [
            'paracetamol', 'acetaminophen', 'aspirin', 'ibuprofen', 'metformin',
            'amlodipine', 'losartan', 'metoprolol', 'atenolol', 'bisoprolol',
            'carvedilol', 'ceftriaxone', 'omeprazole', 'esomeprazole', 'pantoprazole',
            'simvastatin', 'atorvastatin', 'rosuvastatin', 'lisinopril', 'enalapril',
            'ramipril', 'hydrochlorothiazide', 'furosemide', 'spironolactone',
            'azithromycin', 'clarithromycin', 'amoxicillin', 'metronidazole',
            'gliclazide', 'glimepiride', 'sitagliptin', 'vildagliptin',
        ]
        drug_suffix_pattern = re.compile(
            r'\b(' + '|'.join(common_drugs) + r')(?:\s+\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mcg|iu|%))?\b',
            re.IGNORECASE
        )
        for match in drug_suffix_pattern.finditer(text):
            drug_name = match.group(1)
            # Check if this overlap with an existing match
            overlap = False
            for existing in matches:
                if match.start() < existing.end and existing.start < match.end():
                    overlap = True
                    break
            if not overlap:
                matches.append(DrugMatch(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    ingredient=drug_name,
                    confidence=0.85
                ))

        # Pattern 3: Vietnamese drug context (uống/thuốc/tiêm + drug name)
        for pattern_str in VIETNAMESE_DRUG_TERMS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                # Extract drug name after the keyword
                after = text[match.end():match.end() + 50].strip()
                drug_match = re.match(r'([A-Za-z]+(?:\s+[A-Za-z]+)?)(?:\s+\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|g|iu|%))?', after)
                if drug_match:
                    drug_text = drug_match.group(1)
                    if len(drug_text) >= 4:
                        full_text = text[match.start():match.end() + len(drug_text)]
                        # Check overlap
                        overlap = False
                        for existing in matches:
                            if match.start() < existing.end and existing.start < match.end() + len(drug_text):
                                overlap = True
                                break
                        if not overlap:
                            matches.append(DrugMatch(
                                text=full_text,
                                start=match.start(),
                                end=match.end() + len(drug_text),
                                ingredient=drug_text,
                                confidence=0.8
                            ))

        # Deduplicate and merge overlapping matches
        matches = self._deduplicate(matches)

        return matches

    def _deduplicate(self, matches: list[DrugMatch]) -> list[DrugMatch]:
        """Remove duplicate and overlapping matches."""
        if not matches:
            return []

        # Sort by start position, then by length (longer first)
        sorted_matches = sorted(matches, key=lambda x: (x.start, -(x.end - x.start)))

        result = [sorted_matches[0]]
        for match in sorted_matches[1:]:
            last = result[-1]
            # If overlapping, keep the longer one
            if match.start < last.end:
                if (match.end - match.start) > (last.end - last.start):
                    result[-1] = match
            else:
                result.append(match)

        return result

    def parse_drug_text(self, text: str) -> DrugMatch:
        """
        Parse drug mention text thành components.

        Args:
            text: Drug mention text (e.g., "Aspirin 81 mg po daily")

        Returns:
            DrugMatch with parsed components
        """
        match = DrugMatch(text=text, start=0, end=len(text))

        # Extract dosage
        dosage_match = self.dosage_pattern.search(text)
        if dosage_match:
            match.strength = dosage_match.group(0)

        # Extract route
        route_match = self.route_pattern.search(text)
        if route_match:
            match.route = route_match.group(1)

        # Extract frequency
        freq_match = self.freq_pattern.search(text)
        if freq_match:
            match.frequency = freq_match.group(1)

        # Extract ingredient (text before dosage)
        if dosage_match:
            match.ingredient = text[:dosage_match.start()].strip()

        return match


def extract_drugs_simple(text: str) -> list[dict]:
    """
    Simple drug extraction without KB.

    Args:
        text: Input text

    Returns:
        List of drug dicts
    """
    extractor = DrugExtractor()
    matches = extractor.extract(text)

    results = []
    for m in matches:
        results.append({
            "text": m.text,
            "position": [m.start, m.end],
            "type": "THUỐC",
            "ingredient": m.ingredient,
            "strength": m.strength,
            "route": m.route,
            "frequency": m.frequency,
            "confidence": m.confidence
        })

    return results


# =============================================================================
# Tests
# =============================================================================

def test_drug_extractor():
    """Test drug extractor."""
    extractor = DrugExtractor()

    text = """
    Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML,
    Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích tế bào máu.
    Danh sách thuốc: 1. amlodipine 10 mg po daily
    2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily
    """

    print("=== Extracted Drugs ===")
    matches = extractor.extract(text)
    for m in matches:
        print(f"  [{m.start}:{m.end}] {m.text}")
        print(f"    ingredient: {m.ingredient}")
        print(f"    strength: {m.strength}")
        print(f"    route: {m.route}")
        print()


if __name__ == "__main__":
    test_drug_extractor()
