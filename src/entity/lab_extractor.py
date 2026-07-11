"""
Laboratory Test Extractor - Regex patterns cho xét nghiệm

Module để extract tên xét nghiệm và kết quả xét nghiệm từ văn bản y khoa.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Patterns
# =============================================================================

# Common lab test names (tiếng Việt và tiếng Anh)
LAB_TEST_NAMES = [
    # Blood tests
    r"WBC", r"NEUT%", r"NEUT#", r"LYMPH%", r"LYMPH#", r"RBC", r"HGB", r"HCT",
    r"MCV", r"MCH", r"MCHC", r"PLT", r"MPV", r"PDW",
    # Biochemistry
    r"Glucose", r"GLU", r"Creatinin", r"CRE", r"BUN", r"Urea",
    r"AST", r"ALT", r"GGT", r"ALP", r"TP", r"ALB",
    r"Total Bilirubin", r"TBIL", r"Direct Bilirubin", r"DBIL",
    r"Cholesterol", r"CHOL", r"Triglyceride", r"TG",
    r"HDL", r"LDL", r"LDL-C", r"VLDL",
    # Kidney function
    r"eGFR",
    # Liver function
    r"LDH",
    # Cardiac markers
    r"Troponin", r"CPK", r"CK-MB", r"BNP", r"NT-proBNP",
    # Electrolytes
    r"Na", r"K", r"Cl", r"Ca", r"Mg", r"Phosphate",
    # Thyroid
    r"TSH", r"FT3", r"FT4", r"T3", r"T4",
    # Urine
    r"pH", r"Protein", r"Glucose", r"Ketone", r"Blood", r"Leukocyte",
    # Coagulation
    r"PT", r"INR", r"APTT", r"Fibrinogen", r"D-dimer",
    # Inflammatory markers
    r"CRP", r"ESR", r"Procalcitonin", r"PCT",
    # Blood group
    r"ABO", r"Rh",
    # Vietnamese terms
    r"tổng phân tích tế bào máu", r"tbm", r"tế bào máu",
    r"huyết đồ", r"công thức máu", r"CTM",
    r"đường huyết", r"đường máu",
    r"chức năng gan", r"chức năng thận",
    r"ion đồ", r"điện giải",
]

# Combined pattern for lab test names
LAB_TEST_PATTERN = re.compile(
    r'\b(' + '|'.join(LAB_TEST_NAMES) + r')\b',
    re.IGNORECASE
)

# Vietnamese lab test full patterns
VIETNAMESE_LAB_PATTERNS = [
    # Vietnamese full names
    (r"tổng phân tích tế bào máu", "Tổng phân tích tế bào máu"),
    (r"huyết đồ", "Huyết đồ"),
    (r"công thức máu", "Công thức máu"),
    (r"chức năng gan", "Chức năng gan"),
    (r"chức năng thận", "Chức năng thận"),
    (r"điện giải đồ", "Điện giải đồ"),
    (r"ion đồ", "Ion đồ"),
    (r"đường huyết", "Đường huyết"),
]

# Result value patterns
RESULT_VALUE_PATTERN = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(?:[-–]\s*(\d+(?:[.,]\d+)?))?\s*'  # Value or range
    r'(%|g\/dL|g\/L|mg\/dL|mg\/L|mmol\/L|mmol\/l|'
    r'u\/L|u\/l|IU\/L|ng\/mL|pg\/mL|mg\/mL|'
    r'|UNT|%|mEq\/L)?',  # Units
    re.IGNORECASE
)

# Reference range pattern (e.g., "3.5-5.0")
REFERENCE_RANGE_PATTERN = re.compile(
    r'\(([^)]+)\)',  # Text in parentheses
)

# Common lab units
LAB_UNITS = [
    "mg/dL", "g/dL", "g/L", "mmol/L", "mmol/l",
    "mEq/L", "u/L", "u/l", "IU/L",
    "ng/mL", "pg/mL", "mg/mL", "mg/g",
    "%", "UNT", "10^3/uL", "10^6/uL",
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LabTestMatch:
    """Kết quả match của một lab test."""
    text: str
    start: int
    end: int
    normalized_name: str
    test_type: str = "UNKNOWN"  # TÊN_XÉT_NGHIỆM

    def is_result(self) -> bool:
        """Check if this is a result value."""
        return self.test_type == "KẾT_QUẢ_XÉT_NGHIỆM"


@dataclass
class LabResultMatch:
    """Kết quả match của một giá trị xét nghiệm."""
    text: str
    start: int
    end: int
    value: Optional[float] = None
    unit: Optional[str] = None
    is_abnormal: bool = False
    reference_range: Optional[str] = None


# =============================================================================
# Lab Extractor
# =============================================================================

class LabTestExtractor:
    """
    Extractor cho laboratory tests và results.

    Extracts:
    - TÊN_XÉT_NGHIỆM: Tên xét nghiệm (WBC, Glucose, etc.)
    - KẾT_QUẢ_XÉT_NGHIỆM: Giá trị kết quả (14.43, 76.4, etc.)
    """

    def __init__(self):
        self.lab_pattern = LAB_TEST_PATTERN
        self.result_pattern = RESULT_VALUE_PATTERN
        self.ref_pattern = REFERENCE_RANGE_PATTERN

        # Vietnamese lab patterns
        self.viet_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in VIETNAMESE_LAB_PATTERNS
        ]

        # Common lab test synonyms (normalized name -> standard name)
        self.synonym_map = {
            "wbc": "WBC",
            "rbc": "RBC",
            "hgb": "HGB",
            "hct": "HCT",
            "plt": "PLT",
            "glu": "Glucose",
            "cre": "Creatinin",
            "bun": "Urea",
            "ast": "AST",
            "alt": "ALT",
            "tbil": "Total Bilirubin",
            "chol": "Cholesterol",
            "tg": "Triglyceride",
            "tbm": "Tổng phân tích tế bào máu",
            "ctm": "Công thức máu",
        }

    def extract_tests(self, text: str) -> list[LabTestMatch]:
        """
        Extract all lab test names from text.

        Args:
            text: Input text

        Returns:
            List of LabTestMatch
        """
        results = []

        # Extract using main pattern
        for match in self.lab_pattern.finditer(text):
            normalized = self.synonym_map.get(match.group().lower(), match.group())
            results.append(LabTestMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                normalized_name=normalized,
                test_type="TÊN_XÉT_NGHIỆM"
            ))

        # Extract Vietnamese full patterns
        for pattern, name in self.viet_patterns:
            for match in pattern.finditer(text):
                results.append(LabTestMatch(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    normalized_name=name,
                    test_type="TÊN_XÉT_NGHIỆM"
                ))

        return results

    def extract_results(self, text: str) -> list[LabResultMatch]:
        """
        Extract all lab result values from text.

        Args:
            text: Input text

        Returns:
            List of LabResultMatch
        """
        results = []

        for match in self.result_pattern.finditer(text):
            value_str = match.group(1)
            value = None
            try:
                value = float(value_str.replace(',', '.'))
            except ValueError:
                continue

            unit = match.group(2) if match.group(2) else None

            # Check if abnormal (value followed by H/L or *)
            is_abnormal = False
            after_match = text[match.end():match.end()+5].upper()
            if 'H' in after_match or 'L' in after_match or '*' in after_match:
                is_abnormal = True

            results.append(LabResultMatch(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                value=value,
                unit=unit,
                is_abnormal=is_abnormal
            ))

        return results

    def extract_all(self, text: str) -> tuple[list[LabTestMatch], list[LabResultMatch]]:
        """
        Extract both test names and results.

        Args:
            text: Input text

        Returns:
            Tuple of (lab_tests, lab_results)
        """
        tests = self.extract_tests(text)
        results = self.extract_results(text)
        return tests, results

    def extract_lab_panel(self, text: str) -> dict:
        """
        Extract complete lab panel information.

        Finds test names and their associated results.

        Args:
            text: Input text

        Returns:
            Dict mapping test names to results
        """
        tests, results = self.extract_all(text)

        panel = {}

        # Associate results with nearby tests
        for result in results:
            # Find nearest test before this result
            nearest_test = None
            min_distance = float('inf')

            for test in tests:
                if test.end <= result.start:
                    distance = result.start - test.end
                    if distance < min_distance:
                        min_distance = distance
                        nearest_test = test

            if nearest_test and min_distance < 20:  # Within 20 chars
                test_name = nearest_test.normalized_name
                if test_name not in panel:
                    panel[test_name] = []
                panel[test_name].append({
                    "value": result.value,
                    "unit": result.unit,
                    "position": [result.start, result.end],
                    "is_abnormal": result.is_abnormal
                })

        return panel


# =============================================================================
# Unit Tests
# =============================================================================

def test_lab_extractor():
    """Test lab extractor với sample text."""
    extractor = LabTestExtractor()

    text = """
    Đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm):
    WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4;
    LYPH% (Tỷ lệ bạch cầu lympho):12,8;
    """

    print("=== Lab Tests ===")
    tests = extractor.extract_tests(text)
    for test in tests:
        print(f"  [{test.start}:{test.end}] {test.text} -> {test.normalized_name}")

    print("\n=== Lab Results ===")
    results = extractor.extract_results(text)
    for result in results:
        print(f"  [{result.start}:{result.end}] {result.text} = {result.value} {result.unit}")

    print("\n=== Lab Panel ===")
    panel = extractor.extract_lab_panel(text)
    for test_name, values in panel.items():
        print(f"  {test_name}: {values}")


if __name__ == "__main__":
    test_lab_extractor()
