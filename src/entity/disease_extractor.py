"""
Disease Entity Extractor - Rule-based disease extraction

Module để extract disease/diagnosis entities từ văn bản y khoa.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Patterns
# =============================================================================

# Vietnamese disease-related keywords
VIETNAMESE_DISEASE_KEYWORDS = [
    # Disease terms (order matters - more specific first)
    r"viêm phổi",
    r"viêm phế quản",
    r"viêm dạ dày",
    r"viêm gan",
    r"viêm ruột",
    r"viêm mũi",
    r"viêm họng",
    r"viêm màng não",
    r"viêm amidan",
    r"viêm xoang",
    r"viêm túi mật",
    r"viêm khớp",
    r"viêm thần kinh",
    r"viêm cơ tim",
    r"viêm phúc mạc",
    r"viêm bìu",
    r"viêm bao quy đầu",
    r"viêm cổ tử cung",
    r"viêm nội mạc tử cung",
    r"viêm ống cổ tử cung",
    r"viêm phần phụ",
    r"viêm tử cung",
    r"viêm da cơ địa",
    r"viêm da dị ứng",
    r"viêm da tiếp xúc",
    r"viêm loét đại trực tràng",
    r"viêm loét miệng",
    r"viêm miệng",
    r"viêm mô tế bào",

    # Common conditions
    r"trào ngược dạ dày",
    r"trào ngược",
    r"đái tháo đường",
    r"tiểu đường",
    r"tăng huyết áp",
    r"cao huyết áp",
    r"huyết áp cao",
    r"hen suyễn",
    r"hen phế quản",
    r"phổi tắc nghẽn",
    r"bệnh phổi tắc nghẽn",
    r"đau thượng vị",
    r"đau dạ dày",
    r"khó tiêu",
    r"ợ hơi",
    r"ợ chua",
    r"tức ngực",
    r"đau ngực",
    r"đau bụng",
    r"đau đầu",
    r"đau lưng",
    r"đau cơ",
    r"đau khớp",
    r"đau tim",
    r"ho gà",
    r"ho khan",
    r"ho đờm",
    r"sốt xuất huyết",
    r"sốt ret",
    r"sốt dengue",
    r"sốt malaria",
    r"sốt phát ban",
    r"sốt virus",
    r"sốt kèm",
    r"tiêu chảy cấp",
    r"tiêu chảy mãn",
    r"táo bón mãn",
    r"mất ngủ",
    r"lo âu",
    r"trầm cảm",
    r"rối loạn lo âu",
    r"rối loạn giấc ngủ",
    r"rối loạn tiêu hóa",
    r"rối loạn nhịp tim",
    r"rối loạn lipid máu",
    r"rối loạn chuyển hóa",
    r"rối loạn tuyến giáp",
    r"suy thận",
    r"suy tim",
    r"suy gan",
    r"suy hô hấp",
    r"suy giảm miễn dịch",
    r"tăng cholesterol",
    r"tăng triglyceride",
    r"tăng đường huyết",
    r"tăng acid uric",
    r"hạ đường huyết",
    r"hạ canxi máu",
    r"hạ kali máu",
    r"hạ natri máu",
    r"tăng kali máu",
    r"tăng canxi máu",
    r"bướu cổ",
    r"basedow",
    r"hashimoto",
    r"ung thư",
    r"khối u",
    r"u lành tính",
    r"u ác tính",
    r"polyp",
    r"nhiễm trùng",
    r"nhiễm nấm",
    r"nhiễm khuẩn",
    r"nhiễm virus",
    r"lao phổi",
    r"lao",
    r"nhồi máu cơ tim",
    r"tai biến mạch máu não",
    r"đột quỵ",
    r"xuất huyết não",
    r"xơ gan",
    r"xơ vữa động mạch",
    r"gout",
    r"thống phong",
    r"lupus ban đỏ",
    r"dị ứng",
    r"quá mẫn",
    r"phù quinck",
    r"sốc phản vệ",
    r"hen",
    r"co thắt phế quản",
    r"cơn hen",
    r"cơn đau thắt ngực",
    r"đau thắt ngực",
    r"nhồi máu",
    r"tắc mạch",
    r"thuyên tắc",
    r"giãn tĩnh mạch",
    r"suy giãn tĩnh mạch",
    r"trĩ",
    r"nứt hậu môn",
    r"mụn trứng cá",
    r"viêm da",
    r"chàm",
    r" eczema",
    r"psoriasis",
    r"vảy nến",
    r" zona",
    r"herpes",
    r"HIV",
    r"AIDS",
    r"viêm não",
    r"viêm màng não",
    r"viêm tủy",
    r"động kinh",
    r" Parkinson",
    r"Alzheimer",
    r"sa sút trí tuệ",
    r"loạn thần",
    r"tâm thần phân liệt",
    r"rối loạn lưỡng cực",
    r"rối loạn cảm xúc",
    r"chán nản",
    r"stress",
    r"mệt mỏi mãn tính",
    r"xạ trị",
    r"hóa trị",
    r"cắt cụt",
    r"ghép",
    r"ghép thận",
    r"ghép gan",
    r"ghép tim",
    r"chạy thận",
    r"lọc máu",
    r"thẩm phân",
    r"oxy",
    r"thở máy",
    r"đặt nội khí quản",
    r"đặt sonde",
    r" sonde dạ dày",
    r"ống thông",
    r" catheter",
]

# English disease patterns
ENGLISH_DISEASE_PATTERNS = [
    r"\b(?:GERD|GORD|GERD)\b",
    r"\b(?:diabetes|DM|T1DM|T2DM)\b",
    r"\b(?:hypertension|HTN|high blood pressure)\b",
    r"\b(?:asthma|COPD|pneumonia|bronchitis)\b",
    r"\b(?:gastritis|peptic ulcer|GERD)\b",
    r"\b(?:hepatitis|liver disease)\b",
    r"\b(?:arthritis|osteoarthritis|rheumatoid)\b",
    r"\b(?:cancer|carcinoma|tumor|malignancy)\b",
    r"\b(?:infection|inflammation|injury)\b",
]

# Combine all patterns
DISEASE_PATTERN = re.compile(
    '|'.join(VIETNAMESE_DISEASE_KEYWORDS + ENGLISH_DISEASE_PATTERNS),
    re.IGNORECASE
)

# Diagnostic context markers
DIAGNOSTIC_MARKERS = [
    r"chẩn\s+đoán",
    r"được\s+chẩn\s+đoán",
    r"chẩn\s+đoán\s+là",
    r"bệnh\s+chính",
    r"bệnh\t+hiểu",
    r"primary\s+diagnosis",
]

DIAGNOSTIC_PATTERN = re.compile(
    '|'.join(DIAGNOSTIC_MARKERS),
    re.IGNORECASE
)


@dataclass
class DiseaseMatch:
    """Kết quả match của một disease mention."""
    text: str
    start: int
    end: int
    context: str = "UNKNOWN"  # CHẨN_ĐOÁN, TRIỆU_CHỨNG, etc.
    confidence: float = 1.0
    is_diagnosed: bool = False


class DiseaseExtractor:
    """
    Rule-based disease extractor.

    Extracts disease mentions và classifies context.
    """

    def __init__(self, disease_kb: Optional = None):
        self.disease_kb = disease_kb
        self.pattern = DISEASE_PATTERN
        self.diagnostic_pattern = DIAGNOSTIC_PATTERN

    def extract(self, text: str) -> list[DiseaseMatch]:
        """
        Extract disease mentions từ text.

        Args:
            text: Input text

        Returns:
            List of DiseaseMatch
        """
        matches = []

        # Find all disease mentions
        for match in self.pattern.finditer(text):
            disease_text = match.group().strip()

            # Determine context by looking at surrounding text
            context = self._determine_context(text, match.start(), match.end())

            # Check if this is a diagnosed condition
            is_diagnosed = self._is_diagnosed(text, match.start())

            matches.append(DiseaseMatch(
                text=disease_text,
                start=match.start(),
                end=match.end(),
                context=context,
                is_diagnosed=is_diagnosed,
                confidence=0.8 if context == "TRIỆU_CHỨNG" else 0.9
            ))

        # Remove duplicates and overlapping
        matches = self._deduplicate(matches)

        return matches

    def _determine_context(self, text: str, start: int, end: int) -> str:
        """
        Determine if the disease mention is a diagnosis or symptom.

        Returns:
            CHẨN_ĐOÁN or TRIỆU_CHỨNG
        """
        # Look backwards for diagnostic markers
        look_back = 100
        before_text = text[max(0, start - look_back):start].lower()

        # Check for diagnostic markers - if found, it's a diagnosis
        if self.diagnostic_pattern.search(before_text):
            return "CHẨN_ĐOÁN"

        # Check for symptom markers - if found, it's a symptom
        symptom_markers = [
            "triệu chứng", "biểu hiện", "phàn nàn", "thấy", "kêu"
        ]

        for marker in symptom_markers:
            if marker in before_text:
                return "TRIỆU_CHỨNG"

        # Check for context keywords that suggest symptoms
        symptom_context = ["ho", "đau", "sốt", "tức", "khó", "mệt", "ngứa", "ợ"]
        for keyword in symptom_context:
            # Only if keyword immediately precedes (within 10 chars)
            prefix = before_text[-15:] if len(before_text) > 15 else before_text
            if keyword in prefix:
                return "TRIỆU_CHỨNG"

        # Check for negative markers (loại trừ, không) - suggests it's a diagnosis being ruled out
        if "loại trừ" in before_text or "không" in before_text:
            return "CHẨN_ĐOÁN"

        # Default based on pattern type - specific diseases are likely diagnoses
        match_text = text[start:end].lower()
        diagnosis_patterns = ["viêm phổi", "viêm phế quản", "viêm gan", "viêm dạ dày",
                              "đái tháo đường", "tăng huyết áp", "trào ngược", "hen suyễn",
                              "nhồi máu", "tai biến"]
        for pattern in diagnosis_patterns:
            if pattern in match_text:
                return "CHẨN_ĐOÁN"

        # Default to symptom if still unclear
        return "TRIỆU_CHỨNG"

    def _is_diagnosed(self, text: str, position: int) -> bool:
        """Check if the disease is mentioned as diagnosed."""
        look_back = 100
        before_text = text[max(0, position - look_back):position].lower()

        diagnosed_markers = [
            "chẩn đoán", "được chẩn đoán", "xác định",
            "chẩn đoán:", "chẩn đoán là",
            "confirm", "diagnosed", "confirmed", "diagnosis:"
        ]

        for marker in diagnosed_markers:
            if marker in before_text:
                return True

        return False

    def _deduplicate(self, matches: list[DiseaseMatch]) -> list[DiseaseMatch]:
        """Remove duplicate and overlapping matches."""
        if not matches:
            return []

        sorted_matches = sorted(matches, key=lambda x: (x.start, -(x.end - x.start)))

        result = [sorted_matches[0]]
        for match in sorted_matches[1:]:
            last = result[-1]
            if match.start < last.end:
                if (match.end - match.start) > (last.end - last.start):
                    result[-1] = match
            else:
                result.append(match)

        return result


def extract_diseases_simple(text: str) -> list[dict]:
    """
    Simple disease extraction.

    Args:
        text: Input text

    Returns:
        List of disease dicts
    """
    extractor = DiseaseExtractor()
    matches = extractor.extract(text)

    results = []
    for m in matches:
        results.append({
            "text": m.text,
            "position": [m.start, m.end],
            "type": m.context,
            "confidence": m.confidence,
            "is_diagnosed": m.is_diagnosed
        })

    return results


# =============================================================================
# Tests
# =============================================================================

def test_disease_extractor():
    """Test disease extractor."""
    extractor = DiseaseExtractor()

    text = """
    Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực,
    đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản.
    Tiền sử tăng huyết áp, đái tháo đường type 2.
    """

    print("=== Extracted Diseases ===")
    matches = extractor.extract(text)
    for m in matches:
        print(f"  [{m.start}:{m.end}] {m.text}")
        print(f"    type: {m.context}")
        print(f"    diagnosed: {m.is_diagnosed}")
        print()


if __name__ == "__main__":
    test_disease_extractor()
