"""
Text Normalization for ICD-10 Retrieval

- Lowercase, whitespace normalization, dash normalization
- Optional tone removal for Vietnamese
- Abbreviation expansion
- Synonym expansion
"""

import re
import unicodedata
from typing import Optional

# Vietnamese tone marks
VIETNAMESE_TONE_MARKS = "àáảãạầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđÀÁẢÃẠẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴĐ"

# Abbreviation / shorthand expansions
VIETNAMESE_ABBREVIATIONS: dict[str, str] = {
    "ts": "tiền sử",
    "tsử": "tiền sử",
    "tăng huyết áp": "tăng huyết áp",
    "huyết áp": "huyết áp",
    "b/a": "bệnh án",
    "bt": "bình thường",
    "kt": "không thấy",
    "kn": "không",
    "k": "không",
    "kp": "không phải",
    "ktg": "không ghi nhận",
    "ct": "chưa",
    "kq": "kết quả",
    "cls": "cận lâm sàng",
    "ls": "lâm sàng",
    "shs": "số huyết sắc tố",
    "bc": "bạch cầu",
    "hct": "hematocrit",
    "hb": "hemoglobin",
    "ht": "hồng cầu",
    "pt": " prothrombin time",
    "aptt": "activated partial thromboplastin time",
    "ck": "creatinine kinase",
    "ckm": "creatinine kinase myocardial fraction",
    "nt": "nhịp tim",
    "huyết động": "huyết động",
    "cl": "chẩn đoán",
    "cp": "cận phẫu thuật",
    "tt": "tiến triển",
    "td": "triệu chứng",
    "thdt": "theo dõi",
    "tn": "tăng",
    "cgd": "cơn gió",
    "cv": "cơn xoắn",
    "cơn": "cơn",
    "nmct": "nhồi máu cơ tim",
    "tdn": "tiểu đường",
    "blt": "bệnh lý tim",
    "bld": "bệnh lý đường",
    "tmc": "tổn thương mạch vành",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "gerd": "bệnh trào ngược dạ dày",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
    "copd": "bệnh phổi tắc nghẽn mạn tính",
}

# Synonym expansions for common medical terms
VIETNAMESE_SYNONYMS: dict[str, str] = {
    "viêm phổi": "viêm phổi",
    "viêm phế quản": "viêm phế quản",
    "viêm mũi dị ứng": "viêm mũi dị ứng",
    "viêm dạ dày": "viêm dạ dày",
    "viêm dạ dày tá tràng": "viêm dạ dày tá tràng",
    "viêm khớp dạng thấp": "viêm khớp dạng thấp",
    "viêm gan": "viêm gan",
    "viêm bàng quang": "viêm bàng quang",
    "viêm đại tràng": "viêm đại tràng",
    "viêm mũi họng": "viêm mũi họng",
    "viêm phổi cộng đồng": "viêm phổi cộng đồng",
    "viêm phổi": "viêm phổi",
    "bệnh phổi tắc nghẽn mạn tính": "bệnh phổi tắc nghẽn mạn tính",
    "bệnh trào ngược dạ dày": "bệnh trào ngược dạ dày",
    "trào ngược dạ dày": "trào ngược dạ dày",
    "suy tim": "suy tim",
    "suy thận": "suy thận",
    "suy giáp": "suy giáp",
    "cường giáp": "cường giáp",
    "tăng huyết áp": "tăng huyết áp",
    "huyết áp cao": "tăng huyết áp",
    "cao huyết áp": "tăng huyết áp",
    "đái tháo đường": "đái tháo đường",
    "tiểu đường": "đái tháo đường",
    "động kinh": "động kinh",
    "co giật": "động kinh",
    "đau nửa đầu": "đau nửa đầu",
    "migraine": "đau nửa đầu",
    "mất ngủ": "mất ngủ",
    "khó ngủ": "mất ngủ",
    "rối loạn giấc ngủ": "mất ngủ",
    "trầm cảm": "trầm cảm",
    "rối loạn lo âu": "rối loạn lo âu",
    "lo âu": "rối loạn lo âu",
    "béo phì": "béo phì",
    "thừa cân": "béo phì",
    "hen suyễn": "hen phế quản",
    "hen": "hen phế quản",
    "viêm phế quản dị ứng": "hen phế quản",
    "nhồi máu cơ tim": "nhồi máu cơ tim",
    "đau tim": "đau thắt ngực",
    "đau ngực": "đau thắt ngực",
    "đau thắt ngực": "đau thắt ngực",
    "rung nhĩ": "rung nhĩ",
    "đột quỵ": "đột quỵ",
    "tai biến mạch não": "đột quỵ",
    "nghiện thuốc lá": "nghiện thuốc lá",
    "hút thuốc": "nghiện thuốc lá",
    "sỏi thận": "sỏi thận",
    "sỏi đường tiết niệu": "sỏi thận",
    "nhiễm trùng tiết niệu": "nhiễm trùng tiết niệu",
    "nhiễm trùng đường tiết niệu": "nhiễm trùng tiết niệu",
    "viêm đường tiết niệu": "nhiễm trùng tiết niệu",
    "viêm đường tiết niệu": "nhiễm trùng tiết niệu",
    "xơ gan": "xơ gan",
    "đau thắt lưng": "đau thắt lưng",
    "đau lưng": "đau thắt lưng",
    "đau bụng": "đau bụng",
    "bệnh mạch vành": "bệnh mạch vành",
    "bệnh tim thiếu máu": "bệnh mạch vành",
    "bệnh tim": "bệnh tim",
    "viêm dạ dày": "viêm dạ dày",
    "loét dạ dày": "loét dạ dày",
    "loét tá tràng": "loét tá tràng",
    "hội chứng ruột kích thích": "hội chứng ruột kích thích",
    "nhiễm trùng hô hấp": "nhiễm trùng hô hấp",
    "nhiễm trùng hô hấp trên": "nhiễm trùng hô hấp trên",
    "viêm mũi họng": "viêm mũi họng",
}


def remove_tones(text: str) -> str:
    """Remove Vietnamese tone marks, keeping base characters."""
    decomposed = unicodedata.normalize("NFD", text)
    result = []
    for char in decomposed:
        if unicodedata.category(char) == "Mn":
            continue
        result.append(char)
    return "".join(result)


def remove_diacritics(text: str) -> str:
    """Remove all diacritics (tone marks and other accents)."""
    decomposed = unicodedata.normalize("NFD", text)
    result = []
    for char in decomposed:
        if unicodedata.category(char) == "Mn":
            continue
        result.append(char)
    return "".join(result)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple spaces/tabs/newlines."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_dashes(text: str) -> str:
    """Normalize various dash characters to hyphen, then to space."""
    result = re.sub(r"[‐‑‒–—―–—−]", "-", text)
    # Replace hyphens with spaces for better word matching
    result = result.replace("-", " ")
    return result


def expand_abbreviations(text: str) -> str:
    """Expand known abbreviations in text."""
    result = text.lower()
    for abbr, expansion in VIETNAMESE_ABBREVIATIONS.items():
        result = re.sub(rf"\b{re.escape(abbr)}\b", expansion, result)
    return result


def expand_synonyms(text: str) -> str:
    """Normalize synonyms to canonical forms."""
    result = text.lower()
    for term, canonical in VIETNAMESE_SYNONYMS.items():
        result = re.sub(rf"\b{re.escape(term)}\b", canonical, result)
    return result


class TextNormalizer:
    """Configurable text normalizer for ICD-10 retrieval."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_tones: bool = False,
        normalize_whitespace: bool = True,
        normalize_dashes: bool = True,
        expand_abbreviations: bool = True,
        expand_synonyms: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_tones = remove_tones
        self.normalize_whitespace = normalize_whitespace
        self.normalize_dashes = normalize_dashes
        self.expand_abbreviations = expand_abbreviations
        self.expand_synonyms = expand_synonyms

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

        if self.expand_abbreviations:
            result = expand_abbreviations(result)

        if self.expand_synonyms:
            result = expand_synonyms(result)

        if self.remove_tones:
            result = remove_tones(result)

        return result

    def normalize_for_alias(self, text: str) -> str:
        """Normalize specifically for alias matching (lighter normalization)."""
        if not text:
            return ""
        result = text.lower()
        result = normalize_dashes(result)
        result = normalize_whitespace(result)
        result = expand_abbreviations(result)
        return result

    def normalize_for_fuzzy(self, text: str) -> str:
        """Normalize for fuzzy matching (tone-insensitive)."""
        if not text:
            return ""
        result = text.lower()
        result = normalize_dashes(result)
        result = normalize_whitespace(result)
        result = expand_abbreviations(result)
        result = expand_synonyms(result)
        result = remove_tones(result)
        return result
