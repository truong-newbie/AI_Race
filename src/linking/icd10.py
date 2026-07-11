"""
ICD-10 Knowledge Base Loader và Preprocessing

Module để load và preprocess ICD-10 codes thành structured format
phục vụ cho entity linking.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ICD10Entry:
    """Một entry trong ICD-10 knowledge base."""
    code: str                           # Mã ICD-10 (VD: K21.0)
    name: str                           # Tên tiếng Anh
    name_vi: Optional[str] = None       # Tên tiếng Việt (nếu có)
    parent_code: Optional[str] = None   # Mã cha
    chapter: Optional[str] = None       # Chapter
    description: Optional[str] = None    # Mô tả
    synonyms: list[str] = field(default_factory=list)  # Các tên gọi khác
    aliases: list[str] = field(default_factory=list)   # Các alias
    include_terms: list[str] = field(default_factory=list)  # Terms bao gồm
    exclude_terms: list[str] = field(default_factory=list)    # Terms loại trừ

    def to_dict(self) -> dict:
        """Convert sang dict."""
        return {
            "code": self.code,
            "name": self.name,
            "name_vi": self.name_vi,
            "parent_code": self.parent_code,
            "chapter": self.chapter,
            "description": self.description,
            "synonyms": self.synonyms,
            "aliases": self.aliases,
            "include_terms": self.include_terms,
            "exclude_terms": self.exclude_terms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ICD10Entry":
        """Tạo từ dict."""
        return cls(
            code=data["code"],
            name=data.get("name", ""),
            name_vi=data.get("name_vi"),
            parent_code=data.get("parent_code"),
            chapter=data.get("chapter"),
            description=data.get("description"),
            synonyms=data.get("synonyms", []),
            aliases=data.get("aliases", []),
            include_terms=data.get("include_terms", []),
            exclude_terms=data.get("exclude_terms", []),
        )


class ICD10KnowledgeBase:
    """
    ICD-10 Knowledge Base để tra cứu và fuzzy matching.

    Supported operations:
    - Exact code lookup
    - Name lookup (exact và fuzzy)
    - Code tree traversal (parent/children)
    - Synonym search
    """

    def __init__(self):
        self.entries: dict[str, ICD10Entry] = {}
        self.code_to_codes: dict[str, list[str]] = {}  # normalized -> original codes
        self.name_index: dict[str, str] = {}  # normalized name -> code
        self.synonym_index: dict[str, str] = {}  # normalized synonym -> code

    def add_entry(self, entry: ICD10Entry) -> None:
        """Thêm một entry vào KB."""
        self.entries[entry.code] = entry

        # Index theo normalized name
        normalized = self._normalize(entry.name)
        if normalized and normalized not in self.name_index:
            self.name_index[normalized] = entry.code

        # Index synonyms
        for syn in entry.synonyms:
            norm_syn = self._normalize(syn)
            if norm_syn and norm_syn not in self.synonym_index:
                self.synonym_index[norm_syn] = entry.code

        # Index aliases
        for alias in entry.aliases:
            norm_alias = self._normalize(alias)
            if norm_alias and norm_alias not in self.synonym_index:
                self.synonym_index[norm_alias] = entry.code

        # Index Vietnamese name
        if entry.name_vi:
            norm_vi = self._normalize(entry.name_vi)
            if norm_vi and norm_vi not in self.name_index:
                self.name_index[norm_vi] = entry.code

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text cho việc indexing."""
        if not text:
            return ""
        # Lowercase, strip, remove extra spaces
        return " ".join(text.lower().strip().split())

    def get_by_code(self, code: str) -> Optional[ICD10Entry]:
        """Lấy entry theo code."""
        return self.entries.get(code)

    def get_by_name(self, name: str) -> Optional[ICD10Entry]:
        """Lấy entry theo exact name match."""
        normalized = self._normalize(name)
        code = self.name_index.get(normalized)
        if code:
            return self.entries.get(code)
        return None

    def get_by_synonym(self, synonym: str) -> Optional[ICD10Entry]:
        """Lấy entry theo synonym match."""
        normalized = self._normalize(synonym)
        code = self.synonym_index.get(normalized)
        if code:
            return self.entries.get(code)
        return None

    def get_children(self, code: str) -> list[ICD10Entry]:
        """Lấy tất cả entries có parent_code = code."""
        return [
            entry for entry in self.entries.values()
            if entry.parent_code == code
        ]

    def get_ancestors(self, code: str) -> list[ICD10Entry]:
        """Lấy tất cả ancestors của một code."""
        ancestors = []
        current = self.get_by_code(code)
        while current and current.parent_code:
            parent = self.get_by_code(current.parent_code)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def search(self, query: str, limit: int = 10) -> list[ICD10Entry]:
        """
        Search entries by name, synonym, hoặc alias.

        Returns entries sorted by relevance (exact match > starts with > contains).
        """
        normalized = self._normalize(query)
        if not normalized:
            return []

        results = []

        # 1. Exact match on name
        if normalized in self.name_index:
            code = self.name_index[normalized]
            if code in self.entries:
                results.append(self.entries[code])

        # 2. Exact match on synonym
        if normalized in self.synonym_index:
            code = self.synonym_index[normalized]
            if code in self.entries and self.entries[code] not in results:
                results.append(self.entries[code])

        # 3. Starts with match
        for norm_name, code in self.name_index.items():
            if norm_name.startswith(normalized) and code in self.entries:
                if self.entries[code] not in results:
                    results.append(self.entries[code])

        # 4. Contains match
        for norm_name, code in self.name_index.items():
            if normalized in norm_name and code in self.entries:
                if self.entries[code] not in results:
                    results.append(self.entries[code])

        return results[:limit]

    def get_all_codes(self) -> set[str]:
        """Lấy tất cả codes trong KB."""
        return set(self.entries.keys())

    def size(self) -> int:
        """Số lượng entries trong KB."""
        return len(self.entries)

    @classmethod
    def from_dict_list(cls, data: list[dict]) -> "ICD10KnowledgeBase":
        """Tạo KB từ list of dicts."""
        kb = cls()
        for item in data:
            entry = ICD10Entry.from_dict(item)
            kb.add_entry(entry)
        return kb

    def to_dict_list(self) -> list[dict]:
        """Convert KB sang list of dicts."""
        return [entry.to_dict() for entry in self.entries.values()]

    def save(self, path: Union[str, Path]) -> None:
        """Save KB ra file JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict_list(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved ICD-10 KB to {path} ({self.size()} entries)")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ICD10KnowledgeBase":
        """Load KB từ file JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        kb = cls.from_dict_list(data)
        logger.info(f"Loaded ICD-10 KB from {path} ({kb.size()} entries)")
        return kb


def create_sample_icd10_kb() -> ICD10KnowledgeBase:
    """
    Tạo sample ICD-10 KB với các codes phổ biến.

    Đây là sample data, trong thực tế cần load từ official ICD-10 source.
    """
    sample_data = [
        {
            "code": "K21.0",
            "name": "Gastro-oesophageal reflux disease with oesophagitis",
            "name_vi": "Bệnh trào ngược dạ dày thực quản có viêm thực quản",
            "parent_code": "K21",
            "chapter": "XI",
            "synonyms": ["GERD with oesophagitis", "reflux oesophagitis"],
            "aliases": []
        },
        {
            "code": "K21.9",
            "name": "Gastro-oesophageal reflux disease without oesophagitis",
            "name_vi": "Bệnh trào ngược dạ dày thực quản không có viêm thực quản",
            "parent_code": "K21",
            "chapter": "XI",
            "synonyms": ["GERD", "GORD", "acid reflux"],
            "aliases": ["trào ngược dạ dày", "trào ngược", "gastroesophageal reflux"]
        },
        {
            "code": "K21",
            "name": "Gastro-oesophageal reflux disease",
            "name_vi": "Bệnh trào ngược dạ dày thực quản",
            "parent_code": "K20",
            "chapter": "XI",
            "synonyms": ["GERD", "GORD"],
            "aliases": []
        },
        {
            "code": "I10",
            "name": "Essential (primary) hypertension",
            "name_vi": "Tăng huyết áp nguyên phát",
            "parent_code": "I10",
            "chapter": "IX",
            "synonyms": ["high blood pressure", "HTN"],
            "aliases": ["tăng huyết áp", "huyết áp cao", "cao huyết áp"]
        },
        {
            "code": "E11.9",
            "name": "Type 2 diabetes mellitus without complications",
            "name_vi": "Đái tháo đường type 2 không biến chứng",
            "parent_code": "E11",
            "chapter": "IV",
            "synonyms": ["diabetes type 2", "T2DM", "NIDDM"],
            "aliases": ["đái tháo đường", "tiểu đường", "bệnh đường"]
        },
        {
            "code": "E11",
            "name": "Type 2 diabetes mellitus",
            "name_vi": "Đái tháo đường type 2",
            "parent_code": "E10",
            "chapter": "IV",
            "synonyms": ["diabetes type 2", "T2DM"],
            "aliases": []
        },
        {
            "code": "J06.9",
            "name": "Acute upper respiratory infection, unspecified",
            "name_vi": "Nhiễm trùng hô hấp trên cấp tính không xác định",
            "parent_code": "J06",
            "chapter": "X",
            "synonyms": ["URI", "common cold", "upper respiratory infection"],
            "aliases": ["nhiễm trùng hô hấp", "cảm cúm", "viêm mũi"]
        },
        {
            "code": "J18.9",
            "name": "Pneumonia, unspecified",
            "name_vi": "Viêm phổi không xác định",
            "parent_code": "J18",
            "chapter": "X",
            "synonyms": ["pneumonia"],
            "aliases": ["viêm phổi"]
        },
        {
            "code": "R50.9",
            "name": "Fever, unspecified",
            "name_vi": "Sốt không xác định",
            "parent_code": "R50",
            "chapter": "XVIII",
            "synonyms": ["fever", "pyrexia"],
            "aliases": ["sốt"]
        },
        {
            "code": "R51",
            "name": "Headache",
            "name_vi": "Đau đầu",
            "parent_code": "R50",
            "chapter": "XVIII",
            "synonyms": ["cephalgia", "head pain"],
            "aliases": ["đau đầu"]
        },
        {
            "code": "R07.0",
            "name": "Pain in throat",
            "name_vi": "Đau họng",
            "parent_code": "R07",
            "chapter": "XVIII",
            "synonyms": ["sore throat", "throat pain"],
            "aliases": ["đau họng", "viêm họng"]
        },
        {
            "code": "R05",
            "name": "Cough",
            "name_vi": "Ho",
            "parent_code": "R00",
            "chapter": "XVIII",
            "synonyms": ["tussis"],
            "aliases": ["ho"]
        },
        {
            "code": "J45.9",
            "name": "Other and unspecified asthma",
            "name_vi": "Hen phế quản không xác định",
            "parent_code": "J45",
            "chapter": "X",
            "synonyms": ["asthma", "bronchial asthma"],
            "aliases": ["hen suyễn", "hen", "viêm phế quản dị ứng"]
        },
        {
            "code": "K29.7",
            "name": "Gastritis, unspecified",
            "name_vi": "Viêm dạ dày không xác định",
            "parent_code": "K29",
            "chapter": "XI",
            "synonyms": ["gastritis"],
            "aliases": ["viêm dạ dày", "đau dạ dày"]
        },
        {
            "code": "K30",
            "name": "Functional dyspepsia",
            "name_vi": "Khó tiêu chức năng",
            "parent_code": "K30",
            "chapter": "XI",
            "synonyms": ["indigestion", "dyspepsia"],
            "aliases": ["khó tiêu", "đầy hơi", "ợ hơi"]
        },
    ]

    return ICD10KnowledgeBase.from_dict_list(sample_data)
