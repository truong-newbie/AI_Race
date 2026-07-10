"""
Medical Ontology Schema Definitions

Định nghĩa schema cho entity types, assertions, candidates
và quy ước position (offset 0-based).
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Entity Types
# =============================================================================

class EntityType(str, Enum):
    """5 loại entity trong bài toán."""
    TRIEU_CHUNG = "TRIỆU_CHỨNG"
    TEN_XET_NGHIEM = "TÊN_XÉT_NGHIỆM"
    KET_QUA_XET_NGHIEM = "KẾT_QUẢ_XÉT_NGHIỆM"
    CHAN_DOAN = "CHẨN_ĐOÁN"
    THUOC = "THUỐC"


# =============================================================================
# Assertions
# =============================================================================

class AssertionType(str, Enum):
    """3 loại assertion - mối liên hệ của entity trong ngữ cảnh."""
    NEGATED = "isNegated"      # Bị phủ định
    FAMILY = "isFamily"         # Liên quan người nhà
    HISTORICAL = "isHistorical" # Liên quan tiền sử


# =============================================================================
# Position Convention
# =============================================================================

# Position convention:
# - 0-based indexing (ký tự đầu tiên có index 0)
# - [start, end) - half-open interval (start inclusive, end exclusive)
# - start >= 0
# - end > start
# - end <= len(original_text)


# =============================================================================
# Entity Model
# =============================================================================

class Entity(BaseModel):
    """
    Một entity được phát hiện trong văn bản y khoa.

    Attributes:
        text: Cụm từ trong input mà hệ thống xác định là entity
        position: [start, end) - vị trí bắt đầu và kết thúc trong input (0-based)
        type: Loại entity (EntityType enum)
        assertions: Danh sách các assertions (tối đa 3 phần tử)
        candidates: Danh sách mã ICD/RxNorm (chỉ cho CHẨN_ĐOÁN và THUỐC)
    """
    text: str = Field(..., min_length=1, description="Cụm từ chứa entity")
    position: List[int] = Field(..., min_length=2, max_length=2, description="[start, end) position")
    type: EntityType = Field(..., description="Loại entity")
    assertions: List[AssertionType] = Field(
        default_factory=list,
        max_length=3,
        description="Các mối liên hệ (isNegated, isFamily, isHistorical)"
    )
    candidates: List[str] = Field(
        default_factory=list,
        description="Mã ICD cho CHẨN_ĐOÁN, RxNorm cho THUỐC"
    )

    @field_validator('position')
    @classmethod
    def validate_position(cls, v: List[int]) -> List[int]:
        """Kiểm tra position hợp lệ."""
        if len(v) != 2:
            raise ValueError(f"position must have exactly 2 elements, got {len(v)}")
        start, end = v
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")
        return v

    @field_validator('assertions', mode='before')
    @classmethod
    def validate_assertions(cls, v) -> List[AssertionType]:
        """Chuyển string assertions sang AssertionType."""
        if isinstance(v, list):
            result = []
            for item in v:
                if isinstance(item, str):
                    try:
                        result.append(AssertionType(item))
                    except ValueError:
                        raise ValueError(f"Invalid assertion: {item}")
                else:
                    result.append(item)
            return result
        return v

    def model_post_init(self, __context) -> None:
        """Validate sau khi khởi tạo."""
        # Assertions chỉ áp dụng cho TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC
        allowed_types = {EntityType.TRIEU_CHUNG, EntityType.CHAN_DOAN, EntityType.THUOC}
        if self.type not in allowed_types and self.assertions:
            raise ValueError(
                f"Assertions only allowed for TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC, "
                f"got {self.type} with assertions"
            )

        # Candidates chỉ áp dụng cho CHẨN_ĐOÁN và THUỐC
        candidate_types = {EntityType.CHAN_DOAN, EntityType.THUOC}
        if self.type not in candidate_types and self.candidates:
            raise ValueError(
                f"candidates only allowed for CHẨN_ĐOÁN and THUỐC, "
                f"got {self.type}"
            )


# =============================================================================
# Document Model
# =============================================================================

class MedicalDocument(BaseModel):
    """Toàn bộ document với entities."""
    text: str = Field(..., description="Original text")
    entities: List[Entity] = Field(default_factory=list, description="List of entities")

    def get_entity_at_position(self, start: int, end: int) -> Optional[Entity]:
        """Tìm entity tại vị trí [start, end)."""
        for entity in self.entities:
            if entity.position == [start, end]:
                return entity
        return None

    def validate_against_text(self) -> List[str]:
        """
        Kiểm tra tất cả entities có match với original text.

        Returns:
            Danh sách lỗi (empty nếu không có lỗi)
        """
        errors = []
        for i, entity in enumerate(self.entities):
            start, end = entity.position
            if end > len(self.text):
                errors.append(
                    f"Entity {i}: position {entity.position} exceeds text length {len(self.text)}"
                )
            extracted = self.text[start:end]
            if extracted != entity.text:
                errors.append(
                    f"Entity {i}: text mismatch. "
                    f"Expected '{entity.text}', got '{extracted}' at position {entity.position}"
                )
        return errors


# =============================================================================
# Validation Functions
# =============================================================================

def validate_span(text: str, start: int, end: int) -> bool:
    """
    Kiểm tra span [start, end) có hợp lệ với text không.

    Args:
        text: Original text
        start: Start position (inclusive)
        end: End position (exclusive)

    Returns:
        True nếu hợp lệ
    """
    if start < 0:
        return False
    if end <= start:
        return False
    if end > len(text):
        return False
    return True


def extract_span(text: str, start: int, end: int) -> str:
    """
    Trích xuất text từ span [start, end).

    Args:
        text: Original text
        start: Start position (inclusive)
        end: End position (exclusive)

    Returns:
        Extracted substring

    Raises:
        ValueError: Nếu span không hợp lệ
    """
    if not validate_span(text, start, end):
        raise ValueError(f"Invalid span [{start}, {end}) for text of length {len(text)}")
    return text[start:end]


def span_from_match(text: str, matched_text: str, start_hint: Optional[int] = None) -> tuple[int, int]:
    """
    Tìm vị trí của matched_text trong text và trả về span.

    Args:
        text: Original text
        matched_text: Text cần tìm
        start_hint: Hint về vị trí bắt đầu tìm kiếm

    Returns:
        Tuple (start, end)

    Raises:
        ValueError: Nếu không tìm thấy
    """
    if start_hint is not None:
        search_text = text[start_hint:]
        idx = search_text.find(matched_text)
        if idx != -1:
            return (start_hint + idx, start_hint + idx + len(matched_text))

    idx = text.find(matched_text)
    if idx == -1:
        raise ValueError(f"Text '{matched_text}' not found in '{text}'")
    return (idx, idx + len(matched_text))
