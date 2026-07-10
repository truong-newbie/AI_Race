"""
Tests cho schema module.
"""

import pytest
from src.schema import (
    Entity,
    EntityType,
    AssertionType,
    MedicalDocument,
    validate_span,
    extract_span,
    span_from_match,
)


class TestEntityType:
    """Tests cho EntityType enum."""

    def test_all_types_exist(self):
        """Test tất cả entity types được định nghĩa."""
        assert EntityType.TRIEU_CHUNG.value == "TRIỆU_CHỨNG"
        assert EntityType.TEN_XET_NGHIEM.value == "TÊN_XÉT_NGHIỆM"
        assert EntityType.KET_QUA_XET_NGHIEM.value == "KẾT_QUẢ_XÉT_NGHIỆM"
        assert EntityType.CHAN_DOAN.value == "CHẨN_ĐOÁN"
        assert EntityType.THUOC.value == "THUỐC"

    def test_entity_type_from_string(self):
        """Test tạo EntityType từ string."""
        assert EntityType("TRIỆU_CHỨNG") == EntityType.TRIEU_CHUNG
        assert EntityType("THUỐC") == EntityType.THUOC


class TestAssertionType:
    """Tests cho AssertionType enum."""

    def test_all_assertions_exist(self):
        """Test tất cả assertion types được định nghĩa."""
        assert AssertionType.NEGATED.value == "isNegated"
        assert AssertionType.FAMILY.value == "isFamily"
        assert AssertionType.HISTORICAL.value == "isHistorical"


class TestEntity:
    """Tests cho Entity model."""

    def test_basic_entity_creation(self):
        """Test tạo entity cơ bản."""
        entity = Entity(
            text="ho đờm",
            position=[11, 17],
            type=EntityType.TRIEU_CHUNG
        )
        assert entity.text == "ho đờm"
        assert entity.position == [11, 17]
        assert entity.type == EntityType.TRIEU_CHUNG
        assert entity.assertions == []
        assert entity.candidates == []

    def test_entity_with_assertions(self):
        """Test entity với assertions."""
        entity = Entity(
            text="tăng huyết áp",
            position=[11, 25],
            type=EntityType.CHAN_DOAN,
            assertions=[AssertionType.HISTORICAL]
        )
        assert entity.assertions == [AssertionType.HISTORICAL]

    def test_entity_with_candidates(self):
        """Test entity với candidates."""
        entity = Entity(
            text="trào ngược",
            position=[0, 10],
            type=EntityType.CHAN_DOAN,
            candidates=["K21.0", "K21.9"]
        )
        assert entity.candidates == ["K21.0", "K21.9"]

    def test_invalid_position_negative(self):
        """Test position với start < 0."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            Entity(
                text="test",
                position=[-1, 4],
                type=EntityType.TRIEU_CHUNG
            )

    def test_invalid_position_end_before_start(self):
        """Test position với end <= start."""
        with pytest.raises(ValueError, match="end must be > start"):
            Entity(
                text="test",
                position=[5, 3],
                type=EntityType.TRIEU_CHUNG
            )

    def test_invalid_position_wrong_length(self):
        """Test position với độ dài != 2."""
        with pytest.raises(ValueError):
            Entity(
                text="test",
                position=[0],
                type=EntityType.TRIEU_CHUNG
            )


class TestMedicalDocument:
    """Tests cho MedicalDocument model."""

    def test_document_with_entities(self):
        """Test document với entities."""
        doc = MedicalDocument(
            text="BN ho đờm xanh",
            entities=[
                Entity(
                    text="ho đờm",
                    position=[3, 9],
                    type=EntityType.TRIEU_CHUNG
                )
            ]
        )
        assert len(doc.entities) == 1
        assert doc.entities[0].text == "ho đờm"

    def test_validate_against_text_success(self):
        """Test validate text thành công."""
        text = "BN ho đờm"
        doc = MedicalDocument(
            text=text,
            entities=[
                Entity(
                    text="ho đờm",
                    position=[3, 9],
                    type=EntityType.TRIEU_CHUNG
                )
            ]
        )
        errors = doc.validate_against_text()
        assert errors == []

    def test_validate_against_text_mismatch(self):
        """Test validate text với mismatch."""
        text = "BN ho đờm"
        doc = MedicalDocument(
            text=text,
            entities=[
                Entity(
                    text="khác",
                    position=[3, 7],
                    type=EntityType.TRIEU_CHUNG
                )
            ]
        )
        errors = doc.validate_against_text()
        assert len(errors) == 1
        assert "mismatch" in errors[0]


class TestSpanFunctions:
    """Tests cho span utility functions."""

    def test_validate_span_valid(self):
        """Test validate_span với span hợp lệ."""
        text = "BN ho đờm xanh"
        assert validate_span(text, 0, 2) is True
        assert validate_span(text, 3, 9) is True
        assert validate_span(text, 0, len(text)) is True

    def test_validate_span_invalid(self):
        """Test validate_span với span không hợp lệ."""
        text = "BN ho đờm"
        assert validate_span(text, -1, 3) is False
        assert validate_span(text, 3, 1) is False
        assert validate_span(text, 0, len(text) + 1) is False

    def test_extract_span(self):
        """Test extract_span."""
        text = "BN ho đờm xanh"
        assert extract_span(text, 0, 2) == "BN"
        assert extract_span(text, 3, 9) == "ho đờm"
        assert extract_span(text, 0, len(text)) == text

    def test_extract_span_invalid(self):
        """Test extract_span với span không hợp lệ."""
        text = "test"
        with pytest.raises(ValueError):
            extract_span(text, -1, 3)

    def test_span_from_match(self):
        """Test span_from_match."""
        text = "BN ho đờm xanh"
        start, end = span_from_match(text, "ho đờm")
        assert text[start:end] == "ho đờm"

    def test_span_from_match_not_found(self):
        """Test span_from_match khi không tìm thấy."""
        text = "BN ho đờm xanh"
        with pytest.raises(ValueError):
            span_from_match(text, "không có")


class TestVietnameseUnicode:
    """Tests cho Unicode tiếng Việt."""

    def test_vietnamese_positions(self):
        """Test position với ký tự tiếng Việt có dấu."""
        text = "BN bị trào ngược dạ dày."

        # Tính positions chính xác
        # B=0, N=1, space=2, b=3, ị=4, space=5, t=6, r=7, à=8, o=9, space=10, n=11, g=12, ượ=13, c=14, space=15, d=16, ạ=17, space=18, d=19, à=20, y=21, .=22

        assert text[0:2] == "BN"
        assert text[6:10] == "trào"
        assert text[11:15] == "ngượ"

        # Verify positions - trào ngược = indices 6-16 (10 chars)
        entity = Entity(
            text="trào ngược",
            position=[6, 16],
            type=EntityType.CHAN_DOAN
        )
        doc = MedicalDocument(text=text, entities=[entity])
        errors = doc.validate_against_text()
        assert errors == []

    def test_vietnamese_diacritics(self):
        """Test với nhiều loại dấu tiếng Việt."""
        text = "àáảãạăằắặẳẵâầấậẩẫèéẻẽẹêềếệểễìíỉĩịòóỏõọôồốộổỗơờớợởỡùúủũụưừứựửữỳýỷỹỵđ"

        # Verify length và substrings
        assert len(text) > 0
        assert text[0:3] == "àáả"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
