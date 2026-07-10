"""
Tests cho preprocessing/loader module.
"""

import pytest
import tempfile
from pathlib import Path

from src.preprocessing.loader import (
    load_text,
    load_texts_from_directory,
    save_text,
    TextLoadError,
    FileNotFoundError,
    EncodingError,
)


class TestLoadText:
    """Tests cho load_text function."""

    def test_basic_loading(self, tmp_path):
        """Test đọc file cơ bản."""
        test_file = tmp_path / "test.txt"
        content = "BN ho đờm xanh"
        test_file.write_text(content, encoding='utf-8')

        result = load_text(test_file)
        assert result == content

    def test_vietnamese_text(self, tmp_path):
        """Test đọc text tiếng Việt."""
        test_file = tmp_path / "vietnamese.txt"
        content = "BN bị bệnh trào ngược"
        test_file.write_text(content, encoding='utf-8')

        result = load_text(test_file)
        assert result == content
        # Verify character positions
        assert result[0:2] == "BN"

    def test_strip_whitespace(self, tmp_path):
        """Test strip whitespace."""
        test_file = tmp_path / "whitespace.txt"
        content = "  Test content  \n\n"
        test_file.write_text(content, encoding='utf-8')

        result = load_text(test_file, strip_whitespace=True)
        assert result == "Test content"

    def test_no_strip_whitespace(self, tmp_path):
        """Test không strip whitespace."""
        test_file = tmp_path / "whitespace.txt"
        content = "  Test content  \n\n"
        test_file.write_text(content, encoding='utf-8')

        result = load_text(test_file, strip_whitespace=False)
        assert result == content

    def test_file_not_found(self):
        """Test file không tồn tại."""
        with pytest.raises(FileNotFoundError):
            load_text("/nonexistent/file.txt")

    def test_empty_file_raises_error(self, tmp_path):
        """Test file rỗng raise error."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding='utf-8')

        with pytest.raises(TextLoadError):
            load_text(test_file)

    def test_null_bytes_raises_error(self, tmp_path):
        """Test file có null bytes raise error."""
        test_file = tmp_path / "null.txt"
        # Write binary with null bytes
        with open(test_file, 'wb') as f:
            f.write(b"Test\x00content")

        with pytest.raises(TextLoadError):
            load_text(test_file)


class TestLoadTextsFromDirectory:
    """Tests cho load_texts_from_directory function."""

    def test_load_multiple_files(self, tmp_path):
        """Test đọc nhiều files."""
        (tmp_path / "1.txt").write_text("Content 1", encoding='utf-8')
        (tmp_path / "2.txt").write_text("Content 2", encoding='utf-8')
        (tmp_path / "3.txt").write_text("Content 3", encoding='utf-8')

        results = load_texts_from_directory(tmp_path)

        assert len(results) == 3
        assert results["1"] == "Content 1"
        assert results["2"] == "Content 2"
        assert results["3"] == "Content 3"

    def test_sorted_order(self, tmp_path):
        """Test files được sorted theo tên."""
        (tmp_path / "3.txt").write_text("Content 3", encoding='utf-8')
        (tmp_path / "1.txt").write_text("Content 1", encoding='utf-8')
        (tmp_path / "2.txt").write_text("Content 2", encoding='utf-8')

        results = load_texts_from_directory(tmp_path)
        keys = list(results.keys())

        assert keys == ["1", "2", "3"]

    def test_pattern_filter(self, tmp_path):
        """Test glob pattern filter."""
        (tmp_path / "1.txt").write_text("Content 1", encoding='utf-8')
        (tmp_path / "2.txt").write_text("Content 2", encoding='utf-8')
        (tmp_path / "3.md").write_text("Content 3", encoding='utf-8')

        results = load_texts_from_directory(tmp_path, pattern="*.txt")

        assert len(results) == 2
        assert "3" not in results

    def test_directory_not_found(self):
        """Test directory không tồn tại."""
        with pytest.raises(FileNotFoundError):
            load_texts_from_directory("/nonexistent/dir")


class TestSaveText:
    """Tests cho save_text function."""

    def test_save_basic(self, tmp_path):
        """Test lưu file cơ bản."""
        file_path = tmp_path / "output.txt"
        content = "Test content"

        save_text(content, file_path)

        assert file_path.exists()
        assert file_path.read_text(encoding='utf-8') == content

    def test_create_parent_dirs(self, tmp_path):
        """Test tạo parent directories."""
        file_path = tmp_path / "nested" / "dir" / "output.txt"
        content = "Test content"

        save_text(content, file_path, create_dirs=True)

        assert file_path.exists()
        assert file_path.read_text(encoding='utf-8') == content

    def test_overwrite_existing(self, tmp_path):
        """Test ghi đè file existing."""
        file_path = tmp_path / "output.txt"
        file_path.write_text("Old content", encoding='utf-8')

        save_text("New content", file_path)

        assert file_path.read_text(encoding='utf-8') == "New content"


class TestPositionAccuracy:
    """Tests cho position accuracy sau khi load."""

    def test_character_positions_preserved(self, tmp_path):
        """Test character positions được preserve."""
        test_file = tmp_path / "position.txt"
        content = "BN ho đờm, tức ngực."
        test_file.write_text(content, encoding='utf-8')

        loaded = load_text(test_file)

        # Verify lengths
        assert len(loaded) == len(content)

        # Verify specific positions
        # B=0, N=1, space=2, h=3, o=4, space=5, đ=6, ờ=7, m=8, ,=9, space=10, t=11, ứ=12, c=13, space=14, n=15, g=16, ự=17, c=18, .=19
        assert loaded[0:2] == "BN"
        assert loaded[3:9] == "ho đờm"
        assert loaded[11:15] == "tức " or loaded[11:15] == "tức ng"

    def test_vietnamese_complex_characters(self, tmp_path):
        """Test với ký tự tiếng Việt phức tạp."""
        test_file = tmp_path / "complex.txt"
        content = "ê ệ ơ ư ơ ư"
        test_file.write_text(content, encoding='utf-8')

        loaded = load_text(test_file)
        assert loaded == content
        assert len(loaded) == len(content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
