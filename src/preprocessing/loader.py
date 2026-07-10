"""
UTF-8 Text Loader cho Medical Ontology

Module đọc file .txt UTF-8 an toàn với error handling và validation.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import pytest

logger = logging.getLogger(__name__)


class TextLoadError(Exception):
    """Exception raised when text loading fails."""
    pass


class EncodingError(TextLoadError):
    """Exception raised for encoding issues."""
    pass


class FileNotFoundError(TextLoadError):
    """Exception raised when file not found."""
    pass


def load_text(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    strip_whitespace: bool = True,
    validate: bool = True
) -> str:
    """
    Đọc file text với UTF-8 encoding.

    Args:
        file_path: Đường dẫn đến file
        encoding: Encoding của file (default: utf-8)
        strip_whitespace: Có loại bỏ whitespace ở đầu/cuối không
        validate: Có validate nội dung không

    Returns:
        Nội dung text

    Raises:
        FileNotFoundError: Nếu file không tồn tại
        EncodingError: Nếu có lỗi encoding
        TextLoadError: Các lỗi khác
    """
    path = Path(file_path)

    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise TextLoadError(f"Path is not a file: {file_path}")

    # Try to read with specified encoding
    try:
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError as e:
        # Try fallback encodings
        fallback_encodings = ['utf-8-sig', 'latin-1', 'cp1252']
        for fallback in fallback_encodings:
            if fallback == encoding:
                continue
            try:
                with open(path, 'r', encoding=fallback) as f:
                    content = f.read()
                logger.warning(
                    f"File {file_path} read with {fallback} instead of {encoding}"
                )
                break
            except UnicodeDecodeError:
                continue
        else:
            raise EncodingError(
                f"Failed to decode {file_path} with {encoding} or fallback encodings"
            ) from e

    # Strip whitespace
    if strip_whitespace:
        content = content.strip()

    # Validate content
    if validate:
        _validate_content(content, file_path)

    logger.info(f"Loaded {path} ({len(content)} chars)")
    return content


def _validate_content(content: str, source: Union[str, Path]) -> None:
    """
    Validate nội dung text.

    Args:
        content: Nội dung text
        source: Nguồn file (để log)

    Raises:
        TextLoadError: Nếu nội dung không hợp lệ
    """
    # Check not empty
    if not content:
        raise TextLoadError(f"Empty file: {source}")

    # Check for null bytes
    if '\x00' in content:
        raise TextLoadError(f"File contains null bytes: {source}")

    # Check for valid Unicode
    try:
        content.encode('utf-8')
    except UnicodeEncodeError as e:
        raise TextLoadError(f"Invalid Unicode in file: {source}") from e


def load_texts_from_directory(
    directory: Union[str, Path],
    pattern: str = '*.txt',
    encoding: str = 'utf-8'
) -> dict[str, str]:
    """
    Đọc tất cả files text từ directory.

    Args:
        directory: Thư mục chứa files
        pattern: Glob pattern để match files
        encoding: Encoding của files

    Returns:
        Dict với key là file name (không có extension) và value là content
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not dir_path.is_dir():
        raise TextLoadError(f"Path is not a directory: {directory}")

    results = {}

    for file_path in sorted(dir_path.glob(pattern)):
        try:
            content = load_text(file_path, encoding=encoding)
            key = file_path.stem  # filename without extension
            results[key] = content
        except TextLoadError as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

    logger.info(f"Loaded {len(results)} files from {directory}")
    return results


def save_text(
    content: str,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> None:
    """
    Ghi text ra file.

    Args:
        content: Nội dung cần ghi
        file_path: Đường dẫn file
        encoding: Encoding
        create_dirs: Có tạo thư mục cha nếu chưa có không
    """
    path = Path(file_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding=encoding) as f:
        f.write(content)

    logger.info(f"Saved {path} ({len(content)} chars)")


# =============================================================================
# Tests
# =============================================================================

def test_load_text_basic(tmp_path):
    """Test đọc file cơ bản."""
    test_file = tmp_path / "test.txt"
    test_content = "Bệnh nhân ho đờm xanh."
    test_file.write_text(test_content, encoding='utf-8')

    result = load_text(test_file)
    assert result == test_content


def test_load_text_vietnamese(tmp_path):
    """Test đọc file tiếng Việt."""
    test_file = tmp_path / "vietnamese.txt"
    # Vietnamese text with various diacritics
    test_content = "Bệnh nhân bị bệnh trào ngược dạ dày - thực quản."
    test_file.write_text(test_content, encoding='utf-8')

    result = load_text(test_file)
    assert result == test_content
    # Verify character positions
    assert result[0:6] == "Bệnh "


def test_load_text_empty():
    """Test đọc file empty."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        with pytest.raises(TextLoadError):
            load_text(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_text_not_found():
    """Test đọc file không tồn tại."""
    with pytest.raises(FileNotFoundError):
        load_text("/nonexistent/file.txt")


def test_load_texts_from_directory(tmp_path):
    """Test đọc nhiều files từ directory."""
    # Create test files
    (tmp_path / "1.txt").write_text("Content 1", encoding='utf-8')
    (tmp_path / "2.txt").write_text("Content 2", encoding='utf-8')
    (tmp_path / "3.txt").write_text("Content 3", encoding='utf-8')

    results = load_texts_from_directory(tmp_path)

    assert len(results) == 3
    assert results["1"] == "Content 1"
    assert results["2"] == "Content 2"
    assert results["3"] == "Content 3"


def test_position_accuracy(tmp_path):
    """Test độ chính xác của position sau khi load."""
    test_file = tmp_path / "position.txt"
    original = "Bệnh nhân ho đờm xanh, tức ngực."
    test_file.write_text(original, encoding='utf-8')

    content = load_text(test_file)

    # Test various substrings
    assert content[0:11] == "Bệnh nhân"
    assert content[12:19] == "ho đờm"
    assert content[20:29] == "xanh, tức"

    # Verify positions match
    assert len(content) == len(original)
