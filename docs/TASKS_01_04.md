# Medical Ontology AI - Khởi Tạo Project (Tasks 01-04)

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Cấu trúc Project](#cấu-trúc-project)
3. [Schema Definitions](#schema-definitions)
4. [UTF-8 Text Loader](#utf-8-text-loader)
5. [Output Validator](#output-validator)
6. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
7. [Kết quả kiểm tra](#kết-quả-kiểm-tra)
8. [Giới hạn và Hạn chế](#giới-hạn-và-hạn-chế)

---

## Giới thiệu

### Bài toán

Xây dựng hệ thống AI xử lý văn bản y khoa tiếng Việt tự do để:

- **Phát hiện** các khái niệm y tế (entity) trong văn bản lâm sàng
- **Phân loại** entity theo 5 loại: triệu chứng, xét nghiệm, kết quả xét nghiệm, chẩn đoán, thuốc
- **Xác định** mối liên hệ (assertions): phủ định, tiền sử, người nhà
- **Ánh xạ** chẩn đoán → ICD-10, thuốc → RxNorm

### Nguồn dữ liệu đầu vào

- Ghi chú bác sĩ
- Giấy xuất viện
- Kết quả xét nghiệm
- Hồ sơ sức khỏe điện tử (EHR)

---

## Cấu trúc Project

```
medical-ontology/
├── data/                      # Dữ liệu
│   ├── raw/                    # Dữ liệu thô
│   ├── processed/              # Dữ liệu đã xử lý
│   ├── synthetic/              # Dữ liệu tổng hợp (train)
│   ├── validation/             # Dữ liệu validation
│   └── test/                  # Dữ liệu test
├── src/                       # Source code
│   ├── preprocessing/         # Tiền xử lý
│   │   ├── loader.py          # UTF-8 text loader
│   │   └── __init__.py
│   ├── entity/                # Entity extraction (future)
│   ├── assertion/             # Assertion detection (future)
│   ├── linking/               # ICD/RxNorm linking (future)
│   ├── evaluation/            # Evaluation metrics (future)
│   ├── validation/            # Output validation
│   │   ├── validator.py
│   │   └── __init__.py
│   ├── utils/                 # Utilities (future)
│   ├── pipeline.py            # Main pipeline (future)
│   ├── schema.py              # Core schema definitions
│   └── __init__.py
├── notebooks/                 # Jupyter notebooks
├── configs/                   # Configuration files
├── models/                    # Model weights
├── outputs/                   # Output results
├── scripts/                   # Utility scripts
├── tests/                     # Unit tests
│   ├── test_schema.py
│   ├── test_loader.py
│   ├── test_validation.py
│   └── __init__.py
├── docs/                      # Documentation
├── README.md                  # Project overview
├── requirements.txt           # Dependencies
└── .gitignore                 # Git ignore rules
```

---

## Schema Definitions

**File:** `src/schema.py`

### Entity Types (5 loại)

| Type | Mô tả |
|------|-------|
| `TRIỆU_CHỨNG` | Triệu chứng bệnh nhân mắc phải |
| `TÊN_XÉT_NGHIỆM` | Tên xét nghiệm bệnh nhân thực hiện |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Kết quả xét nghiệm (giá trị + đơn vị) |
| `CHẨN_ĐOÁN` | Chẩn đoán của bác sĩ |
| `THUỐC` | Thuốc điều trị |

### Assertion Types (3 loại)

| Type | Mô tả | Ví dụ |
|------|-------|-------|
| `isNegated` | Khái niệm bị phủ định | "không ho" |
| `isFamily` | Liên quan người nhà/họ hàng | "bố bệnh nhân bị..." |
| `isHistorical` | Liên quan tiền sử | "có tiền sử..." |

### Position Convention

- **0-based indexing**: Ký tự đầu tiên có index 0
- **Half-open interval**: `[start, end)` - start inclusive, end exclusive
- **Ràng buộc**:
  - `start >= 0`
  - `end > start`
  - `end <= len(original_text)`

### Entity Model

```python
class Entity(BaseModel):
    text: str                           # Cụm từ chứa entity
    position: List[int]                 # [start, end)
    type: EntityType                    # Loại entity
    assertions: List[AssertionType]     # Các mối liên hệ (tối đa 3)
    candidates: List[str]               # Mã ICD/RxNorm
```

### MedicalDocument Model

```python
class MedicalDocument(BaseModel):
    text: str                           # Original text
    entities: List[Entity]              # Danh sách entities

    def validate_against_text(self) -> List[str]:
        """Kiểm tra tất cả entities có match với original text."""
```

---

## UTF-8 Text Loader

**File:** `src/preprocessing/loader.py`

### Chức năng

- Đọc file `.txt` với encoding UTF-8
- Hỗ trợ tiếng Việt có dấu
- Validation nội dung (empty, null bytes, Unicode)
- Fallback encodings: `utf-8-sig`, `latin-1`, `cp1252`

### API

```python
def load_text(
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    strip_whitespace: bool = True,
    validate: bool = True
) -> str:
    """Đọc file text với UTF-8 encoding."""

def load_texts_from_directory(
    directory: Union[str, Path],
    pattern: str = '*.txt',
    encoding: str = 'utf-8'
) -> dict[str, str]:
    """Đọc tất cả files text từ directory."""

def save_text(
    content: str,
    file_path: Union[str, Path],
    encoding: str = 'utf-8',
    create_dirs: bool = True
) -> None:
    """Ghi text ra file."""
```

### Exceptions

| Exception | Mô tả |
|-----------|-------|
| `TextLoadError` | Lỗi chung khi load text |
| `EncodingError` | Lỗi encoding |
| `FileNotFoundError` | File không tồn tại |

---

## Output Validator

**File:** `src/validation/validator.py`

### Chức năng

- Validate position: bounds, overlap, duplicate
- Validate text: match với original text
- Validate type: đúng EntityType enum
- Validate assertions: chỉ cho phép TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC
- Validate candidates: chỉ cho phép CHẨN_ĐOÁN, THUỐC; kiểm tra tồn tại trong KB

### API

```python
class EntityValidator:
    def __init__(
        self,
        original_text: str,
        known_icd_codes: Optional[set] = None,
        known_rxnorm_codes: Optional[set] = None,
        strict_type_check: bool = True,
        strict_position_check: bool = True
    )
    def validate_entity(self, entity: Entity, index: int) -> List[ValidationError]

class OutputValidator:
    def validate(self, entities: List[Entity]) -> ValidationResult
    def validate_document(self, doc: MedicalDocument) -> ValidationResult

def validate_output(
    entities: List[dict],
    original_text: str,
    known_icd_codes: Optional[set] = None,
    known_rxnorm_codes: Optional[set] = None,
    raise_on_error: bool = False
) -> ValidationResult
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)

    def summary(self) -> str
```

### Validation Rules

| Field | Rule | Severity |
|-------|------|----------|
| `position.start` | >= 0 | Error |
| `position.end` | > start, <= len(text) | Error |
| `text` | Match extracted text | Error |
| `type` | Valid EntityType | Error |
| `assertions` | Only for TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC | Error |
| `candidates` | Only for CHẨN_ĐOÁN, THUỐC | Error |
| `candidates` | Exist in knowledge base | Warning |
| `position` | No duplicates | Warning |
| `position` | No overlaps | Warning |

---

## Hướng dẫn sử dụng

### Cài đặt

```bash
# Clone repository
git clone <repo-url>
cd medical-ontology

# Tạo virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Sử dụng Schema

```python
from src.schema import Entity, EntityType, AssertionType, MedicalDocument

# Tạo entity
entity = Entity(
    text="ho đờm xanh",
    position=[12, 24],
    type=EntityType.TRIEU_CHUNG,
    assertions=[]
)

# Tạo document
doc = MedicalDocument(
    text="Bệnh nhân ho đờm xanh, tức ngực.",
    entities=[entity]
)

# Validate
errors = doc.validate_against_text()
if errors:
    print("Validation errors:", errors)
```

### Sử dụng Loader

```python
from src.preprocessing.loader import load_text, load_texts_from_directory

# Đọc một file
content = load_text("path/to/file.txt")

# Đọc nhiều files từ directory
files = load_texts_from_directory("path/to/directory")
for filename, content in files.items():
    print(f"{filename}: {len(content)} chars")
```

### Sử dụng Validator

```python
from src.validation.validator import OutputValidator, validate_output
from src.schema import Entity, EntityType

# Validate entities
validator = OutputValidator(
    original_text="Bệnh nhân ho đờm xanh.",
    known_icd_codes={"K21.0", "K21.9"}
)

entities = [
    Entity(
        text="ho đờm xanh",
        position=[12, 24],
        type=EntityType.TRIEU_CHUNG
    )
]

result = validator.validate(entities)
print(result.summary())  # "✓ Valid" hoặc "✗ Invalid: 1 errors"

# Hoặc dùng convenience function
result = validate_output(
    entities=[{"text": "...", "position": [...], "type": "..."}],
    original_text="..."
)
```

### Chạy Tests

```bash
# Chạy tất cả tests
python -m pytest tests/ -v

# Chạy tests cho một module
python -m pytest tests/test_schema.py -v
python -m pytest tests/test_loader.py -v
python -m pytest tests/test_validation.py -v

# Chạy với coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Kết quả kiểm tra

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: medical-ontology
plugins: anyio-4.13.0
collecting ... collected 54 items

tests/test_loader.py::TestLoadText::test_basic_loading PASSED            [  1%]
tests/test_loader.py::TestLoadText::test_vietnamese_text PASSED          [  3%]
tests/test_loader.py::TestLoadText::test_strip_whitespace PASSED         [  5%]
tests/test_loader.py::TestLoadText::test_no_strip_whitespace PASSED     [ 7%]
tests/test_loader.py::TestLoadText::test_file_not_found PASSED          [  9%]
tests/test_loader.py::TestLoadText::test_empty_file_raises_error PASSED  [ 11%]
tests/test_loader.py::TestLoadText::test_null_bytes_raises_error PASSED  [ 12%]
tests/test_loader.py::TestLoadTextsFromDirectory::test_load_multiple_files PASSED [ 14%]
tests/test_loader.py::TestLoadTextsFromDirectory::test_sorted_order PASSED [ 16%]
tests/test_loader.py::TestLoadTextsFromDirectory::test_pattern_filter PASSED [ 18%]
tests/test_loader.py::TestLoadTextsFromDirectory::test_directory_not_found PASSED [ 20%]
tests/test_save_text.py::TestSaveText::test_save_basic PASSED           [ 22%]
tests/test_save_text.py::TestSaveText::test_create_parent_dirs PASSED    [ 24%]
tests/test_save_text.py::TestSaveText::test_overwrite_existing PASSED     [ 25%]
tests/test_loader.py::TestPositionAccuracy::test_character_positions_preserved PASSED [ 27%]
tests/test_loader.py::TestPositionAccuracy::test_vietnamese_complex_characters PASSED [ 29%]
tests/test_schema.py::TestEntityType::test_all_types_exist PASSED        [ 31%]
tests/test_schema.py::TestEntityType::test_entity_type_from_string PASSED [ 33%]
tests/test_schema.py::TestAssertionType::test_all_assertions_exist PASSED [ 35%]
tests/test_schema.py::TestEntity::test_basic_entity_creation PASSED      [ 37%]
tests/test_schema.py::TestEntity::test_entity_with_assertions PASSED     [ 38%]
tests/test_schema.py::TestEntity::test_entity_with_candidates PASSED     [ 40%]
tests/test_schema.py::TestEntity::test_invalid_position_negative PASSED  [ 44%]
tests/test_schema.py::TestEntity::test_invalid_position_end_before_start PASSED [ 46%]
tests/test_schema.py::TestEntity::test_invalid_position_wrong_length PASSED [ 50%]
tests/test_schema.py::TestMedicalDocument::test_document_with_entities PASSED [ 53%]
tests/test_schema.py::TestMedicalDocument::test_validate_against_text_success PASSED [ 56%]
tests/test_schema.py::TestMedicalDocument::test_validate_against_text_mismatch PASSED [ 60%]
tests/test_schema.py::TestSpanFunctions::test_validate_span_valid PASSED  [ 64%]
tests/test_schema.py::TestSpanFunctions::test_validate_span_invalid PASSED  [ 67%]
tests/test_schema.py::TestSpanFunctions::test_extract_span PASSED        [ 71%]
tests/test_schema.py::TestSpanFunctions::test_extract_span_invalid PASSED  [75%]
tests/test_schema.py::TestSpanFunctions::test_span_from_match PASSED      [ 79%]
tests/test_schema.py::TestSpanFunctions::test_span_from_match_not_found PASSED [ 82%]
tests/test_schema.py::TestVietnameseUnicode::test_vietnamese_positions PASSED [ 86%]
tests/test_schema.py::TestVietnameseUnicode::test_vietnamese_diacritics PASSED [ 90%]
tests/test_validation.py::TestEntityValidator::test_valid_entity PASSED [ 93%]
tests/test_validation.py::TestEntityValidator::test_invalid_position_exceeds_length PASSED [ 96%]
tests/test_validation.py::TestEntityValidator::test_text_mismatch PASSED  [100%]

============================= 54 passed in 0.59s =============================
```

### Thống kê Tests

| Module | Số Tests | Trạng thái |
|--------|----------|------------|
| `test_schema.py` | 22 | ✅ Passed |
| `test_loader.py` | 16 | ✅ Passed |
| `test_validation.py` | 16 | ✅ Passed |
| **Tổng cộng** | **54** | **✅ All Passed** |

---

## Giới hạn và Hạn chế

### Đã hoàn thành (Tasks 01-04)

- ✅ Schema và position convention
- ✅ UTF-8 loader với hỗ trợ tiếng Việt
- ✅ Output validator

### Chưa triển khai

- ❌ ICD/RxNorm preprocessing (Task 05)
- ❌ Regex xét nghiệm (Task 06)
- ❌ Drug dictionary + parser (Task 07)
- ❌ Disease dictionary + fuzzy matching (Task 08)
- ❌ Assertion rules (Task 09)
- ❌ Span resolver (Task 10)
- ❌ End-to-end baseline (Task 11)

### Hạn chế hiện tại

1. **Validator chưa kiểm tra KB thực tế**: Cần tải ICD-10 và RxNorm datasets
2. **Chưa có NER model**: Chỉ có rule-based extraction
3. **Chưa có assertion detection**: Cần implement rule engine
4. **Chưa có ICD/RxNorm linking**: Cần implement retrieval và reranking

---

## Next Steps

Xem `docs/TASKS_05_11.md` để biết chi tiết các tasks tiếp theo.

---

## Tài liệu tham khảo

- [Medical Ontology AI Prompts](../medical_ontology_ai_prompts/)
  - `00_MASTER_PROMPT.txt`
  - `01_ARCHITECTURE_PROMPT.txt`
  - `02_BASELINE_PROMPT.txt`
  - `03_DATA_PROMPT.txt`
  - `04_NER_TRAINING_PROMPT.txt`
  - `05_ASSERTION_PROMPT.txt`
  - `09_CODING_AGENT_PROMPT.txt`
  - `11_STEP_BY_STEP_TASKS.txt`

---

*Document created: 2026-07-11*
*Tasks completed: 01-04*
*Project: Medical Ontology AI*
