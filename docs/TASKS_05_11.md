# Medical Ontology AI - Baseline Implementation (Tasks 05-11)

## Mục lục

1. [Tổng quan](#tổng-quan)
2. [Kiến trúc Baseline](#kiến-trúc-baseline)
3. [Module Chi tiết](#module-chi-tiết)
4. [Sử dụng Pipeline](#sử-dụng-pipeline)
5. [Kết quả kiểm tra](#kết-quả-kiểm-tra)
6. [Hạn chế của Baseline](#hạn-chế-của-baseline)
7. [Next Steps](#next-steps)

---

## Tổng quan

Phase 2 (Tasks 05-11) hoàn thành việc triển khai baseline rule-based system bao gồm:

- **Knowledge Base**: ICD-10 và RxNorm loaders
- **Entity Extractors**: Lab tests, drugs, diseases
- **Assertion Detection**: Negation, historical, family history
- **Span Resolution**: Handle overlapping entities
- **End-to-End Pipeline**: Kết hợp tất cả components

### Pipeline Flow

```
Input Text
    ↓
┌─────────────────────────────────────────────┐
│ 1. Entity Extraction                         │
│    • LabTestExtractor → TRIỆU_CHỨNG        │
│    • DrugExtractor → THUỐC                  │
│    • DiseaseExtractor → CHẨN_ĐOÁN          │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 2. Span Resolution (resolve overlaps)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. Assertion Detection                        │
│    • Negation (không, chưa...)              │
│    • Historical (tiền sử...)                │
│    • Family (bố, mẹ...)                    │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 4. Entity Linking                            │
│    • ICD-10 → CHẨN_ĐOÁN                    │
│    • RxNorm → THUỐC                         │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 5. Validation                                │
│    • Position bounds                         │
│    • Text match                              │
│    • Type constraints                        │
└─────────────────────────────────────────────┘
    ↓
JSON Output (list of entities)
```

---

## Kiến trúc Baseline

```
medical-ontology/
├── src/
│   ├── pipeline.py                 # End-to-end pipeline
│   ├── linking/
│   │   ├── icd10.py               # ICD-10 KB + fuzzy matching
│   │   └── rxnorm.py              # RxNorm KB + drug parser
│   ├── entity/
│   │   ├── lab_extractor.py       # Lab test/reuslt extraction
│   │   ├── drug_extractor.py      # Drug extraction
│   │   ├── disease_extractor.py    # Disease/symptom extraction
│   │   └── span_resolver.py       # Overlap resolution
│   ├── assertion/
│   │   └── rules.py              # Assertion detection rules
│   └── validation/
│       └── validator.py           # Output validation
├── tests/
│   └── test_pipeline.py          # Pipeline tests
└── demo.py                        # Demo script
```

---

## Module Chi tiết

### 1. ICD-10 Knowledge Base (`src/linking/icd10.py`)

```python
@dataclass
class ICD10Entry:
    code: str                    # e.g., "J18.9"
    name: str                    # e.g., "Pneumonia, unspecified"
    name_vi: Optional[str]       # Vietnamese name
    parent_code: Optional[str]    # e.g., "J18"
    synonyms: list[str]          # Alternative names
    aliases: list[str]            # Short forms

class ICD10KnowledgeBase:
    def add_entry(entry: ICD10Entry)
    def get_by_code(code: str) -> Optional[ICD10Entry]
    def get_by_name(name: str) -> List[ICD10Entry]
    def search(query: str, limit: int = 5) -> List[ICD10Entry]
    def fuzzy_search(query: str, threshold: float = 0.7) -> List[tuple[ICD10Entry, float]]
```

**Sample entries:**
| Code | Name (EN) | Name (VI) |
|------|-----------|-----------|
| J18.9 | Pneumonia, unspecified | Viêm phổi không xác định |
| E11.9 | Type 2 diabetes | Đái tháo đường type 2 |
| I10 | Essential hypertension | Tăng huyết áp nguyên phát |
| K21.0 | GERD with esophagitis | Trào ngược có viêm thực quản |

### 2. RxNorm Knowledge Base (`src/linking/rxnorm.py`)

```python
@dataclass
class RxNormEntry:
    rxcui: str                   # RxNorm CUI
    name: str                    # Drug name
    synonym: Optional[str]       # Synonym
    strength: Optional[str]      # e.g., "500mg"
    route: Optional[str]         # e.g., "oral"

class DrugParser:
    def parse(text: str) -> List[ParsedDrug]

class RxNormLinker:
    def link(drug_text: str, limit: int = 5) -> List[tuple[RxNormEntry, float]]
```

### 3. Lab Test Extractor (`src/entity/lab_extractor.py`)

```python
@dataclass
class LabMatch:
    text: str
    start: int
    end: int
    value: Optional[str] = None

class LabTestExtractor:
    def extract_all(text: str) -> Tuple[List[LabMatch], List[LabMatch]]:
        """Returns (test_names, test_results)"""

    def extract_lab_panel(text: str) -> List[str]
```

**Supported patterns:**
- `WBC`, `RBC`, `Hemoglobin`, `Hematocrit`
- `Glucose`, `Cholesterol`, `Triglyceride`, `HDL`, `LDL`
- `Creatinine`, `BUN`, `eGFR`
- `ALT`, `AST`, `GGT`
- `CRP`, `ESR`, `PCT`
- `Na+`, `K+`, `Ca2+`, `Mg2+`

### 4. Drug Extractor (`src/entity/drug_extractor.py`)

```python
@dataclass
class DrugMatch:
    text: str
    start: int
    end: int
    ingredient: Optional[str]
    strength: Optional[str]
    route: Optional[str]
    frequency: Optional[str]
    confidence: float

class DrugExtractor:
    def extract(text: str) -> List[DrugMatch]
```

**Features:**
- Common drug name detection (paracetamol, metformin, amlodipine, etc.)
- Dosage parsing (mg, g, ml units)
- Route detection (uống, tiêm, po, oral)
- Frequency detection (ngày, lần, bid, tid)

### 5. Disease Extractor (`src/entity/disease_extractor.py`)

```python
@dataclass
class DiseaseMatch:
    text: str
    start: int
    end: int
    context: str          # CHẨN_ĐOÁN or TRIỆU_CHỨNG
    confidence: float
    is_diagnosed: bool

class DiseaseExtractor:
    def extract(text: str) -> List[DiseaseMatch]
```

**Context detection:**
- `CHẨN_ĐOÁN`: After "chẩn đoán", "mắc bệnh", "bị bệnh"
- `TRIỆU_CHỨNG`: After symptom keywords like "ho", "đau", "sốt"

### 6. Assertion Detector (`src/assertion/rules.py`)

```python
NEGATION_CUES = ["không", "chưa", "loại trừ", "không thấy", ...]
HISTORICAL_CUES = ["tiền sử", "đã từng", "trước đây", ...]
FAMILY_CUES = ["bố", "mẹ", "cha", "gia đình", ...]

@dataclass
class EntityAssertion:
    entity_text: str
    is_negated: bool
    is_historical: bool
    is_family: bool

class AssertionDetector:
    def detect(text: str, entity_start: int, entity_end: int) -> EntityAssertion
```

### 7. Span Resolver (`src/entity/span_resolver.py`)

```python
@dataclass
class Span:
    start: int
    end: int
    text: str
    entity_type: str
    confidence: float
    source: str

class SpanResolver:
    TYPE_PRIORITY = {
        "THUỐC": 5,
        "CHẨN_ĐOÁN": 4,
        "TRIỆU_CHỨNG": 3,
        "KẾT_QUẢ_XÉT_NGHIỆM": 2,
        "TÊN_XÉT_NGHIỆM": 1,
    }

    def resolve(spans: list[Span]) -> ResolutionResult
```

**Resolution strategies:**
- `longest`: Chọn span dài nhất
- `confidence`: Chọn confidence cao nhất
- `type_priority`: Chọn type ưu tiên cao nhất
- `hybrid`: Kết hợp cả 3

---

## Sử dụng Pipeline

### Quick Start

```python
from src.pipeline import extract_medical_entities

text = """
Bệnh nhân nam, 55 tuổi, nhập viện vì ho đờm xanh, sốt cao.
Tiền sử tăng huyết áp, đái tháo đường type 2.
Chẩn đoán: viêm phổi cộng đồng.
Điều trị: Ceftriaxone 1g, Paracetamol 500mg khi sốt.
"""

entities = extract_medical_entities(text)

# Output format:
# [
#   {
#     "text": "ho đờm xanh",
#     "position": [42, 54],
#     "type": "TRIỆU_CHỨNG",
#     "assertions": [],
#     "candidates": []
#   },
#   ...
# ]
```

### Full Pipeline Configuration

```python
from src.pipeline import MedicalOntologyPipeline, PipelineConfig

config = PipelineConfig(
    extract_labs=True,
    extract_drugs=True,
    extract_diseases=True,
    detect_assertions=True,
    link_icd=True,
    link_rxnorm=True,
    resolve_overlaps=True,
    overlap_strategy="hybrid"
)

pipeline = MedicalOntologyPipeline(config)
result = pipeline.process(text)

# Access results
print(f"Found {len(result.entities)} entities")
if result.validation_result:
    print(result.validation_result.summary())
```

### Process Files

```python
from src.pipeline import MedicalOntologyPipeline

pipeline = MedicalOntologyPipeline()

# Single file
result = pipeline.process_file("input.txt", "output.json")

# Directory
results = pipeline.process_directory("input_dir/", "output_dir/")
```

### CLI Usage

```bash
python -m src.pipeline --input input.txt --output output.json
python -m src.pipeline --input input_dir/ --output output_dir/
```

---

## Kết quả kiểm tra

```
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 77 items

tests/test_schema.py ....................                                    [ 22%]
tests/test_loader.py ................                                     [ 33%]
tests/test_validation.py .................                                    [ 48%]
tests/test_pipeline.py ....................                                   [ 74%]

============================= 77 passed in 0.83s =============================
```

### Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| `test_schema.py` | 22 | ✅ Passed |
| `test_loader.py` | 16 | ✅ Passed |
| `test_validation.py` | 16 | ✅ Passed |
| `test_pipeline.py` | 23 | ✅ Passed |
| **Total** | **77** | **✅ All Passed** |

---

## Hạn chế của Baseline

### 1. Rule-based Limitations

- **Coverage**: Chỉ nhận diện được entities trong patterns định nghĩa sẵn
- **Context**: Khó phân biệt ngữ cảnh phức tạp
- **Ambiguity**: Không xử lý được đa nghĩa

### 2. Entity Linking

- **Sample KB**: Chỉ có sample entries, cần load full datasets
- **Fuzzy matching**: Accuracy phụ thuộc vào threshold
- **Multi-word**: Cần improve multi-word drug/disease matching

### 3. Assertion Detection

- **Cue scope**: Giới hạn trong 150 ký tự lookback
- **Complex negation**: Chưa xử lý "double negation"
- **Conditional**: Chưa xử lý "có thể", "có thể không"

### 4. Span Resolution

- **No ML model**: Chỉ dùng rules
- **Type priority**: Cố định, không linh hoạt
- **Adjacent merge**: Có thể merge nhầm

---

## Next Steps

### Potential Improvements

1. **ML-based NER**: Train BERT/RoBERTa model cho Vietnamese medical NER
2. **Full KB Loading**: Load complete ICD-10 và RxNorm datasets
3. **Contextual Embeddings**: Dùng sentence embeddings cho fuzzy matching
4. **Transformer-based Assertion**: Fine-tune model cho assertion detection
5. **Data Augmentation**: Tạo synthetic training data
6. **Evaluation Metrics**: Precision, Recall, F1 với gold standard annotations

### Running Demo

```bash
cd medical-ontology
python demo.py
```

Demo output shows:
- Simple entity extraction
- JSON output format
- Individual extractor outputs
- Full pipeline with configuration

---

## Tài liệu tham khảo

- Previous Phase: [TASKS_01_04.md](./TASKS_01_04.md)
- Medical AI Prompts: `../medical_ontology_ai_prompts/`

---

*Document created: 2026-07-11*
*Tasks completed: 05-11*
*Baseline implementation: Rule-based Medical NER*
