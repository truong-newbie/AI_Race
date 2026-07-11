"""
Template Generator for Synthetic Medical Data

Tạo các template mẫu cho việc sinh synthetic data:
- CHẨN_ĐOÁN: 10 patterns
- TRIỆU_CHỨNG: 6 patterns
- THUỐC: 6 patterns
- XÉT_NGHIỆM: 6 patterns
- Multi-entity: 6 patterns
"""

import random
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .schema import Sample, Entity, find_span, EntityType


# =============================================================================
# Template Data
# =============================================================================

# Diseases/conditions for CHẨN_ĐOÁN
DIAGNOSES = {
    "viêm phổi": {"icd10": "J18.9", "name_vi": "Viêm phổi không xác định"},
    "viêm phổi cộng đồng": {"icd10": "J18.9", "name_vi": "Viêm phổi cộng đồng"},
    "tăng huyết áp": {"icd10": "I10", "name_vi": "Tăng huyết áp nguyên phát"},
    "đái tháo đường type 2": {"icd10": "E11.9", "name_vi": "Đái tháo đường type 2"},
    "đái tháo đường": {"icd10": "E11.9", "name_vi": "Đái tháo đường"},
    "trào ngược dạ dày thực quản": {"icd10": "K21.9", "name_vi": "Trào ngược dạ dày thực quản"},
    "viêm dạ dày": {"icd10": "K29.9", "name_vi": "Viêm dạ dày"},
    "viêm phổi mắc phải": {"icd10": "J18.9", "name_vi": "Viêm phổi mắc phải"},
    "hen suyễn": {"icd10": "J45.9", "name_vi": "Hen suyễn"},
    "bệnh phổi tắc nghẽn mạn tính": {"icd10": "J44.9", "name_vi": "Bệnh phổi tắc nghẽn mạn tính"},
    "suy tim": {"icd10": "I50.9", "name_vi": "Suy tim"},
    "nhồi máu cơ tim": {"icd10": "I21.9", "name_vi": "Nhồi máu cơ tim"},
    "đột quỵ": {"icd10": "I64", "name_vi": "Đột quỵ"},
    "viêm gan B": {"icd10": "B18.1", "name_vi": "Viêm gan B mạn tính"},
    "viêm khớp dạng thấp": {"icd10": "M06.9", "name_vi": "Viêm khớp dạng thấp"},
    "loét dạ dày": {"icd10": "K25.9", "name_vi": "Loét dạ dày"},
    "sỏi thận": {"icd10": "N20.0", "name_vi": "Sỏi thận"},
    "viêm bàng quang": {"icd10": "N30.9", "name_vi": "Viêm bàng quang"},
    "nhiễm trùng tiết niệu": {"icd10": "N39.0", "name_vi": "Nhiễm trùng tiết niệu"},
    "viêm mũi dị ứng": {"icd10": "J30.4", "name_vi": "Viêm mũi dị ứng"},
}

# Symptoms for TRIỆU_CHỨNG
SYMPTOMS = {
    "ho": ["ho", "ho khan", "ho đờm", "ho ra máu"],
    "sốt": ["sốt cao", "sốt nhẹ", "sốt không", "ủn ỉn sốt"],
    "đau": ["đau đầu", "đau bụng", "đau ngực", "đau lưng", "đau khớp"],
    "khó thở": ["khó thở", "thở nhanh", "thở khò khè", "thở hụt hơi"],
    "mệt": ["mệt mỏi", "mệt", "kiệt sức", "uể oải"],
    "buồn nôn": ["buồn nôn", "nôn", "nôn ói"],
    "tiêu chảy": ["tiêu chảy", "đi lỏng", "phân lỏng"],
    "táo bón": ["táo bón", "đi cầu khó"],
    "chóng mặt": ["chóng mặt", "hoa mắt", "ý whơi"],
    "phù": ["phù chân", "phù mặt", "phù toàn thân"],
    "vàng da": ["vàng da", "vàng mắt", "vàng da niêm mạc"],
    "xuất huyết": ["xuất huyết da", "bầm tím", "chảy máu"],
    "đờm": ["đờm", "đờm xanh", "đờm vàng", "đờm trắng"],
    "nghẹt mũi": ["nghẹt mũi", "chảy mũi", "mũi tắc"],
    "đau họng": ["đau họng", "nuốt đau", "nuốt khó"],
}

# Drugs for THUỐC
DRUGS = {
    "paracetamol": {"rxcui": "161", "name": "Acetaminophen", "strength": "500mg"},
    "paracetamol 500mg": {"rxcui": "161", "name": "Acetaminophen 500mg", "strength": "500mg"},
    "paracetamol 1g": {"rxcui": "161", "name": "Acetaminophen 1g", "strength": "1g"},
    "ceftriaxone": {"rxcui": "8628", "name": "Ceftriaxone", "strength": "1g"},
    "ceftriaxone 1g": {"rxcui": "8628", "name": "Ceftriaxone 1g", "strength": "1g"},
    "amoxicillin": {"rxcui": "723", "name": "Amoxicillin", "strength": "500mg"},
    "amoxicillin 500mg": {"rxcui": "723", "name": "Amoxicillin 500mg", "strength": "500mg"},
    "metformin": {"rxcui": "6809", "name": "Metformin", "strength": "500mg"},
    "metformin 500mg": {"rxcui": "6809", "name": "Metformin 500mg", "strength": "500mg"},
    "omeprazole": {"rxcui": "7646", "name": "Omeprazole", "strength": "20mg"},
    "omeprazole 20mg": {"rxcui": "7646", "name": "Omeprazole 20mg", "strength": "20mg"},
    "amlodipine": {"rxcui": "32937", "name": "Amlodipine", "strength": "5mg"},
    "amlodipine 5mg": {"rxcui": "32937", "name": "Amlodipine 5mg", "strength": "5mg"},
    "aspirin": {"rxcui": "1191", "name": "Aspirin", "strength": "81mg"},
    "aspirin 81mg": {"rxcui": "1191", "name": "Aspirin 81mg", "strength": "81mg"},
    "losartan": {"rxcui": "52175", "name": "Losartan", "strength": "50mg"},
    "atorvastatin": {"rxcui": "617312", "name": "Atorvastatin", "strength": "20mg"},
    "pantoprazole": {"rxcui": "40790", "name": "Pantoprazole", "strength": "40mg"},
    "ciprofloxacin": {"rxcui": "2556", "name": "Ciprofloxacin", "strength": "500mg"},
    "azithromycin": {"rxcui": "18631", "name": "Azithromycin", "strength": "500mg"},
    "prednisolone": {"rxcui": "8640", "name": "Prednisolone", "strength": "5mg"},
}

# Lab tests for XÉT_NGHIỆM
LAB_TESTS = {
    "CBC": ["WBC", "RBC", "Hemoglobin", "Hematocrit"],
    "sinh hóa": ["Glucose", "Creatinine", "BUN", "ALT", "AST", "Cholesterol", "Triglyceride", "HDL", "LDL"],
    "điện giải": ["Na+", "K+", "Ca2+", "Mg2+"],
    "viêm": ["CRP", "ESR", "PCT"],
    "chức năng gan": ["ALT", "AST", "GGT", "Bilirubin"],
    "chức năng thận": ["Creatinine", "BUN", "eGFR"],
}

# Lab test values
LAB_VALUES = {
    "WBC": ("5.0-10.0 G/L", "12.5 G/L"),
    "RBC": ("4.5-5.5 T/L", "4.2 T/L"),
    "Hemoglobin": ("120-160 g/L", "95 g/L"),
    "Hematocrit": ("36-46%", "32%"),
    "Glucose": ("70-100 mg/dL", "126 mg/dL"),
    "Creatinine": ("0.6-1.2 mg/dL", "1.5 mg/dL"),
    "BUN": ("7-20 mg/dL", "25 mg/dL"),
    "ALT": ("7-56 U/L", "85 U/L"),
    "AST": ("10-40 U/L", "72 U/L"),
    "Cholesterol": ("<200 mg/dL", "245 mg/dL"),
    "Triglyceride": ("<150 mg/dL", "180 mg/dL"),
    "HDL": (">40 mg/dL", "35 mg/dL"),
    "LDL": ("<100 mg/dL", "165 mg/dL"),
    "CRP": ("<10 mg/L", "85 mg/L"),
    "ESR": ("0-20 mm/h", "45 mm/h"),
    "Na+": ("136-145 mmol/L", "132 mmol/L"),
    "K+": ("3.5-5.0 mmol/L", "5.8 mmol/L"),
}


# =============================================================================
# Template Patterns
# =============================================================================

@dataclass
class Template:
    """Một template pattern."""
    pattern: str  # Text pattern with placeholders like {diagnosis}
    entity_type: str
    entity_key: str  # Key to extract entity text from context
    context_keywords: List[str] = field(default_factory=list)  # Keywords to help context detection
    assertion_context: Optional[Dict[str, Any]] = None  # For generating assertions


# CHẨN_ĐOÁN Templates (10 patterns)
DIAGNOSIS_TEMPLATES = [
    # Simple diagnosis patterns
    Template(
        pattern="Chẩn đoán: {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["chẩn đoán", "bị bệnh", "mắc bệnh"],
    ),
    Template(
        pattern="Chẩn đoán bệnh: {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["chẩn đoán"],
    ),
    Template(
        pattern="Bệnh chính: {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["bệnh chính"],
    ),
    Template(
        pattern="Chẩn đoán xác định: {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["chẩn đoán xác định"],
    ),
    Template(
        pattern="Chẩn đoán: bệnh nhân bị {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["bị"],
    ),
    Template(
        pattern="Bệnh nhân được chẩn đoán {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["chẩn đoán"],
    ),
    Template(
        pattern="Phát hiện {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["phát hiện"],
    ),
    Template(
        pattern="Xác định {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["xác định"],
    ),
    Template(
        pattern="Có {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=[],
    ),
    Template(
        pattern="Tiền sử {diagnosis}.",
        entity_type=EntityType.CHAN_DOAN.value,
        entity_key="diagnosis",
        context_keywords=["tiền sử"],
        assertion_context={"isHistorical": True},
    ),
]


# TRIỆU_CHỨNG Templates (6 patterns)
SYMPTOM_TEMPLATES = [
    Template(
        pattern="Bệnh nhân {symptom_verb}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom_verb",
        context_keywords=[],
    ),
    Template(
        pattern="Khám: {symptom}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom",
        context_keywords=["khám"],
    ),
    Template(
        pattern="BN {symptom}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom",
        context_keywords=[],
    ),
    Template(
        pattern="Bệnh nhân có biểu hiện {symptom}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom",
        context_keywords=["biểu hiện"],
    ),
    Template(
        pattern="Triệu chứng: {symptom}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom",
        context_keywords=["triệu chứng"],
    ),
    Template(
        pattern="Nhập viện vì {symptom}.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom",
        context_keywords=["nhập viện"],
    ),
]


# THUỐC Templates (6 patterns)
DRUG_TEMPLATES = [
    Template(
        pattern="Điều trị: {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["điều trị"],
    ),
    Template(
        pattern="Kê đơn {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["kê đơn"],
    ),
    Template(
        pattern="Dùng {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["dùng"],
    ),
    Template(
        pattern="Uống {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["uống"],
    ),
    Template(
        pattern="Tiêm {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["tiêm"],
    ),
    Template(
        pattern="Cho dùng {drug}.",
        entity_type=EntityType.THUOC.value,
        entity_key="drug",
        context_keywords=["cho dùng"],
    ),
]


# XÉT_NGHIỆM Templates (6 patterns)
LAB_TEMPLATES = [
    Template(
        pattern="Xét nghiệm: {lab_test} {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=["xét nghiệm"],
    ),
    Template(
        pattern="Kết quả {lab_test}: {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=["kết quả"],
    ),
    Template(
        pattern="XN {lab_test}: {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=["xn"],
    ),
    Template(
        pattern="{lab_test} {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=[],
    ),
    Template(
        pattern="Đo {lab_test}: {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=["đo"],
    ),
    Template(
        pattern="Lab: {lab_test} {value}.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="lab_with_value",
        context_keywords=["lab"],
    ),
]


# Multi-entity Templates (6 patterns)
MULTI_ENTITY_TEMPLATES = [
    Template(
        pattern="BN ho đờm xanh, sốt cao.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="multi_symptom",
        context_keywords=["ho", "sốt"],
    ),
    Template(
        pattern="Điều trị: Ceftriaxone 1g, Paracetamol 500mg.",
        entity_type=EntityType.THUOC.value,
        entity_key="multi_drug",
        context_keywords=["điều trị"],
    ),
    Template(
        pattern="Chẩn đoán: viêm phổi. Điều trị: Ceftriaxone 1g.",
        entity_type=EntityType.THUOC.value,
        entity_key="diagnosis_drug",
        context_keywords=["chẩn đoán", "điều trị"],
    ),
    Template(
        pattern="WBC 12.5 G/L, CRP 85 mg/L.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="multi_lab",
        context_keywords=["wbc", "crp"],
    ),
    Template(
        pattern="Bệnh nhân đau ngực, khó thở. Tiền sử tăng huyết áp.",
        entity_type=EntityType.TRIEU_CHUNG.value,
        entity_key="symptom_history",
        context_keywords=["đau ngực", "khó thở", "tiền sử"],
    ),
    Template(
        pattern="Xét nghiệm: Glucose 126 mg/dL, Cholesterol 245 mg/dL.",
        entity_type=EntityType.KET_QUA_XET_NGHIEM.value,
        entity_key="multi_lab_value",
        context_keywords=["xét nghiệm"],
    ),
]


# =============================================================================
# Assertion Templates
# =============================================================================

NEGATION_TEMPLATES = [
    "Bệnh nhân không {symptom}.",
    "Không có biểu hiện {symptom}.",
    "Không thấy {symptom}.",
    "Chưa ghi nhận {symptom}.",
    "Loại trừ {diagnosis}.",
    "Không phải {diagnosis}.",
    "Bệnh nhân hết {symptom}.",
]

HISTORICAL_TEMPLATES = [
    "Tiền sử {diagnosis}.",
    "Có tiền sử {diagnosis}.",
    "Đã từng bị {diagnosis}.",
    "Trước đây mắc {diagnosis}.",
    "BN có {diagnosis} từ trước.",
]

FAMILY_TEMPLATES = [
    "Bố bệnh nhân bị {diagnosis}.",
    "Mẹ có tiền sử {diagnosis}.",
    "Gia đình có người bị {diagnosis}.",
    "Ông bà bệnh nhân từng bị {diagnosis}.",
]


# =============================================================================
# Template Generator
# =============================================================================

class TemplateGenerator:
    """
    Generator để tạo synthetic samples từ templates.

    Usage:
        generator = TemplateGenerator()
        samples = generator.generate_all(count=100)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.counter = 0

    def _next_id(self, prefix: str = "tmpl") -> str:
        """Generate next sample ID."""
        self.counter += 1
        return f"{prefix}_{self.counter:05d}"

    def _get_diagnosis(self) -> str:
        """Get random diagnosis."""
        return random.choice(list(DIAGNOSES.keys()))

    def _get_symptom(self) -> Tuple[str, str]:
        """Get random symptom. Returns (symptom_type, symptom_text)."""
        symptom_type = random.choice(list(SYMPTOMS.keys()))
        symptom_text = random.choice(SYMPTOMS[symptom_type])
        return symptom_type, symptom_text

    def _get_drug(self) -> Tuple[str, str]:
        """Get random drug. Returns (drug_key, drug_text)."""
        drug_key = random.choice(list(DRUGS.keys()))
        return drug_key, drug_key  # Use key as text for now

    def _get_lab_test(self) -> Tuple[str, str, str]:
        """Get random lab test. Returns (test_name, normal_value, abnormal_value)."""
        test_name = random.choice(list(LAB_VALUES.keys()))
        normal, abnormal = LAB_VALUES[test_name]
        return test_name, normal, abnormal

    # -------------------------------------------------------------------------
    # Generate single entity samples
    # -------------------------------------------------------------------------

    def generate_diagnosis(self, count: int = 1, with_assertions: bool = True) -> List[Sample]:
        """Generate diagnosis samples."""
        samples = []
        for _ in range(count):
            template = random.choice(DIAGNOSIS_TEMPLATES)
            diagnosis = self._get_diagnosis()

            # Format text
            text = template.pattern.format(diagnosis=diagnosis)

            # Find entity span
            start, end = find_span(text, diagnosis)

            # Create entity
            entity = Entity(
                text=diagnosis,
                start=start,
                end=end,
                type=template.entity_type,
                assertions=[],
                candidates=[DIAGNOSES[diagnosis]["icd10"]] if diagnosis in DIAGNOSES else [],
            )

            # Add assertions if specified
            if with_assertions and template.assertion_context:
                if template.assertion_context.get("isHistorical"):
                    entity.assertions.append("isHistorical")

            samples.append(Sample(
                id=self._next_id("diag"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    def generate_symptom(self, count: int = 1, with_assertions: bool = True) -> List[Sample]:
        """Generate symptom samples."""
        samples = []
        for _ in range(count):
            template = random.choice(SYMPTOM_TEMPLATES)
            symptom_type, symptom_text = self._get_symptom()

            # Handle multi-symptom templates
            if template.entity_key == "multi_symptom":
                text = template.pattern
                # Find first symptom span
                start = text.find("ho")
                if start == -1:
                    start = text.find("sốt")
                end = start + len("ho đờm xanh")
                entity = Entity(
                    text="ho đờm xanh",
                    start=start,
                    end=end,
                    type=template.entity_type,
                    assertions=[],
                    candidates=[],
                )
            else:
                text = template.pattern.format(
                    symptom=symptom_text,
                    symptom_verb=f"{symptom_text}"
                )
                start, end = find_span(text, symptom_text)
                entity = Entity(
                    text=symptom_text,
                    start=start,
                    end=end,
                    type=template.entity_type,
                    assertions=[],
                    candidates=[],
                )

            samples.append(Sample(
                id=self._next_id("symp"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    def generate_drug(self, count: int = 1, with_assertions: bool = True) -> List[Sample]:
        """Generate drug samples."""
        samples = []
        for _ in range(count):
            template = random.choice(DRUG_TEMPLATES)
            drug_key, drug_text = self._get_drug()
            drug_info = DRUGS[drug_key]

            # Format text - use drug_key directly
            text = template.pattern.format(drug=drug_key)

            # Find entity span
            start, end = find_span(text, drug_key)

            entity = Entity(
                text=drug_key,
                start=start,
                end=end,
                type=template.entity_type,
                assertions=[],
                candidates=[drug_info["rxcui"]],
            )

            samples.append(Sample(
                id=self._next_id("drug"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    def generate_lab(self, count: int = 1, with_assertions: bool = True) -> List[Sample]:
        """Generate lab test samples."""
        samples = []
        for _ in range(count):
            template = random.choice(LAB_TEMPLATES)
            test_name, normal_val, abnormal_val = self._get_lab_test()

            # Use abnormal value for synthetic data
            value = abnormal_val

            # Format text
            if template.entity_key == "lab_with_value":
                text = template.pattern.format(lab_test=test_name, value=value)
            else:
                text = template.pattern

            # Entity text depends on pattern:
            # - "{test} {value}." -> entity is "{test} {value}"
            # - "Xét nghiệm: {test} {value}." -> entity is "{test} {value}"
            # - "Kết quả {test}: {value}." -> entity is "{test}: {value}"
            # - "Lab: {test} {value}." -> entity is "{test} {value}"
            # - "Đo {test}: {value}." -> entity is "{test}: {value}"
            # - "XN {test}: {value}." -> entity is "{test}: {value}"
            colon_patterns = ["Kết quả", "Đo", "XN"]
            if any(p in template.pattern for p in colon_patterns):
                entity_text = f"{test_name}: {value}"
            else:
                entity_text = f"{test_name} {value}"

            # Try to find in formatted text (without trailing period)
            text_no_period = text.rstrip('.')
            start, end = find_span(text_no_period, entity_text)

            # Adjust end to not include period if present
            actual_end = end
            if end < len(text_no_period) and text_no_period[end:end+1] in ['.', ',']:
                actual_end = end + 1

            entity = Entity(
                text=entity_text,
                start=start,
                end=actual_end,
                type=template.entity_type,
                assertions=[],
                candidates=[],
            )

            samples.append(Sample(
                id=self._next_id("lab"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    # -------------------------------------------------------------------------
    # Generate assertion samples
    # -------------------------------------------------------------------------

    def generate_negated(self, count: int = 10) -> List[Sample]:
        """Generate negated entity samples."""
        samples = []
        for _ in range(count):
            template_str = random.choice(NEGATION_TEMPLATES)

            # Try to fill template
            if "{symptom}" in template_str:
                _, symptom = self._get_symptom()
                text = template_str.format(symptom=symptom)
                entity_text = symptom
                entity_type = EntityType.TRIEU_CHUNG.value
            elif "{diagnosis}" in template_str:
                diagnosis = self._get_diagnosis()
                text = template_str.format(diagnosis=diagnosis)
                entity_text = diagnosis
                entity_type = EntityType.CHAN_DOAN.value
            else:
                continue

            start, end = find_span(text, entity_text)
            entity = Entity(
                text=entity_text,
                start=start,
                end=end,
                type=entity_type,
                assertions=["isNegated"],
                candidates=[],
            )

            samples.append(Sample(
                id=self._next_id("neg"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    def generate_historical(self, count: int = 10) -> List[Sample]:
        """Generate historical entity samples."""
        samples = []
        for _ in range(count):
            template_str = random.choice(HISTORICAL_TEMPLATES)
            diagnosis = self._get_diagnosis()
            text = template_str.format(diagnosis=diagnosis)

            start, end = find_span(text, diagnosis)
            entity = Entity(
                text=diagnosis,
                start=start,
                end=end,
                type=EntityType.CHAN_DOAN.value,
                assertions=["isHistorical"],
                candidates=[DIAGNOSES[diagnosis]["icd10"]] if diagnosis in DIAGNOSES else [],
            )

            samples.append(Sample(
                id=self._next_id("hist"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    def generate_family(self, count: int = 10) -> List[Sample]:
        """Generate family history entity samples."""
        samples = []
        for _ in range(count):
            template_str = random.choice(FAMILY_TEMPLATES)
            diagnosis = self._get_diagnosis()
            text = template_str.format(diagnosis=diagnosis)

            start, end = find_span(text, diagnosis)
            entity = Entity(
                text=diagnosis,
                start=start,
                end=end,
                type=EntityType.CHAN_DOAN.value,
                assertions=["isFamily"],
                candidates=[DIAGNOSES[diagnosis]["icd10"]] if diagnosis in DIAGNOSES else [],
            )

            samples.append(Sample(
                id=self._next_id("fam"),
                text=text,
                entities=[entity],
                source="template",
                review_status="auto_validated",
            ))

        return samples

    # -------------------------------------------------------------------------
    # Generate multi-entity samples
    # -------------------------------------------------------------------------

    def generate_multi_entity(self, count: int = 1) -> List[Sample]:
        """Generate samples with multiple entities."""
        samples = []
        for _ in range(count):
            template = random.choice(MULTI_ENTITY_TEMPLATES)

            if template.entity_key == "multi_symptom":
                text = template.pattern
                # Calculate spans dynamically
                s1_start, s1_end = find_span(text, "ho đờm xanh")
                s2_start, s2_end = find_span(text, "sốt cao")
                entities = [
                    Entity(text="ho đờm xanh", start=s1_start, end=s1_end, type=EntityType.TRIEU_CHUNG.value, assertions=[], candidates=[]),
                    Entity(text="sốt cao", start=s2_start, end=s2_end, type=EntityType.TRIEU_CHUNG.value, assertions=[], candidates=[]),
                ]
            elif template.entity_key == "multi_drug":
                text = template.pattern
                s1_start, s1_end = find_span(text, "Ceftriaxone 1g")
                s2_start, s2_end = find_span(text, "Paracetamol 500mg")
                entities = [
                    Entity(text="Ceftriaxone 1g", start=s1_start, end=s1_end, type=EntityType.THUOC.value, assertions=[], candidates=["8628"]),
                    Entity(text="Paracetamol 500mg", start=s2_start, end=s2_end, type=EntityType.THUOC.value, assertions=[], candidates=["161"]),
                ]
            elif template.entity_key == "diagnosis_drug":
                text = template.pattern
                d_start, d_end = find_span(text, "viêm phổi")
                dr_start, dr_end = find_span(text, "Ceftriaxone 1g")
                entities = [
                    Entity(text="viêm phổi", start=d_start, end=d_end, type=EntityType.CHAN_DOAN.value, assertions=[], candidates=["J18.9"]),
                    Entity(text="Ceftriaxone 1g", start=dr_start, end=dr_end, type=EntityType.THUOC.value, assertions=[], candidates=["8628"]),
                ]
            elif template.entity_key == "multi_lab":
                text = template.pattern
                l1_start, l1_end = find_span(text, "WBC 12.5 G/L")
                l2_start, l2_end = find_span(text, "CRP 85 mg/L")
                entities = [
                    Entity(text="WBC 12.5 G/L", start=l1_start, end=l1_end, type=EntityType.KET_QUA_XET_NGHIEM.value, assertions=[], candidates=[]),
                    Entity(text="CRP 85 mg/L", start=l2_start, end=l2_end, type=EntityType.KET_QUA_XET_NGHIEM.value, assertions=[], candidates=[]),
                ]
            elif template.entity_key == "symptom_history":
                text = template.pattern
                s1_start, s1_end = find_span(text, "đau ngực")
                s2_start, s2_end = find_span(text, "khó thở")
                d_start, d_end = find_span(text, "tăng huyết áp")
                entities = [
                    Entity(text="đau ngực", start=s1_start, end=s1_end, type=EntityType.TRIEU_CHUNG.value, assertions=[], candidates=[]),
                    Entity(text="khó thở", start=s2_start, end=s2_end, type=EntityType.TRIEU_CHUNG.value, assertions=[], candidates=[]),
                    Entity(text="tăng huyết áp", start=d_start, end=d_end, type=EntityType.CHAN_DOAN.value, assertions=["isHistorical"], candidates=["I10"]),
                ]
            elif template.entity_key == "multi_lab_value":
                text = template.pattern
                l1_start, l1_end = find_span(text, "Glucose 126 mg/dL")
                l2_start, l2_end = find_span(text, "Cholesterol 245 mg/dL")
                entities = [
                    Entity(text="Glucose 126 mg/dL", start=l1_start, end=l1_end, type=EntityType.KET_QUA_XET_NGHIEM.value, assertions=[], candidates=[]),
                    Entity(text="Cholesterol 245 mg/dL", start=l2_start, end=l2_end, type=EntityType.KET_QUA_XET_NGHIEM.value, assertions=[], candidates=[]),
                ]
            else:
                continue

            samples.append(Sample(
                id=self._next_id("multi"),
                text=text,
                entities=entities,
                source="template",
                review_status="auto_validated",
            ))

        return samples

    # -------------------------------------------------------------------------
    # Generate all
    # -------------------------------------------------------------------------

    def generate_all(self, count: int = 100) -> List[Sample]:
        """
        Generate all types of samples.

        Args:
            count: Approximate total count (distributed across types)

        Returns:
            List of samples
        """
        # Distribute counts
        n_diagnosis = count // 5
        n_symptom = count // 5
        n_drug = count // 5
        n_lab = count // 5
        n_assertion = count // 10
        n_multi = count // 10

        samples = []

        # Generate each type
        samples.extend(self.generate_diagnosis(n_diagnosis))
        samples.extend(self.generate_symptom(n_symptom))
        samples.extend(self.generate_drug(n_drug))
        samples.extend(self.generate_lab(n_lab))
        samples.extend(self.generate_negated(n_assertion))
        samples.extend(self.generate_historical(n_assertion))
        samples.extend(self.generate_family(n_assertion))
        samples.extend(self.generate_multi_entity(n_multi))

        return samples


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for template generator."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate synthetic medical data from templates")
    parser.add_argument("--count", "-n", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic/template_samples.jsonl", help="Output file")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = TemplateGenerator(seed=args.seed)
    samples = generator.generate_all(count=args.count)

    # Save to JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} samples")
    print(f"Saved to {args.output}")

    # Show distribution
    by_type: Dict[str, int] = {}
    for sample in samples:
        for entity in sample.entities:
            by_type[entity.type] = by_type.get(entity.type, 0) + 1

    print("\nDistribution by entity type:")
    for etype, count in sorted(by_type.items()):
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
