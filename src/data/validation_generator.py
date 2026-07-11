"""
Validation Template Generator

Tạo manual validation set templates (210 samples):
- Symptom category: 40 samples
- Diagnosis category: 40 samples
- Drug category: 40 samples
- Lab test category: 40 samples
- Multi-entity category: 30 samples
- Edge cases: 20 samples
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .schema import ValidationTemplate


# =============================================================================
# Validation Text Patterns
# =============================================================================

# Complex/multi-symptom patterns
SYMPTOM_PATTERNS = [
    # Simple symptoms
    {"text": "Bệnh nhân ho khan 2 ngày.", "expected": [{"text": "ho khan", "type": "TRIỆU_CHỨNG"}], "difficulty": "easy"},
    {"text": "BN sốt cao 39 độ C.", "expected": [{"text": "sốt cao", "type": "TRIỆU_CHỨNG"}], "difficulty": "easy"},
    {"text": "Người bệnh đau đầu vùng trán.", "expected": [{"text": "đau đầu", "type": "TRIỆU_CHỨNG"}], "difficulty": "easy"},

    # With negation
    {"text": "Bệnh nhân không ho, không sốt.", "expected": [{"text": "ho", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}, {"text": "sốt", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}], "difficulty": "medium"},
    {"text": "Không ghi nhận đau ngực.", "expected": [{"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}], "difficulty": "medium"},
    {"text": "Loại trừ triệu chứng đau bụng.", "expected": [{"text": "đau bụng", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}], "difficulty": "medium"},

    # Multiple symptoms
    {"text": "BN ho đờm, sốt cao, mệt mỏi.", "expected": [{"text": "ho đờm", "type": "TRIỆU_CHỨNG"}, {"text": "sốt cao", "type": "TRIỆU_CHỨNG"}, {"text": "mệt mỏi", "type": "TRIỆU_CHỨNG"}], "difficulty": "medium"},
    {"text": "Bệnh nhân đau đầu, chóng mặt, buồn nôn.", "expected": [{"text": "đau đầu", "type": "TRIỆU_CHỨNG"}, {"text": "chóng mặt", "type": "TRIỆU_CHỨNG"}, {"text": "buồn nôn", "type": "TRIỆU_CHỨNG"}], "difficulty": "medium"},
    {"text": "Có triệu chứng khó thở, thở nhanh, tím môi.", "expected": [{"text": "khó thở", "type": "TRIỆU_CHỨNG"}, {"text": "thở nhanh", "type": "TRIỆU_CHỨNG"}], "difficulty": "medium"},

    # Complex context
    {"text": "Khám: phổi không ran, tim không có tiếng mờ.", "expected": [{"text": "ran", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}], "difficulty": "hard"},
    {"text": "Bụng chướng, gan to, lách không to.", "expected": [{"text": "chướng", "type": "TRIỆU_CHỨNG"}, {"text": "gan to", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},
    {"text": "Da vàng, niêm mạc không vàng.", "expected": [{"text": "Da vàng", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},

    # Historical symptoms
    {"text": "Tiền sử ho khan kéo dài 2 tháng trước.", "expected": [{"text": "ho khan", "type": "TRIỆU_CHỨNG", "assertion": "isHistorical"}], "difficulty": "medium"},
    {"text": "Trước đây bệnh nhân từng bị đau ngực.", "expected": [{"text": "đau ngực", "type": "TRIỆU_CHỨNG", "assertion": "isHistorical"}], "difficulty": "medium"},

    # Ambiguous symptoms
    {"text": "Bệnh nhân mệt, có thể do thiếu máu.", "expected": [{"text": "mệt", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},
    {"text": "BN ho nhẹ, không rõ nguyên nhân.", "expected": [{"text": "ho", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},
    {"text": "Người bệnh đau bụng, điều trị không đỡ.", "expected": [{"text": "đau bụng", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},
]

# Diagnosis patterns
DIAGNOSIS_PATTERNS = [
    # Simple diagnosis
    {"text": "Chẩn đoán: viêm phổi cộng đồng.", "expected": [{"text": "viêm phổi cộng đồng", "type": "CHẨN_ĐOÁN"}], "difficulty": "easy"},
    {"text": "Bệnh chính: tăng huyết áp.", "expected": [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}], "difficulty": "easy"},
    {"text": "Chẩn đoán bệnh: đái tháo đường type 2.", "expected": [{"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN"}], "difficulty": "easy"},

    # Multiple diagnoses
    {"text": "Chẩn đoán: viêm phổi, suy tim độ A.", "expected": [{"text": "viêm phổi", "type": "CHẨN_ĐOÁN"}, {"text": "suy tim độ A", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},
    {"text": "Bệnh nhân mắc tăng huyết áp và đái tháo đường type 2.", "expected": [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}, {"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},
    {"text": "Phát hiện trào ngược dạ dày thực quản, viêm dạ dày.", "expected": [{"text": "trào ngược dạ dày thực quản", "type": "CHẨN_ĐOÁN"}, {"text": "viêm dạ dày", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},

    # Historical diagnosis
    {"text": "Tiền sử viêm gan B.", "expected": [{"text": "viêm gan B", "type": "CHẨN_ĐOÁN", "assertion": "isHistorical"}], "difficulty": "easy"},
    {"text": "Có tiền sử nhồi máu cơ tim năm 2020.", "expected": [{"text": "nhồi máu cơ tim", "type": "CHẨN_ĐOÁN", "assertion": "isHistorical"}], "difficulty": "medium"},
    {"text": "Trước đây bệnh nhân đã từng bị hen suyễn.", "expected": [{"text": "hen suyễn", "type": "CHẨN_ĐOÁN", "assertion": "isHistorical"}], "difficulty": "medium"},

    # Family history
    {"text": "Gia đình có người bị đái tháo đường.", "expected": [{"text": "đái tháo đường", "type": "CHẨN_ĐOÁN", "assertion": "isFamily"}], "difficulty": "medium"},
    {"text": "Bố bệnh nhân bị tăng huyết áp.", "expected": [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertion": "isFamily"}], "difficulty": "easy"},
    {"text": "Mẹ có tiền sử ung thư vú.", "expected": [{"text": "ung thư vú", "type": "CHẨN_ĐOÁN", "assertion": "isFamily"}], "difficulty": "medium"},

    # With negation
    {"text": "Loại trừ bệnh lao phổi.", "expected": [{"text": "lao phổi", "type": "CHẨN_ĐOÁN", "assertion": "isNegated"}], "difficulty": "medium"},
    {"text": "Không phải viêm màng não.", "expected": [{"text": "viêm màng não", "type": "CHẨN_ĐOÁN", "assertion": "isNegated"}], "difficulty": "medium"},
    {"text": "Chẩn đoán loại trừ: bệnh mạch vành.", "expected": [{"text": "bệnh mạch vành", "type": "CHẨN_ĐOÁN", "assertion": "isNegated"}], "difficulty": "hard"},

    # Complex/uncertain
    {"text": "Chẩn đoán: theo dõi viêm phổi, chưa xác định.", "expected": [{"text": "viêm phổi", "type": "CHẨN_ĐOÁN"}], "difficulty": "hard"},
    {"text": "Nghi ngờ bệnh thận mạn tính, cần xét nghiệm thêm.", "expected": [], "difficulty": "hard"},  # Suspicion, not confirmed
    {"text": "Có thể là viêm ruột thừa hoặc viêm phúc mạc.", "expected": [], "difficulty": "hard"},  # Differential diagnosis

    # Rare/edge cases
    {"text": "Phát hiện bệnh Pompe.", "expected": [{"text": "bệnh Pompe", "type": "CHẨN_ĐOÁN"}], "difficulty": "hard"},
    {"text": "Chẩn đoán: Wilson disease.", "expected": [{"text": "Wilson disease", "type": "CHẨN_ĐOÁN"}], "difficulty": "hard"},
    {"text": "Xác định bệnh Fabry.", "expected": [{"text": "bệnh Fabry", "type": "CHẨN_ĐOÁN"}], "difficulty": "hard"},
]

# Drug patterns
DRUG_PATTERNS = [
    # Simple drug
    {"text": "Điều trị: Paracetamol 500mg.", "expected": [{"text": "Paracetamol 500mg", "type": "THUỐC"}], "difficulty": "easy"},
    {"text": "Kê đơn: Metformin 500mg x 2 viên/ngày.", "expected": [{"text": "Metformin 500mg", "type": "THUỐC"}], "difficulty": "easy"},
    {"text": "Dùng Amoxicillin 500mg.", "expected": [{"text": "Amoxicillin 500mg", "type": "THUỐC"}], "difficulty": "easy"},

    # Multiple drugs
    {"text": "Điều trị: Ceftriaxone 1g và Paracetamol 500mg.", "expected": [{"text": "Ceftriaxone 1g", "type": "THUỐC"}, {"text": "Paracetamol 500mg", "type": "THUỐC"}], "difficulty": "medium"},
    {"text": "Phác đồ: Amoxicillin 500mg, Metronidazole 500mg, Omeprazole 20mg.", "expected": [{"text": "Amoxicillin 500mg", "type": "THUỐC"}, {"text": "Metronidazole 500mg", "type": "THUỐC"}, {"text": "Omeprazole 20mg", "type": "THUỐC"}], "difficulty": "medium"},

    # With frequency
    {"text": "Omeprazole 20mg x 2 lần/ngày.", "expected": [{"text": "Omeprazole 20mg", "type": "THUỐC"}], "difficulty": "easy"},
    {"text": "Metformin 500mg uống 3 lần/ngày sau ăn.", "expected": [{"text": "Metformin 500mg", "type": "THUỐC"}], "difficulty": "medium"},
    {"text": "Paracetamol 500mg khi sốt trên 38.5 độ.", "expected": [{"text": "Paracetamol 500mg", "type": "THUỐC"}], "difficulty": "easy"},

    # Injection
    {"text": "Tiêm Ceftriaxone 1g truyền tĩnh mạch.", "expected": [{"text": "Ceftriaxone 1g", "type": "THUỐC"}], "difficulty": "easy"},
    {"text": "Tiêm truyền: Metronidazole 500mg.", "expected": [{"text": "Metronidazole 500mg", "type": "THUỐC"}], "difficulty": "medium"},

    # Historical
    {"text": "Tiền sử dùng Aspirin 81mg.", "expected": [{"text": "Aspirin 81mg", "type": "THUỐC", "assertion": "isHistorical"}], "difficulty": "medium"},
    {"text": "Trước đây sử dụng Metformin 1000mg.", "expected": [{"text": "Metformin 1000mg", "type": "THUỐC", "assertion": "isHistorical"}], "difficulty": "medium"},

    # Family (drug allergy context)
    {"text": "Bố dị ứng với Penicillin.", "expected": [], "difficulty": "medium"},  # Drug allergy context, not actual drug use
    {"text": "Gia đình có người dùng Warfarin.", "expected": [], "difficulty": "hard"},  # Family context, not patient's drug

    # Complex instructions
    {"text": "Amoxicillin 500mg uống 8 giờ/lần, 7 ngày.", "expected": [{"text": "Amoxicillin 500mg", "type": "THUỐC"}], "difficulty": "medium"},
    {"text": "Prednisolone 5mg giảm liều dần trong 2 tuần.", "expected": [{"text": "Prednisolone 5mg", "type": "THUỐC"}], "difficulty": "hard"},
    {"text": "Lợi tiểu furosemide 40mg x 1 lần/ngày sáng.", "expected": [{"text": "furosemide 40mg", "type": "THUỐC"}], "difficulty": "hard"},

    # Edge cases
    {"text": "BN tự ý dùng thuốc nam.", "expected": [], "difficulty": "hard"},  # Not a specific drug
    {"text": "Uống thuốc bổ gan.", "expected": [], "difficulty": "hard"},  # Not specific enough
    {"text": "Dùng mỡ trị viêm khớp.", "expected": [], "difficulty": "hard"},  # Topical, not specific drug
]

# Lab test patterns
LAB_PATTERNS = [
    # Simple lab
    {"text": "Xét nghiệm: WBC 12.5 G/L.", "expected": [{"text": "WBC 12.5 G/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},
    {"text": "Kết quả: Glucose 126 mg/dL.", "expected": [{"text": "Glucose 126 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},
    {"text": "CRP 85 mg/L.", "expected": [{"text": "CRP 85 mg/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},

    # Multiple labs
    {"text": "Xét nghiệm: WBC 12.5 G/L, CRP 85 mg/L.", "expected": [{"text": "WBC 12.5 G/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "CRP 85 mg/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "medium"},
    {"text": "Kết quả XN: Glucose 126 mg/dL, Cholesterol 245 mg/dL, Triglyceride 180 mg/dL.", "expected": [{"text": "Glucose 126 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "Cholesterol 245 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "Triglyceride 180 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "medium"},

    # Normal/abnormal
    {"text": "Creatinine 1.5 mg/dL (tăng).", "expected": [{"text": "Creatinine 1.5 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},
    {"text": "Hemoglobin 95 g/L (giảm).", "expected": [{"text": "Hemoglobin 95 g/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},

    # Unit variations
    {"text": "Glucose: 126 mg/dL.", "expected": [{"text": "Glucose: 126 mg/dL", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "easy"},
    {"text": "Na+ = 132 mmol/L.", "expected": [{"text": "Na+ = 132 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "medium"},
    {"text": "K+ 5.8 mmol/L (cao).", "expected": [{"text": "K+ 5.8 mmol/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "medium"},

    # Panel/group
    {"text": "CBC: WBC 12.5 G/L, RBC 4.2 T/L, Hgb 95 g/L.", "expected": [{"text": "WBC 12.5 G/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "RBC 4.2 T/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "Hgb 95 g/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "hard"},
    {"text": "Sinh hóa máu: ALT 85 U/L, AST 72 U/L, GGT 95 U/L.", "expected": [{"text": "ALT 85 U/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "AST 72 U/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "GGT 95 U/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}], "difficulty": "hard"},

    # Normal values
    {"text": "Xét nghiệm trong giới hạn bình thường: WBC, RBC, Glucose.", "expected": [], "difficulty": "hard"},  # Not extracting specific values
    {"text": "Tất cả các chỉ số xét nghiệm đều bình thường.", "expected": [], "difficulty": "hard"},  # No specific values

    # Edge cases
    {"text": "XN cho thấy cần sinh thiết.", "expected": [], "difficulty": "hard"},  # Not lab value
    {"text": "Kết quả chụp CT: bình thường.", "expected": [], "difficulty": "hard"},  # Imaging, not lab
    {"text": "Xét nghiệm máu.", "expected": [], "difficulty": "hard"},  # Too vague
]

# Multi-entity patterns
MULTI_ENTITY_PATTERNS = [
    # Symptom + Diagnosis
    {"text": "BN ho đờm xanh, sốt cao. Chẩn đoán: viêm phổi.", "expected": [{"text": "ho đờm xanh", "type": "TRIỆU_CHỨNG"}, {"text": "sốt cao", "type": "TRIỆU_CHỨNG"}, {"text": "viêm phổi", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},
    {"text": "Bệnh nhân đau thượng vị, ợ chua. Điều trị: Omeprazole 20mg.", "expected": [{"text": "đau thượng vị", "type": "TRIỆU_CHỨNG"}, {"text": "ợ chua", "type": "TRIỆU_CHỨNG"}, {"text": "Omeprazole 20mg", "type": "THUỐC"}], "difficulty": "medium"},

    # Diagnosis + Drug
    {"text": "Chẩn đoán: đái tháo đường type 2. Điều trị: Metformin 500mg.", "expected": [{"text": "đái tháo đường type 2", "type": "CHẨN_ĐOÁN"}, {"text": "Metformin 500mg", "type": "THUỐC"}], "difficulty": "easy"},
    {"text": "Xác định tăng huyết áp. Kê đơn: Amlodipine 5mg.", "expected": [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN"}, {"text": "Amlodipine 5mg", "type": "THUỐC"}], "difficulty": "easy"},

    # All types
    {"text": "BN ho, sốt. Xét nghiệm WBC 12.5 G/L. Chẩn đoán viêm phổi. Điều trị Ceftriaxone 1g.", "expected": [{"text": "ho", "type": "TRIỆU_CHỨNG"}, {"text": "sốt", "type": "TRIỆU_CHỨNG"}, {"text": "WBC 12.5 G/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "viêm phổi", "type": "CHẨN_ĐOÁN"}, {"text": "Ceftriaxone 1g", "type": "THUỐC"}], "difficulty": "hard"},

    # With assertions
    {"text": "Tiền sử hen suyễn. Hiện tại BN ho khan. Điều trị: Prednisolone 5mg.", "expected": [{"text": "hen suyễn", "type": "CHẨN_ĐOÁN", "assertion": "isHistorical"}, {"text": "ho khan", "type": "TRIỆU_CHỨNG"}, {"text": "Prednisolone 5mg", "type": "THUỐC"}], "difficulty": "medium"},
    {"text": "Bệnh nhân đau ngực. Bố có tiền sử nhồi máu cơ tim.", "expected": [{"text": "đau ngực", "type": "TRIỆU_CHỨNG"}, {"text": "nhồi máu cơ tim", "type": "CHẨN_ĐOÁN", "assertion": "isFamily"}], "difficulty": "hard"},

    # Complex cases
    {"text": "BN nhập viện vì khó thở, ho đờm. XN: CRP 85 mg/L. Chẩn đoán: viêm phổi. Điều trị: Ceftriaxone 1g, Paracetamol 500mg.", "expected": [{"text": "khó thở", "type": "TRIỆU_CHỨNG"}, {"text": "ho đờm", "type": "TRIỆU_CHỨNG"}, {"text": "CRP 85 mg/L", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}, {"text": "viêm phổi", "type": "CHẨN_ĐOÁN"}, {"text": "Ceftriaxone 1g", "type": "THUỐC"}, {"text": "Paracetamol 500mg", "type": "THUỐC"}], "difficulty": "hard"},
]

# Edge cases
EDGE_CASE_PATTERNS = [
    # Ambiguous mentions
    {"text": "Bệnh nhân có vấn đề về đường huyết.", "expected": [], "difficulty": "hard"},  # Not specific
    {"text": "BN bị bệnh tim.", "expected": [], "difficulty": "hard"},  # Too vague
    {"text": "Người bệnh dùng thuốc huyết áp.", "expected": [], "difficulty": "hard"},  # Not specific drug name

    # Negation edge cases
    {"text": "Bệnh nhân không ho nhưng vẫn sốt.", "expected": [{"text": "ho", "type": "TRIỆU_CHỨNG", "assertion": "isNegated"}, {"text": "sốt", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},
    {"text": "Có thể không phải viêm phổi.", "expected": [], "difficulty": "hard"},  # Uncertainty
    {"text": "BN không có bệnh lý gì đáng kể.", "expected": [], "difficulty": "hard"},  # No specific entity

    # Numeric ambiguity
    {"text": "WBC tăng nhẹ.", "expected": [], "difficulty": "hard"},  # No value
    {"text": "Glucose cao hơn bình thường.", "expected": [], "difficulty": "hard"},  # No specific value
    {"text": "Xét nghiệm cho thấy bất thường.", "expected": [], "difficulty": "hard"},  # No specific test

    # Mixed language
    {"text": "BN được chẩn đoán pneumonia.", "expected": [{"text": "pneumonia", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},
    {"text": "Treatment: Aspirin 81mg.", "expected": [{"text": "Aspirin 81mg", "type": "THUỐC"}], "difficulty": "medium"},
    {"text": "Patient has hypertension.", "expected": [{"text": "hypertension", "type": "CHẨN_ĐOÁN"}], "difficulty": "medium"},

    # Temporal ambiguity
    {"text": "BN ho từ hôm qua.", "expected": [{"text": "ho", "type": "TRIỆU_CHỨNG"}], "difficulty": "medium"},
    {"text": "Tiền sử dùng thuốc 2 năm trước.", "expected": [], "difficulty": "hard"},  # Drug history, not specific

    # Overlapping entities
    {"text": "Viêm phổi cộng đồng mắc phải.", "expected": [{"text": "viêm phổi cộng đồng", "type": "CHẨN_ĐOÁN"}], "difficulty": "hard"},  # "mắc phải" is not part of diagnosis
    {"text": "Đau đầu vùng trán right.", "expected": [{"text": "đau đầu", "type": "TRIỆU_CHỨNG"}], "difficulty": "hard"},  # English word mixed
]


# =============================================================================
# Validation Template Generator
# =============================================================================

class ValidationTemplateGenerator:
    """
    Generator để tạo manual validation templates.

    Usage:
        generator = ValidationTemplateGenerator()
        templates = generator.generate_all()
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.counter = 0

    def _next_id(self) -> str:
        self.counter += 1
        return f"val_{self.counter:04d}"

    def _convert_expected(self, expected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert expected entities to validation format (without positions)."""
        result = []
        for exp in expected:
            entity = {
                "text": exp["text"],
                "type": exp["type"],
            }
            if "assertion" in exp:
                entity["assertions"] = [exp["assertion"]]
            result.append(entity)
        return result

    def generate_symptom_templates(self, count: int = 40) -> List[ValidationTemplate]:
        """Generate symptom validation templates."""
        templates = []
        shuffled = random.sample(SYMPTOM_PATTERNS, len(SYMPTOM_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="symptom",
            ))

        return templates

    def generate_diagnosis_templates(self, count: int = 40) -> List[ValidationTemplate]:
        """Generate diagnosis validation templates."""
        templates = []
        shuffled = random.sample(DIAGNOSIS_PATTERNS, len(DIAGNOSIS_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="diagnosis",
            ))

        return templates

    def generate_drug_templates(self, count: int = 40) -> List[ValidationTemplate]:
        """Generate drug validation templates."""
        templates = []
        shuffled = random.sample(DRUG_PATTERNS, len(DRUG_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="drug",
            ))

        return templates

    def generate_lab_templates(self, count: int = 40) -> List[ValidationTemplate]:
        """Generate lab test validation templates."""
        templates = []
        shuffled = random.sample(LAB_PATTERNS, len(LAB_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="lab",
            ))

        return templates

    def generate_multi_entity_templates(self, count: int = 30) -> List[ValidationTemplate]:
        """Generate multi-entity validation templates."""
        templates = []
        shuffled = random.sample(MULTI_ENTITY_PATTERNS, len(MULTI_ENTITY_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="multi",
            ))

        return templates

    def generate_edge_case_templates(self, count: int = 20) -> List[ValidationTemplate]:
        """Generate edge case validation templates."""
        templates = []
        shuffled = random.sample(EDGE_CASE_PATTERNS, len(EDGE_CASE_PATTERNS))

        for i in range(min(count, len(shuffled))):
            pattern = shuffled[i]
            templates.append(ValidationTemplate(
                id=self._next_id(),
                text=pattern["text"],
                expected_entities=self._convert_expected(pattern["expected"]),
                difficulty=pattern["difficulty"],
                category="edge_case",
            ))

        return templates

    def generate_all(self) -> List[ValidationTemplate]:
        """Generate all validation templates."""
        templates = []

        templates.extend(self.generate_symptom_templates(40))
        templates.extend(self.generate_diagnosis_templates(40))
        templates.extend(self.generate_drug_templates(40))
        templates.extend(self.generate_lab_templates(40))
        templates.extend(self.generate_multi_entity_templates(30))
        templates.extend(self.generate_edge_case_templates(20))

        return templates

    def generate_summary(self, templates: List[ValidationTemplate]) -> Dict[str, Any]:
        """Generate summary statistics for templates."""
        by_category: Dict[str, int] = {}
        by_difficulty: Dict[str, int] = {}
        total_entities = 0

        for t in templates:
            by_category[t.category] = by_category.get(t.category, 0) + 1
            by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1
            total_entities += len(t.expected_entities)

        return {
            "total_templates": len(templates),
            "total_expected_entities": total_entities,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for validation template generator."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate manual validation templates")
    parser.add_argument("--output", "-o", type=str, default="data/validation/manual_validation_template.jsonl", help="Output file")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = ValidationTemplateGenerator(seed=args.seed)
    templates = generator.generate_all()

    # Save to JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for template in templates:
            f.write(json.dumps(template.to_dict(), ensure_ascii=False) + "\n")

    # Print summary
    summary = generator.generate_summary(templates)

    print(f"Generated {summary['total_templates']} validation templates")
    print(f"Total expected entities: {summary['total_expected_entities']}")
    print(f"\nBy category:")
    for cat, count in summary['by_category'].items():
        print(f"  {cat}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in summary['by_difficulty'].items():
        print(f"  {diff}: {count}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
