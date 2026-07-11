"""
ICD-10 Knowledge Base Schema

Processed schema for ICD-10 candidate retrieval:
  code, name_vi, name_en, synonyms, aliases, description,
  parent_code, chapter, include_terms, exclude_terms, normalized_text.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ICD10Entry:
    """ICD-10 entry with all searchable fields."""
    code: str
    name_vi: Optional[str] = None
    name_en: Optional[str] = None
    synonyms: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    description: Optional[str] = None
    parent_code: Optional[str] = None
    chapter: Optional[str] = None
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    normalized_text: Optional[str] = None

    def get_all_searchable_texts(self) -> list[str]:
        """All texts used for retrieval matching."""
        texts = []
        if self.name_vi:
            texts.append(self.name_vi)
        if self.name_en:
            texts.append(self.name_en)
        texts.extend(self.synonyms)
        texts.extend(self.aliases)
        if self.description:
            texts.append(self.description)
        texts.extend(self.include_terms)
        return texts

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name_vi": self.name_vi,
            "name_en": self.name_en,
            "synonyms": self.synonyms,
            "aliases": self.aliases,
            "description": self.description,
            "parent_code": self.parent_code,
            "chapter": self.chapter,
            "include_terms": self.include_terms,
            "exclude_terms": self.exclude_terms,
            "normalized_text": self.normalized_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ICD10Entry":
        return cls(
            code=data["code"],
            name_vi=data.get("name_vi"),
            name_en=data.get("name_en"),
            synonyms=data.get("synonyms", []),
            aliases=data.get("aliases", []),
            description=data.get("description"),
            parent_code=data.get("parent_code"),
            chapter=data.get("chapter"),
            include_terms=data.get("include_terms", []),
            exclude_terms=data.get("exclude_terms", []),
            normalized_text=data.get("normalized_text"),
        )


# All 38 codes found in the competition data
COMPETITION_ICD10_KB: list[dict] = [
    # --- Digestive (K) ---
    {
        "code": "K21.9",
        "name_en": "Gastro-oesophageal reflux disease without oesophagitis",
        "name_vi": "Bệnh trào ngược dạ dày thực quản không có viêm thực quản",
        "parent_code": "K21",
        "chapter": "XI",
        "synonyms": ["GERD", "GORD", "acid reflux disease", "reflux disease"],
        "aliases": ["trào ngược dạ dày", "trào ngược", "GERD", "bệnh trào ngược"],
        "include_terms": ["ợ nóng", "ợ chua", "acid reflux", "gastro-oesophageal reflux"],
    },
    {
        "code": "K25.9",
        "name_en": "Gastric ulcer, unspecified, without haemorrhage or perforation",
        "name_vi": "Loét dạ dày không xác định không chảy máu hoặc thủng",
        "parent_code": "K25",
        "chapter": "XI",
        "synonyms": ["gastric ulcer", "stomach ulcer"],
        "aliases": ["loét dạ dày", "loét bao tử", "dạ dày"],
        "include_terms": ["đau dạ dày", "loét dạ dày"],
    },
    {
        "code": "K26.9",
        "name_en": "Duodenal ulcer, unspecified, without haemorrhage or perforation",
        "name_vi": "Loét tá tràng không xác định không chảy máu hoặc thủng",
        "parent_code": "K26",
        "chapter": "XI",
        "synonyms": ["duodenal ulcer", "peptic ulcer duodenum"],
        "aliases": ["loét tá tràng", "loét tá tràng không biến chứng"],
        "include_terms": ["đau tá tràng", "loét tá tràng"],
    },
    {
        "code": "K29.9",
        "name_en": "Gastritis and duodenitis, unspecified",
        "name_vi": "Viêm dạ dày và viêm tá tràng không xác định",
        "parent_code": "K29",
        "chapter": "XI",
        "synonyms": ["gastritis", "duodenitis", "gastric inflammation"],
        "aliases": ["viêm dạ dày", "viêm dạ dày tá tràng", "đau dạ dày"],
        "include_terms": ["đau dạ dày", "nóng rát dạ dày", "viêm dạ dày"],
    },
    {
        "code": "K58.9",
        "name_en": "Irritable bowel syndrome, unspecified",
        "name_vi": "Hội chứng ruột kích thích không xác định",
        "parent_code": "K58",
        "chapter": "XI",
        "synonyms": ["IBS", "irritable bowel syndrome", "spastic colon"],
        "aliases": ["hội chứng ruột kích thích", "ruột kích thích"],
        "include_terms": ["đau bụng", "rối loạn tiêu hóa", "đi cầu không đều"],
    },
    {
        "code": "K70.30",
        "name_en": "Alcoholic cirrhosis of liver without ascites",
        "name_vi": "Xơ gan do rượu không có cổ trướng",
        "parent_code": "K70",
        "chapter": "XI",
        "synonyms": ["alcoholic cirrhosis", "alcoholic liver cirrhosis"],
        "aliases": ["xơ gan do rượu", "xơ gan", "xơ gan nghiện rượu"],
        "include_terms": ["gan", "xơ gan", "cirrhosis"],
    },
    {
        "code": "A09",
        "name_en": "Infectious gastroenteritis and colitis, unspecified",
        "name_vi": "Viêm dạ dày ruột nhiễm trùng và viêm đại tràng không xác định",
        "parent_code": "A09",
        "chapter": "I",
        "synonyms": ["infectious gastroenteritis", "gastroenteritis", "food poisoning"],
        "aliases": ["viêm dạ dày ruột", "tiêu chảy nhiễm trùng", "ngộ độc thức ăn"],
        "include_terms": ["nôn", "tiêu chảy", "đau bụng", "sốt"],
    },
    # --- Circulatory (I) ---
    {
        "code": "I10",
        "name_en": "Essential (primary) hypertension",
        "name_vi": "Tăng huyết áp nguyên phát",
        "parent_code": "I10",
        "chapter": "IX",
        "synonyms": ["high blood pressure", "HTN", "hypertension", "elevated blood pressure"],
        "aliases": ["tăng huyết áp", "huyết áp cao", "cao huyết áp", "huyết áp"],
        "include_terms": ["huyết áp cao", "tăng huyết áp"],
    },
    {
        "code": "I21.9",
        "name_en": "Acute myocardial infarction, unspecified",
        "name_vi": "Nhồi máu cơ tim cấp không xác định",
        "parent_code": "I21",
        "chapter": "IX",
        "synonyms": ["myocardial infarction", "MI", "heart attack", "acute MI"],
        "aliases": ["nhồi máu cơ tim", "nhồi máu", "đau tim", "tai biến mạch vành"],
        "include_terms": ["đau ngực", "khó thở", "đổ mồ hôi", "buồn nôn"],
    },
    {
        "code": "I25.10",
        "name_en": "Atherosclerotic heart disease of native coronary artery without angina pectoris",
        "name_vi": "Bệnh tim do xơ vữa động mạch vành tự nhiên không có đau thắt ngực",
        "parent_code": "I25",
        "chapter": "IX",
        "synonyms": ["coronary artery disease", "CAD", "ischemic heart disease", "atherosclerotic heart disease"],
        "aliases": ["bệnh mạch vành", "xơ vữa động mạch vành", "bệnh tim thiếu máu cơ tim"],
        "include_terms": ["đau ngực", "mệt", "khó thở khi gắng sức"],
    },
    {
        "code": "I48.9",
        "name_en": "Atrial fibrillation and atrial flutter, unspecified",
        "name_vi": "Rung nhĩ và cuồng động nhĩ không xác định",
        "parent_code": "I48",
        "chapter": "IX",
        "synonyms": ["atrial fibrillation", "AF", "AFib", "irregular heartbeat"],
        "aliases": ["rung nhĩ", "nhịp tim không đều", "cuồng động nhĩ", "rung tâm nhĩ"],
        "include_terms": ["nhịp tim nhanh", "tim đập không đều", "đánh trống ngực"],
    },
    {
        "code": "I50.9",
        "name_en": "Heart failure, unspecified",
        "name_vi": "Suy tim không xác định",
        "parent_code": "I50",
        "chapter": "IX",
        "synonyms": ["heart failure", "cardiac failure", "congestive heart failure"],
        "aliases": ["suy tim", "suy tim sung huyết", "suy tim cấp"],
        "include_terms": ["khó thở", "phù chân", "mệt", "gan to"],
    },
    {
        "code": "I64",
        "name_en": "Stroke, not specified as haemorrhage or infarction",
        "name_vi": "Đột quỵ không xác định là xuất huyết hay nhồi máu",
        "parent_code": "I64",
        "chapter": "IX",
        "synonyms": ["stroke", "cerebrovascular accident", "CVA", "brain attack"],
        "aliases": ["đột quỵ", "tai biến mạch não", "tai biến mạch máu não", "đột quỵ não"],
        "include_terms": ["liệt", "nói khó", "mất ý thức", "yếu nửa người"],
    },
    # --- Respiratory (J) ---
    {
        "code": "J06.9",
        "name_en": "Acute upper respiratory infection, unspecified",
        "name_vi": "Nhiễm trùng hô hấp trên cấp tính không xác định",
        "parent_code": "J06",
        "chapter": "X",
        "synonyms": ["URI", "upper respiratory infection", "common cold", "acute upper respiratory infection"],
        "aliases": ["nhiễm trùng hô hấp", "cảm cúm", "viêm mũi họng", "nhiễm trùng hô hấp trên"],
        "include_terms": ["ho", "sổ mũi", "đau họng", "hắt hơi", "nghẹt mũi"],
    },
    {
        "code": "J18.9",
        "name_en": "Pneumonia, unspecified",
        "name_vi": "Viêm phổi không xác định",
        "parent_code": "J18",
        "chapter": "X",
        "synonyms": ["pneumonia", "lung infection", "community acquired pneumonia"],
        "aliases": ["viêm phổi", "viêm phổi cộng đồng", "pneumonia"],
        "include_terms": ["sốt", "ho", "khó thở", "đau ngực", "đờm"],
    },
    {
        "code": "J20.9",
        "name_en": "Acute bronchitis, unspecified",
        "name_vi": "Viêm phế quản cấp tính không xác định",
        "parent_code": "J20",
        "chapter": "X",
        "synonyms": ["acute bronchitis", "bronchitis", "acute bronchiolitis"],
        "aliases": ["viêm phế quản cấp", "viêm phế quản", "phế quản"],
        "include_terms": ["ho", "khó thở", "sốt", "đờm", "thở khò khè"],
    },
    {
        "code": "J30.4",
        "name_en": "Allergic rhinitis, unspecified",
        "name_vi": "Viêm mũi dị ứng không xác định",
        "parent_code": "J30",
        "chapter": "X",
        "synonyms": ["allergic rhinitis", "hay fever", "pollen allergy", "seasonal allergies"],
        "aliases": ["viêm mũi dị ứng", "sốt mùa", "dị ứng mũi", "viêm mũi do dị ứng"],
        "include_terms": ["hắt hơi", "nghẹt mũi", "chảy nước mũi", "ngứa mũi"],
    },
    {
        "code": "J44.9",
        "name_en": "Chronic obstructive pulmonary disease, unspecified",
        "name_vi": "Bệnh phổi tắc nghẽn mạn tính không xác định",
        "parent_code": "J44",
        "chapter": "X",
        "synonyms": ["COPD", "chronic obstructive pulmonary disease", "chronic bronchitis", "emphysema"],
        "aliases": ["bệnh phổi tắc nghẽn mạn tính", "COPD", "phổi tắc nghẽn mạn"],
        "include_terms": ["ho", "khó thở", "thở khò khè", "đờm"],
    },
    {
        "code": "J45.9",
        "name_en": "Other and unspecified asthma",
        "name_vi": "Hen phế quản không xác định",
        "parent_code": "J45",
        "chapter": "X",
        "synonyms": ["asthma", "bronchial asthma", "asthmatic bronchitis"],
        "aliases": ["hen suyễn", "hen", "viêm phế quản dị ứng", "bệnh hen"],
        "include_terms": ["thở khò khè", "khó thở", "ho", "co thắt phế quản"],
    },
    # --- Endocrine (E) ---
    {
        "code": "E03.9",
        "name_en": "Hypothyroidism, unspecified",
        "name_vi": "Suy giáp không xác định",
        "parent_code": "E03",
        "chapter": "IV",
        "synonyms": ["hypothyroidism", "underactive thyroid", "thyroid insufficiency"],
        "aliases": ["suy giáp", "giảm chức năng tuyến giáp", "nhiễm độc giáp", "tuyến giáp kém"],
        "include_terms": ["mệt", "tăng cân", "lạnh", "táo bón", "da khô"],
    },
    {
        "code": "E05.90",
        "name_en": "Thyrotoxicosis, unspecified without thyrotoxic crisis or storm",
        "name_vi": "Nhiễm độc giáp không xác định không có khủng hoảng nhiễm độc giáp",
        "parent_code": "E05",
        "chapter": "IV",
        "synonyms": ["hyperthyroidism", "thyrotoxicosis", "overactive thyroid", "Graves disease"],
        "aliases": ["nhiễm độc giáp", "cường giáp", "basedow", " Graves"],
        "include_terms": ["tim đập nhanh", "sụt cân", "run tay", "đổ mồ hôi"],
    },
    {
        "code": "E11.9",
        "name_en": "Type 2 diabetes mellitus without complications",
        "name_vi": "Đái tháo đường type 2 không biến chứng",
        "parent_code": "E11",
        "chapter": "IV",
        "synonyms": ["diabetes type 2", "T2DM", "type 2 diabetes", "NIDDM"],
        "aliases": ["đái tháo đường", "tiểu đường", "bệnh đường", "tiểu đường type 2"],
        "include_terms": ["đường huyết cao", "uống nhiều nước", "tiểu nhiều"],
    },
    {
        "code": "E66.9",
        "name_en": "Obesity, unspecified",
        "name_vi": "Béo phì không xác định",
        "parent_code": "E66",
        "chapter": "IV",
        "synonyms": ["obesity", "overweight", "adiposity"],
        "aliases": ["béo phì", "thừa cân", "béo", "overweight"],
        "include_terms": ["BMI cao", "tích mỡ"],
    },
    # --- Mental (F) ---
    {
        "code": "F17.210",
        "name_en": "Nicotine dependence, cigarettes, uncomplicated",
        "name_vi": "Phụ thuộc nicotin, thuốc lá, không biến chứng",
        "parent_code": "F17",
        "chapter": "V",
        "synonyms": ["nicotine dependence", "tobacco dependence", "smoking addiction", "cigarette dependence"],
        "aliases": ["nghiện thuốc lá", "nghiện thuốc", "hút thuốc", "phụ thuộc nicotin"],
        "include_terms": ["thuốc lá", "hút thuốc", "nicotine"],
    },
    {
        "code": "F32.9",
        "name_en": "Major depressive disorder, single episode, unspecified",
        "name_vi": "Rối loạn trầm cảm nặng, giai đoạn đơn, không xác định",
        "parent_code": "F32",
        "chapter": "V",
        "synonyms": ["depression", "major depression", "depressive disorder", "clinical depression"],
        "aliases": ["trầm cảm", "rối loạn trầm cảm", "mất tinh thần", "u sầm"],
        "include_terms": ["buồn", "mất hứng thú", "rối loạn giấc ngủ", "mệt"],
    },
    {
        "code": "F41.1",
        "name_en": "Generalized anxiety disorder",
        "name_vi": "Rối loạn lo âu tổng quát",
        "parent_code": "F41",
        "chapter": "V",
        "synonyms": ["generalized anxiety disorder", "GAD", "anxiety neurosis", "anxiety disorder"],
        "aliases": ["rối loạn lo âu", "lo âu", "rối loạn lo âu tổng quát", "GAD"],
        "include_terms": ["lo âu", "bồn chồn", "căng thẳng", "khó ngủ", "mệt"],
    },
    # --- Neurological (G) ---
    {
        "code": "B18.1",
        "name_en": "Chronic viral hepatitis B without delta-agent",
        "name_vi": "Viêm gan virus B mạn tính không có delta agent",
        "parent_code": "B18",
        "chapter": "I",
        "synonyms": ["chronic hepatitis B", "HBV", "hep B"],
        "aliases": ["viêm gan B mạn", "viêm gan B", "viêm gan virus B", "gan nhiễm mỡ B"],
        "include_terms": ["gan", "viêm gan", "vàng da", "mệt"],
    },
    {
        "code": "B18.2",
        "name_en": "Chronic viral hepatitis C",
        "name_vi": "Viêm gan virus C mạn tính",
        "parent_code": "B18",
        "chapter": "I",
        "synonyms": ["chronic hepatitis C", "HCV", "hep C"],
        "aliases": ["viêm gan C mạn", "viêm gan C", "viêm gan virus C"],
        "include_terms": ["gan", "viêm gan", "mệt"],
    },
    {
        "code": "B34.9",
        "name_en": "Viral infection, unspecified",
        "name_vi": "Nhiễm virus không xác định",
        "parent_code": "B34",
        "chapter": "I",
        "synonyms": ["viral infection", "virus infection", "viraemia"],
        "aliases": ["nhiễm virus", "nhiễm trùng virus", "cảm virus"],
        "include_terms": ["sốt", "mệt", "đau cơ"],
    },
    {
        "code": "G40.909",
        "name_en": "Epilepsy, unspecified, not intractable, without status epilepticus",
        "name_vi": "Động kinh không xác định, không khó trị, không có trạng thái động kinh",
        "parent_code": "G40",
        "chapter": "VI",
        "synonyms": ["epilepsy", "seizure disorder", "convulsive disorder", "epileptic"],
        "aliases": ["động kinh", "co giật", "bệnh động kinh", "suy giáp"],
        "include_terms": ["co giật", "ngã", "mất ý thức", "cứng người"],
    },
    {
        "code": "G43.909",
        "name_en": "Migraine, unspecified, not intractable, without status migrainosus",
        "name_vi": "Đau nửa đầu không xác định, không khó trị, không có trạng thái đau nửa đầu",
        "parent_code": "G43",
        "chapter": "VI",
        "synonyms": ["migraine", "migraine headache", "hemicrania", "sick headache"],
        "aliases": ["đau nửa đầu", "migraine", "đau đầu một bên", "đau đầu migraine"],
        "include_terms": ["đau đầu", "buồn nôn", "nhạy cảm ánh sáng", "hoa mắt"],
    },
    {
        "code": "G47.00",
        "name_en": "Insomnia, unspecified",
        "name_vi": "Mất ngủ không xác định",
        "parent_code": "G47",
        "chapter": "VI",
        "synonyms": ["insomnia", "sleep disorder", "sleeplessness", "difficulty sleeping"],
        "aliases": ["mất ngủ", "khó ngủ", "rối loạn giấc ngủ", "ngủ không sâu"],
        "include_terms": ["khó ngủ", "thức giấc", "ngủ ít", "thiếu ngủ"],
    },
    # --- Musculoskeletal (M) ---
    {
        "code": "M06.9",
        "name_en": "Rheumatoid arthritis, unspecified",
        "name_vi": "Viêm khớp dạng thấp không xác định",
        "parent_code": "M06",
        "chapter": "XIII",
        "synonyms": ["rheumatoid arthritis", "RA", "chronic polyarthritis"],
        "aliases": ["viêm khớp dạng thấp", "viêm khớp", "thấp khớp", "dạng thấp"],
        "include_terms": ["đau khớp", "sưng khớp", "cứng khớp", "biến dạng khớp"],
    },
    {
        "code": "M54.5",
        "name_en": "Low back pain",
        "name_vi": "Đau thắt lưng",
        "parent_code": "M54",
        "chapter": "XIII",
        "synonyms": ["low back pain", "lumbago", "lumbar pain", "lower back pain"],
        "aliases": ["đau thắt lưng", "đau lưng", "đau thắt lưng dưới", "thoái hóa cột sống"],
        "include_terms": ["đau lưng", "cứng lưng", "đau thắt lưng"],
    },
    {
        "code": "M79.3",
        "name_en": "Panniculitis, unspecified",
        "name_vi": "Viêm mô mỡ không xác định",
        "parent_code": "M79",
        "chapter": "XIII",
        "synonyms": ["panniculitis", "lipoedema", "inflammation of subcutaneous fat"],
        "aliases": ["viêm mô mỡ", "panniculitis", "sưng mô mỡ"],
        "include_terms": ["sưng", "đau", "mô mỡ"],
    },
    # --- Genitourinary (N) ---
    {
        "code": "N18.3",
        "name_en": "Chronic kidney disease, stage 3",
        "name_vi": "Bệnh thận mạn tính giai đoạn 3",
        "parent_code": "N18",
        "chapter": "XIV",
        "synonyms": ["chronic kidney disease stage 3", "CKD stage 3", "moderate chronic renal failure"],
        "aliases": ["bệnh thận mạn giai đoạn 3", "suy thận mạn giai đoạn 3", "bệnh thận mạn"],
        "include_terms": ["thận", "suy thận", "creatinine cao", "đạm máu"],
    },
    {
        "code": "N20.0",
        "name_en": "Calculus of kidney",
        "name_vi": "Sỏi thận",
        "parent_code": "N20",
        "chapter": "XIV",
        "synonyms": ["kidney stone", "nephrolithiasis", "renal calculus", "kidney calculus"],
        "aliases": ["sỏi thận", "sỏi đường tiết niệu", "sỏi niệu quản", "sỏi bàng quang"],
        "include_terms": ["đau thắt lưng", "đau bụng dưới", "tiểu máu", "buồn nôn"],
    },
    {
        "code": "N39.0",
        "name_en": "Urinary tract infection, site not specified",
        "name_vi": "Nhiễm trùng đường tiết niệu không xác định vị trí",
        "parent_code": "N39",
        "chapter": "XIV",
        "synonyms": ["urinary tract infection", "UTI", "cystitis", "bladder infection"],
        "aliases": ["nhiễm trùng tiết niệu", "nhiễm trùng đường tiết niệu", "viêm bàng quang", "viêm đường tiết niệu"],
        "include_terms": ["tiểu rắt", "tiểu buốt", "đau khi tiểu", "nước tiểu đục"],
    },
]


def get_knowledge_base() -> list[ICD10Entry]:
    """Return all ICD-10 entries for the competition."""
    return [ICD10Entry.from_dict(d) for d in COMPETITION_ICD10_KB]
