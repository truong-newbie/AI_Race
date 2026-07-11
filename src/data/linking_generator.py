"""
Linking Sample Generator for ICD-10 and RxNorm

Tạo samples cho entity linking:
- ICD-10 linking: positive + hard negatives
- RxNorm linking: positive + hard negatives
"""

import random
from typing import List, Dict, Any, Optional, Tuple
import random
from dataclasses import dataclass, field

from .schema import ICDLinkingSample, RxNormLinkingSample


# =============================================================================
# ICD-10 Linking Data
# =============================================================================

# Extended ICD-10 knowledge base with hard negatives
ICD10_ENTRIES = {
    # Respiratory (J00-J99)
    "J18.9": {
        "name": "Pneumonia, unspecified",
        "name_vi": "viêm phổi",
        "synonyms": ["viêm phổi cộng đồng", "viêm phổi mắc phải", "pneumonia"],
        "category": "respiratory",
        "related_codes": ["J12.9", "J13", "J14", "J15.9", "J16.8", "J17.8", "J20.9"],
    },
    "J20.9": {
        "name": "Acute bronchitis, unspecified",
        "name_vi": "viêm phế quản cấp",
        "synonyms": ["viêm phế quản", "viêm phế quản cấp", "bronchitis"],
        "category": "respiratory",
        "related_codes": ["J18.9", "J21.9", "J40", "J40-J47"],
    },
    "J45.9": {
        "name": "Asthma, unspecified",
        "name_vi": "hen suyễn",
        "synonyms": ["hen phế quản", "hen", "asthma"],
        "category": "respiratory",
        "related_codes": ["J45.0", "J45.1", "J45.2", "J45.3", "J45.4", "J45.5", "J45.8", "J45.9"],
    },
    "J44.9": {
        "name": "COPD, unspecified",
        "name_vi": "bệnh phổi tắc nghẽn mạn tính",
        "synonyms": ["bệnh phổi tắc nghẽn", "copd", "hen phổi mạn"],
        "category": "respiratory",
        "related_codes": ["J44.0", "J44.1", "J44.8", "J43.9", "J40"],
    },
    "J30.4": {
        "name": "Allergic rhinitis, unspecified",
        "name_vi": "viêm mũi dị ứng",
        "synonyms": ["viêm mũi", "dị ứng mũi", "viêm mũi dị ứng"],
        "category": "respiratory",
        "related_codes": ["J30.0", "J30.1", "J30.2", "J30.3", "J30.4", "J31.0"],
    },
    "J06.9": {
        "name": "Acute upper respiratory infection, unspecified",
        "name_vi": "nhiễm trùng hô hấp trên",
        "synonyms": ["nhiễm trùng hô hấp", "cảm cúm", "viêm mũi họng"],
        "category": "respiratory",
        "related_codes": ["J00", "J01.9", "J02.9", "J03.90", "J04.9", "J05.9", "J06.0"],
    },

    # Cardiovascular (I00-I99)
    "I10": {
        "name": "Essential (primary) hypertension",
        "name_vi": "tăng huyết áp",
        "synonyms": ["tăng huyết áp nguyên phát", "huyết áp cao", "hypertension"],
        "category": "cardiovascular",
        "related_codes": ["I11.9", "I12.9", "I13.9", "I15.0", "I15.1", "I15.2", "I15.8", "I15.9"],
    },
    "I50.9": {
        "name": "Heart failure, unspecified",
        "name_vi": "suy tim",
        "synonyms": ["suy tim", "suy tim mạn", "heart failure"],
        "category": "cardiovascular",
        "related_codes": ["I50.0", "I50.1", "I50.2", "I50.3", "I50.4", "I50.8", "I50.9"],
    },
    "I21.9": {
        "name": "Acute myocardial infarction, unspecified",
        "name_vi": "nhồi máu cơ tim",
        "synonyms": ["nhồi máu cơ tim cấp", "nmct", "heart attack"],
        "category": "cardiovascular",
        "related_codes": ["I21.0", "I21.1", "I21.2", "I21.3", "I21.4", "I21.9", "I22.9"],
    },
    "I64": {
        "name": "Stroke, not specified as haemorrhage or infarction",
        "name_vi": "đột quỵ",
        "synonyms": ["tai biến mạch máu não", "đột quỵ não", "stroke"],
        "category": "cardiovascular",
        "related_codes": ["I60.9", "I61.9", "I63.9", "I64", "I65.9", "I66.9", "I67.8"],
    },
    "I48.9": {
        "name": "Atrial fibrillation, unspecified",
        "name_vi": "rung nhĩ",
        "synonyms": ["rung nhĩ", "nghịch thường nhĩ", "fibrillation"],
        "category": "cardiovascular",
        "related_codes": ["I48.0", "I48.1", "I48.2", "I48.3", "I48.4", "I48.9"],
    },
    "I25.10": {
        "name": "Atherosclerotic heart disease of native coronary artery",
        "name_vi": "bệnh tim thiếu máu cơ tim",
        "synonyms": ["bệnh tim thiếu máu", "bệnh mạch vành", "coronary artery disease"],
        "category": "cardiovascular",
        "related_codes": ["I25.0", "I25.1", "I25.10", "I25.11", "I25.2", "I25.3"],
    },

    # Endocrine (E00-E90)
    "E11.9": {
        "name": "Type 2 diabetes mellitus without complications",
        "name_vi": "đái tháo đường type 2",
        "synonyms": ["đái tháo đường", "tiểu đường", "diabetes", "đtđ type 2"],
        "category": "endocrine",
        "related_codes": ["E10.9", "E11.0", "E11.1", "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8", "E11.9"],
    },
    "E03.9": {
        "name": "Hypothyroidism, unspecified",
        "name_vi": "suy giáp",
        "synonyms": ["suy giáp", "giảm chức năng tuyến giáp", "hypothyroidism"],
        "category": "endocrine",
        "related_codes": ["E03.0", "E03.1", "E03.2", "E03.3", "E03.4", "E03.5", "E03.8", "E03.9"],
    },
    "E05.90": {
        "name": "Thyrotoxicosis, unspecified without thyrotoxic crisis",
        "name_vi": "cường giáp",
        "synonyms": ["cường giáp", "nhiễm độc giáp", "hyperthyroidism"],
        "category": "endocrine",
        "related_codes": ["E05.0", "E05.1", "E05.2", "E05.3", "E05.4", "E05.5", "E05.8", "E05.90", "E05.91"],
    },
    "E66.9": {
        "name": "Obesity, unspecified",
        "name_vi": "béo phì",
        "synonyms": ["béo phì", "thừa cân", "obesity"],
        "category": "endocrine",
        "related_codes": ["E66.0", "E66.1", "E66.2", "E66.8", "E66.9", "Z68.30"],
    },

    # Gastrointestinal (K00-K93)
    "K21.9": {
        "name": "Gastro-oesophageal reflux disease without oesophagitis",
        "name_vi": "trào ngược dạ dày thực quản",
        "synonyms": ["trào ngược", "gerd", "trào ngược dạ dày", "reflux"],
        "category": "gastrointestinal",
        "related_codes": ["K21.0", "K21.9", "K22.7", "K30"],
    },
    "K29.9": {
        "name": "Gastritis, unspecified",
        "name_vi": "viêm dạ dày",
        "synonyms": ["viêm dạ dày", "viêm niêm mạc dạ dày", "gastritis"],
        "category": "gastrointestinal",
        "related_codes": ["K29.0", "K29.1", "K29.2", "K29.3", "K29.4", "K29.5", "K29.6", "K29.7", "K29.8", "K29.9"],
    },
    "K25.9": {
        "name": "Gastric ulcer, unspecified, without haemorrhage or perforation",
        "name_vi": "loét dạ dày",
        "synonyms": ["loét dạ dày", "u loét dạ dày", "gastric ulcer"],
        "category": "gastrointestinal",
        "related_codes": ["K25.0", "K25.1", "K25.2", "K25.3", "K25.4", "K25.5", "K25.6", "K25.7", "K25.9"],
    },
    "K26.9": {
        "name": "Duodenal ulcer, unspecified, without haemorrhage or perforation",
        "name_vi": "loét tá tràng",
        "synonyms": ["loét tá tràng", "u loét tá tràng", "duodenal ulcer"],
        "category": "gastrointestinal",
        "related_codes": ["K26.0", "K26.1", "K26.2", "K26.3", "K26.4", "K26.5", "K26.6", "K26.7", "K26.9"],
    },
    "K58.9": {
        "name": "Irritable bowel syndrome, unspecified",
        "name_vi": "hội chứng ruột kích thích",
        "synonyms": ["hội chứng ruột kích thích", "ibs", "đại tràng co thắt"],
        "category": "gastrointestinal",
        "related_codes": ["K58.0", "K58.1", "K58.2", "K58.3", "K58.8", "K58.9"],
    },
    "K70.30": {
        "name": "Cirrhosis of liver without alcohol",
        "name_vi": "xơ gan",
        "synonyms": ["xơ gan", "xơ gan mạn", "cirrhosis"],
        "category": "gastrointestinal",
        "related_codes": ["K70.0", "K70.1", "K70.2", "K70.3", "K70.9", "K74.0", "K74.1", "K74.2", "K74.6"],
    },

    # Musculoskeletal (M00-M99)
    "M06.9": {
        "name": "Rheumatoid arthritis, unspecified",
        "name_vi": "viêm khớp dạng thấp",
        "synonyms": ["viêm khớp dạng thấp", "vklpt", "rheumatoid arthritis"],
        "category": "musculoskeletal",
        "related_codes": ["M05.9", "M06.0", "M06.1", "M06.2", "M06.3", "M06.8", "M06.9"],
    },
    "M54.5": {
        "name": "Low back pain",
        "name_vi": "đau lưng",
        "synonyms": ["đau lưng", "đau thắt lưng", "low back pain"],
        "category": "musculoskeletal",
        "related_codes": ["M54.0", "M54.1", "M54.2", "M54.3", "M54.4", "M54.5", "M54.6", "M54.8", "M54.9"],
    },
    "M79.3": {
        "name": "Panniculitis, unspecified",
        "name_vi": "viêm mô mỡ",
        "synonyms": ["viêm mô mỡ", "panniculitis"],
        "category": "musculoskeletal",
        "related_codes": ["M79.0", "M79.1", "M79.2", "M79.3", "M79.4", "M79.5", "M79.6"],
    },

    # Genitourinary (N00-N99)
    "N18.3": {
        "name": "Chronic kidney disease, stage 3",
        "name_vi": "suy thận mạn giai đoạn 3",
        "synonyms": ["suy thận mạn", "bệnh thận mạn", "ckd", "chronic kidney disease"],
        "category": "genitourinary",
        "related_codes": ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
    },
    "N20.0": {
        "name": "Calculus of kidney",
        "name_vi": "sỏi thận",
        "synonyms": ["sỏi thận", "sỏi bàng quang", "kidney stone"],
        "category": "genitourinary",
        "related_codes": ["N20.0", "N20.1", "N20.2", "N20.9", "N21.0", "N21.1", "N21.8", "N21.9"],
    },
    "N39.0": {
        "name": "Urinary tract infection, site not specified",
        "name_vi": "nhiễm trùng tiết niệu",
        "synonyms": ["nhiễm trùng tiết niệu", "nttn", "viêm bàng quang"],
        "category": "genitourinary",
        "related_codes": ["N30.0", "N30.1", "N30.2", "N30.3", "N30.8", "N30.9", "N39.0"],
    },

    # Infectious (A00-B99)
    "B18.1": {
        "name": "Chronic viral hepatitis B without delta-agent",
        "name_vi": "viêm gan B mạn tính",
        "synonyms": ["viêm gan B", "viêm gan siêu vi B", "hepatitis B"],
        "category": "infectious",
        "related_codes": ["B16.9", "B17.0", "B17.1", "B18.0", "B18.1", "B18.2", "B18.8", "B18.9"],
    },
    "B18.2": {
        "name": "Chronic viral hepatitis C",
        "name_vi": "viêm gan C mạn tính",
        "synonyms": ["viêm gan C", "viêm gan siêu vi C", "hepatitis C"],
        "category": "infectious",
        "related_codes": ["B17.1", "B18.2"],
    },
    "A09": {
        "name": "Infectious gastroenteritis and colitis, unspecified",
        "name_vi": "viêm dạ dày ruột nhiễm trùng",
        "synonyms": ["viêm dạ dày ruột", "ngộ độc thức ăn", "gastroenteritis"],
        "category": "infectious",
        "related_codes": ["A00.9", "A01.0", "A02.0", "A03.9", "A04.9", "A05.9", "A08.4", "A09"],
    },
    "B34.9": {
        "name": "Viral infection, unspecified",
        "name_vi": "nhiễm virus không xác định",
        "synonyms": ["nhiễm virus", "nhiễm siêu vi", "viral infection"],
        "category": "infectious",
        "related_codes": ["B34.0", "B34.1", "B34.2", "B34.3", "B34.4", "B34.8", "B34.9"],
    },

    # Neurological (G00-G99)
    "G43.909": {
        "name": "Migraine, unspecified, not intractable",
        "name_vi": "đau nửa đầu",
        "synonyms": ["đau nửa đầu", "migraine", "chứng nhức đầu"],
        "category": "neurological",
        "related_codes": ["G43.0", "G43.1", "G43.2", "G43.3", "G43.8", "G43.9", "G43.909"],
    },
    "G47.00": {
        "name": "Insomnia, unspecified",
        "name_vi": "mất ngủ",
        "synonyms": ["mất ngủ", "khó ngủ", "insomnia"],
        "category": "neurological",
        "related_codes": ["G47.00", "G47.01", "G47.02", "G47.09", "G47.3", "G47.4"],
    },
    "G40.909": {
        "name": "Epilepsy, unspecified",
        "name_vi": "động kinh",
        "synonyms": ["động kinh", "co giật", "epilepsy"],
        "category": "neurological",
        "related_codes": ["G40.0", "G40.1", "G40.2", "G40.3", "G40.4", "G40.5", "G40.8", "G40.9", "G40.909"],
    },

    # Mental/Behavioral (F00-F99)
    "F32.9": {
        "name": "Major depressive disorder, single episode, unspecified",
        "name_vi": "trầm cảm",
        "synonyms": ["trầm cảm", "rối loạn lo âu", "depression"],
        "category": "mental",
        "related_codes": ["F20.9", "F31.9", "F32.0", "F32.1", "F32.2", "F32.3", "F32.8", "F32.9"],
    },
    "F41.1": {
        "name": "Generalized anxiety disorder",
        "name_vi": "rối loạn lo âu tổng quát",
        "synonyms": ["rối loạn lo âu", "lo âu", "anxiety"],
        "category": "mental",
        "related_codes": ["F40.0", "F40.1", "F40.2", "F40.8", "F41.0", "F41.1", "F41.2", "F41.3", "F41.8", "F41.9"],
    },
    "F17.210": {
        "name": "Tobacco use disorder, uncomplicated",
        "name_vi": "nghiện thuốc lá",
        "synonyms": ["nghiện thuốc lá", "hút thuốc", "tobacco"],
        "category": "mental",
        "related_codes": ["F17.200", "F17.201", "F17.210", "F17.211", "F17.220", "F17.221", "F17.290", "F17.291"],
    },
}


# =============================================================================
# RxNorm Linking Data
# =============================================================================

RXNORM_ENTRIES = {
    # Analgesics
    "161": {
        "name": "Acetaminophen 500 MG Oral Tablet",
        "name_short": "Paracetamol 500mg",
        "generic_name": "Acetaminophen",
        "brand_names": ["Panadol", "Efferalgan", "Tylenol"],
        "strength": "500mg",
        "route": "oral",
        "category": "analgesic",
        "related_rxcuis": ["1191", "161", "198211", "198240"],
    },
    "1191": {
        "name": "Aspirin 81 MG Oral Tablet",
        "name_short": "Aspirin 81mg",
        "generic_name": "Aspirin",
        "brand_names": ["Bayer", "Bufferin"],
        "strength": "81mg",
        "route": "oral",
        "category": "analgesic",
        "related_rxcuis": ["1191", "1192", "161", "212448"],
    },
    "1192": {
        "name": "Aspirin 325 MG Oral Tablet",
        "name_short": "Aspirin 325mg",
        "generic_name": "Aspirin",
        "brand_names": ["Bayer"],
        "strength": "325mg",
        "route": "oral",
        "category": "analgesic",
        "related_rxcuis": ["1191", "1192", "212448"],
    },

    # Antibiotics
    "8628": {
        "name": "Ceftriaxone 1 GM Injection",
        "name_short": "Ceftriaxone 1g",
        "generic_name": "Ceftriaxone",
        "brand_names": ["Rocephin"],
        "strength": "1g",
        "route": "injection",
        "category": "antibiotic",
        "related_rxcuis": ["8628", "864563", "864564", "864565"],
    },
    "723": {
        "name": "Amoxicillin 500 MG Oral Capsule",
        "name_short": "Amoxicillin 500mg",
        "generic_name": "Amoxicillin",
        "brand_names": ["Amoxil", "Moxatag"],
        "strength": "500mg",
        "route": "oral",
        "category": "antibiotic",
        "related_rxcuis": ["723", "308136", "308182", "308183"],
    },
    "2556": {
        "name": "Ciprofloxacin 500 MG Oral Tablet",
        "name_short": "Ciprofloxacin 500mg",
        "generic_name": "Ciprofloxacin",
        "brand_names": ["Cipro"],
        "strength": "500mg",
        "route": "oral",
        "category": "antibiotic",
        "related_rxcuis": ["2556", "311354", "311989"],
    },
    "18631": {
        "name": "Azithromycin 500 MG Oral Tablet",
        "name_short": "Azithromycin 500mg",
        "generic_name": "Azithromycin",
        "brand_names": ["Zithromax", "Azithrocin"],
        "strength": "500mg",
        "route": "oral",
        "category": "antibiotic",
        "related_rxcuis": ["18631", "308182", "616382", "1049221"],
    },
    "83367": {
        "name": "Cefuroxime 500 MG Oral Tablet",
        "name_short": "Cefuroxime 500mg",
        "generic_name": "Cefuroxime",
        "brand_names": ["Ceftin", "Zinnat"],
        "strength": "500mg",
        "route": "oral",
        "category": "antibiotic",
        "related_rxcuis": ["83367", "83368", "86333"],
    },
    "1116635": {
        "name": "Metronidazole 500 MG Oral Tablet",
        "name_short": "Metronidazole 500mg",
        "generic_name": "Metronidazole",
        "brand_names": ["Flagyl"],
        "strength": "500mg",
        "route": "oral",
        "category": "antibiotic",
        "related_rxcuis": ["1116635", "313820", "313821"],
    },

    # Antidiabetics
    "6809": {
        "name": "Metformin 500 MG Oral Tablet",
        "name_short": "Metformin 500mg",
        "generic_name": "Metformin",
        "brand_names": ["Glucophage", "Fortamet"],
        "strength": "500mg",
        "route": "oral",
        "category": "antidiabetic",
        "related_rxcuis": ["6809", "860975", "860983", "861007"],
    },
    "860975": {
        "name": "Metformin 850 MG Oral Tablet",
        "name_short": "Metformin 850mg",
        "generic_name": "Metformin",
        "brand_names": ["Glucophage"],
        "strength": "850mg",
        "route": "oral",
        "category": "antidiabetic",
        "related_rxcuis": ["6809", "860975", "861007"],
    },
    "316672": {
        "name": "Glibenclamide 5 MG Oral Tablet",
        "name_short": "Glibenclamide 5mg",
        "generic_name": "Glibenclamide",
        "brand_names": ["Glyburide", "DiaBeta"],
        "strength": "5mg",
        "route": "oral",
        "category": "antidiabetic",
        "related_rxcuis": ["316672", "310965", "310798"],
    },
    "861007": {
        "name": "Metformin 1000 MG Oral Tablet",
        "name_short": "Metformin 1000mg",
        "generic_name": "Metformin",
        "brand_names": ["Glucophage"],
        "strength": "1000mg",
        "route": "oral",
        "category": "antidiabetic",
        "related_rxcuis": ["6809", "860975", "861007"],
    },
    "317541": {
        "name": "Sitagliptin 100 MG Oral Tablet",
        "name_short": "Sitagliptin 100mg",
        "generic_name": "Sitagliptin",
        "brand_names": ["Januvia"],
        "strength": "100mg",
        "route": "oral",
        "category": "antidiabetic",
        "related_rxcuis": ["317541", "317542", "317543", "317544"],
    },

    # Cardiovascular
    "7646": {
        "name": "Omeprazole 20 MG Delayed Release Oral Tablet",
        "name_short": "Omeprazole 20mg",
        "generic_name": "Omeprazole",
        "brand_names": ["Prilosec", "Losec"],
        "strength": "20mg",
        "route": "oral",
        "category": "gastrointestinal",
        "related_rxcuis": ["7646", "197361", "197517", "197518"],
    },
    "32937": {
        "name": "Amlodipine 5 MG Oral Tablet",
        "name_short": "Amlodipine 5mg",
        "generic_name": "Amlodipine",
        "brand_names": ["Norvasc"],
        "strength": "5mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["32937", "17767", "197380", "197381"],
    },
    "52175": {
        "name": "Losartan 50 MG Oral Tablet",
        "name_short": "Losartan 50mg",
        "generic_name": "Losartan",
        "brand_names": ["Cozaar"],
        "strength": "50mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["52175", "314768", "314769", "314770"],
    },
    "617312": {
        "name": "Atorvastatin 20 MG Oral Tablet",
        "name_short": "Atorvastatin 20mg",
        "generic_name": "Atorvastatin",
        "brand_names": ["Lipitor"],
        "strength": "20mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["617312", "617334", "617369", "83367"],
    },
    "197361": {
        "name": "Pantoprazole 40 MG Delayed Release Oral Tablet",
        "name_short": "Pantoprazole 40mg",
        "generic_name": "Pantoprazole",
        "brand_names": ["Protonix"],
        "strength": "40mg",
        "route": "oral",
        "category": "gastrointestinal",
        "related_rxcuis": ["197361", "7646", "40790"],
    },
    "314076": {
        "name": "Bisoprolol 5 MG Oral Tablet",
        "name_short": "Bisoprolol 5mg",
        "generic_name": "Bisoprolol",
        "brand_names": ["Zebeta"],
        "strength": "5mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["314076", "314077", "314078"],
    },
    "198211": {
        "name": "Spironolactone 25 MG Oral Tablet",
        "name_short": "Spironolactone 25mg",
        "generic_name": "Spironolactone",
        "brand_names": ["Aldactone"],
        "strength": "25mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["198211", "198212", "198213"],
    },
    "855332": {
        "name": "Clopidogrel 75 MG Oral Tablet",
        "name_short": "Clopidogrel 75mg",
        "generic_name": "Clopidogrel",
        "brand_names": ["Plavix"],
        "strength": "75mg",
        "route": "oral",
        "category": "cardiovascular",
        "related_rxcuis": ["855332", "308136", "308182"],
    },

    # Steroids/Anti-inflammatory
    "8640": {
        "name": "Prednisolone 5 MG Oral Tablet",
        "name_short": "Prednisolone 5mg",
        "generic_name": "Prednisolone",
        "brand_names": ["Prelone", "Orapred"],
        "strength": "5mg",
        "route": "oral",
        "category": "steroid",
        "related_rxcuis": ["8640", "1160162", "1160193"],
    },
    "285070": {
        "name": "Dexamethasone 4 MG Oral Tablet",
        "name_short": "Dexamethasone 4mg",
        "generic_name": "Dexamethasone",
        "brand_names": ["Decadron"],
        "strength": "4mg",
        "route": "oral",
        "category": "steroid",
        "related_rxcuis": ["285070", "285071", "285072"],
    },

    # Psychiatric/Neurological
    "72407": {
        "name": "Sertraline 50 MG Oral Tablet",
        "name_short": "Sertraline 50mg",
        "generic_name": "Sertraline",
        "brand_names": ["Zoloft"],
        "strength": "50mg",
        "route": "oral",
        "category": "psychiatric",
        "related_rxcuis": ["72407", "72408", "312961"],
    },
    "72509": {
        "name": "Alprazolam 0.5 MG Oral Tablet",
        "name_short": "Alprazolam 0.5mg",
        "generic_name": "Alprazolam",
        "brand_names": ["Xanax"],
        "strength": "0.5mg",
        "route": "oral",
        "category": "psychiatric",
        "related_rxcuis": ["72509", "72510", "72511"],
    },
    "312961": {
        "name": "Diazepam 5 MG Oral Tablet",
        "name_short": "Diazepam 5mg",
        "generic_name": "Diazepam",
        "brand_names": ["Valium"],
        "strength": "5mg",
        "route": "oral",
        "category": "psychiatric",
        "related_rxcuis": ["312961", "58031", "58032", "58033"],
    },
    "206977": {
        "name": "Zopiclone 7.5 MG Oral Tablet",
        "name_short": "Zopiclone 7.5mg",
        "generic_name": "Zopiclone",
        "brand_names": ["Imovane"],
        "strength": "7.5mg",
        "route": "oral",
        "category": "psychiatric",
        "related_rxcuis": ["206977", "197370"],
    },
}


# =============================================================================
# Query Contexts
# =============================================================================

ICD10_QUERY_CONTEXTS = [
    "Chẩn đoán: {mention}",
    "Bệnh nhân bị {mention}",
    "Phát hiện {mention}",
    "Tiền sử {mention}",
    "Có {mention}",
    "Bị {mention}",
    "Mắc {mention}",
    "{mention} được ghi nhận",
    "Xác định {mention}",
    "BN được chẩn đoán {mention}",
]

RXNORM_QUERY_CONTEXTS = [
    "Điều trị: {mention}",
    "Kê đơn {mention}",
    "Dùng {mention}",
    "Uống {mention}",
    "Tiêm {mention}",
    "Sử dụng {mention}",
    "BN được prescrible {mention}",
    "{mention} được chỉ định",
    "Dùng thuốc {mention}",
    "Uống thuốc {mention}",
]


# =============================================================================
# Linking Sample Generator
# =============================================================================

class LinkingGenerator:
    """
    Generator để tạo linking samples cho ICD-10 và RxNorm.

    Usage:
        generator = LinkingGenerator(seed=42)
        icd_samples = generator.generate_icd10_samples(100)
        rx_samples = generator.generate_rxnorm_samples(100)
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.icd_counter = 0
        self.rx_counter = 0

    def _next_icd_id(self) -> str:
        self.icd_counter += 1
        return f"icd_{self.icd_counter:05d}"

    def _next_rx_id(self) -> str:
        self.rx_counter += 1
        return f"rx_{self.rx_counter:05d}"

    def _get_hard_negatives_icd10(self, positive_code: str, count: int = 4) -> List[str]:
        """
        Get hard negative codes for ICD-10.

        Hard negatives are codes from the same category or with similar names
        but not the correct code.
        """
        if positive_code not in ICD10_ENTRIES:
            return []

        entry = ICD10_ENTRIES[positive_code]
        negatives = []

        # Add related codes (same category, different condition)
        for related in entry.get("related_codes", []):
            if related != positive_code and related in ICD10_ENTRIES:
                negatives.append(related)

        # Add codes from same category
        category = entry.get("category", "")
        for code, info in ICD10_ENTRIES.items():
            if code != positive_code and info.get("category") == category:
                if code not in negatives:
                    negatives.append(code)

        # Shuffle and limit
        random.shuffle(negatives)
        return negatives[:count]

    def _get_hard_negatives_rxnorm(self, positive_rxcui: str, count: int = 4) -> List[Tuple[str, str]]:
        """
        Get hard negative entries for RxNorm.

        Returns list of (rxcui, name) tuples.
        """
        if positive_rxcui not in RXNORM_ENTRIES:
            return []

        entry = RXNORM_ENTRIES[positive_rxcui]
        negatives = []

        # Add related rxcuis (same category, different drug)
        for related in entry.get("related_rxcuis", []):
            if related != positive_rxcui and related in RXNORM_ENTRIES:
                neg_entry = RXNORM_ENTRIES[related]
                negatives.append((related, neg_entry["name_short"]))

        # Add drugs from same category
        category = entry.get("category", "")
        for rxcui, info in RXNORM_ENTRIES.items():
            if rxcui != positive_rxcui and info.get("category") == category:
                if (rxcui, info["name_short"]) not in negatives:
                    negatives.append((rxcui, info["name_short"]))

        # Shuffle and limit
        random.shuffle(negatives)
        return negatives[:count]

    def generate_icd10_samples(self, count: int = 100) -> List[ICDLinkingSample]:
        """
        Generate ICD-10 linking samples.

        Args:
            count: Number of samples to generate

        Returns:
            List of ICDLinkingSample
        """
        samples = []
        codes = list(ICD10_ENTRIES.keys())

        for _ in range(count):
            # Pick random code
            code = random.choice(codes)
            entry = ICD10_ENTRIES[code]

            # Pick mention text (either Vietnamese or English)
            if random.random() < 0.7:  # 70% Vietnamese
                mention = entry["name_vi"]
            else:  # 30% English synonyms
                mention = random.choice(entry.get("synonyms", [entry["name"]]))

            # Pick query context
            query_template = random.choice(ICD10_QUERY_CONTEXTS)
            query_text = query_template.format(mention=mention)

            # Get hard negatives
            negative_codes = self._get_hard_negatives_icd10(code, count=4)

            samples.append(ICDLinkingSample(
                id=self._next_icd_id(),
                query_text=query_text,
                mention=mention,
                positive_code=code,
                negative_codes=negative_codes,
                source="synthetic_hard_negative",
            ))

        return samples

    def generate_rxnorm_samples(self, count: int = 100) -> List[RxNormLinkingSample]:
        """
        Generate RxNorm linking samples.

        Args:
            count: Number of samples to generate

        Returns:
            List of RxNormLinkingSample
        """
        samples = []
        rxcuis = list(RXNORM_ENTRIES.keys())

        for _ in range(count):
            # Pick random drug
            rxcui = random.choice(rxcuis)
            entry = RXNORM_ENTRIES[rxcui]

            # Pick mention text (either short name or generic name)
            if random.random() < 0.8:  # 80% short name
                mention = entry["name_short"]
            else:  # 20% generic name
                mention = entry["generic_name"]

            # Pick query context
            query_template = random.choice(RXNORM_QUERY_CONTEXTS)
            query_text = query_template.format(mention=mention)

            # Get hard negatives
            negative_list = self._get_hard_negatives_rxnorm(rxcui, count=4)
            negative_rxcuis = [n[0] for n in negative_list]
            negative_names = [n[1] for n in negative_list]

            samples.append(RxNormLinkingSample(
                id=self._next_rx_id(),
                query_text=query_text,
                mention=mention,
                positive_rxcui=rxcui,
                positive_name=entry["name_short"],
                negative_rxcuis=negative_rxcuis,
                negative_names=negative_names,
                source="synthetic_hard_negative",
            ))

        return samples

    def generate_all(self, icd_count: int = 100, rx_count: int = 100) -> Tuple[List[ICDLinkingSample], List[RxNormLinkingSample]]:
        """
        Generate all linking samples.

        Args:
            icd_count: Number of ICD-10 samples
            rx_count: Number of RxNorm samples

        Returns:
            Tuple of (icd_samples, rx_samples)
        """
        return (
            self.generate_icd10_samples(icd_count),
            self.generate_rxnorm_samples(rx_count),
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for linking generator."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate linking samples for ICD-10 and RxNorm")
    parser.add_argument("--icd-count", type=int, default=100, help="Number of ICD-10 samples")
    parser.add_argument("--rx-count", type=int, default=100, help="Number of RxNorm samples")
    parser.add_argument("--icd-output", type=str, default="data/synthetic/icd_linking_samples.jsonl", help="ICD-10 output file")
    parser.add_argument("--rx-output", type=str, default="data/synthetic/rxnorm_linking_samples.jsonl", help="RxNorm output file")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generator = LinkingGenerator(seed=args.seed)
    icd_samples, rx_samples = generator.generate_all(args.icd_count, args.rx_count)

    # Save ICD-10 samples
    with open(args.icd_output, "w", encoding="utf-8") as f:
        for sample in icd_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    # Save RxNorm samples
    with open(args.rx_output, "w", encoding="utf-8") as f:
        for sample in rx_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    print(f"Generated {len(icd_samples)} ICD-10 linking samples")
    print(f"Generated {len(rx_samples)} RxNorm linking samples")
    print(f"Saved to {args.icd_output} and {args.rx_output}")


if __name__ == "__main__":
    main()
