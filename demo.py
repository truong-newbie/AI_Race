"""
Demo script - Medical Ontology Pipeline

Shows how to use the pipeline to extract medical entities from Vietnamese text.
"""

import json
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.pipeline import MedicalOntologyPipeline, extract_medical_entities


def demo_simple():
    """Simple extraction demo."""
    print("=" * 60)
    print("Medical Ontology Pipeline - Simple Demo")
    print("=" * 60)
    print()

    text = """
    Bệnh nhân nam, 55 tuổi, nhập viện vì ho đờm xanh, sốt cao.
    Tiền sử tăng huyết áp, đái tháo đường type 2.
    Khám: phổi có ran ẩm 2 bên.
    Xét nghiệm: WBC 12.5 G/L, CRP 85 mg/L.
    Chẩn đoán: viêm phổi cộng đồng.
    Điều trị: Ceftriaxone 1g, Paracetamol 500mg khi sốt.
    """

    print("Input text:")
    print(text.strip())
    print()

    # Extract entities
    entities = extract_medical_entities(text)

    print(f"Extracted {len(entities)} entities:")
    print("-" * 60)

    # Group by type
    by_type = {}
    for e in entities:
        t = e["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    for entity_type, items in by_type.items():
        print(f"\n{entity_type}:")
        for item in items:
            assertions = ", ".join(item["assertions"]) if item["assertions"] else "none"
            candidates = ", ".join(item["candidates"]) if item["candidates"] else "none"
            print(f"  • '{item['text']}' @ {item['position']}")
            print(f"    assertions: {assertions}")
            print(f"    candidates: {candidates}")

    print()
    return entities


def demo_json_output():
    """Show JSON output format."""
    print()
    print("=" * 60)
    print("JSON Output Format")
    print("=" * 60)
    print()

    text = "Bệnh nhân ho đờm xanh. Dùng Paracetamol 500mg."
    entities = extract_medical_entities(text)

    print(json.dumps(entities, ensure_ascii=False, indent=2))


def demo_individual_extraction():
    """Show individual extractor outputs."""
    print()
    print("=" * 60)
    print("Individual Extractor Demos")
    print("=" * 60)
    print()

    # Lab extractor
    from src.entity.lab_extractor import LabTestExtractor
    lab_ext = LabTestExtractor()

    text1 = "WBC 12.5 G/L, CRP 85 mg/L, Glucose 126 mg/dL"
    tests, results = lab_ext.extract_all(text1)
    print("Lab Tests:")
    print(f"  Tests found: {[(t.text, t.start, t.end) for t in tests]}")
    print(f"  Results found: {[(r.text, r.start, r.end) for r in results]}")
    print()

    # Drug extractor
    from src.entity.drug_extractor import DrugExtractor
    drug_ext = DrugExtractor()

    text2 = "Ceftriaxone 1g, Paracetamol 500mg, Aspirin 81mg"
    drugs = drug_ext.extract(text2)
    print("Drugs:")
    for d in drugs:
        print(f"  • '{d.text}' [{d.start}:{d.end}] - {d.ingredient} {d.strength}")
    print()

    # Disease extractor
    from src.entity.disease_extractor import DiseaseExtractor
    disease_ext = DiseaseExtractor()

    text3 = "Chẩn đoán: viêm phổi cộng đồng. Tiền sử tăng huyết áp."
    diseases = disease_ext.extract(text3)
    print("Diseases:")
    for d in diseases:
        print(f"  • '{d.text}' [{d.start}:{d.end}] - {d.context}")
    print()

    # Assertion detector
    from src.assertion.rules import AssertionDetector
    detector = AssertionDetector()

    text4 = "Bệnh nhân ho. Tiền sử hen suyễn. Bố bệnh nhân bị đái tháo đường."
    print("Assertions:")
    # Check specific positions
    test_cases = [
        ("ho", 12, 14, "no assertion"),
        ("hen suyễn", 24, 33, "isHistorical"),
        ("đái tháo đường", 52, 66, "isFamily"),
    ]
    for desc, start, end, expected in test_cases:
        result = detector.detect(text4, start, end)
        assertions = result.to_list()
        print(f"  '{desc}': {assertions} (expected: {expected})")


def demo_full_pipeline():
    """Use the full pipeline with all options."""
    print()
    print("=" * 60)
    print("Full Pipeline with Configuration")
    print("=" * 60)
    print()

    from src.pipeline import PipelineConfig, MedicalOntologyPipeline

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

    text = """
    Bệnh nhân nữ, 45 tuổi, vào viện vì đau thượng vị, ợ chua 2 tháng nay.
    Tiền sử: lo âu, mất ngủ.
    Khám: bụng mềm, đau epigastric.
    Nội soi: viêm dạ dày mức độ trung bình.
    Sinh thiết: không phát hiện ung thư.
    Chẩn đoán: viêm dạ dày, trào ngược dạ dày thực quản.
    Điều trị: Omeprazole 20mg x 2 lần/ngày, Paracetamol 500mg khi đau.
    Tái khám sau 2 tuần.
    """

    print("Processing full medical note...")
    result = pipeline.process(text)

    print(f"\nFound {len(result.entities)} entities")
    print()

    if result.validation_result:
        print(f"Validation: {result.validation_result.summary()}")

    # Show entities grouped by type
    by_type = {}
    for e in result.entities:
        t = e.type.value
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e)

    for entity_type, items in by_type.items():
        print(f"\n{entity_type}:")
        for item in items:
            assertions = [a.value for a in item.assertions]
            assertions_str = ", ".join(assertions) if assertions else "-"
            candidates_str = ", ".join(item.candidates) if item.candidates else "-"
            print(f"  • '{item.text}' [{item.position[0]}:{item.position[1]}]")
            print(f"    assertions: {assertions_str}")
            print(f"    candidates: {candidates_str}")


if __name__ == "__main__":
    demo_simple()
    demo_json_output()
    demo_individual_extraction()
    demo_full_pipeline()
