"""Direct test of assertion detection on file 1."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import MedicalOntologyPipeline

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
output_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\medical-ontology\predictions")

text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()

cfg = PipelineConfig(
    extract_labs=True, extract_drugs=True, extract_diseases=True,
    extract_symptoms=True, detect_assertions=True, link_icd=True,
    link_rxnorm=True, resolve_overlaps=True, overlap_strategy="hybrid",
    deterministic=True,
)
pipeline = MedicalOntologyPipeline(cfg)
result = pipeline.process(text)
predictions = result.to_dict()

# Write fresh prediction
out_path = output_dir / "1.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

# Analyze
hist = [p for p in predictions if "isHistorical" in p["assertions"]]
drugs = [p for p in predictions if p["type"] == "THUỐC"]
drugs_hist = [p for p in hist if p["type"] == "THUỐC"]

output = {
    "total_entities": len(predictions),
    "total_historical": len(hist),
    "drugs_total": len(drugs),
    "drugs_historical": len(drugs_hist),
    "drug_entities": [{"text": d["text"], "assertions": d["assertions"]} for d in drugs],
}

with open("direct_test.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Done")
