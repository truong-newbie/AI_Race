"""Diagnostic for assertion detection."""
import json
from pathlib import Path

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
pred_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\medical-ontology\predictions")

# Read file 1
text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()
preds = json.loads((pred_dir / "1.json").read_text(encoding="utf-8"))

# Count isHistorical
hist = [p for p in preds if "isHistorical" in p["assertions"]]
drugs_hist = [p for p in hist if p["type"] == "THUỐC"]
symptoms_hist = [p for p in hist if p["type"] == "TRIỆU_CHỨNG"]
diagnoses_hist = [p for p in hist if p["type"] == "CHẨN_ĐOÁN"]

output = {
    "total_historical": len(hist),
    "drugs_historical": len(drugs_hist),
    "symptoms_historical": len(symptoms_hist),
    "diagnoses_historical": len(diagnoses_hist),
    "total_entities": len(preds),
}

# Find atenolol
for p in preds:
    if "atenolol" in p["text"].lower():
        output["atenolol"] = p

# Find metoprolol
for p in preds:
    if "metoprolol" in p["text"].lower():
        output["metoprolol"] = p

# Print drug entities
output["drug_entities"] = [p for p in preds if p["type"] == "THUỐC"]

with open("diagnostic.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Diagnostic saved. Results:")
for k, v in output.items():
    if k not in ("atenolol", "metoprolol", "drug_entities"):
        print(f"  {k}: {v}")
