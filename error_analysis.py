"""
Error analysis for validation predictions.
Categorizes common error patterns.
"""
import json
from collections import defaultdict
from pathlib import Path

def analyze_predictions():
    pred_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\medical-ontology\predictions")
    input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
    report_path = pred_dir / "error_report.txt"

    errors = defaultdict(list)
    total = 0

    import io, codecs
    _out = io.StringIO()

    def log(msg=""):
        _out.write(msg + "\n")
    fp_symptom = []  # False positive: "bệnh nhân", fragments
    fp_lab_number = []  # False positive: numbers as lab results
    fp_lab_vitals = []  # False positive: vital signs as single entity
    truncated_entities = []  # Truncated spans
    wrong_type = []  # Wrong entity type
    overmarked_historical = []  # isHistorical on current treatments
    noisy_entities = []  # Garbage/incomplete entities

    for i in range(1, 101):
        fname = f"{i}.txt"
        json_name = f"{i}.json"

        text = input_dir / fname
        pred_file = pred_dir / json_name

        if not pred_file.exists():
            continue

        text_content = text.read_text(encoding="utf-8").strip()
        predictions = json.loads(pred_file.read_text(encoding="utf-8"))

        for entity in predictions:
            total += 1
            text_str = entity["text"]
            etype = entity["type"]
            pos = entity["position"]

            # Check 1: Number-only entities as lab results
            stripped = text_str.strip()
            if etype in ("KẾT_QUẢ_XÉT_NGHIỆM", "TÊN_XÉT_NGHIỆM"):
                if stripped.isdigit():
                    fp_lab_number.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                elif stripped.replace(".", "").isdigit():
                    fp_lab_number.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                elif " " not in stripped and len(stripped) < 5:
                    fp_lab_number.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                elif len(stripped.split()) > 3:
                    fp_lab_vitals.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })

            # Check 2: Known false-positive symptom words
            if etype == "TRIỆU_CHỨNG":
                # "bệnh nhân" should NOT be a symptom
                if text_str.strip() in ("bệnh nhân", "Bệnh nhân", "bệnh nhân\n"):
                    fp_symptom.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-20):pos[1]+20]
                    })
                # Fragment entities (too short, weird cuts)
                elif len(text_str.strip()) <= 4 and "bệnh" in text_str:
                    fp_symptom.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                # Check for truncated entities (ending mid-word)
                elif text_str.endswith(" ") or text_str.endswith("n"):
                    truncated_entities.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })

            # Check 3: Truncated drug entities (missing full name)
            if etype == "THUỐC":
                if text_str.startswith("Thuốc trước"):
                    truncated_entities.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                # Drug names that are fragments
                if text_str in ("oxy",) or text_str.startswith("oxy"):
                    truncated_entities.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })

            # Check 4: isHistorical on drugs in current treatment section
            if etype == "THUỐC" and "isHistorical" in entity["assertions"]:
                # Get context to check if it's a current drug
                context = text_content[max(0,pos[0]-30):pos[1]+30]
                if "kê" in context or "uống" in context or "tiêm" in context:
                    overmarked_historical.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "assertions": entity["assertions"],
                        "context": context
                    })

            # Check 5: Noisy entities (weird fragments)
            if etype == "TRIỆU_CHỨNG":
                if "bệnh viện" in text_str.strip() or "bệnh có" in text_str.strip() or "bệnh hiện" in text_str.strip():
                    fp_symptom.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })
                if len(text_str.strip()) <= 3 and not text_str.strip().isalpha():
                    noisy_entities.append({
                        "file": fname,
                        "text": repr(text_str),
                        "position": pos,
                        "context": text_content[max(0,pos[0]-10):pos[1]+10]
                    })

    # Deduplicate
    def dedup(lst):
        seen = set()
        result = []
        for item in lst:
            key = (item["file"], item["text"], tuple(item["position"]))
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    fp_symptom = dedup(fp_symptom)
    fp_lab_number = dedup(fp_lab_number)
    fp_lab_vitals = dedup(fp_lab_vitals)
    truncated_entities = dedup(truncated_entities)
    overmarked_historical = dedup(overmarked_historical)
    noisy_entities = dedup(noisy_entities)

    log("=" * 80)
    log("ERROR ANALYSIS REPORT")
    log("=" * 80)
    log(f"\nTotal entities extracted: {total}")
    log(f"\n--- CATEGORY 1: FALSE POSITIVE SYMPTOMS ({len(fp_symptom)} unique) ---")
    for item in fp_symptom[:20]:
        log(f"  [{item['file']}] {item['text']} at {item['position']}")
        log(f"    Context: ...{item['context']}...")

    log(f"\n--- CATEGORY 2: FALSE POSITIVE NUMBERS AS LAB RESULTS ({len(fp_lab_number)} unique) ---")
    for item in fp_lab_number[:20]:
        log(f"  [{item['file']}] {item['text']} at {item['position']}")
        log(f"    Context: ...{item['context']}...")

    log(f"\n--- CATEGORY 3: VITAL SIGNS CLUSTERED AS ONE ENTITY ({len(fp_lab_vitals)} unique) ---")
    for item in fp_lab_vitals[:10]:
        log(f"  [{item['file']}] {item['text']} at {item['position']}")

    log(f"\n--- CATEGORY 4: TRUNCATED ENTITIES ({len(truncated_entities)} unique) ---")
    for item in truncated_entities[:15]:
        log(f"  [{item['file']}] {item['text']} at {item['position']}")
        log(f"    Context: ...{item['context']}...")

    log(f"\n--- CATEGORY 5: isHistorical ON CURRENT TREATMENTS ({len(overmarked_historical)} unique) ---")
    for item in overmarked_historical[:10]:
        log(f"  [{item['file']}] {item['text']} assertions={item['assertions']}")
        log(f"    Context: ...{item['context']}...")

    log(f"\n--- CATEGORY 6: NOISY/INCOMPLETE ENTITIES ({len(noisy_entities)} unique) ---")
    for item in noisy_entities[:15]:
        log(f"  [{item['file']}] {item['text']} at {item['position']}")
        log(f"    Context: ...{item['context']}...")

    total_errors = len(fp_symptom) + len(fp_lab_number) + len(fp_lab_vitals) + len(truncated_entities) + len(overmarked_historical) + len(noisy_entities)
    log(f"\n{'=' * 80}")
    log(f"SUMMARY:")
    log(f"  False positive symptoms:       {len(fp_symptom)}")
    log(f"  False positive lab numbers:     {len(fp_lab_number)}")
    log(f"  Vital signs clusters:           {len(fp_lab_vitals)}")
    log(f"  Truncated entities:             {len(truncated_entities)}")
    log(f"  Over-marked historical:         {len(overmarked_historical)}")
    log(f"  Noisy/incomplete entities:      {len(noisy_entities)}")
    log(f"  TOTAL identified issues:        {total_errors}")
    log(f"  Total entities:                 {total}")
    log(f"  Error rate (lower bound):       {total_errors}/{total} = {total_errors/total*100:.1f}%")

    # Save detailed analysis
    analysis = {
        "total_entities": total,
        "categories": {
            "false_positive_symptoms": fp_symptom,
            "false_positive_lab_numbers": fp_lab_number,
            "vital_signs_clusters": fp_lab_vitals,
            "truncated_entities": truncated_entities,
            "overmarked_historical": overmarked_historical,
            "noisy_entities": noisy_entities,
        }
    }
    out_path = pred_dir / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    report_text = _out.getvalue()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log(f"\nReport saved to: {report_path}")
    log(f"Detailed analysis saved to: {out_path}")

if __name__ == "__main__":
    analyze_predictions()
