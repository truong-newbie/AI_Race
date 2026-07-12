"""
Run pipeline on validation data and export predictions.
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import MedicalOntologyPipeline

def main():
    input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
    output_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\medical-ontology\predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        extract_labs=True,
        extract_drugs=True,
        extract_diseases=True,
        extract_symptoms=True,
        detect_assertions=True,
        link_icd=True,
        link_rxnorm=True,
        resolve_overlaps=True,
        overlap_strategy="hybrid",
        deterministic=True,
    )
    pipeline = MedicalOntologyPipeline(cfg)

    # Collect all input files (1.txt through 100.txt)
    all_predictions = {}
    stats = {
        "total_files": 0,
        "total_entities": 0,
        "by_type": {},
        "by_assertion": {},
        "files_with_entities": 0,
        "total_errors": 0,
        "empty_files": 0,
    }

    for i in range(1, 101):
        fname = f"{i}.txt"
        in_path = input_dir / fname
        if not in_path.exists():
            print(f"WARNING: {in_path} not found, skipping")
            continue

        text = in_path.read_text(encoding="utf-8").strip()
        result = pipeline.process(text)
        predictions = result.to_dict()

        all_predictions[fname] = predictions

        # Write individual file prediction
        out_path = output_dir / fname.replace(".txt", ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        stats["total_files"] += 1
        stats["total_entities"] += len(predictions)

        if predictions:
            stats["files_with_entities"] += 1
        else:
            stats["empty_files"] += 1

        for entity in predictions:
            etype = entity["type"]
            stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1
            for assertion in entity["assertions"]:
                stats["by_assertion"][assertion] = stats["by_assertion"].get(assertion, 0) + 1

        stats["total_errors"] += len(result.errors)

        if i % 10 == 0:
            print(f"Processed {i}/100 files...")

    # Write combined predictions
    combined_path = output_dir / "predictions.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    # Write stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n=== VALIDATION RUN COMPLETE ===")
    print(f"Files processed: {stats['total_files']}/100")
    print(f"Files with entities: {stats['files_with_entities']}")
    print(f"Empty files: {stats['empty_files']}")
    print(f"Total entities extracted: {stats['total_entities']}")
    print(f"Total errors: {stats['total_errors']}")
    print(f"\nBy entity type:")
    for etype, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    print(f"\nBy assertion:")
    for assertion, count in sorted(stats["by_assertion"].items(), key=lambda x: -x[1]):
        print(f"  {assertion}: {count}")
    print(f"\nPredictions exported to: {output_dir}")
    print(f"  - Individual files: {output_dir}/*.json")
    print(f"  - Combined: {combined_path}")
    print(f"  - Stats: {stats_path}")

if __name__ == "__main__":
    main()
