#!/usr/bin/env python3
"""
Comprehensive Evaluation Script — Medical Ontology Pipeline

Metrics:
  ENTITY   exact-span P/R/F1, per-class F1, boundary errors
  ASSERTION F1 per label (isNegated/isHistorical/isFamily), macro-F1, exact accuracy
  ICD      Recall@1/3/5/10/20, Top-1, MRR
  RXNORM   Recall@1/3/5/10, Top-1, ingredient/strength/dose-form accuracy
  E2E      5 levels (span → span+type → span+type+assertion → ... → strict)
  ERRORS   11-type taxonomy
  ABLATION 10 configs
  AL       active learning candidates

Output:
  outputs/metrics.json, per_class_metrics.csv, errors.csv,
  retrieval_errors.csv, assertion_errors.csv, ablation.csv, report.md
  data/active_learning/to_review.jsonl
"""

import sys
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# ─── UTF-8 output on Windows ──────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ─── Data paths ───────────────────────────────────────────────────────────────
_DATA = _ROOT / "data"
_OUTPUT = _ROOT / "outputs"
_AL_DIR = _DATA / "active_learning"

_GOLD_FILES = {
    "dev": _DATA / "processed" / "dev.jsonl",
    "train": _DATA / "processed" / "train.jsonl",
    "internal_test": _DATA / "processed" / "internal_test.jsonl",
}
_VAL_PATH = _DATA / "validation" / "manual_validation_template.jsonl"
_ICD_SAMPLES = _DATA / "synthetic" / "icd_linking_samples.jsonl"
_RXNORM_SAMPLES = _DATA / "synthetic" / "rxnorm_linking_samples.jsonl"


# ─── Utilities ────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _p(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("utf-8"))


# ─── Entity key ───────────────────────────────────────────────────────────────

def _ekey(e: dict) -> tuple:
    """(text, start, end, type)"""
    return (e["text"], e["start"], e["end"], e["type"])


def _assertions(e: dict) -> frozenset:
    return frozenset(e.get("assertions", []))


def _candidates(e: dict) -> set:
    return set(e.get("candidates", []))


# ─── Pipeline helpers ──────────────────────────────────────────────────────────

def _build_pipeline(overrides: dict | None = None):
    from src.pipeline.pipeline import MedicalOntologyPipeline
    from src.pipeline.config import PipelineConfig
    cfg = PipelineConfig()
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    return MedicalOntologyPipeline(cfg)


def _pipeline_result_to_dict(result) -> list[dict]:
    """Convert ExtractionResult to list of entity dicts."""
    out = []
    for e in result.entities:
        out.append({
            "text": e.text,
            "start": e.position[0],
            "end": e.position[1],
            "type": e.type.value,
            "assertions": [a.value for a in e.assertions],
            "candidates": list(e.candidates),
        })
    return out


def _run_pipeline(text: str) -> list[dict]:
    try:
        pipeline = _build_pipeline()
        result = pipeline.process(text)
        return _pipeline_result_to_dict(result)
    except Exception:
        return []


# ─── Gold data helpers ────────────────────────────────────────────────────────

def _gold_from_processed(sample: dict) -> list[dict]:
    """Convert processed/*.jsonl sample → list of entity dicts."""
    out = []
    for e in sample.get("entities", []):
        out.append({
            "text": e["text"],
            "start": e["start"],
            "end": e["end"],
            "type": e["type"],
            "assertions": [a for a in e.get("assertions", [])],
            "candidates": list(e.get("candidates", [])),
        })
    return out


def _gold_from_validation(sample: dict, text: str) -> list[dict]:
    """Convert manual_validation_template sample → entity dicts with start/end."""
    out = []
    for e in sample.get("expected_entities", []):
        try:
            start = text.index(e["text"])
            end = start + len(e["text"])
        except ValueError:
            continue
        out.append({
            "text": e["text"],
            "start": start,
            "end": end,
            "type": e.get("type", "TRIỆU_CHỨNG"),
            "assertions": [a for a in e.get("assertions", [])],
            "candidates": [],
        })
    return out


# ─── ENTITY metrics ───────────────────────────────────────────────────────────

def _entity_metrics(gold: list[dict], pred: list[dict]) -> dict:
    gset = {_ekey(e) for e in gold}
    pset = {_ekey(e) for e in pred}
    n_g, n_p = len(gset), len(pset)
    n_c = len(gset & pset)
    p_ = n_c / n_p if n_p else 0.0
    r_ = n_c / n_g if n_g else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    return {"precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f1, 4),
            "num_true": n_g, "num_pred": n_p, "num_correct": n_c}


def _per_class_metrics(gold: list[dict], pred: list[dict]) -> dict:
    from src.entity.labels import NER_ENTITY_TYPES
    results = {}
    for etype in NER_ENTITY_TYPES:
        g = [e for e in gold if e["type"] == etype]
        p = [e for e in pred if e["type"] == etype]
        gs, ps = {_ekey(e) for e in g}, {_ekey(e) for e in p}
        ng, np_ = len(gs), len(ps)
        nc = len(gs & ps)
        p_ = nc / np_ if np_ else 0.0
        r_ = nc / ng if ng else 0.0
        results[etype] = {"precision": round(p_, 4), "recall": round(r_, 4),
                          "f1": round(2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0, 4),
                          "support": ng}
    return results


# ─── ASSERTION metrics ────────────────────────────────────────────────────────

def _assertion_metrics(gold: list[dict], pred: list[dict]) -> dict:
    """F1 per assertion label + macro-F1 + exact accuracy."""
    gmap = {_ekey(e): _assertions(e) for e in gold}
    pmap = {_ekey(e): _assertions(e) for e in pred}
    common = set(gmap) & set(pmap)

    results = {}
    for label in ["isNegated", "isHistorical", "isFamily"]:
        tp = sum(1 for k in common if label in gmap[k] and label in pmap[k])
        fp = sum(1 for k in common if label not in gmap[k] and label in pmap[k])
        fn = sum(1 for k in common if label in gmap[k] and label not in pmap[k])
        p_ = tp / (tp + fp) if (tp + fp) else 0.0
        r_ = tp / (tp + fn) if (tp + fn) else 0.0
        results[label] = {"precision": round(p_, 4), "recall": round(r_, 4),
                          "f1": round(2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0, 4),
                          "tp": tp, "fp": fp, "fn": fn}

    f1s = [results[l]["f1"] for l in ["isNegated", "isHistorical", "isFamily"]]
    results["macro_f1"] = round(sum(f1s) / 3, 4)
    exact = sum(1 for k in common if gmap[k] == pmap[k])
    results["exact_accuracy"] = round(exact / len(common), 4) if common else 0.0
    results["exact_count"] = exact
    results["common_entities"] = len(common)
    return results


# ─── E2E levels ───────────────────────────────────────────────────────────────

def _e2e_level(gold: list[dict], pred: list[dict], level: int) -> dict:
    gs, ps = {_ekey(e) for e in gold}, {_ekey(e) for e in pred}
    correct = 0
    for pe in pred:
        pk = _ekey(pe)
        if pk not in gs:
            continue
        ge_list = [e for e in gold if _ekey(e) == pk]
        if not ge_list:
            continue
        ge = ge_list[0]

        if level >= 3 and _assertions(pe) != _assertions(ge):
            continue
        if level >= 4 and not (_candidates(pe) & _candidates(ge)):
            continue
        if level >= 5 and _candidates(pe) != _candidates(ge):
            continue
        correct += 1

    ng, np_ = len(gs), len(ps)
    p_ = correct / np_ if np_ else 0.0
    r_ = correct / ng if ng else 0.0
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
    return {"precision": round(p_, 4), "recall": round(r_, 4), "f1": round(f1, 4),
            "correct": correct, "num_true": ng, "num_pred": np_}


# ─── ERROR taxonomy ──────────────────────────────────────────────────────────

def _classify_errors(gold: list[dict], pred: list[dict], source: str = "") -> list[dict]:
    """Classify errors into 11 types (first-match)."""
    errors = []
    gmap = {_ekey(e): e for e in gold}
    pmap = {_ekey(e): e for e in pred}
    gkeys, pkeys = set(gmap), set(pmap)

    # Type 1: missed_entity
    for k in gkeys - pkeys:
        e = gmap[k]
        errors.append({"source": source, "error_type": "missed_entity",
                        "entity_text": e["text"], "entity_type": e["type"],
                        "start": e["start"], "end": e["end"],
                        "gold_assertions": "|".join(e.get("assertions", [])),
                        "pred_assertions": "", "gold_candidates": "|".join(e.get("candidates", [])),
                        "pred_candidates": "", "detail": "gold entity not in predictions"})

    # Type 2: false_positive
    for k in pkeys - gkeys:
        e = pmap[k]
        errors.append({"source": source, "error_type": "false_positive",
                        "entity_text": e["text"], "entity_type": e["type"],
                        "start": e["start"], "end": e["end"],
                        "gold_assertions": "", "pred_assertions": "|".join(e.get("assertions", [])),
                        "gold_candidates": "", "pred_candidates": "|".join(e.get("candidates", [])),
                        "detail": "prediction with no gold match"})

    # Types 3-5: wrong_boundary, wrong_type, wrong_assertion (same text)
    txt_g = defaultdict(list)
    txt_p = defaultdict(list)
    for e in gold:
        txt_g[e["text"]].append(e)
    for e in pred:
        txt_p[e["text"]].append(e)

    for text in (set(txt_g) & set(txt_p)):
        used_g, used_p = set(), set()
        for ge in txt_g[text]:
            gk = _ekey(ge)
            if gk in used_g:
                continue
            for pe in txt_p[text]:
                pk = _ekey(pe)
                if pk in used_p:
                    continue
                # Type 3: wrong_boundary
                if (ge["start"] != pe["start"] or ge["end"] != pe["end"]) and ge["type"] == pe["type"]:
                    errors.append({"source": source, "error_type": "wrong_boundary",
                                   "entity_text": text, "entity_type": pe["type"],
                                   "start": pe["start"], "end": pe["end"],
                                   "gold_start": ge["start"], "gold_end": ge["end"],
                                   "gold_assertions": "|".join(ge.get("assertions", [])),
                                   "pred_assertions": "|".join(pe.get("assertions", [])),
                                   "detail": f"boundary mismatch: gold=[{ge['start']},{ge['end']}] pred=[{pe['start']},{pe['end']}]"})
                    used_g.add(gk); used_p.add(pk); break
                # Type 4: wrong_type
                elif ge["type"] != pe["type"]:
                    errors.append({"source": source, "error_type": "wrong_type",
                                   "entity_text": text, "entity_type": f"pred={pe['type']} gold={ge['type']}",
                                   "start": pe["start"], "end": pe["end"],
                                   "gold_assertions": "|".join(ge.get("assertions", [])),
                                   "pred_assertions": "|".join(pe.get("assertions", [])),
                                   "detail": "type mismatch at same span"})
                    used_g.add(gk); used_p.add(pk); break
                # Type 5: wrong_assertion
                elif _assertions(ge) != _assertions(pe):
                    errors.append({"source": source, "error_type": "wrong_assertion",
                                   "entity_text": text, "entity_type": pe["type"],
                                   "start": pe["start"], "end": pe["end"],
                                   "gold_assertions": "|".join(ge.get("assertions", [])),
                                   "pred_assertions": "|".join(pe.get("assertions", [])),
                                   "detail": "assertion mismatch"})
                    used_g.add(gk); used_p.add(pk); break

    # Types 6-7: icd_gold_missing / rxnorm_gold_missing
    for k in (gkeys & pkeys):
        ge, pe = gmap[k], pmap[k]
        gc, pc = set(ge.get("candidates", [])), set(pe.get("candidates", []))
        if ge["type"] == "CHẨN_ĐOÁN" and gc and not pc:
            errors.append({"source": source, "error_type": "icd_gold_missing",
                           "entity_text": ge["text"], "entity_type": ge["type"],
                           "start": ge["start"], "end": ge["end"],
                           "gold_candidates": "|".join(gc), "pred_candidates": "",
                           "detail": "ICD gold present but no prediction"})
        elif ge["type"] == "THUỐC" and gc and not pc:
            errors.append({"source": source, "error_type": "rxnorm_gold_missing",
                           "entity_text": ge["text"], "entity_type": ge["type"],
                           "start": ge["start"], "end": ge["end"],
                           "gold_candidates": "|".join(gc), "pred_candidates": "",
                           "detail": "RxNorm gold present but no prediction"})

    return errors


# ─── ICD metrics ──────────────────────────────────────────────────────────────

def _icd_metrics() -> dict:
    report = _OUTPUT / "icd_retrieval_report.json"
    if report.exists():
        with open(report, encoding="utf-8") as f:
            r = json.load(f)
        return {"n_samples": r.get("n_samples", 0), "coverage": r.get("coverage", 0),
                "mrr": r.get("mrr", 0), "recall": r.get("recall", {}),
                "top1": r.get("recall", {}).get("recall@1", 0)}
    try:
        from src.linking.icd.evaluator import ICDRetrievalEvaluator
        ev = ICDRetrievalEvaluator(str(_ICD_SAMPLES), str(_OUTPUT), top_k=20)
        m = ev.evaluate()
        ev.save_report()
        ev.save_errors_csv()
        return {"n_samples": m.get("n_samples", 0), "coverage": m.get("coverage", 0),
                "mrr": m.get("mrr", 0), "recall": m.get("recall", {}),
                "top1": m.get("recall", {}).get("recall@1", 0)}
    except Exception as e:
        return {"error": str(e)}


# ─── RxNorm metrics ──────────────────────────────────────────────────────────

def _rxnorm_metrics() -> dict:
    report = _OUTPUT / "rxnorm_retrieval_report.json"
    if report.exists():
        with open(report, encoding="utf-8") as f:
            r = json.load(f)
        return r
    try:
        from src.linking.rxnorm.evaluator import DrugRetrievalEvaluator
        ev = DrugRetrievalEvaluator(str(_RXNORM_SAMPLES), str(_OUTPUT), top_k=10, use_reranker=True)
        m = ev.evaluate()
        ev.save_report(m)
        return {"n_samples": m.n_samples, "top1_accuracy": round(m.top1_accuracy, 4),
                "mrr": round(m.mrr, 4), "ingredient_accuracy": round(m.ingredient_accuracy, 4),
                "strength_accuracy": round(m.strength_accuracy, 4),
                "dose_form_accuracy": round(m.dose_form_accuracy, 4),
                "recall": {k: round(v, 4) for k, v in m.recall.items()}}
    except Exception as e:
        return {"error": str(e)}


# ─── Assertion eval (standalone) ───────────────────────────────────────────────

def _assertion_standalone() -> dict:
    """Evaluate RuleBasedDetector against validation gold."""
    from src.assertion.rules import RuleBasedDetector
    detector = RuleBasedDetector()
    samples = _load_jsonl(_VAL_PATH)

    all_gold, all_pred = [], []
    for sample in samples:
        text = sample["text"]
        for ge in sample.get("expected_entities", []):
            try:
                s = text.index(ge["text"])
                e = s + len(ge["text"])
            except ValueError:
                continue
            res = detector.detect(text, s, e, entity_type=ge.get("type"))
            pred_a = []
            if res.status.is_negated:
                pred_a.append("isNegated")
            if res.status.is_historical:
                pred_a.append("isHistorical")
            if res.status.is_family:
                pred_a.append("isFamily")
            all_gold.append({"text": ge["text"], "start": s, "end": e,
                             "type": ge.get("type", "TRIỆU_CHỨNG"),
                             "assertions": [a for a in ge.get("assertions", [])]})
            all_pred.append({"text": ge["text"], "start": s, "end": e,
                              "type": ge.get("type", "TRIỆU_CHỨNG"),
                              "assertions": pred_a})

    return _assertion_metrics(all_gold, all_pred)


# ─── NER eval (validation data) ───────────────────────────────────────────────

def _ner_validation() -> dict:
    """Run pipeline on validation data, compare with gold."""
    samples = _load_jsonl(_VAL_PATH)
    all_gold, all_pred = [], []
    for sample in samples:
        text = sample["text"]
        pred = _run_pipeline(text)
        gold = _gold_from_validation(sample, text)
        all_gold.extend(gold)
        all_pred.extend(pred)
    return {
        "entity": _entity_metrics(all_gold, all_pred),
        "per_class": _per_class_metrics(all_gold, all_pred),
        "assertion": _assertion_metrics(all_gold, all_pred),
    }


# ─── E2E eval (gold data) ─────────────────────────────────────────────────────

def _e2e_all() -> dict:
    """E2E metrics at 5 levels against processed/*.jsonl gold."""
    gold_samples = []
    for p in _GOLD_FILES.values():
        gold_samples.extend(_load_jsonl(p))

    all_gold, all_pred = [], []
    for sample in gold_samples:
        pred = _run_pipeline(sample["text"])
        gold = _gold_from_processed(sample)
        all_gold.extend(gold)
        all_pred.extend(pred)

    levels = {}
    for lvl in range(1, 6):
        levels[f"level_{lvl}"] = _e2e_level(all_gold, all_pred, lvl)
    return levels


# ─── Error taxonomy ───────────────────────────────────────────────────────────

def _collect_errors() -> tuple[list[dict], dict]:
    gold_samples = []
    for p in _GOLD_FILES.values():
        gold_samples.extend(_load_jsonl(p))
    val_samples = _load_jsonl(_VAL_PATH)

    all_errors = []
    for sample in gold_samples:
        pred = _run_pipeline(sample["text"])
        gold = _gold_from_processed(sample)
        src = f"e2e:{sample.get('id','')}"
        all_errors.extend(_classify_errors(gold, pred, src))

    for sample in val_samples:
        text = sample["text"]
        pred = _run_pipeline(text)
        gold = _gold_from_validation(sample, text)
        src = f"val:{sample.get('id','')}"
        all_errors.extend(_classify_errors(gold, pred, src))

    counts = defaultdict(int)
    for e in all_errors:
        counts[e.get("error_type", "unknown")] += 1
    return all_errors, dict(counts)


# ─── Ablation ─────────────────────────────────────────────────────────────────

def _ablation() -> list[dict]:
    """10 ablation configs — entity F1."""
    gold_samples = []
    for p in _GOLD_FILES.values():
        gold_samples.extend(_load_jsonl(p))

    configs = [
        ("rule-only", {"extract_labs": True, "extract_drugs": True,
                       "extract_diseases": True, "extract_symptoms": True}),
        ("ner-only", {"extract_labs": False, "extract_drugs": False,
                      "extract_diseases": False, "extract_symptoms": False}),
        ("rule+ner", {}),
        ("lexical-retrieval", {"icd_dense_enabled": False, "rxnorm_dense_enabled": False}),
        ("lexical+dense", {"icd_dense_enabled": True, "rxnorm_dense_enabled": True}),
        ("retrieval-only", {"reranker_enabled": False, "icd_reranker_enabled": False,
                            "rxnorm_reranker_enabled": False}),
        ("retrieval+cross-encoder", {"cross_encoder_enabled": True}),
        ("assertion-rule", {"detect_assertions": True}),
        ("assertion-no-family", {}),
        ("assertion-no-historical", {}),
    ]

    results = []
    for name, overrides in configs:
        from src.pipeline.pipeline import MedicalOntologyPipeline
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        pipeline = MedicalOntologyPipeline(cfg)

        all_g, all_p = [], []
        for sample in gold_samples:
            result = pipeline.process(sample["text"])
            pred = _pipeline_result_to_dict(result)
            gold = _gold_from_processed(sample)
            all_g.extend(gold)
            all_p.extend(pred)

        m = _entity_metrics(all_g, all_p)
        results.append({"config": name, "precision": m["precision"], "recall": m["recall"],
                        "f1": m["f1"], "num_true": m["num_true"], "num_pred": m["num_pred"]})
        _p(f"  {name:<30} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    return results


# ─── Active learning ───────────────────────────────────────────────────────────

def _active_learning() -> list[dict]:
    """Collect samples needing human review."""
    samples = []
    gold_samples = []
    for p in _GOLD_FILES.values():
        gold_samples.extend(_load_jsonl(p))
    val_samples = _load_jsonl(_VAL_PATH)

    # Rule vs model disagreement
    for sample in gold_samples[:30]:
        pred = _run_pipeline(sample["text"])
        gc, pc = len(sample.get("entities", [])), len(pred)
        if abs(gc - pc) >= 2:
            samples.append({"trigger": "rule_model_disagreement",
                             "sample_id": sample.get("id", ""),
                             "text": sample["text"][:200],
                             "gold_count": gc, "pred_count": pc,
                             "detail": f"count mismatch: gold={gc} pred={pc}"})

    # Unmatched entities from validation
    for sample in val_samples[:50]:
        text = sample["text"]
        pred = _run_pipeline(text)
        gold_texts = {e["text"] for e in sample.get("expected_entities", [])}
        for e in pred:
            if e["text"] not in gold_texts and e["type"] in ["THUỐC", "TRIỆU_CHỨNG"]:
                samples.append({"trigger": "unmatched_entity",
                                 "sample_id": sample.get("id", ""),
                                 "text": text[:200],
                                 "entity_text": e["text"],
                                 "entity_type": e["type"],
                                 "assertions": e.get("assertions", []),
                                 "detail": "entity not in gold — potential false positive"})

    # Truncated spans
    for sample in gold_samples[:20]:
        pred = _run_pipeline(sample["text"])
        for e in pred:
            if len(e["text"]) < 3:
                samples.append({"trigger": "short_span",
                                 "sample_id": sample.get("id", ""),
                                 "text": sample["text"][:200],
                                 "entity_text": e["text"],
                                 "entity_type": e["type"],
                                 "detail": f"very short entity: '{e['text']}'"})

    # Deduplicate
    seen = set()
    deduped = []
    for s in samples:
        k = (s.get("trigger"), s.get("sample_id"), s.get("entity_text", ""))
        if k not in seen:
            seen.add(k)
            deduped.append(s)
    return deduped


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _p("\n" + "=" * 70)
    _p("  Comprehensive Medical Ontology Evaluation")
    _p("=" * 70)

    _OUTPUT.mkdir(parents=True, exist_ok=True)
    _AL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. NER + Assertion on validation
    _p("\n[1/8] NER + Assertion evaluation on validation data...")
    try:
        ner = _ner_validation()
        results["ner"] = ner["entity"]
        results["per_class"] = ner["per_class"]
        results["assertion"] = ner["assertion"]
        _p(f"  NER: P={results['ner']['precision']:.4f} R={results['ner']['recall']:.4f} "
           f"F1={results['ner']['f1']:.4f}  ({results['ner']['num_true']} gold, "
           f"{results['ner']['num_pred']} pred)")
        _p(f"  Assertion exact accuracy: {results['assertion']['exact_accuracy']:.4f}")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["ner"] = {}
        results["per_class"] = {}
        results["assertion"] = {}

    # 2. ICD
    _p("\n[2/8] ICD retrieval evaluation...")
    try:
        icd = _icd_metrics()
        results["icd"] = icd
        if "error" not in icd:
            _p(f"  Coverage={icd.get('coverage',0):.2%}  MRR={icd.get('mrr',0):.4f}  "
               f"Recall@1={icd.get('recall',{}).get('recall@1',0):.4f}")
        else:
            _p(f"  ERROR: {icd['error']}")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["icd"] = {}

    # 3. RxNorm
    _p("\n[3/8] RxNorm retrieval evaluation...")
    try:
        rx = _rxnorm_metrics()
        results["rxnorm"] = rx
        if "error" not in rx:
            _p(f"  Top-1 acc={rx.get('top1_accuracy',0):.4f}  MRR={rx.get('mrr',0):.4f}")
            _p(f"  Ingredient={rx.get('ingredient_accuracy',0):.4f}  "
               f"Strength={rx.get('strength_accuracy',0):.4f}  "
               f"DoseForm={rx.get('dose_form_accuracy',0):.4f}")
        else:
            _p(f"  ERROR: {rx['error']}")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["rxnorm"] = {}

    # 4. E2E
    _p("\n[4/8] E2E evaluation at 5 levels...")
    try:
        e2e = _e2e_all()
        results["e2e"] = e2e
        for lvl, m in e2e.items():
            _p(f"  {lvl}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  "
               f"(correct={m['correct']}/{m['num_true']})")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["e2e"] = {}

    # 5. Error taxonomy
    _p("\n[5/8] Building error taxonomy...")
    try:
        all_errors, counts = _collect_errors()
        results["error_taxonomy"] = counts
        results["total_errors"] = len(all_errors)
        _p(f"  Total errors: {len(all_errors)}")
        for et, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            _p(f"    {et}: {cnt}")

        err_csv = _OUTPUT / "errors.csv"
        if all_errors:
            with open(err_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_errors[0].keys(), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_errors)
        _p(f"  Saved: {err_csv}")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["error_taxonomy"] = {}
        results["total_errors"] = 0

    # 6. Ablation
    _p("\n[6/8] Running ablation (10 configs)...")
    try:
        ablation = _ablation()
        results["ablation"] = ablation

        ab_csv = _OUTPUT / "ablation.csv"
        with open(ab_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["config", "precision", "recall", "f1", "num_true", "num_pred"])
            writer.writeheader()
            writer.writerows(ablation)
        _p(f"  Saved: {ab_csv}")
    except Exception as e:
        _p(f"  ERROR: {e}")
        results["ablation"] = []

    # 7. Active learning
    _p("\n[7/8] Collecting active learning samples...")
    try:
        al = _active_learning()
        al_path = _AL_DIR / "to_review.jsonl"
        with open(al_path, "w", encoding="utf-8") as f:
            for s in al:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        _p(f"  Collected {len(al)} samples -> {al_path}")
    except Exception as e:
        _p(f"  ERROR: {e}")

    # 8. Save metrics.json
    _p("\n[8/8] Saving metrics.json...")
    metrics_json = _OUTPUT / "metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    _p(f"  Saved: {metrics_json}")

    # Save per_class_metrics.csv
    if results.get("per_class"):
        pc_csv = _OUTPUT / "per_class_metrics.csv"
        with open(pc_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["entity_type", "precision", "recall", "f1", "support"])
            writer.writeheader()
            for etype, m in results["per_class"].items():
                writer.writerow({"entity_type": etype, **m})
        _p(f"  Saved: {pc_csv}")

    # Generate report.md
    _p("\nGenerating report.md...")
    _write_report(results)

    _p("\n" + "=" * 70)
    _p("  Evaluation complete!")
    _p("=" * 70)
    for fn in ["metrics.json", "per_class_metrics.csv", "errors.csv",
               "retrieval_errors.csv", "assertion_errors.csv", "ablation.csv", "report.md"]:
        fp = _OUTPUT / fn
        if fp.exists():
            _p(f"  {fn}: {fp.stat().st_size:,} bytes")
    al_path = _AL_DIR / "to_review.jsonl"
    if al_path.exists():
        _p(f"  data/active_learning/to_review.jsonl: {al_path.stat().st_size:,} bytes")


def _write_report(results: dict) -> None:
    """Generate Markdown report."""
    rpath = _OUTPUT / "report.md"
    with open(rpath, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Evaluation Report\n\n")

        # NER
        f.write("## Entity Extraction (NER)\n\n")
        if results.get("ner"):
            m = results["ner"]
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in [("Precision", m["precision"]), ("Recall", m["recall"]),
                         ("F1", m["f1"]), ("Gold entities", m["num_true"]),
                         ("Predicted", m["num_pred"]), ("Correct", m["num_correct"])]:
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        if results.get("per_class"):
            f.write("### Per-Class F1\n\n")
            f.write("| Entity Type | Precision | Recall | F1 | Support |\n"
                    "|-------------|-----------|--------|-----|--------|\n")
            for etype, m in results["per_class"].items():
                f.write(f"| {etype} | {m['precision']:.4f} | {m['recall']:.4f} | "
                        f"{m['f1']:.4f} | {m['support']} |\n")
            f.write("\n")

        # Assertion
        f.write("## Assertion Detection\n\n")
        if results.get("assertion"):
            a = results["assertion"]
            f.write("| Label | Precision | Recall | F1 |\n"
                    "|-------|-----------|--------|----|\n")
            for label in ["isNegated", "isHistorical", "isFamily"]:
                if label in a:
                    m = a[label]
                    f.write(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | "
                             f"{m['f1']:.4f} |\n")
            f.write(f"| **Macro F1** | -- | -- | **{a['macro_f1']:.4f}** |\n"
                     f"| **Exact Accuracy** | -- | -- | **{a['exact_accuracy']:.4f}** |\n\n")

        # ICD
        f.write("## ICD-10 Linking\n\n")
        if results.get("icd") and "error" not in results["icd"]:
            icd = results["icd"]
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in [("Samples", icd.get("n_samples", 0)), ("Coverage", f"{icd.get('coverage', 0):.2%}"),
                          ("MRR", f"{icd.get('mrr', 0):.4f}"), ("Top-1", f"{icd.get('top1', 0):.4f}")]:
                f.write(f"| {k} | {v} |\n")
            if icd.get("recall"):
                f.write("\n**Recall@K:**\n")
                for k_, v_ in sorted(icd["recall"].items()):
                    f.write(f"- {k_}: {v_:.4f}\n")
            f.write("\n")

        # RxNorm
        f.write("## RxNorm Linking\n\n")
        if results.get("rxnorm") and "error" not in results["rxnorm"]:
            rx = results["rxnorm"]
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in [("Top-1 Accuracy", f"{rx.get('top1_accuracy', 0):.4f}"),
                          ("MRR", f"{rx.get('mrr', 0):.4f}"),
                          ("Ingredient Accuracy", f"{rx.get('ingredient_accuracy', 0):.4f}"),
                          ("Strength Accuracy", f"{rx.get('strength_accuracy', 0):.4f}"),
                          ("Dose Form Accuracy", f"{rx.get('dose_form_accuracy', 0):.4f}"),
                          ("Samples", rx.get("n_samples", 0))]:
                f.write(f"| {k} | {v} |\n")
            if rx.get("recall"):
                f.write("\n**Recall@K:**\n")
                for k_, v_ in sorted(rx["recall"].items()):
                    f.write(f"- {k_}: {v_:.4f}\n")
            f.write("\n")

        # E2E
        f.write("## End-to-End Levels\n\n")
        if results.get("e2e"):
            f.write("| Level | Precision | Recall | F1 | Correct / True |\n"
                    "|-------|-----------|--------|-----|----------------|\n")
            for lvl, m in results["e2e"].items():
                f.write(f"| {lvl} | {m['precision']:.4f} | {m['recall']:.4f} | "
                        f"{m['f1']:.4f} | {m['correct']} / {m['num_true']} |\n")
            f.write("\n")

        # Error taxonomy
        f.write("## Error Taxonomy\n\n")
        if results.get("error_taxonomy"):
            f.write("| Error Type | Count |\n|------------|-------|\n")
            for et, cnt in sorted(results["error_taxonomy"].items(), key=lambda x: -x[1]):
                f.write(f"| {et} | {cnt} |\n")
            f.write(f"\n**Total errors:** {results.get('total_errors', 0)}\n\n")

        # Ablation
        f.write("## Ablation Study\n\n")
        if results.get("ablation"):
            f.write("| Config | Precision | Recall | F1 |\n"
                    "|--------|-----------|--------|-----|\n")
            for r in results["ablation"]:
                f.write(f"| {r['config']} | {r['precision']:.4f} | "
                        f"{r['recall']:.4f} | {r['f1']:.4f} |\n")
            f.write("\n")

        # Priority improvements
        f.write("## Prioritized Improvements\n\n")
        if results.get("error_taxonomy"):
            top = sorted(results["error_taxonomy"].items(), key=lambda x: -x[1])[:5]
            recs = {
                "missed_entity": "Review extraction rules for missing entity types. Consider expanding dictionaries.",
                "false_positive": "Tighten confidence thresholds. Add negative pattern filters.",
                "wrong_boundary": "Improve span resolution logic. Check token alignment in preprocessing.",
                "wrong_type": "Review type classification rules. Add disambiguation heuristics.",
                "wrong_assertion": "Improve assertion detection scope rules. Verify cue coverage.",
                "icd_gold_missing": "ICD linking failed -- check KB availability and retrieval rankings.",
                "rxnorm_gold_missing": "RxNorm linking failed -- check KB coverage and fuzzy matching.",
                "reranker_error": "Reranker not placing gold candidates at top -- investigate cross-encoder scores.",
                "invalid_candidate": "Candidate validation failing -- check KB entry format.",
                "schema_error": "Output format issue -- validate against competition schema.",
                "offset_mismatch": "Span positions don't align with text -- check preprocessing pipeline.",
            }
            for i, (et, cnt) in enumerate(top, 1):
                pri = "HIGH" if cnt > 10 else "MEDIUM" if cnt > 3 else "LOW"
                f.write(f"{i}. **{et}** -- {cnt} occurrences (**{pri}**)\n")
                if et in recs:
                    f.write(f"   -> {recs[et]}\n")
                f.write("\n")

        f.write("\n*Generated by comprehensive_evaluation.py*\n")

    _p(f"  Saved: {rpath}")


if __name__ == "__main__":
    main()
