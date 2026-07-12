"""Debug: import from actual module and test directly."""
import sys, json, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
import src.assertion.scope as scope_module
importlib.reload(scope_module)

from src.assertion.scope import ClauseSegmenter, SECTION_HEADER_HISTORICAL_PATTERN, HISTORICAL_SECTION_PATTERNS

results = {}

# Test 1: simple text "Bố có tiền sử bệnh tim."
text1 = "Bố có tiền sử bệnh tim."
seg = ClauseSegmenter()
result1 = seg._is_historical_section(text1)
stripped1 = text1.lower().lstrip()
pat_m1 = SECTION_HEADER_HISTORICAL_PATTERN.search(stripped1)
hist_match1 = None
for pat in HISTORICAL_SECTION_PATTERNS:
    m = re.search(pat, stripped1, re.DOTALL)
    if m:
        hist_match1 = {"pattern": pat, "matched": m.group()}
        break

results["test1"] = {
    "text": text1,
    "stripped": stripped1,
    "has_historical_section": result1,
    "header_pattern_match": bool(pat_m1),
    "header_match_text": pat_m1.group() if pat_m1 else None,
    "hist_pattern_match": hist_match1,
}

# Test 2: actual file 1 section header text
file_text = open(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input\1.txt", encoding="utf-8").read()
# Find the first sentence that starts with "Tiền sử bệnh"
sents = seg.segment_sentences(file_text)
for s in sents:
    if s.text.startswith("Tiền sử bệnh"):
        stripped_sent = s.text.lower().lstrip()
        pm = SECTION_HEADER_HISTORICAL_PATTERN.search(stripped_sent)
        hm = None
        for pat in HISTORICAL_SECTION_PATTERNS:
            m = re.search(pat, stripped_sent, re.DOTALL)
            if m:
                hm = {"pattern": pat, "matched": m.group()}
                break
        results["test2_section_header"] = {
            "sentence_text": s.text[:60],
            "stripped": stripped_sent[:60],
            "has_historical_section": s.has_historical_section,
            "header_pattern_match": bool(pm),
            "header_match_text": pm.group() if pm else None,
            "hist_pattern_match": hm,
        }
        break

# Also: what does the HISTORICAL_SECTION_PATTERNS loop return for the section header?
results["SECTION_HEADER_HISTORICAL_PATTERN_flags"] = SECTION_HEADER_HISTORICAL_PATTERN.flags
results["SECTION_HEADER_HISTORICAL_PATTERN_pattern"] = SECTION_HEADER_HISTORICAL_PATTERN.pattern

with open("_dbg_scope.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
