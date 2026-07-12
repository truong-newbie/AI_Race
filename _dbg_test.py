"""Debug test_family_with_tien_su"""
from src.assertion.cues import find_cue_matches, CueType
import re

text = "Bố có tiền sử bệnh tim."
cues = find_cue_matches(text)

results = {}
results["text"] = text
results["cues"] = [
    {"start": c.start, "end": c.end, "type": c.cue_type.name, "text": c.text}
    for c in cues
]

# Test gap logic
hist_cues = [c for c in cues if c.cue_type == CueType.HISTORICAL and c.text.strip() in ("tiền sử", "có tiền sử")]
fam_cues = [c for c in cues if c.cue_type == CueType.FAMILY]

results["hist_after_filter"] = [c.text for c in hist_cues]
results["fam_cues"] = [{"text": c.text, "start": c.start, "end": c.end} for c in fam_cues]
results["gap_check"] = []
for hcue in hist_cues:
    for fcue in fam_cues:
        gap = hcue.start - fcue.end
        results["gap_check"].append({
            "hist": hcue.text, "hist_start": hcue.start,
            "fam": fcue.text, "fam_end": fcue.end,
            "gap": gap, "in_range": 0 < gap <= 10
        })

# Also: what does apply_scope_rules return for this?
from src.assertion.scope import ClauseSegmenter, apply_scope_rules
seg = ClauseSegmenter()
result = apply_scope_rules(text, 14, 22, cues, seg)
results["scope_result"] = result
results["has_hist_section"] = seg._is_historical_section(text)

import json
with open("_dbg_test.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
