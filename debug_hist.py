"""Debug the historical cue filtering."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.assertion.cues import find_cue_matches, CueType
from src.assertion.scope import ClauseSegmenter, apply_scope_rules

SECTION_HEADER_CUE_PATTERN = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()

cues = find_cue_matches(text)
hist_cues = [c for c in cues if c.cue_type == CueType.HISTORICAL]

output = {
    "all_hist_cues": [(c.start, c.end, c.text[:30]) for c in hist_cues],
    "cues_count": len(hist_cues),
}

# Filter test
filtered = []
for c in hist_cues:
    window = text[max(0, c.start - 20):c.end + 20]
    is_header = bool(SECTION_HEADER_CUE_PATTERN.search(window))
    if not is_header:
        filtered.append(c)
    output[f"cue_{c.start}_{c.end}"] = {
        "text": c.text[:30],
        "window": window[:50],
        "is_header": is_header,
    }

output["filtered_count"] = len(filtered)

# Now test apply_scope_rules for atenolol
aten_text = "atenolol"
aten_pos = text.find(aten_text)
if aten_pos >= 0:
    aten_end = aten_pos + len(aten_text)
    seg = ClauseSegmenter()
    result_all = apply_scope_rules(text, aten_pos, aten_end, hist_cues, seg)
    result_filt = apply_scope_rules(text, aten_pos, aten_end, filtered, seg)
    output["atenolol_all_hist"] = result_all["is_historical"]
    output["atenolol_filtered_hist"] = result_filt["is_historical"]

output["remaining_cues"] = [(c.start, c.end, text[c.start:c.end]) for c in filtered]
output["window_atenolol"] = text[max(0, aten_pos-100):aten_pos+50] if aten_pos >= 0 else None
output["atenolol_pos"] = aten_pos

with open("debug_hist.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
