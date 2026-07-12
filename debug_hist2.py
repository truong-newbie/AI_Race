"""Debug the historical cue filtering - write to file."""
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

# Filter
filtered = []
for c in hist_cues:
    window = text[max(0, c.start - 20):c.end + 20]
    is_header = bool(SECTION_HEADER_CUE_PATTERN.search(window))
    if not is_header:
        filtered.append(c)
    else:
        pass  # header, filtered out

# Test atenolol
aten_text = "atenolol"
aten_pos = text.find(aten_text)
aten_end = aten_pos + len(aten_text)
seg = ClauseSegmenter()
result_filt = apply_scope_rules(text, aten_pos, aten_end, filtered, seg)

# For each remaining cue, check if it's before atenolol and in the same sentence
remaining_before_aten = [c for c in filtered if c.end <= aten_pos]
remaining_before_aten.sort(key=lambda c: c.start)

# Also check sentences around atenolol
sents = seg.segment_sentences(text)
sent_containing_aten = None
for s in sents:
    if s.start <= aten_pos < s.end:
        sent_containing_aten = s
        break

output = {
    "atenolol_pos": aten_pos,
    "atenolol_filtered_hist": result_filt["is_historical"],
    "remaining_hist_cues": [{"start": c.start, "end": c.end, "text": text[c.start:c.end]} for c in filtered],
    "remaining_before_aten": [{"start": c.start, "end": c.end, "text": text[c.start:c.end]} for c in remaining_before_aten],
    "sent_count": len(sents),
    "sent_containing_aten": {"start": sent_containing_aten.start, "end": sent_containing_aten.end, "hist": sent_containing_aten.has_historical_section} if sent_containing_aten else None,
}

with open("debug_hist2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
