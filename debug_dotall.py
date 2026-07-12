"""Debug: test the exact SECTION_HEADER_HISTORICAL_PATTERN with DOTALL on real text."""
import json, re
from pathlib import Path

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8")

# SECTION_HEADER_PATTERN = r'^[ \t]*\d+[\.\)][ \t]+'
SECTION_HEADER_PATTERN = re.compile(r'^[ \t]*\d+[\.\)][ \t]+', re.MULTILINE)

# SECTION_HEADER_HISTORICAL_PATTERN with DOTALL added
PAT = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

results = {}

# Simulate what segment_sentences does
boundary_positions = []
for match in SECTION_HEADER_PATTERN.finditer(text):
    boundary_positions.append(match.start())
for match in re.finditer(r'[.。;；!！？]', text):
    boundary_positions.append(match.end())
boundary_positions = sorted(set(boundary_positions))

start = 0
sentences_info = []
for i, end in enumerate(boundary_positions):
    if end <= start:
        continue
    raw = text[start:end]
    stripped = raw.strip(' \t\n')
    while stripped and stripped[-1] in '.!?':
        stripped = stripped[:-1]

    stripped_lower = stripped.lower().lstrip()

    m = PAT.search(stripped_lower)
    sentences_info.append({
        "idx": i,
        "start": start,
        "end": end,
        "stripped_repr": repr(stripped_lower[:80]),
        "pattern_matched": bool(m),
        "matched_text": repr(m.group()) if m else None,
        "stripped_len": len(stripped),
    })
    start = end

results["boundary_positions"] = boundary_positions[:10]
results["sentences"] = sentences_info
results["total_sentences"] = len(sentences_info)

# Also test the exact third alternative on stripped text
alt3 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.DOTALL)
# Find the sentence that starts with "Tiền sử bệnh"
for s in sentences_info:
    if 'tiền sử bệnh' in s['stripped_repr'].lower():
        stext = s['stripped_repr'].lower().lstrip()
        r = alt3.search(stext)
        results["alt3_on_sent"] = {
            "text": stext[:60],
            "matched": bool(r),
            "match": repr(r.group()) if r else None
        }
        # Also test with MULTILINE
        alt3_ml = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE | re.DOTALL)
        r2 = alt3_ml.search(stext)
        results["alt3_ml_on_sent"] = {
            "matched": bool(r2),
            "match": repr(r2.group()) if r2 else None
        }
        break

with open("debug_dotall.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
