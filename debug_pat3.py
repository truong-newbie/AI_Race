"""Debug: test SECTION_HEADER_HISTORICAL_PATTERN against actual sentence text."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()

# The actual sentence at [2:405]
raw_sent = text[2:405]
stripped = raw_sent.strip(' \t\n')
while stripped and stripped[-1] in '.!?':
    stripped = stripped[:-1]

output = {
    "raw_sent_head": repr(raw_sent[:80]),
    "stripped_head": repr(stripped[:80]),
    "stripped_len": len(stripped),
}

# Test each pattern separately
patterns = {
    "numbered_tien_su": (r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b', re.IGNORECASE | re.MULTILINE),
    "tioen_su_benh_hien_tai_dash": (r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE),
    "tioen_su_benh_hien_tai_plain": (r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE),
    "tioen_su_only": (r'^tiền\s+sử\b', re.IGNORECASE | re.MULTILINE),
    "tiền_sử_bệnh": (r'tiền\s+sử\s+bệnh\b', re.IGNORECASE | re.MULTILINE),
}

for name, (pat, flags) in patterns.items():
    compiled = re.compile(pat, flags)
    matches = [(m.start(), m.end(), stripped[m.start():m.end()]) for m in compiled.finditer(stripped)]
    output[name] = {"pattern": pat, "matches": matches}

# Combined pattern
combined = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)
output["combined"] = [(m.start(), m.end(), stripped[m.start():m.end()]) for m in combined.finditer(stripped)]

with open("debug_pat3.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
