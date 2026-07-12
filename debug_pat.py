"""Debug the SECTION_HEADER_HISTORICAL_PATTERN."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Reload to avoid cached module
import importlib
import src.assertion.scope as scope_mod
importlib.reload(scope_mod)

SECTION_HEADER_HISTORICAL_PATTERN = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()

# Get the sentence starting at position 2
sent2 = text[2:405]

output = {
    "sent2_repr": repr(sent2[:50]),
    "sent2_first_line": sent2.split('\n')[0],
    "matches": [],
}

for m in SECTION_HEADER_HISTORICAL_PATTERN.finditer(sent2):
    output["matches"].append({"start": m.start(), "end": m.end(), "group": sent2[m.start():m.end()]})

# Also test with lower case
matches_lower = list(SECTION_HEADER_HISTORICAL_PATTERN.finditer(sent2.lower()))
output["matches_lower"] = [{"start": m.start(), "end": m.end()} for m in matches_lower]

# Test the individual patterns
p1 = re.compile(r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b', re.IGNORECASE | re.MULTILINE)
p2 = re.compile(r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE)
output["p1_matches"] = [{"start": m.start(), "end": m.end()} for m in p1.finditer(sent2)]
output["p2_matches"] = [{"start": m.start(), "end": m.end()} for m in p2.finditer(sent2)]

with open("debug_pat.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
