"""Debug: show actual file structure of file 1."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8")

output = {}
lines = text.split('\n')
output["lines_0_20"] = [(i, repr(l[:60])) for i, l in enumerate(lines[:20])]

# Find all lines containing "tiền sử"
output["tien_su_lines"] = [i for i, l in enumerate(lines) if 'tiền' in l.lower() and 'sử' in l.lower()]

# For each tien_su line, show context
output["tien_su_context"] = []
for i in output["tien_su_lines"]:
    context = lines[max(0,i-1):i+3]
    output["tien_su_context"].append({
        "line_idx": i,
        "lines": [repr(l[:80]) for l in context]
    })

# Show actual byte positions of "Tiền sử bệnh hiện tại" in raw text
m = re.search(r'Tiền\s+sử\s+bệnh\s+hiện\s+tại', text)
if m:
    output["tioen_su_benh_hien_tai_pos"] = (m.start(), m.end())
    output["around_tioen_su_benh_hien_tai"] = repr(text[max(0,m.start()-20):m.end()+50])
else:
    output["tioen_su_benh_hien_tai_pos"] = None
    output["search_note"] = "Not found - checking individual parts"
    for pname, pat in [
        ("tiền sử bệnh", r'tiền\s+sử\s+bệnh'),
        ("tiền sử", r'tiền\s+sử'),
        ("hiện tại", r'hiện\s+tại'),
    ]:
        matches = [(m.start(), m.end()) for m in re.finditer(pat, text, re.IGNORECASE)]
        output[pname] = matches[:3]

# Also: check what's in the raw text at the section boundaries
for pos in [0, 2, 400, 405, 407, 410]:
    if pos < len(text):
        output[f"text_at_{pos}"] = repr(text[pos:pos+30])

with open("debug_file.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
