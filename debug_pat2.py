"""Debug: see raw text around the Tiền sử section."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

input_dir = Path(r"C:\Users\DELL\Downloads\AI_Race_Bai2(1)\input")
text = (input_dir / "1.txt").read_text(encoding="utf-8").strip()

output = {}

# Show first 30 chars with byte positions
output["first_30"] = [(i, repr(text[i])) for i in range(min(30, len(text)))]

# Show what's around position 2-10
output["pos_0_to_30"] = repr(text[0:30])

# What does the segmenter produce for sentence 2?
import importlib
import src.assertion.scope as sm
importlib.reload(sm)
seg = sm.ClauseSegmenter()
sents = seg.segment_sentences(text)
output["sentences"] = [
    {"start": s.start, "end": s.end, "text_len": len(s.text), "text_head": repr(s.text[:40]), "hist": s.has_historical_section}
    for s in sents[:5]
]

# The text passed to _is_historical_section for sentence 2
if len(sents) > 1:
    s2 = sents[1]
    raw_text = text[s2.start:s2.end]
    stripped_text = raw_text.strip(' \t\n')
    while stripped_text and stripped_text[-1] in '.!?':
        stripped_text = stripped_text[:-1]
    output["sent2_raw"] = repr(raw_text[:50])
    output["sent2_stripped"] = repr(stripped_text[:50])

with open("debug_pat2.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("Done")
