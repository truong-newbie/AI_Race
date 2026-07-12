"""Minimal debug: test pattern directly on exact stripped sentence text."""
import json, re

PAT = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+(?=hiện)',
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

# Exact text of sentence [2:405] after strip
sent = 'tiền sử bệnh\n    thuốc trước khi nhập viện\n    - metoprolol 25mg po bid\n    - doxycycline cho viêm tuyến mồ hôi\n    - atenolol (uống hôm nay)'

results = {}

results["sent_first50"] = repr(sent[:50])
results["sent_last20"] = repr(sent[-20:])

# Test each alternative separately
alt1 = re.compile(r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b', re.IGNORECASE | re.MULTILINE)
alt2 = re.compile(r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE | re.DOTALL)
alt3 = re.compile(r'^tiền\s+sử\s+bệnh\s+(?=hiện)', re.IGNORECASE | re.MULTILINE | re.DOTALL)

results["alt1_match"] = bool(alt1.search(sent))
results["alt2_match"] = bool(alt2.search(sent))
results["alt3_match"] = bool(alt3.search(sent))

# Full pattern
results["full_match"] = bool(PAT.search(sent))
if PAT.search(sent):
    m = PAT.search(sent)
    results["full_match_text"] = repr(sent[m.start():m.end()])

# Try: what if there's a carriage return?
sent_crlf = sent.replace('\n', '\r\n')
results["crlf_match"] = bool(PAT.search(sent_crlf))

# What if the stripped text starts differently?
sent_stripped = sent.lstrip()
results["lstrip_match"] = bool(PAT.search(sent_stripped))

# Check: what is the actual start of the sentence?
results["char0"] = ord(sent[0]) if sent else None
results["char1"] = ord(sent[1]) if len(sent) > 1 else None
results["char2"] = ord(sent[2]) if len(sent) > 2 else None

# With IGNORECASE: does 't' match '^'?
results["first4"] = repr(sent[:4])

# Maybe the issue: stripped text is bytes, not str?
results["type"] = type(sent).__name__

# Test: what does HISTORICAL_SECTION_PATTERNS match?
for i, pat in enumerate([
    r"tiền\s+sử\b",
    r"quá\s+khứ\b",
    r"bệnh\s+sử\b",
    r"tiền\s+sử\s+bệnh\b",
    r"past\s+history\b",
    r"history\b",
]):
    m = re.search(pat, sent, re.IGNORECASE | re.DOTALL)
    results[f"hist_pat_{i}"] = bool(m)
    if m:
        results[f"hist_pat_{i}_match"] = repr(sent[m.start():m.end()])

with open("debug_pattern.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
