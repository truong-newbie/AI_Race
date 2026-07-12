"""Debug: test pattern matching with file output."""
import json, re, sys
from pathlib import Path

s = 'Tiền sử bệnh\n    Thuốc trước khi'

results = {}
tests = [
    ("no_flags_tien_su_benh_hien_tai", r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', 0),
    ("no_flags_tien_su_benh", r'^tiền\s+sử\s+bệnh', 0),
    ("no_flags_tien_su_hien_tai", r'^tiền\s+sử\s+hiện\s+tại\b', 0),
    ("MULTILINE_tien_su_benh_hien_tai", r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.MULTILINE),
    ("MULTILINE_tien_su_benh", r'^tiền\s+sử\s+bệnh', re.MULTILINE),
    ("MULTILINE_tien_su_only", r'^tiền\s+sử\b', re.MULTILINE),
    # With dotall to let . match newline
    ("DOTALL_tien_su_benh_hien_tai", r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.DOTALL),
    ("DOTALL_MULTILINE_tien_su_benh_hien_tai", r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.DOTALL | re.MULTILINE),
    # Check what's ACTUALLY in the string after "Tiền sử bệnh"
    ("char_after_tien_su_benh", {"bytes": [repr(c) for c in s[12:18]]}),
]

# Add character analysis
results["chars_after_tien_su_benh"] = [(i, ord(c), hex(ord(c))) for i, c in enumerate(s[12:18])]

# Test the exact combined pattern
combined = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)
results["combined_matches"] = [(m.start(), m.end(), s[m.start():m.end()]) for m in combined.finditer(s)]

# Also test without MULTILINE
combined_no_ml = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE
)
results["combined_no_ml_matches"] = [(m.start(), m.end(), s[m.start():m.end()]) for m in combined_no_ml.finditer(s)]

# Test the pattern from scope.py
combined_sc = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b',
    re.IGNORECASE | re.MULTILINE
)
results["scope_combined_matches"] = [(m.start(), m.end(), s[m.start():m.end()]) for m in combined_sc.finditer(s)]

# Test each component pattern
p3 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE)
results["p3_matches"] = [(m.start(), m.end(), s[m.start():m.end()]) for m in p3.finditer(s)]

# Check: what does the \s match here?
results["s_repr"] = repr(s)
results["s_split"] = s.split('\n')

# Also: maybe the space is NOT a regular space?
results["bytes_at_12"] = list(s[12].encode('utf-8'))
results["bytes_at_13"] = list(s[13].encode('utf-8'))
results["bytes_at_14"] = list(s[14].encode('utf-8'))
results["bytes_at_15"] = list(s[15].encode('utf-8'))
results["bytes_at_16"] = list(s[16].encode('utf-8'))
results["bytes_at_17"] = list(s[17].encode('utf-8'))

with open("debug_pat4.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
