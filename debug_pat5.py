"""Debug: minimal pattern test."""
import json, re

s = 'Tiền sử bệnh\n    Thuốc trước khi'

results = {}

# Step by step
results["s_repr"] = repr(s)
results["s_first_20"] = repr(s[:20])

# Test parts
p1 = re.compile(r'^tiền', re.IGNORECASE)
p2 = re.compile(r'^tiền\s+sử', re.IGNORECASE)
p3 = re.compile(r'^tiền\s+sử\s+bệnh', re.IGNORECASE)
p4 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện', re.IGNORECASE)
p5 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE)
p6 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.MULTILINE)

results["p1"] = bool(p1.search(s))
results["p2"] = bool(p2.search(s))
results["p3"] = bool(p3.search(s))
results["p4"] = bool(p4.search(s))
results["p5"] = bool(p5.search(s))
results["p6"] = bool(p6.search(s))

if p5.search(s):
    m = p5.search(s)
    results["p5_match"] = repr(s[m.start():m.end()])

# With DOTALL
p7 = re.compile(r'^tiền\s+sử\s+bệnh\s+hiện\s+tại\b', re.IGNORECASE | re.DOTALL)
results["p7"] = bool(p7.search(s))

# Maybe the problem is that \s matches newline but then it can't find "hiện"?
# Check character by character after "Tiền sử bệnh"
# Position 0-11 = "Tiền sử bệnh" (12 chars)
results["s_12_20"] = repr(s[12:20])
results["s_ord_12_20"] = [(i+12, ord(c)) for i, c in enumerate(s[12:20])]

# Test: is the space between "bệnh" and what follows actually a space?
results["char_12"] = ord(s[12])  # should be 10 (newline)
results["char_13"] = ord(s[13])  # should be 32 (space)
results["char_14"] = ord(s[14])  # should be 32 (space)
results["char_15"] = ord(s[15])  # should be 32 (space)
results["char_16"] = ord(s[16])  # should be 32 (space)
results["char_17"] = ord(s[17])  # should be ord('T') = 84

# Try: match \s+ greedily
p_greedy = re.compile(r'tiền\s+sử\s+bệnh(\s+)hiện', re.IGNORECASE)
m_g = p_greedy.search(s)
if m_g:
    results["greedy_group"] = repr(m_g.group(1))
    results["greedy_match"] = repr(s[m_g.start():m_g.end()])

# Try with raw string
p_raw = re.compile(r'tiền\s+sử\s+bệnh\s+hiện\s+tại', re.IGNORECASE)
results["p_raw"] = bool(p_raw.search(s))

with open("debug_pat5.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
