"""Test lookbehind with \w on Vietnamese letters."""
import re, json

results = {}

# Test: what does \w match in Python 3?
s = "Bố có"
for i, c in enumerate(s):
    results[f"char_{i}"] = {"char": c, "is_w": c.isalnum(), "is_space": c.isspace(), "is_alpha": c.isalpha()}
    results[f"char_{i}_ord"] = ord(c)

# The text: "Bố có tiền sử bệnh tim."
# Positions: B(0) ố(1) space(2) c(3) ó(4) space(5) t(6) iền(7-10)
text1 = "Bố có tiền sử bệnh tim."
text2 = "Tiền sử bệnh\n    Thuốc"

# Test \w lookbehind
p = re.compile(r'(?<!\w)tiền\s+sử\s+bệnh\b', re.UNICODE | re.IGNORECASE)
results["t1_match"] = bool(p.search(text1))
results["t2_match"] = bool(p.search(text2))
if p.search(text1):
    m = p.search(text1)
    results["t1_match_pos"] = (m.start(), m.end())
    results["t1_char_before"] = text1[m.start()-1] if m.start() > 0 else "START"
    results["t1_char_before_ord"] = ord(text1[m.start()-1]) if m.start() > 0 else None

# Check: what's before 'tiền' in t1?
for i in range(max(0, 5), 11):
    if i < len(text1):
        results[f"t1_pos_{i}"] = {"char": text1[i], "is_w": text1[i].isalnum()}

# Try using str.isalpha() explicitly in a conditional check
# Instead of regex lookbehind, use Python to check
for label, text in [("t1", text1), ("t2", text2)]:
    idx = text.lower().find("tiền sử bệnh")
    if idx >= 0:
        char_before = text[idx-1] if idx > 0 else None
        results[f"{label}_before_tiensu"] = {
            "idx": idx, "char_before": char_before,
            "is_alpha": char_before.isalpha() if char_before else None,
            "is_space": char_before.isspace() if char_before else None
        }

with open("_dbg_lb.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
