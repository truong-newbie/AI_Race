"""Direct pattern test on actual section header text."""
import json, re

# Test: what does (?=\s)(?!\S) actually do?
PAT = re.compile(
    r'^[ \t]*\d+[\.\)][ \t]+tiền\s+sử\b|'
    r'^[ \t]*-*[ \t]*tiền\s+sử\s+bệnh\s+hiện\s+tại\b|'
    r'^tiền\s+sử\s+bệnh\s+(?=\s)(?!\S)',
    re.IGNORECASE | re.MULTILINE | re.DOTALL
)

test_cases = [
    ("Tiền sử bệnh\n    Thuốc trước khi", "Section 1 header (newline after bệnh)"),
    ("Tiền sử bệnh hiện tại\n    Lý do", "Section 2 header (space after bệnh)"),
    ("Bố có tiền sử bệnh tim.", "Body text with tiền sử"),
    ("Tiền sử bệnh tiểu đường", "Body: tiền sử bệnh followed by letter"),
    ("1.  Tiền sử bệnh", "With number prefix"),
    ("Tiền sử bệnh ", "With trailing space"),
    ("Tiền sử bệnh\n", "With trailing newline"),
]

results = {}
for text, desc in test_cases:
    m = PAT.search(text.lower())
    results[desc] = {
        "text": repr(text),
        "matched": bool(m),
        "match": repr(m.group()) if m else None,
        "match_pos": (m.start(), m.end()) if m else None
    }

# Also test each alternative individually on the section header
alt3 = re.compile(r'^tiền\s+sử\s+bệnh\s+(?=\s)(?!\S)', re.IGNORECASE | re.MULTILINE | re.DOTALL)
for text, desc in test_cases[:4]:
    m = alt3.search(text.lower())
    results[f"alt3_only_{desc[:20]}"] = {
        "matched": bool(m),
        "match": repr(m.group()) if m else None
    }

# Check: after "Tiền sử bệnh\n" in lowercase, what's at position 12?
s = "tiền sử bệnh\n    thuốc"
results["char_after_benh"] = {
    "pos_12": repr(s[12:15]),
    "ord_12": ord(s[12]),
    "is_space_12": s[12].isspace(),
    "is_S_12": not s[12].isspace()
}

# The key question: (?!\S) means "NOT followed by non-whitespace"
# After "bệnh" in "tiền sử bệnh\n": next char is \n = whitespace
# So (?!\S) should PASS (nothing follows that's non-whitespace) → match succeeds
# But wait: (?=\s) means "followed by whitespace" — yes, \n is whitespace
# So (?=\s)(?!\S) should match at "tiền sử bệnh\n"
results["logic_check"] = {
    "newline_is_space": "\n".isspace(),
    "newline_not_S": not "\n".isspace(),
    "should_match": "\n".isspace() and not "\n".isspace()
}

with open("_dbg_pat.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Done")
