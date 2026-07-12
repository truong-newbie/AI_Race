import sys, csv, json
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

errors = list(csv.DictReader(open('outputs/errors.csv', encoding='utf-8')))
missed = [e for e in errors if e['error_type'] == 'missed_entity']
fp = [e for e in errors if e['error_type'] == 'false_positive']

# Group missed by entity type
from collections import Counter
by_type = Counter(e['entity_text'] for e in missed)
print("=== MISSED ENTITY frequency ===")
for text, cnt in by_type.most_common(50):
    types = Counter(e['entity_type'] for e in missed if e['entity_text'] == text)
    print("  [%d] '%s' (%s)" % (cnt, text, dict(types)))

print()
print("=== FALSE POSITIVE frequency ===")
by_fp = Counter(e['entity_text'] for e in fp)
for text, cnt in by_fp.most_common(30):
    types = Counter(e['entity_type'] for e in fp if e['entity_text'] == text)
    print("  [%d] '%s' (%s)" % (cnt, text, dict(types)))
