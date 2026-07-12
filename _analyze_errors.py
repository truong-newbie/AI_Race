import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import csv
from collections import Counter

errors = list(csv.DictReader(open('outputs/errors.csv', encoding='utf-8')))
missed = [e for e in errors if e['error_type'] == 'missed_entity']
fp = [e for e in errors if e['error_type'] == 'false_positive']
wt = [e for e in errors if e['error_type'] == 'wrong_type']

print('missed_entity:', len(missed))
for t, c in Counter(e['entity_type'] for e in missed).most_common():
    print('  %s: %d' % (t, c))
print('  Samples:')
for e in missed[:8]:
    print('    [%s] "%s"' % (e['entity_type'], e['entity_text']))

print()
print('false_positive:', len(fp))
for t, c in Counter(e['entity_type'] for e in fp).most_common():
    print('  %s: %d' % (t, c))
print('  Samples:')
for e in fp[:8]:
    print('    [%s] "%s"' % (e['entity_type'], e['entity_text']))

print()
print('wrong_type:', len(wt))
for e in wt[:10]:
    print('  [%s] text="%s"' % (e['entity_type'], e['entity_text']))

print()
print('=== E2E sources ===')
sources = Counter(e['source'] for e in errors)
for s, c in sources.most_common(5):
    print('  %s: %d' % (s, c))
