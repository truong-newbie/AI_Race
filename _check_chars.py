import sys, json
sys.path.insert(0, '.')
text = open('src/entity/disease_extractor.py', encoding='utf-8').read()
results = []
for i, line in enumerate(text.split('\n'), 1):
    if ('viêm ph' in line or 'viêm gan' in line or
        'viêm dạ' in line or 'hen suy' in line or
        'tăng huyết' in line or 'nhiễm trùng' in line or
        'viêm mũi' in line or 'viêm phổi cộng' in line or
        'nhiễm trùng tiết' in line):
        results.append({'line': i, 'text': line.rstrip()})

# Also check what characters are in specific patterns
patterns_to_check = ['hen suyễn', 'viêm phổi', 'tăng huyết áp', 'nhiễm trùng']
for p in patterns_to_check:
    chars = [(c, hex(ord(c)), repr(c)) for c in p]
    results.append({'pattern': p, 'chars': chars})

with open('_debug_chars.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
