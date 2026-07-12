import sys, json
sys.path.insert(0, '.')
text = open('src/entity/disease_extractor.py', encoding='utf-8').read()
results = []
for i, line in enumerate(text.split('\n'), 1):
    if 'hen suy' in line or 'huy' in line.lower() and i >= 55 and i <= 75:
        results.append({'line': i, 'text': repr(line)})
with open('_debug_lines.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
