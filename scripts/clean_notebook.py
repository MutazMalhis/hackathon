"""Git clean filter: remove notebook execution outputs without editing the local file."""
import json
import sys

nb = json.load(sys.stdin)
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    cell.get('metadata', {}).pop('execution', None)
nb.get('metadata', {}).pop('widgets', None)
json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write('\n')
