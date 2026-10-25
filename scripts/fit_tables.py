#!/usr/bin/env python
"""Wrap every tabular of the paper in an adjustbox with max width = column (table) or page (table*) width.
Idempotent. Usage: python scripts/fit_tables.py paper/article paper/article_fr"""
import re, sys
from pathlib import Path

def fix(text):
    text = re.sub(r'\\resizebox\{\\linewidth\}\{!\}\{%\n(\\begin\{tabular\})', r'\1', text)
    text = text.replace('\\end{tabular}}', '\\end{tabular}')
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', text, re.S):
        block = m.group(0)
        if 'adjustbox' not in block:
            width = '\\textwidth' if m.group(1) == 'table*' else '\\linewidth'
            block = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})',
                           lambda t: f'\\begin{{adjustbox}}{{max width={width}}}\n{t.group(1)}\n\\end{{adjustbox}}', block, flags=re.S)
        out.append(text[pos:m.start()]); out.append(block); pos = m.end()
    out.append(text[pos:])
    return ''.join(out)

for d in sys.argv[1:]:
    d = Path(d)
    main = d / 'main.tex'
    t = main.read_text()
    if 'adjustbox' not in t:
        main.write_text(t.replace('\\usepackage{booktabs}', '\\usepackage{booktabs}\n\\usepackage{adjustbox}', 1))
    for f in list((d / 'sections').glob('*.tex')) + list((d / 'appendices').glob('*.tex')):
        f.write_text(fix(f.read_text()))
