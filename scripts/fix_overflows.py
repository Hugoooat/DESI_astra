#!/usr/bin/env python
"""Avoid overfull lines in the two-column paper: long \\texttt paths -> \\url (breakable, xurl),
emulator-loss equation on two lines. Idempotent. Usage: python scripts/fix_overflows.py paper/article paper/article_fr"""
import re, sys
from pathlib import Path
for d in map(Path, sys.argv[1:]):
    m = d / 'main.tex'; t = m.read_text()
    if 'xurl' not in t:
        m.write_text(t.replace('\\usepackage{hyperref}', '\\usepackage{hyperref}\n\\usepackage{xurl}', 1))
    for f in d.glob('appendices/*.tex'):
        t = f.read_text()
        def to_url(mm):
            s = mm.group(1)
            return '\\url{' + s.replace('\\_', '_').replace('\\{', '{').replace('\\}', '}') + '}'
        t = re.sub(r'\\texttt\{((?:/|\.\.\./)[^{}]*(?:\\\{[^{}]*\\\}[^{}]*)*)\}', to_url, t)
        t = re.sub(r'\\begin\{equation\}\n(\\mathcal L = \\Big\\langle .*?\\Big\\rangle) (\+ \\lambda .*?)\n\\end\{equation\}',
                   r'\\begin{multline}\n\1 \\\\\n\2\n\\end{multline}', t, flags=re.S)
        f.write_text(t)
