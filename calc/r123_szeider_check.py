"""Check against Szeider's evaluation of Anthropic's recursion (Zenodo, doi:10.5281/zenodo.23688832, addendum):
the baseline model of r123_baseline.py with exhaustive search 2^n as the competitor.

For m = Delta n the main term beats 2^n once e_l < 1 / (2 k Delta), k = 3.  At Delta = 5 Szeider finds 2^275616
clauses; at Delta = 1 the paper of Anthropic remarks log m > 8000.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r123_baseline import threshold

lines = []
for delta in (1, 5):
    bar = 1 / (2 * 3 * delta)
    l, x, e, lm = threshold(bar)
    lines.append(f'exhaustive search, m = {delta} n: bar e < {bar:.4f}; level {l}, log2 K = {x:.0f}, '
                 f'e = {e:.4f}, log2 m* >= {lm:.0f}')
out = '\n'.join(lines)
print(out)
with open(__file__.replace('.py', '_out.txt'), 'w') as f:
    f.write(out + '\n')
