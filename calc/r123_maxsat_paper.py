"""Max-3-SAT spectrum: main-term threshold of the recursion AS WRITTEN (column (P) of tab:p2-density: paper's schedule,
tau 1, x0 18, q = 9, brute-force base, the paper's U < 2km) against the Max-3-SAT bar min{log2 1.2886, 1/Delta}/6
(eq:p2-bar-maxsat with 6 in place of 6 - 2/Delta). Gives the 'from' value for the Max-3-SAT threshold 2^1016."""
import sys, math
sys.path.insert(0, 'calc')
from r123_levers import run, LEVERS

L = 'paper: schedule 2^{-l-5}, tau 1, x0 18'
with open('calc/r123_maxsat_paper_out.txt', 'w') as out:
    for D in [1.0, 2.0, 3.0, 4.26]:
        bar = min(math.log2(1.2886), 1 / D) / 6
        kw = dict(dict(LEVERS)[L]); kw['x0'] = 18
        r = run(bar, **kw)
        out.write(f'density {D}: bar {bar:.4f}; {L}: levels {r[0]}, log2 K = {r[1]:.0f}; log2 m* >= {9 * r[1]:.0f} (q = 9)\n')
