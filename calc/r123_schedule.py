"""Remark 3.3 of the paper: the closed-form slack schedule zeta_l = c 2^{-l} (Lagrange: early slack is doubled by every
later level) against the dynamic-programming optimum of calc/r123_levers.py. Same model (main term, Lemma 8 (ii)
s >= ln(2 s K^2)/(2 zeta^2), e' = (rho + 3 zeta) e + 3/K, x0 = 7). Scans c; reports the best c per bar."""
import math, sys
sys.path.insert(0, 'calc')
from r123_levers import s_min, run

def closed(bar, c, rho=2 ** -0.5, x0=7, L=14):
    x, e = float(x0), 0.5
    for l in range(L):
        if e < bar:
            return l, x
        z = min(0.5, c * 2.0 ** -l)
        s = s_min(x, z, 1.0)
        e = (rho + 3 * z) * e + 3 * 2.0 ** -x
        x = 2 * (x + math.log2(s))
    return None

with open('calc/r123_schedule_out.txt', 'w') as out:
    for name, bar in (('density 1 (0.0881)', 0.5283 / 6), ('density 2 (0.0592)', 0.711 / 12), ('density 4.26 (0.0278)', 0.711 / (6 * 4.26))):
        best = min((r[1], c, r[0]) for c in [2 ** (-j / 4) for j in range(0, 60)] for r in [closed(bar, c)] if r)
        dp = run(bar, x0=7)
        out.write(f'{name}: closed form c = {best[1]:.4f}: levels {best[2]}, log2 K = {best[0]:.0f}; DP: levels {dp[0]}, log2 K = {dp[1]:.0f}\n')
        out.flush()
print(open('calc/r123_schedule_out.txt').read())
