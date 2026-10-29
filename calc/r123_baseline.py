"""Baseline threshold of Anthropic's #k-SAT algorithm (Anthropic's paper, Theorem A, (40), (45)); Section 3.3 and
Table 7 of the paper.

Main term only: the log of the running time is 2 e_l k m + (lower-order terms whose constants depend on the unknown
constant c of Theorem B). Ignoring them gives a LOWER bound on the true threshold n*.

Recursion (40): K_0 = 2^x0, zeta_l = 2^{-l-5}, g_l = 2^{3l+x0} K_l, K_{l+1} = g_l^2,
                e_0 = 1/2, e_{l+1} = (rho + 3 zeta_l) e_l + 3/K_l.
Top level (proof of (c)): the largest l with K_l <= m^{1/q}, q = 9 (g = 3K^8 <= m, Chebyshev balancing).
Applicable from m* := the least m whose level l has 2 e_l k m < c n, i.e. e_l < c / (2 k Delta) for m = Delta n.

Levers, each a hypothetical change of one constant (not a proved construction):
  q      top-level balancing (9 = Chebyshev; ~2 would need g ~ K polylog at the top)
  x0     base size (18 = the union-bound requirement 1.4 (3 x0 + 6) <= 2^{x0-9})
  rho    per-level factor (1/sqrt2 = App. B pieces; 11/12 = Lemma 2 pieces)
  growth 'square' = (40); ('const', beta) = K_{l+1} = beta K_l with the same e-recursion (unequal-weights question open)
"""
import math, json, sys


def levels(rho, x0, growth='square', L=40):
    """Yield (l, log2 K_l, e_l)."""
    x, e = float(x0), 0.5
    for l in range(L):
        yield l, x, e
        zeta = 2.0 ** (-l - 5)
        if growth == 'square':
            xn = 2 * x + 6 * l + 2 * x0          # log K_{l+1} = 2 (3l + x0 + log K_l)
        else:
            xn = x + math.log2(growth[1])
        e = (rho + 3 * zeta) * e + 3 * 2.0 ** (-x)
        x = xn


def threshold(bar, rho=2 ** -0.5, x0=18, q=9, growth='square'):
    """least log2 m at which the main term beats the bar: (l, log2 K_l, e_l, log2 m*)."""
    for l, x, e in levels(rho, x0, growth):
        if e < bar:
            return l, x, e, q * x
    return None


# class bars e < c / (2 k Delta), k = 3. Competitor exponents c are recalled values, not checked here.
CLASSES = [
    ('brute force, Delta = 1 (paper\'s own remark)', 1.0, 1.0),
    ('<= 3 occurrences (m <= n), pathwidth DP 2^{n/3}', 1 / 3, 1.0),
    ('counting, Delta = 1, pathwidth DP 2^{0.52 n}', 0.52, 1.0),
    ('decision, Delta = 1, 2^{0.268 m}', 0.268, 1.0),
    ('counting, Delta = 4.26, Kutzkov 2^{0.716 n}', 0.716, 4.26),
]

LEVERS = [
    ('baseline (40): rho = 1/sqrt2, x0 = 18, q = 9', {}),
    ('q = 2 (top balancing g ~ K)', {'q': 2}),
    ('x0 = 6', {'x0': 6}),
    ('rho = 0.6', {'rho': 0.6}),
    ('rho = 0.5', {'rho': 0.5}),
    ('constant growth beta = 16', {'growth': ('const', 16)}),
    ('constant growth beta = 16, x0 = 6, q = 2', {'growth': ('const', 16), 'x0': 6, 'q': 2}),
]

if __name__ == '__main__':
    rows = []
    for cname, c, delta in CLASSES:
        bar = c / (2 * 3 * delta)
        print(f'\n== {cname}: bar e < {bar:.4f}')
        for lname, kw in LEVERS:
            t = threshold(bar, **kw)
            if t is None:
                print(f'  {lname:52s} not reached in 40 levels')
                continue
            l, x, e, lm = t
            print(f'  {lname:52s} level {l:2d}, log2 K = {x:9.0f}, e = {e:.4f}, log2 m* >= {lm:11.0f}')
            rows.append({'class': cname, 'bar': bar, 'lever': lname, 'level': l, 'log2K': x, 'e': e, 'log2m': lm})
    out = sys.argv[1] if len(sys.argv) > 1 else 'calc/r123_baseline_out.json'
    with open(out, 'w') as f:
        json.dump(rows, f, indent=1)
