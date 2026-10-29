"""Parametric model of Anthropic's recursion (Theorem A, (40), Lemma 8 case (ii)) with every lever free
(Section 3.4, Table 7 and Appendix B.1 of the paper).

Normalisation of the paper's Sections 3-8: e_0 = 1/2 (brute force), main term 2 e k m, bar e < c / (2 k Delta).
Level step (paper (40)/(42)):  g_l = s_l K_l,  K_{l+1} = g_l^2,  e_{l+1} = (rho + a_z zeta_l) e_l + a_K / K_l.
Balancing (Lemma 8 (ii)): 2 tau zeta^2 g >= K ln(2 g K), i.e. s >= ln(2 s K^2) / (2 tau zeta^2), where tau = 1 is
Hoeffding with the worst-case range of the edge weights (the paper) and tau > 1 a sharper tail (e.g. tau = 9 if the
weights lie in a band of width 1/3 of the maximum; tau -> the Cramer/KL rate is the Legendre lever). The paper also
needs 1/g <= zeta. Its schedule zeta_l = 2^{-l-5} and s_l = 2^{3l+x0} satisfy this for x0 >= 18.
Top level: K_l <= m^{1/q}; q = 9 (Chebyshev, the paper's proof of (c)); q ~ 2 if Lemma 8 (ii) is used at the top.
The free schedule: zeta_l chosen per level from a grid by dynamic programming over levels (minimise the final log K
that reaches the bar). log2 m* = q log2 K_final, main term only (a lower bound on the true threshold)."""
import math, json, sys


def s_min(xK, zeta, tau):
    """least s with s >= ln(2 s K^2) / (2 tau zeta^2), K = 2^xK; also s K >= 1/zeta."""
    s = 1.0
    for _ in range(100):
        s = max(1.0, (math.log(2) * (1 + math.log2(s) + 2 * xK)) / (2 * tau * zeta ** 2))
    return max(s, 2.0 ** -xK / zeta)


def run(bar, rho=2 ** -0.5, x0=18, tau=1.0, az=3.0, aK=3.0, schedule='opt', e0=0.5, L=14):
    """return (levels, log2 K_final, e_final) of the cheapest schedule, or None."""
    if schedule == 'paper':
        x, e = float(x0), e0
        for l in range(L):
            if e < bar:
                return l, x, e
            s = 2.0 ** (3 * l + x0)
            zeta = 2.0 ** (-l - 5)
            e = (rho + az * zeta) * e + aK * 2.0 ** -x
            x = 2 * (x + math.log2(s))
        return None
    # dynamic programming: states (x, e) after each level, keep Pareto front (smaller x and smaller e both good)
    zetas = [2.0 ** (-j / 2) for j in range(4, 36)]
    front = [(float(x0), e0)]
    for l in range(L):
        done = [st for st in front if st[1] < bar]
        if done:
            return l, min(st[0] for st in done), min(st[1] for st in done if st[0] == min(d[0] for d in done))
        new = []
        for x, e in front:
            for z in zetas:
                s = s_min(x, z, tau)
                new.append((2 * (x + math.log2(s)), (rho + az * z) * e + aK * 2.0 ** -x))
        new.sort()
        front, best_e = [], float('inf')
        for x, e in new:                       # Pareto filter
            if e < best_e - 1e-12:
                front.append((x, e)); best_e = e
    return None


CLASSES = [('<= 3 occurrences, c = 1/3', 1 / 3, 1.0), ('counting Delta = 1, c = 0.52', 0.52, 1.0),
           ('counting Delta = 4.26, c = 0.716', 0.716, 4.26)]
LEVERS = [
    ('paper: schedule 2^{-l-5}, tau 1, x0 18', dict(schedule='paper')),
    ('free slack schedule (DP), tau 1, x0 18', dict()),
    ('free schedule, tau 1, best x0', dict(x0='best')),
    ('free schedule, tau 9 (band 1/3), best x0', dict(x0='best', tau=9.0)),
    ('free schedule, tau 9, best x0, overheads a = 1', dict(x0='best', tau=9.0, az=1.0, aK=1.0)),
    ('free schedule, tau 9, best x0, rho = 1/2 (ideal pieces)', dict(x0='best', tau=9.0, rho=0.5)),
    ('no balancing slack (s = 1, ideal), best x0', dict(x0='best', tau=1e12)),
]

def job(j):
    """one (class, lever, x0) run; x0 None means the lever's own x0."""
    c, delta, lname, x0 = j['c'], j['delta'], j['lever'], j['x0']
    kw = dict(dict(LEVERS)[lname])
    kw['x0'] = 18 if x0 is None else x0
    r = run(c / (2 * 3 * delta), **kw)
    return None if r is None else {'levels': r[0], 'log2K': r[1], 'e': r[2]}


if __name__ == '__main__':
    sys.path.insert(0, 'calc')
    from resumable import run_jobs
    jobs = []
    for cname, c, delta in CLASSES:
        for lname, kw in LEVERS:
            for x0 in (range(3, 25) if kw.get('x0') == 'best' else [None]):
                jobs.append({'c': c, 'delta': delta, 'lever': lname, 'x0': x0})
    res = run_jobs(jobs, job, 'calc/r123_levers_jobs.jsonl', processes=2)
    out = open('calc/r123_levers_out.txt', 'w')
    for cname, c, delta in CLASSES:
        out.write(f'\n== {cname}: bar e < {c / (6 * delta):.4f}\n')
        for lname, kw in LEVERS:
            cand = [(j['x0'], res[json.dumps(j, sort_keys=True)]) for j in jobs
                    if j['c'] == c and j['delta'] == delta and j['lever'] == lname]
            cand = [(x0, r) for x0, r in cand if r]
            if not cand:
                out.write(f'  {lname:58s} not reached\n'); continue
            x0, r = min(cand, key=lambda t: t[1]['log2K'])
            out.write(f'  {lname:58s} x0 {x0 if x0 is not None else 18:2d}: levels {r["levels"]}, '
                      f'log2 K = {r["log2K"]:8.0f}, e = {r["e"]:.4f}; log2 m* >= {9 * r["log2K"]:9.0f} (q = 9), '
                      f'{2 * r["log2K"]:8.0f} (q = 2)\n')
    out.close()
