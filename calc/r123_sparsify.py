"""Tier-2 threshold of calc/r123_general.py with other sparsification densities Delta(eps) (Table 7, Proposition 3.15).

Imports r123_general (unchanged) and replaces its delta(eps, crude) by each bound below, clamped to Delta >= 8.
Delta = largest number of clauses one variable occurs in (Theorem D's Delta), k = 3.
  calabro_paper     Anthropic's paper, from [Cal09 (5.4)]: 2 (2 + 8192 eps^-2 lg^2(128/eps)) + 2      (counting)
  calabro_exact_cnt Calabro's own lemmas without his last relaxations (own arithmetic):
                    alpha = least alpha >= 2 with 16 alpha h(1/(8 alpha^2)) <= eps (Lemmas 5.11-5.13 with
                    beta_1 = 2, beta_2 = 16 alpha); a literal occurs <= theta_0 + theta_1 + theta_2 = 2 + 2 alpha + 16 alpha^2
                    times (Lemma 5.6.1); Delta = 2 (2 + 2 alpha + 16 alpha^2) + 2 (the +2: N / DHMTW unit clauses) (counting)
  calabro_exact_dec same without the +2                                                                   (decision)
  floor_log         Delta = lg(1/eps): heuristic floor from a block family (not a theorem)
  ideal_inv_eps     Delta = 1/eps (hypothetical)
  const_8           Delta = 8 for every eps (the clamp)
  const_2_noclamp   Delta = 2: Theorem D takes Delta >= 2 (Anthropic's paper), so no sparsifier does better inside it
eps optimised over e 2^{j/4}, j = -40..59 (r123_general uses j >= 0, i.e. eps >= e; the wider grid matters only
when Delta grows slowly)."""
import math, json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r123_general as G
from mpmath import mp, loggamma, mpf

mp.dps = 50
LN2 = math.log(2)


def lgamma2_hp(x):
    """lg Gamma(x) in 50-digit arithmetic. r123_general.lgamma2 is a double: for m = Delta n ~ 1e25 the difference
    lgamma2(m + 1) - lgamma2(m - n_bar) loses most digits (at eps = 3.56e-4 it gives lg C(m, n_bar + 1)/n = 1.65
    instead of 1.13; at eps = 1e-4 it gives 0)."""
    return loggamma(mpf(x))


def U_per_n_hp(eps, tier, n=1e12):
    """r123_general.U_per_n for tier 2 with the binomial term computed in high precision; otherwise identical."""
    assert tier == 2
    k = G.k
    D = G.delta(eps, False)
    nb = n / math.log2(2 * D)
    m = D * n
    beta = n + float((lgamma2_hp(mpf(m) + 1) - lgamma2_hp(mpf(nb) + 2) - lgamma2_hp(mpf(m) - mpf(nb))) / mp.log(2)) + 1
    rbar = beta / 1.0
    M = rbar + beta - 1
    sbar = 16 * m / nb
    kind0 = 2 * (k - 1) * (nb + 1) * math.log2(math.e * m / (nb + 1))
    kind1 = 2 * k * M * math.log2(math.e * rbar * sbar / M)
    return (kind0 + kind1) / n


def h2(x):
    return 0.0 if x <= 0 or x >= 1 else -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def alpha_exact(eps):
    f = lambda a: 16 * a * h2(1 / (8 * a * a))
    lo, hi = 2.0, 2.0
    if f(lo) <= eps:
        return lo
    while f(hi) > eps:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > eps else (lo, mid)
    return hi


def lit_exact(eps):
    a = alpha_exact(eps)
    return 2 + 2 * a + 16 * a * a


BOUNDS = {
    'calabro_paper': lambda e: 2 * (2 + 8192 * e ** -2 * math.log2(128 / e) ** 2) + 2,
    'calabro_exact_cnt': lambda e: 2 * lit_exact(e) + 2,
    'calabro_exact_dec': lambda e: 2 * lit_exact(e),
    'floor_log': lambda e: math.log2(1 / e),
    'ideal_inv_eps': lambda e: 1 / e,
    'const_8': lambda e: 8.0,
    'const_2_noclamp': lambda e: 2.0,
}


def use(name):
    f = BOUNDS[name]
    lo = 2.0 if name.endswith('_noclamp') else 8.0
    G.delta = lambda eps, crude: max(lo, f(eps))


MODE = {'U': U_per_n_hp, 'jmin': -40}


def exponent2(e):
    best = None
    for j in range(MODE['jmin'], 60):
        eps = e * 2 ** (j / 4)
        v = eps + e * MODE['U'](eps, 2)
        if best is None or v < best[0]:
            best = (v, eps)
    return best


def e_star2(bar):
    lo, hi = 1e-12, 0.5
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if exponent2(mid)[0] < bar else (lo, mid)
    return lo


BARS = [('deterministic 3-SAT (Liu) 0.4092', 0.4092), ('#3-SAT (Kutzkov) 0.7157', 0.7157)]
PARAMS = ((2 ** -0.5, 18, 17), (2 ** -0.5, 7, 2.2), (0.5, 7, 2.2))
# runs: (label, U function, jmin, bounds)
RUNS = [('A: r123_general as is (double-precision U, eps >= e), Calabro as quoted: reproduces r123_general_out.txt',
         G.U_per_n, 0, ['calabro_paper']),
        ('B: high-precision U, eps >= e (the paper takes eps in [e, 2e])', U_per_n_hp, 0, list(BOUNDS)),
        ('C: high-precision U, eps optimised over e 2^{j/4}, j >= -40 (eps < e allowed)', U_per_n_hp, -40, list(BOUNDS))]

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    rows = []
    with open(os.path.join(here, 'r123_sparsify_out.txt'), 'w') as out:
        out.write('eps, alpha_exact, alpha_Calabro = 8/eps lg(128/eps), lg Delta: paper / exact_cnt\n')
        for eps in (1e-2, 1e-3, 3.56e-4, 1e-4):
            out.write(f'  {eps:.2e}  {alpha_exact(eps):9.1f}  {8 / eps * math.log2(128 / eps):9.1f}  '
                      f'{math.log2(BOUNDS["calabro_paper"](eps)):5.1f} / {math.log2(BOUNDS["calabro_exact_cnt"](eps)):5.1f}\n')
        out.write('lg C(m, n_bar+1)/n at Calabro-as-quoted Delta, double vs 50 digits (n = 1e12):\n')
        for eps in (1e-2, 1e-3, 3.56e-4, 1e-4):
            use('calabro_paper')
            D = G.delta(eps, False); n = 1e12; nb = n / math.log2(2 * D); m = D * n
            dbl = (G.lgamma2(m + 1) - G.lgamma2(nb + 2) - G.lgamma2(m - nb)) / n
            hp = float((lgamma2_hp(mpf(m) + 1) - lgamma2_hp(mpf(nb) + 2) - lgamma2_hp(mpf(m) - mpf(nb))) / mp.log(2)) / n
            out.write(f'  eps {eps:.2e}: lg Delta {math.log2(D):5.1f}, double {dbl:.4f}, 50-digit {hp:.4f}\n')
        for label, Uf, jmin, names in RUNS:
            MODE['U'], MODE['jmin'] = Uf, jmin
            for bname, bar in BARS:
                out.write(f'== {label}\n   bar {bname}\n')
                for name in names:
                    use(name)
                    es = e_star2(bar)
                    _, eps_opt = exponent2(es)
                    D = G.delta(eps_opt, False)
                    Un = Uf(eps_opt, 2)
                    for rho, x0, q in PARAMS:
                        l, xK, ln = G.threshold(es, rho, x0, q)
                        out.write(f'  {name:18s} e* = {es:.2e}, eps = {eps_opt:.2e}, lg Delta = {math.log2(D):5.1f}, U/n = {Un:7.0f};'
                                  f' rho = {rho:.3f}, x0 = {x0}, q = {q}: level {l}, lg K = {xK:.2e}, lg n* >= {ln:.2e}\n')
                        rows.append(dict(run=label[0], bar=bar, bound=name, e_star=es, eps=eps_opt, log2Delta=math.log2(D),
                                         U_per_n=Un, rho=rho, x0=x0, q=q, level=l, log2K=xK, log2n=ln))
    json.dump(rows, open(os.path.join(here, 'r123_sparsify_out.json'), 'w'), indent=1)
    print(open(os.path.join(here, 'r123_sparsify_out.txt')).read())
