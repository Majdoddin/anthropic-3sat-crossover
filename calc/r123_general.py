"""General 3-SAT via Theorem D: the tiers of Table 5 and Section 3.10 of the paper.

Main term of log2 T(n): eps n + e U (U in the paper's units, exponent e U as 2ekm in Theorem A(c)); n-network of
Theorem D Steps 1-4 on the sparsified outputs (density Delta), Anthropic's paper (60), (61), (67), (68).
Delta(eps), k = 3: crude (Anthropic's paper, Theorem D) 2 (8k/eps)^{3k} + 2; Calabro (as quoted there) 2 (2 + 8192 eps^-2
log2^2(128/eps)) + 2. Tiers:
  0  paper's inequalities: eps = e, crude Delta, U/n = 6k(1 + log2(2D)/n) + 28k log2(96 D log2(2D))  ((68))
  1  same construction counted sharply: eps = e, Calabro Delta, n_bar = n / log2(2D), m = D n,
     beta = n + log2 C(m, n_bar+1) + 1, r_bar = beta / log2(4/3) (kill rate 1/4), M = r_bar + beta - 1,
     s_bar = 16 m / n_bar; kind 0: 2 (k-1) (n_bar+1) log2(e m/(n_bar+1)); kind 1: 2 k M log2(e r_bar s_bar / M)
  2  tier 1 plus eps optimised and kill rate 1/2 (r_bar = beta / 1)
Level count: e_l <= 0.85 rho^l (Theorem A(a)), log2 K_l = (3 x0 + 6) 2^l - 6l - 2 x0 - 6 ((40)); log2 n* = q log2 K_l,
q = 17 (Theorem D as written) or 2.2 (Hoeffding at the top, an estimate). Main term only: lower bounds."""
import math, json

k = 3
LOG2E = math.log2(math.e)


def lgamma2(x):
    return math.lgamma(x) / math.log(2)


def delta(eps, crude):
    return 2 * (8 * k / eps) ** (3 * k) + 2 if crude else 2 * (2 + 8192 * eps ** -2 * math.log2(128 / eps) ** 2) + 2


def U_per_n(eps, tier, n=1e12):
    D = delta(eps, tier == 0)
    if tier == 0:
        return 6 * k * (1 + math.log2(2 * D) / n) + 28 * k * math.log2(96 * D * math.log2(2 * D))
    nb = n / math.log2(2 * D)
    m = D * n
    beta = n + (lgamma2(m + 1) - lgamma2(nb + 2) - lgamma2(m - nb)) + 1
    rbar = beta / (math.log2(4 / 3) if tier == 1 else 1.0)
    M = rbar + beta - 1
    sbar = 16 * m / nb
    kind0 = 2 * (k - 1) * (nb + 1) * math.log2(math.e * m / (nb + 1))
    kind1 = 2 * k * M * math.log2(math.e * rbar * sbar / M)
    return (kind0 + kind1) / n


def exponent(e, tier):
    """eps n + e U, per n; eps = e for tiers 0, 1, optimised for tier 2."""
    if tier < 2:
        return e + e * U_per_n(e, tier)
    return min(eps + e * U_per_n(eps, tier) for eps in (e * 2 ** (j / 4) for j in range(0, 60)))


def e_star(bar, tier):
    lo, hi = 1e-9, 0.5
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if exponent(mid, tier) < bar else (lo, mid)
    return lo


def threshold(es, rho, x0, q):
    l = 0
    while 0.85 * rho ** l > es:
        l += 1
    xK = (3 * x0 + 6) * 2 ** l - 6 * l - 2 * x0 - 6
    return l, xK, q * xK


BARS = [('deterministic 3-SAT (Liu) 0.4092', 0.4092), ('#3-SAT (Kutzkov) 0.7157', 0.7157),
        ('randomized 3-SAT (Jiang-Cai) 0.3865', 0.3865)]

if __name__ == '__main__':
    rows = []
    with open('calc/r123_general_out.txt', 'w') as out:
        for bname, bar in BARS:
            out.write(f'== bar {bname}\n')
            for tier in (0, 1, 2):
                es = e_star(bar, tier)
                U = U_per_n(es, tier)
                for rho, x0, q in ((2 ** -0.5, 18, 17), (2 ** -0.5, 7, 2.2), (0.5, 7, 2.2)):
                    l, xK, ln = threshold(es, rho, x0, q)
                    out.write(f'  tier {tier}: e* = {es:.2e}, U/n = {U:7.0f} (log2 Delta = {math.log2(delta(es, tier == 0)):5.1f});'
                              f' rho = {rho:.3f}, x0 = {x0}, q = {q}: level {l}, log2 K = {xK:.2e}, log2 n* >= {ln:.2e}\n')
                    rows.append(dict(bar=bar, tier=tier, e_star=es, U_per_n=U, rho=rho, x0=x0, q=q, level=l, log2K=xK, log2n=ln))
    json.dump(rows, open('calc/r123_general_out.json', 'w'), indent=1)
    print(open('calc/r123_general_out.txt').read())
