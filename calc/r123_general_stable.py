# Stable log-binomial recomputation of r123_general.py tiers 1-2 (rows G1, G4 of Table 5); supersedes r123_general_out.txt tier 1.
# Recompute r123_general.py tiers 1 and 2 with a numerically stable log-binomial (lgamma(m) for m ~ 1e26 loses the difference).
import math
k = 3
def delta(eps, crude):
    return 2 * (8 * k / eps) ** (3 * k) + 2 if crude else 2 * (2 + 8192 * eps ** -2 * math.log2(128 / eps) ** 2) + 2
def log2binom(m, r):
    # log2 C(m, r) for r << m: sum_{j<r} log2(m-j) - log2(r!) ; use (r) log2 m - r(r-1)/(2 m ln2) - lgamma(r+1)/ln2
    return r * math.log2(m) - r * (r - 1) / (2 * m * math.log(2)) - math.lgamma(r + 1) / math.log(2)
def U_per_n(eps, tier, n=1e12, stable=True):
    D = delta(eps, tier == 0)
    if tier == 0:
        return 6 * k * (1 + math.log2(2 * D) / n) + 28 * k * math.log2(96 * D * math.log2(2 * D))
    nb = n / math.log2(2 * D); m = D * n
    if stable:
        lb = log2binom(m, nb + 1)
    else:
        lg = lambda x: math.lgamma(x) / math.log(2)
        lb = lg(m + 1) - lg(nb + 2) - lg(m - nb)
    beta = n + lb + 1
    rbar = beta / (math.log2(4 / 3) if tier == 1 else 1.0)
    M = rbar + beta - 1
    sbar = 16 * m / nb
    kind0 = 2 * (k - 1) * (nb + 1) * math.log2(math.e * m / (nb + 1))
    kind1 = 2 * k * M * math.log2(math.e * rbar * sbar / M)
    return (kind0 + kind1) / n, beta / n, M / n, math.log2(D)
def exponent(e, tier, stable=True):
    if tier < 2:
        return e + e * U_per_n(e, tier, stable=stable)[0]
    return min(eps + e * U_per_n(eps, tier, stable=stable)[0] for eps in (e * 2 ** (j / 4) for j in range(0, 60)))
def e_star(bar, tier, stable=True):
    lo, hi = 1e-9, 0.5
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if exponent(mid, tier, stable) < bar else (lo, mid)
    return lo
def threshold(es, rho, x0, q):
    l = 0
    while 0.85 * rho ** l > es: l += 1
    xK = (3 * x0 + 6) * 2 ** l - 6 * l - 2 * x0 - 6
    return l, xK, q * xK
for stable in (False, True):
    print('--- stable' if stable else '--- as in script (unstable lgamma)')
    for tier in (1, 2):
        es = e_star(0.4092, tier, stable)
        U, b, M, lD = U_per_n(es, tier, stable=stable)
        print(f'tier {tier}: e* = {es:.3e}, U/n = {U:.0f}, beta/n = {b:.2f}, M/n = {M:.2f}, log2 Delta = {lD:.1f}, exponent(e*) = {exponent(es,tier,stable):.4f}')
        for rho, x0, q in ((2**-0.5, 18, 17), (2**-0.5, 7, 2.2), (0.5, 7, 2.2)):
            l, xK, ln = threshold(es, rho, x0, q)
            print(f'    rho {rho:.3f} x0 {x0} q {q}: level {l}, log2 K = {xK:.3e}, log2 n* = {ln:.3e}')
# monotonicity check of the unstable exponent near e*
print('--- unstable exponent(e,1) sampled near 1.44e-4:')
for e in (1.2e-4, 1.3e-4, 1.4e-4, 1.44e-4, 1.45e-4, 1.5e-4, 1.6e-4, 1.7e-4):
    print(f'  e={e:.2e}: unstable {exponent(e,1,False):.4f}  stable {exponent(e,1,True):.4f}')
