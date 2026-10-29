"""Section 4.7 (Proposition 4.24): the counted per-level factor inside App. B's shape.

Shape: graded space, f(E) = log2 dim V_E (nondecreasing); strings of lambda bits carried at weight >= h, f(h) = lambda;
piece s (all integer s in [-H, W]) factors row-wise through weights <= s + H and column-wise through <= W - s;
counted cost = f(top of window) up to O(log); counted rho = max_s (c_row + c_col) / (4 lambda).
Pigeonhole (row rank of the sum is 2^{2 lambda}): f(W + H) >= 2 lambda. Middle shift: both windows reach M = (W+H)/2.
Concavity on [h, 2M]: f(M) >= lambda (1 + (M - h)/(2M - h)), so rho >= (1 + (M-h)/(2M-h))/2.
 - the shape W = 2h, H >= h: M >= 3h/2, bound 5/8.
 - App. B's Lemma 9 (H = wt a + wt b >= 2h, W = wt b + wt y >= 2h): M >= 2h, bound 2/3.
Checks:
 (1) the bound function and its minimum at the least M;
 (2) random concave piecewise-linear profiles: f(M)/(2 lambda) >= bound whenever the pigeonhole holds; kinked profile
     attains it;
 (3) power profiles f = c E^alpha: rho(alpha) with M_min = max(M0, 2^{1/alpha - 1} h), for M0 = 3h/2 and 2h;
 (4) fermionic Fock space (exterior algebra on n modes, f(E) = log2 C(n, E), dims nondecreasing up to n/2): least
     counted rho over h for both shapes (about 0.660 at h/n = 0.075 for the first);
 (5) App. B's partitions at moderate h: log2 p(2h) / (2 log2 p(h)) and the pigeonhole margin log2 p(4h) - 2 log2 p(h).
Output: calc/r123_escapes_window_out.txt.
"""
import math
import numpy as np
from scipy.special import gammaln

out = []
bound = lambda M, h: 0.5 * (1 + (M - h) / (2 * M - h))
for M0 in (1.5, 2.0):
    Ms = np.linspace(M0, 10, 200)
    vals = bound(Ms, 1.0)
    out.append(f'(1) M0 = {M0} h: bound at M0 = {bound(M0, 1.0):.6f}, nondecreasing in M: {bool(np.all(np.diff(vals) >= 0))}')

rng = np.random.default_rng(7)
for M0 in (1.5, 2.0):
    worst, n_ok = 9.0, 0
    h, lam = 1.0, 1.0
    for _ in range(20000):
        M = M0 + rng.exponential(0.5)
        # concave piecewise linear on [h, 2M] through (h, lam): decreasing slopes at random breakpoints
        k = rng.integers(1, 6)
        bps = np.sort(rng.uniform(h, 2 * M, size=k - 1))
        xs = np.concatenate([[h], bps, [2 * M]])
        slopes = np.sort(rng.exponential(1.0, size=k))[::-1]
        ys = lam + np.concatenate([[0], np.cumsum(slopes * np.diff(xs))])
        if ys[-1] < 2 * lam:
            continue  # pigeonhole fails: not admissible
        n_ok += 1
        fM = np.interp(M, xs, ys)
        worst = min(worst, fM / (2 * lam) - bound(M, h))
    out.append(f'(2) M0 = {M0} h: {n_ok} admissible random concave profiles, min of f(M)/(2 lambda) - bound = {worst:.3e}')
    M = M0
    xs, ys = np.array([h, 2 * M]), np.array([lam, 2 * lam])  # kinked: affine from (h, lam) to (2M, 2 lam)
    out.append(f'    kinked profile (slope after h = {(lam / (2 * M - h)):.4f} lambda/h): f(M)/(2 lambda) = '
               f'{np.interp(M, xs, ys) / 2:.6f}')

for M0 in (1.5, 2.0):
    alphas = np.linspace(0.05, 1.0, 9501)
    Mmin = np.maximum(M0, 2.0 ** (1 / alphas - 1))
    rho = Mmin ** alphas / 2
    i = int(np.argmin(rho))
    out.append(f'(3) power profiles, M0 = {M0} h: min rho = {rho[i]:.4f} at alpha = {alphas[i]:.4f}; '
               f'rho(1/2) = {rho[np.argmin(abs(alphas - 0.5))]:.4f}; rho(1/3) = {rho[np.argmin(abs(alphas - 1/3))]:.4f}; '
               f'log formula max(2^(a-1), 2^-a) at the same alpha = {max(2 ** (alphas[i] - 1), 2 ** -alphas[i]):.4f}')
out.append(f'    alpha = log_3 2 = {math.log(2, 3):.4f}: 2^(-alpha) = {2 ** -math.log(2, 3):.4f}')

n = 2000
lb = lambda E: (gammaln(n + 1) - gammaln(E + 1) - gammaln(n - E + 1)) / math.log(2)
E = np.arange(0, n + 1)
fE = lb(E)
F = np.maximum.accumulate(fE)  # running max: dims of the exterior powers grow up to n/2
for M0 in (1.5, 2.0):
    best = (9, None)
    for h in range(10, n // 4):
        lam = F[h]
        Mlo = int(math.ceil(M0 * h))
        # least M >= M0 h with F(2M) >= 2 lam
        Ms = np.arange(Mlo, n // 2 + 1)
        okM = Ms[F[np.minimum(2 * Ms, n)] >= 2 * lam - 1e-9]
        if len(okM) == 0:
            continue
        M = okM[0]
        r = F[M] / (2 * lam)
        if r < best[0]:
            best = (r, (h, M))
    h, M = best[1]
    out.append(f'(4) fermions n = {n}, M0 = {M0} h: least counted rho = {best[0]:.4f} at h/n = {h / n:.4f}, M/h = {M / h:.3f}')
h = int(0.075 * n)
out.append(f'    at h/n = 0.075, M = 1.5 h: F(3h) - 2 lambda = {F[3 * h] - 2 * F[h]:.2f}, F(1.5h)/(2 lambda) = '
           f'{F[int(1.5 * h)] / (2 * F[h]):.4f}')


def partitions(nmax):
    p = [0] * (nmax + 1)
    p[0] = 1
    for k in range(1, nmax + 1):
        for e in range(k, nmax + 1):
            p[e] += p[e - k]
    return p


P = partitions(8000)
for h in (100, 500, 1000, 2000):
    lam = math.log2(P[h])
    out.append(f'(5) partitions h = {h}: log2 p(2h)/(2 log2 p(h)) = {math.log2(P[2 * h]) / (2 * lam):.4f}; '
               f'log2 p(4h) - 2 log2 p(h) = {math.log2(P[4 * h]) - 2 * lam:.2f}')
text = '\n'.join(out)
print(text)
with open('calc/r123_escapes_window_out.txt', 'w') as fh:
    fh.write(text + '\n')
