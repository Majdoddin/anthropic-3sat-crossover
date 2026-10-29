"""Explicit C_HR in  log2 p(E) >= c_HR sqrt(E) - log2 E - C_HR  for all E >= 1  (Lemma 3.5 of the paper).

Analytic part (n >= 13), from Lehmer's remainder bound for Rademacher's series with N = 1 term, as quoted in
DeSalvo-Pak, arXiv 1310.7982, eq. (4)-(5):  p(n) > e^mu / (8 sqrt3 n) with mu = (pi/6) sqrt(24n-1) >= 9, hence
C_HR = log2(8 sqrt3) + (pi/(6 sqrt(24*13)))/ln2.  This script
 (1) evaluates that constant with mpmath (50 digits);
 (2) checks the intermediate inequalities against exact p(n), n <= EMAX (sanity only, not needed for the proof);
 (3) checks the final inequality with C_HR on 1 <= E <= EMAX by exact p(E) (covers 1 <= E <= 12);
 (4) re-runs the analytic-range part of r123_trace_c.py with this C_HR (copied, lambda range extended).
"""
import math, os, sys
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r123_trace_c import partitions, lg, need_c, cHR  # noqa: E402

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
EMAX = 24000
N0 = 13

out = open(os.path.join(HERE, 'r123_chr_check_out.txt'), 'w')
w = lambda s: (out.write(s + '\n'), print(s))

# (1) the constant
mu0 = mp.pi / 6 * mp.sqrt(24 * N0 - 1)
f9 = 1 - mp.mpf(4) / 9 - 81 * mp.e ** -9
shift = mp.pi / (6 * mp.sqrt(24 * N0))          # mu >= pi sqrt(2n/3) - shift for n >= N0
C_an = mp.log(8 * mp.sqrt(3), 2) + shift / mp.log(2)
w(f'mu(13) = {mp.nstr(mu0, 8)} >= 9; bracket 1 - 4/mu - mu^2 e^-mu at mu = 9: {mp.nstr(f9, 8)} >= 1/2')
w(f'shift pi/(6 sqrt(24*13)) = {mp.nstr(shift, 8)}; log2(8 sqrt3) = {mp.nstr(mp.log(8*mp.sqrt(3), 2), 8)}')
w(f'analytic C_HR (n >= 13) = {mp.nstr(C_an, 10)}')
CHR = 3.84
assert C_an < CHR
w(f'rounded up: C_HR = {CHR}')

# (2), (3) exact checks
p = partitions(EMAX)
bad2 = []
for n in range(N0, EMAX + 1, 1 if EMAX <= 3000 else 7):
    m = mp.pi / 6 * mp.sqrt(24 * n - 1)
    lehmer_lb = mp.pi ** 2 / (6 * mp.sqrt(3)) * mp.e ** m / m ** 2 * (1 - 4 / m - m ** 2 * mp.e ** -m)
    crude_lb = mp.e ** m / (8 * mp.sqrt(3) * n)
    if not (p[n] > lehmer_lb > crude_lb):
        bad2.append(n)
w(f'(2) exact p(n) > Lehmer-based lower bound > e^mu/(8 sqrt3 n), n = 13..{EMAX} step 7: '
  f'{"holds" if not bad2 else "FAILS at " + str(bad2[:10])}')
worst = max((cHR * math.sqrt(E) - math.log2(E) - lg(p[E]), E) for E in range(1, EMAX + 1))
w(f'(3) exact: max over 1 <= E <= {EMAX} of c_HR sqrt E - log2 E - log2 p(E) = {worst[0]:.4f} at E = {worst[1]} '
  f'(<= C_HR = {CHR}: {worst[0] <= CHR})')

# (4) analytic range of r123_trace_c.py, same formulas, C_HR = 3.84; LAM = 271 from r123_trace_c_out.txt
LAM = 271
worst2, at2 = 0.0, None
lams = list(range(LAM, 10 ** 4)) + [int(10 ** (x / 10)) for x in range(40, 141)]  # beyond ~1e14 the float cancellation in (b) dominates
for lam in lams:
    h = 2.0
    for _ in range(200):
        h = ((lam + math.log2(h) + CHR) / cHR) ** 2 + 1
    sh = math.sqrt(h)
    lD = math.log2(4 * h + 1) + 2 * cHR * sh + CHR
    R = 2 * math.sqrt(2) * cHR * sh + 2 * math.log2(2 * h + 1) + 2 * CHR + 4
    c, _ = need_c(lam, h, ('log2', lD), R, math.lgamma(h + 1) / math.log(2))
    if c > worst2:
        worst2, at2 = c, lam
w(f'(4) analytic range lambda = {LAM}..10^14 (sampled as in r123_trace_c.py) with C_HR = {CHR}: '
  f'max need {worst2:.3f} at lambda = {at2}; limit of (b) as lambda -> oo is 4 + 4 sqrt2 = {4 + 4 * math.sqrt(2):.3f}')
w(f'exact range max (from r123_trace_c_out.txt, independent of C_HR): 27.198 at lambda = 1')
w(f'c = ceil(max(27.198, {worst2:.3f})) + 2 = {math.ceil(max(27.198, worst2)) + 2}')
out.close()
