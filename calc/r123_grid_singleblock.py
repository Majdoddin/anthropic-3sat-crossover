"""Checks for the single-block grid theorem (Lemma 4.17, Theorem 4.18 and Corollary 4.19 of the paper).

(1) Sturmian 0/1 matrix X[i][j] = floor((i+j+1)th) - floor((i+j)th): row sums in {floor(nu th), ceil(nu th)},
    column sums in {floor(mu th), ceil(mu th)}; exhaustive over small mu, nu and a grid of th.
(2) The per-level bound: for random a, b >= 3 and random x1, x2 in (0,1], the max over th in [0,1] of
    min(R'(th), C'(th)) (R', C' the rounded lower bounds) is >= x1 x2 (1 - ebar) / (x1 + x2) in the interior case
    and the closed forms; and the step f(K) <= max(Pa, Pb) log2 K / (1 - ebar) holds when f(a) <= Pa log2 a etc.
(3) The error product for the recursion of Theorem A of the cited paper, (40): g_l = 2^(3l + x0) K_l, K_{l+1} = g_l^2,
    log2 K_l = (3x0 + 6) 2^l - 6l - 2x0 - 6; ebar_l = 4/g_l^2 and the exact eps_l = 4/(g_l (g_l + 1)).
Output: r123_grid_singleblock_out.txt (written by this script).
"""
import math, random, sys
from fractions import Fraction as Fr

out = []
def P(*a):
    s = " ".join(str(x) for x in a); out.append(s); print(s)

# (1) Sturmian matrix
bad = 0; tested = 0
for mu in range(1, 16):
    for nu in range(1, 16):
        for k in range(0, 41):
            th = Fr(k, 40)
            rows = [sum(math.floor((i + j + 1) * th) - math.floor((i + j) * th) for j in range(nu)) for i in range(mu)]
            cols = [sum(math.floor((i + j + 1) * th) - math.floor((i + j) * th) for i in range(mu)) for j in range(nu)]
            ok = all(math.floor(nu * th) <= r <= math.ceil(nu * th) for r in rows) and \
                 all(math.floor(mu * th) <= c <= math.ceil(mu * th) for c in cols)
            tested += 1; bad += (not ok)
# irrational-ish th too
rng = random.Random(1)
for _ in range(3000):
    mu, nu = rng.randint(1, 60), rng.randint(1, 60); th = rng.random()
    rows = [sum(math.floor((i + j + 1) * th) - math.floor((i + j) * th) for j in range(nu)) for i in range(mu)]
    cols = [sum(math.floor((i + j + 1) * th) - math.floor((i + j) * th) for i in range(mu)) for j in range(nu)]
    ok = all(math.floor(nu * th) <= r <= math.ceil(nu * th) for r in rows) and \
         all(math.floor(mu * th) <= c <= math.ceil(mu * th) for c in cols)
    tested += 1; bad += (not ok)
P("(1) Sturmian row/column sums: tested", tested, "cases, failures", bad)

# (2) per-level bound
def level_value(a, b, x1, x2, steps=20001):
    K = a * b; E = K * (K - 1) / 2
    i1 = (b - 1) / (K - 1); i2 = (a - 1) / (K - 1); t = (a - 1) * (b - 1) / (K - 1)
    e1 = a * (a - 1) / E; e2 = b * (b - 1) / E
    best = -1
    for s in range(steps):
        th = s / (steps - 1)
        R = x1 * (i2 + th * t - e1); C = x2 * (i1 + (1 - th) * t - e2)
        best = max(best, min(R, C))
    return best, i1, i2, t, e1, e2

fails = 0; worst = 0.0; n = 0
for _ in range(400):
    a, b = rng.randint(3, 40), rng.randint(3, 40)
    la, lb = math.log2(a), math.log2(b)
    Pa, Pb = 1 + rng.random() * 0.1, 1 + rng.random() * 0.1
    # f(a), f(b) anywhere in [1, P log2]
    fa = rng.uniform(1.0, Pa * la) if Pa * la > 1 else Pa * la
    fb = rng.uniform(1.0, Pb * lb) if Pb * lb > 1 else Pb * lb
    v, i1, i2, t, e1, e2 = level_value(a, b, 1 / fa, 1 / fb, 4001)
    ebar = 2 / a**2 + 2 / b**2
    eps = e1 + e2
    assert eps <= ebar + 1e-15 and 2 / b**2 <= ebar and 2 / a**2 <= ebar
    fK = 1 / v; bound = max(Pa, Pb) * math.log2(a * b) / (1 - ebar)
    n += 1
    if fK > bound * (1 + 1e-6): fails += 1
    worst = max(worst, fK / bound)
P("(2) per-level step f(K) <= max(Pa,Pb) log2 K/(1-ebar): tested", n, "random (a,b,f(a),f(b)), failures", fails,
  "max ratio f(K)/bound = %.6f" % worst)
# 2x2 and 2xb: what the rounded bound gives
for (a, b) in [(2, 2), (2, 3), (2, 10), (3, 3), (3, 100)]:
    K = a * b; v, *_ = level_value(a, b, 1 / max(1, math.log2(a)), 1 / max(1, math.log2(b)), 4001)
    P("    grid %dx%d: ebar = %.4f, eps = %.4f, rounded-mixing bound f(K) <= %.3f vs log2 K = %.3f, input floor K/2 = %g"
      % (a, b, 2 / a**2 + 2 / b**2, 2 * (a*(a-1) + b*(b-1)) / (K*(K-1)), 1 / v if v > 0 else float('inf'),
         math.log2(K), K / 2))

# (3) Anthropic's schedule
P("(3) recursion of (40): ebar_l = 4/g_l^2 and exact eps_l = 4/(g_l(g_l+1)); Pi_L = prod_{l<L} 1/(1-ebar_l)")
for x0 in (18, 19, 20, 24):
    s_bar = 0.0; s_ex = 0.0; logPi = 0.0
    rows = []
    for l in range(0, 12):
        xl = (3 * x0 + 6) * 2**l - 6 * l - 2 * x0 - 6
        if l == 0: assert xl == x0
        lg = 3 * l + x0 + xl                # log2 g_l
        eb = 4.0 * 2.0 ** (-2 * lg)         # 4/g^2 (underflows to 0 for big lg: then exactly negligible)
        s_bar += eb; logPi += -math.log1p(-eb) if eb > 0 else 0.0
        rows.append((l, xl, lg, eb))
    P("  x0 = %d: log2 K_0..K_4 = %s; log2 g_0 = %d; sum_l ebar_l = %.3e; Pi_inf - 1 = %.3e"
      % (x0, [r[1] for r in rows[:5]], rows[0][2], s_bar, math.expm1(logPi)))
# corollary threshold
gam, s = 0.4092, 1270.0
bar = gam / s
for q in (2,):
    x0 = 18; Pi = 1 + 4 * 2.0 ** (-72) * (1 + 1e-6)
    P("  corollary: bar = %.5e, q/bar = %.1f; with f <= Pi log2 K: q/(Pi bar) = q/bar / Pi, Pi - 1 = %.3e"
      % (bar, q / bar, 4 * 2.0 ** (-72) * (1 + 1e-6)))
    # which level of (40) first reaches log2 K >= 1/bar
    l = 0
    while (3 * x0 + 6) * 2**l - 6 * l - 2 * x0 - 6 < 1 / bar: l += 1
    P("  first l with log2 K_l >= 1/bar (x0 = 18): l = %d, log2 K_l = %d; grid levels on the chain: %d"
      % (l, (3 * x0 + 6) * 2**l - 6 * l - 2 * x0 - 6, l))
# general sides >= g0: Pi <= (1 - 4/g0^2)^(-log2 K / log2 g0) at log2 K = 1/bar
for g0 in (3, 4, 8, 16, 64, 1024, 2**18):
    d = (1 / bar) / math.log2(g0)
    lp = -d * math.log1p(-4 / g0**2)
    P("  sides >= %d: at log2 K = 1/bar = %.0f, Pi <= (1-4/g0^2)^(-%.0f) = 2^%.4g" % (g0, 1 / bar, d, lp / math.log(2)))

with open(__file__.replace(".py", "_out.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
