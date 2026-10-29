"""An explicit value for the constant c of Anthropic's Theorem B (pieces with parameters (1/sqrt2, c)); Lemma 3.4
and Table 11 of the paper.

Definition 7 of Anthropic's paper asks, for every lambda >= 1:
  (b)  c_row + c_col <= 4 rho lambda + c log2(2 lambda), rho = 1/sqrt2;
  (c1) |I| <= (2 lambda)^c;
  (c2) d, the elements of I, all values of the A_s and of their factor functions are below 2^{(2 lambda)^c};
  (c3) one algorithm computes each of them (and I, c_row, c_col) within 2^{c lambda} operations;
and l. 1527-1530 add 2 to c for truncating factorizations to 2 lambda bits.
Proof of Theorem B: h = least h >= 2 with p(h) >= 2^lambda; I = {-2h..2h}, d = h!;
R_s^row = sum_{i=0}^{2h} p(s+i), R_s^col = sum_{i=0}^{2h} p(2h-s-i), c = ceil(log2 R); Delta = number of partitions of
0..4h; table entries <= 2^{12 h^2}; coefficient sums <= 2^{3h} (a<i>) and <= h! (b).
Operation count (c3): the paper only says Delta^{O(1)}. Counted here from its description: tables of <= Delta monomials
of a<i>, each Delta^2 entries built by <= h steps of (O2) with <= 8h terms per entry (16 h^2 Delta^2 operations each),
then <= (2h+1) Delta pairs (i, sigma) with Delta lookups each; bounded generously by OPS = 64 h^3 Delta^3.
Values (c2): a factor is <= Delta 2^{3h} 2^{12h^2} or <= Delta h! 2^{12h^2}; A_s <= (2h+1) Delta times their product.
Exact for lambda <= LAM (p computed exactly); beyond, Hardy-Ramanujan (46) with the constant C_HR measured on
E <= EMAX (valid for all E only with an explicit remainder bound; see the out file)."""
import math, os

EMAX = 24000
HERE = os.path.dirname(os.path.abspath(__file__))
cHR = math.pi * math.sqrt(2 / 3) / math.log(2)


def partitions(N):
    p = [1] + [0] * N
    for n in range(1, N + 1):
        s, k = 0, 1
        while True:
            g1 = k * (3 * k - 1) // 2
            if g1 > n:
                break
            sg = 1 if k % 2 else -1
            s += sg * p[n - g1]
            g2 = k * (3 * k + 1) // 2
            if g2 <= n:
                s += sg * p[n - g2]
            k += 1
        p[n] = s
    return p


def lg(x):  # log2 of a big int
    b = x.bit_length()
    return b - 1 + math.log2(x >> (b - 53)) - 52 if b > 53 else math.log2(x)


def need_c(lam, h, Delta, Rmax_bits, logfact):
    """least integer c meeting (b), (c1), (c2), (c3) at this lambda (before the +2)."""
    L2 = math.log2(2 * lam)
    cb = max(0.0, (Rmax_bits - 4 * lam / math.sqrt(2)) / L2) if L2 > 0 else (0 if Rmax_bits <= 4 * lam / math.sqrt(2) else math.inf)
    cc1 = math.log2(4 * h + 1) / L2 if L2 > 0 else 0
    lD = Delta[1] if isinstance(Delta, tuple) else lg(Delta)
    vbits = math.log2(2 * h + 1) + 3 * lD + 3 * h + logfact + 24 * h * h + 2  # A_s bound, the largest value
    cc2 = math.log2(vbits) / L2 if L2 > 0 else math.inf
    cc3 = (6 + 3 * math.log2(h) + 3 * lD) / lam
    return max(cb, cc1, cc2, cc3), dict(b=cb, c1=cc1, c2=cc2, c3=cc3)


if __name__ == '__main__':
    p = partitions(EMAX)
    lp = [lg(x) for x in p]
    # C_HR on 1 <= E <= EMAX: both inequalities of (46)
    CHR_lo = max(cHR * math.sqrt(E) - math.log2(E) - lp[E] for E in range(1, EMAX + 1))
    CHR_up = max(lp[E] - cHR * math.sqrt(E) for E in range(0, EMAX + 1))
    CHR = max(CHR_lo, CHR_up)
    cum = [0] * (EMAX + 1)
    acc = 0
    for E in range(EMAX + 1):
        acc += p[E]
        cum[E] = acc
    out = open(os.path.join(HERE, 'r123_trace_c_out.txt'), 'w')
    out.write(f'c_HR = {cHR:.6f}; on 1 <= E <= {EMAX}: lower-bound constant {CHR_lo:.4f}, upper-bound constant '
              f'{CHR_up:.4f}, so C_HR = {CHR:.4f} on this range\n')
    worst, worst_at, lam = 0.0, None, 1
    rows = []
    while True:
        h = 2
        while h <= EMAX and p[h] < 2 ** lam:
            h += 1
        if 4 * h > EMAX:
            break
        Delta = cum[4 * h]
        Rmax = 0.0
        C = lambda a, b: cum[min(b, EMAX)] - (cum[a - 1] if a >= 1 else 0) if b >= max(a, 0) else 0  # sum p[a..b]
        for s in range(-2 * h, 2 * h + 1):
            Rr = C(max(s, 0), s + 2 * h)          # p(s+i), i = 0..2h
            Rc = C(max(-s, 0), 2 * h - s)         # p(2h-s-i), i = 0..2h
            Rmax = max(Rmax, math.ceil(lg(Rr)) + math.ceil(lg(Rc)))
        c, parts = need_c(lam, h, Delta, Rmax, math.lgamma(h + 1) / math.log(2))
        rows.append((lam, h, lg(Delta), Rmax, c, parts))
        if c > worst:
            worst, worst_at = c, lam
        lam += 1
    LAM = lam - 1
    out.write(f'exact range lambda = 1..{LAM} (4h <= {EMAX})\n')
    out.write('lambda   h   log2 Delta  max(c_row+c_col)  need: (b)  (c1)  (c2)  (c3)  -> c\n')
    for r in rows:
        if r[0] <= 12 or r[0] % 25 == 0 or r[0] == LAM:
            lam_, h, lD, R, c, d = r
            out.write(f'{lam_:6d} {h:5d} {lD:10.1f} {R:12.0f}     {d["b"]:6.2f} {d["c1"]:5.2f} {d["c2"]:5.2f} {d["c3"]:5.2f} -> {c:6.2f}\n')
    out.write(f'max over the exact range: {worst:.3f} at lambda = {worst_at}\n')
    # analytic range via (46): c_HR sqrt(h-1) < lambda + log2 h + C_HR, Delta <= (4h+1) p(4h), log p(4h) <= 2 c_HR sqrt h + C_HR,
    # c_row + c_col <= 2 sqrt2 c_HR sqrt h + 2 log2(2h+1) + 2 C_HR + 4 (two ceilings)
    worst2, at2 = 0.0, None
    for lam in list(range(LAM, 10 ** 4)) + [int(10 ** (x / 10)) for x in range(40, 121)]:
        h = 2.0
        for _ in range(200):  # fixed point for the largest h allowed by the bound
            h = ((lam + math.log2(h) + CHR) / cHR) ** 2 + 1
        sh = math.sqrt(h)
        lD = math.log2(4 * h + 1) + 2 * cHR * sh + CHR
        R = 2 * math.sqrt(2) * cHR * sh + 2 * math.log2(2 * h + 1) + 2 * CHR + 4
        c, _ = need_c(lam, h, ('log2', lD), R, math.lgamma(h + 1) / math.log(2))
        if c > worst2:
            worst2, at2 = c, lam
    out.write(f'analytic range lambda >= {LAM} (needs (46) with C_HR = {CHR:.4f} for all E): max need {worst2:.3f} at lambda = {at2}\n')
    total = math.ceil(max(worst, worst2)) + 2
    out.write(f'c = ceil(max) + 2 (truncation to 2 lambda bits, l. 1527-1530) = {total}\n')
    out.write(f'x0 condition (l. 1670 check): 2^(6 x0 - 1) >= c + 5 holds for x0 >= '
              f'{math.ceil((math.log2(total + 5) + 1) / 6)}; x0 = 7 allows c < 2^41.\n')
    out.close()
    print(open(os.path.join(HERE, 'r123_trace_c_out.txt')).read())
