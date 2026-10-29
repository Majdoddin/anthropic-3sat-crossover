# Exponents and thresholds for the K = 4 constructions of Section 2.3 (units: exponent of N = 2^lambda).
# Regimes for omega(1,k,1): 'today' = VXXZ Table 1 chords (table T below); 'om2' = omega = 2;
# ('chord', al) = chord from (al, 2) to (1, 2.371339).
# Matrix part: <N^O, N^I, N^O> (equal outer dims): O * omega(1, I/O, 1) if I <= O, else (I - O) + O * omega (crude,
# never optimal below).  Factors with (asymptotic) rank add their log_N multiplier.  Every cost is a max with the size
# of the table it produces.
# Constructions (Section 2.3 of the paper):
#  U   ungrouped (Theorem 2.7), = grouped with log C(t,i) naming bits; Ut = grouped with one naming bit per block;
#  E   grouped counted by ranks (P on Z^c, Pperp on Z; rank(G) = 56, the rank of P^perp in Section 2.3);
#  V   covering hybrid (T_D for D containing Q, derive T_Q);
#  B   Appendix B pieces on m-bit blocks, m -> infinity (continuum model r = sqrt(2+u), c = sqrt(2-u), u in [-2,2]);
#  H1  App. H one-bit identity counted by ranks.
import numpy as np
T = [(0.321334, 2), (0.33, 2.000100), (0.34, 2.000600), (0.35, 2.001363), (0.40, 2.009541), (0.45, 2.023788), (0.50, 2.042994),
     (0.527661, 2.055322), (0.55, 2.066134), (0.60, 2.092631), (0.65, 2.121734), (0.70, 2.153048), (0.75, 2.186210), (0.80, 2.220929),
     (0.85, 2.256984), (0.90, 2.294209), (0.95, 2.332440), (1, 2.371552)]
ks, ws = map(np.array, zip(*T)); OMEGA = 2.371552; WCH = 2.371339
def w1k1(k, reg):
    k = np.asarray(k, dtype=float)
    if reg == 'om2': return np.full_like(k, 2.0)
    if reg == 'today': return np.where(k <= ks[0], 2.0, np.interp(np.minimum(k, 1.0), ks, ws))
    al = reg[1]; return np.where(k <= al, 2.0, 2 + (np.minimum(k, 1.0) - al) / (1 - al) * (WCH - 2))
def om(reg): return 2.0 if reg == 'om2' else (OMEGA if reg == 'today' else WCH)
def mm(O, I, reg):
    O = np.asarray(O, dtype=float); I = np.asarray(I, dtype=float)
    k = np.minimum(I / O, 1.0)
    return np.where(I <= O, O * w1k1(k, reg), (I - O) + O * om(reg))
def H(a):
    a = np.clip(np.asarray(a, dtype=float), 1e-300, 1 - 1e-16)
    return np.where((a <= 1e-15) | (a >= 1 - 1e-15), 0.0, -(a * np.log2(a) + (1 - a) * np.log2(1 - a)))
L56 = np.log2(56) / 3   # log_8 56 = 1.9358

# ---------------- U: ungrouped / bit-named grouped ----------------
def C_oct(a, g, reg, RS=81):
    # table with column-choice fraction a (rows merged): r = fraction of all blocks bypassed to EACH side, rest of R
    # blocks octonion factors: <N^{1+a+r}, N^{1+2r}, N^{1+a+r}> x N^{(1-a-2r) g}; floor = size 3 + a.
    a = np.asarray(a, dtype=float)[..., None]
    r = np.linspace(0, 1, RS) * (1 - a) / 2
    c = mm(1 + a + r, 1 + 2 * r, reg) + (1 - a - 2 * r) * g
    return np.maximum(c.min(-1), (3 + a[..., 0]))
AS = np.linspace(0, 0.5, 501)
def E_U(g, reg, naming='log'):
    ent = H(AS) / 3 if naming == 'log' else np.full_like(AS, 1 / 3)
    v = ent + C_oct(AS, g, reg); i = int(np.argmax(v)); return v[i], AS[i]

# ---------------- E: grouped counted by ranks ----------------
GR = np.linspace(0, 1, 26)
def C_E(z, g, reg, gfac=L56):
    # P-blocks (1-z): bypass fraction b (sym), rest octonion factor (g); Pperp-blocks (z): fused f, bypass q, rest
    # G factor (rank 56 -> gfac per block).  Output floor 2 + (1-z) + L56 z.
    b = GR[:, None, None] * (1 - z); f = GR[None, :, None] * z; q = GR[None, None, :] * z
    ok = f + q <= z + 1e-12
    O = 1 + b / 2 + f + q * L56 / 2; I = 1 + b + q
    c = mm(O, I, reg) + (1 - z - b) * g + (z - f - q) * gfac
    c = np.where(ok, c, np.inf)
    return max(float(c.min()), 2 + (1 - z) + L56 * z)
ZS = np.linspace(0, 0.5, 101)
def E_E(g, reg, gfac=L56):
    v = np.array([H(z) / 3 + C_E(z, g, reg, gfac) for z in ZS]); i = int(np.argmax(v)); return v[i], ZS[i]

# ---------------- V: covering hybrid ----------------
DS = np.linspace(0, 1, 2001)
def E_V(g, reg, AS_=np.linspace(0, 0.5, 251), detail=False):
    Cd = C_oct(DS, g, reg)
    worst = (-1, None, None)
    for a in AS_:
        m = DS >= a - 1e-12
        d = DS[m]; p = np.where(d > 0, a / np.maximum(d, 1e-300), 0)
        cover = (H(a) - d * H(p)) / 3          # log_N of (number of D's needed) = C(t,at)/C(dt,at)
        # derivation of all T_Q (Q subset of D, |Q| = at) from T_D, block by block: max state 3 + d + max g(beta,kappa)
        # with g = beta H(kappa/beta)/3 + kappa - beta, kappa <= a, beta - kappa <= d - a  (closed form: kappa = a,
        # beta = clip(8a/7, a, d)); times the number of D's.
        beta = np.clip(8 * a / 7, a, d); gst = beta * H(np.where(beta > 0, a / np.maximum(beta, 1e-300), 0)) / 3 + a - beta
        gst = np.maximum(gst, 0)
        deriv = 3 + d + gst
        val = cover + np.maximum(Cd[m], deriv)
        j = int(np.argmin(val))
        tot = max(val[j], H(a) / 3 + 3 + a)   # per-Q contraction (and the per-Q tables) N^{H(a)/3 + 3 + a}
        if tot > worst[0]: worst = (tot, a, d[j])
    return worst

# ---------------- B: Appendix B pieces on long blocks (continuum limit) ----------------
UG = np.linspace(-2, 2, 161)
rB = lambda u: np.sqrt(2 + u)
def cost_mix_rows(ws_, rs_, reg, BG=np.linspace(0, 1, 401)):
    # bypass the smallest-r blocks first (for fixed bypass weight this minimizes the outer dimension); 1-D search on B
    order = np.argsort(rs_); w = np.asarray(ws_)[order]; r = np.asarray(rs_)[order]
    cw = np.concatenate([[0], np.cumsum(w)]); cwr = np.concatenate([[0], np.cumsum(w * r)])
    best = np.inf
    for B in BG * cw[-1]:
        j = np.searchsorted(cw, B, side='right') - 1; j = min(j, len(w) - 1)
        part = B - cw[j]
        byp_r = cwr[j] + part * r[j]
        O = 1 + (cw[-1] - B) + byp_r / 2; I = 1 + B
        best = min(best, float(mm(O, I, reg)))
    return max(best, 2 + float(np.dot(w, r)))
def cost_mix(ws_, us_, reg):
    us_ = np.asarray(us_)
    return min(cost_mix_rows(ws_, rB(us_), reg), cost_mix_rows(ws_, rB(-us_), reg))
def E_B(reg, n2=41):
    best = (-1, None)
    for u in UG:
        v = cost_mix([1.0], [u], reg)
        if v > best[0]: best = (v, ('pure', round(float(u), 3)))
    U2 = np.linspace(-2, 2, n2)
    for i1, u1 in enumerate(U2):
        for u2 in U2[i1 + 1:]:
            for w in np.linspace(0.1, 0.9, 9):
                v = cost_mix([w, 1 - w], [u1, u2], reg)
                if v > best[0]: best = (v, ('mix', round(float(u1), 2), round(float(u2), 2), round(float(w), 2)))
    return best
def partitions_upto(n):
    # Euler's pentagonal recurrence, exact integers
    p = [1] + [0] * n
    for k in range(1, n + 1):
        tot = 0; j = 1
        while True:
            g1 = j * (3 * j - 1) // 2
            if g1 > k: break
            sg = 1 if j % 2 else -1
            tot += sg * p[k - g1]
            g2 = j * (3 * j + 1) // 2
            if g2 <= k: tot += sg * p[k - g2]
            j += 1
        p[k] = tot
    return p
def appB_block(m):
    # exact per-block numbers: h = least with p(h) >= 2^m; R_s^row = sum_{i=0}^{2h} p(s+i), R_s^col = sum_i p(2h-s-i)
    from math import log2
    n = 64
    while True:
        p = partitions_upto(n)
        hs = [k for k in range(n + 1) if p[k] >= 2 ** m]
        if hs and 4 * hs[0] + 2 <= n: break
        n *= 2
    h = hs[0]
    P = lambda k: p[k] if 0 <= k <= n else 0
    rows = []
    for s in range(-2 * h, 2 * h + 1):
        Rr = sum(P(s + i) for i in range(2 * h + 1)); Rc = sum(P(2 * h - s - i) for i in range(2 * h + 1))
        rows.append((s, log2(Rr) / m, log2(Rc) / m))
    return h, rows

# ---------------- H1: one-bit identity counted by ranks ----------------
def C_H1(z, reg, G=26):
    # rows merged, fraction z of bits in Pi_A (row rank 1; as a 2x2x1 tensor rank 2), 1-z in Pi_S (rank 3 output).
    # Pi_A: bypass (inner +1) or factor (mult 1); Pi_S: fused (outer +1 each), bypass (outer +log2(3)/2 each, inner +1),
    # factor (mult log2 3).
    l3 = np.log2(3)
    a = np.linspace(0, 1, G)[:, None, None] * z
    f = np.linspace(0, 1, G)[None, :, None] * (1 - z); q = np.linspace(0, 1, G)[None, None, :] * (1 - z)
    ok = f + q <= (1 - z) + 1e-12
    O = 1 + f + q * l3 / 2; I = 1 + a + q
    c = mm(O, I, reg) + (z - a) * 1.0 + ((1 - z) - f - q) * l3
    c = np.where(ok, c, np.inf)
    return max(float(c.min()), 2 + (1 - z) * l3)
def E_H1(reg):
    zs = np.linspace(0.5, 1, 101)
    v = np.array([H(z) + C_H1(z, reg) for z in zs]); i = int(np.argmax(v)); return v[i], zs[i]

def bisect(f, lo, hi, it=40):
    # f(lo) True (gain), f(hi) False; returns boundary
    for _ in range(it):
        m = (lo + hi) / 2
        (lo, hi) = (m, hi) if f(m) else (lo, m)
    return (lo + hi) / 2

if __name__ == '__main__':
    import sys
    tree = 2 * float(w1k1(0.5, 'today'))
    gAL = np.log(np.sqrt(8 ** (4 * OMEGA / 3) - 2 * (2 ** (OMEGA / 3) - 1) * 8 ** (2 * OMEGA / 3))) / np.log(8)
    gAL2 = np.log(np.sqrt(8 ** (8 / 3) - 2 * (2 ** (2 / 3) - 1) * 8 ** (4 / 3))) / np.log(8)
    print(f"tree today 2 omega(1,1/2,1) = {tree:.4f}; Alman-Li gamma today {gAL:.4f} (rank {8**gAL:.2f}), at omega=2 {gAL2:.4f} (rank {8**gAL2:.2f})")
    print("\n== U: ungrouped (= grouped, log C(t,i) naming bits) and Ut (one naming bit per block) ==")
    for g in [1.0, 1.1088, gAL2, gAL]:
        print(f"gamma={g:.4f}: U today {E_U(g,'today')[0]:.4f} (a={E_U(g,'today')[1]:.3f}), omega=2 {E_U(g,'om2')[0]:.4f};"
              f"  Ut today {E_U(g,'today','bits')[0]:.4f}, omega=2 {E_U(g,'om2','bits')[0]:.4f}")
    gU = bisect(lambda g: E_U(g, 'today')[0] < tree, 1.0, 1.6)
    print(f"U threshold today: gamma < {gU:.4f}, rank < {8**gU:.3f}")
    print("\n== E: grouped counted by ranks ==")
    for g in [1.0, 1.1088, gAL2, gAL]:
        e1 = E_E(g, 'today'); e2 = E_E(g, 'om2')
        print(f"gamma={g:.4f}: E today {e1[0]:.4f} (zeta={e1[1]:.3f}), omega=2 {e2[0]:.4f} (zeta={e2[1]:.3f})")
    print(f"E at omega=2 with G factor forbidden (fused/bypass only), gamma=1: {E_E(1.0,'om2',gfac=9.0)[0]:.4f}")
    print(f"E floor (sizes only) max_z H(z)/3 + 3 + (L56-1) z = {max(H(z)/3+3+(L56-1)*z for z in ZS):.4f}")
    gE = bisect(lambda g: E_E(g, 'today')[0] < tree, 1.0, 1.6)
    print(f"E threshold today: gamma < {gE:.4f}, rank < {8**gE:.3f}")
    gE2 = bisect(lambda g: E_E(g, 'om2')[0] < 23 / 6 - 1e-9, 1.0, 1.6)
    print(f"E beats 23/6 at omega=2 iff gamma < {gE2:.4f} (rank < {8**gE2:.3f}); known bound at omega=2: gamma <= {gAL2:.4f}")
    print("\n== V: covering hybrid ==")
    for g in [1.0, 1.1088, gAL]:
        v = E_V(g, 'today'); print(f"gamma={g:.4f}: V today {v[0]:.4f} (worst a={v[1]:.3f}, d={v[2]:.3f})")
    v = E_V(9.0, 'today'); print(f"gamma=9 (no octonion factor used): V today {v[0]:.4f} (worst a={v[1]:.3f}, d={v[2]:.3f})")
    v = E_V(gAL2, 'om2'); print(f"omega=2: V {v[0]:.4f} (worst a={v[1]:.3f}, d={v[2]:.3f})")
    # what V would give at omega=2 if the per-Q contraction could be grouped exactly (no exact designs exist)
    a = 0.5; d = DS[DS >= a]; val = (H(a) - d * H(a / d)) / 3 + 3 + d
    print(f"omega=2, a=1/2, hypothetical exact grouping over D-slices: min_d {val.min():.4f} at d={d[np.argmin(val)]:.4f} (8/7 * 1/2 = {4/7:.4f})")
    print("\n== B: Appendix B pieces on m-bit blocks, m -> infinity ==")
    for reg in ['om2', 'today']:
        b = E_B(reg); print(f"{reg}: worst mixture cost {b[0]:.4f} at {b[1]}; pure u=0: {cost_mix([1.0],[0.0],reg):.4f}")
    print("pure u=0 closed form at omega=2: 2 + 4/(4 - sqrt 2) =", 2 + 4 / (4 - np.sqrt(2)))
    for m in [64, 256]:
        h, rows = appB_block(m)
        ent = np.log2(4 * h + 1) / m
        mx = max(min(r, c) for _, r, c in rows)
        print(f"m={m}: h={h}, {4*h+1} pieces, entropy bound log2(4h+1)/m = {ent:.4f}, max_s min(r_s,c_s)/m = {mx:.4f} (limit sqrt 2 = 1.4142)")
    print("\n== H1: one-bit identity counted by ranks ==")
    for reg in ['om2', 'today']:
        e = E_H1(reg); print(f"{reg}: {e[0]:.4f} (z={e[1]:.3f})")
    print(f"H1 floor (sizes only) max_(z>=1/2) H(z) + 2 + (1-z) log2 3 = {max(H(z)+2+(1-z)*np.log2(3) for z in np.linspace(0.5,1,501)):.4f}")
    print("\n== alpha thresholds (chord model; tree = 4 for alpha >= 1/2) ==")
    for g in [1.0, 1.1088, gAL2, gAL]:
        aU = bisect(lambda al: not (E_U(g, ('chord', al))[0] < 4 - 1e-9), 0.5, 1.0)
        aE = bisect(lambda al: not (E_E(g, ('chord', al))[0] < 4 - 1e-9), 0.5, 1.0, it=25)
        print(f"gamma={g:.4f} (rank {8**g:.2f}): U alpha* = {aU:.4f}, E alpha* = {aE:.4f}")
    for al in [0.5, 0.505, 0.52, 0.55, 0.6, 0.7, 0.8]:
        v = E_V(9.0, ('chord', al), AS_=np.linspace(0, 0.5, 101))
        b = E_B(('chord', al), n2=21)
        print(f"alpha={al}: V (no octonion factor) {v[0]:.4f} (a={v[1]:.2f}, d={v[2]:.3f}); B {b[0]:.4f}; U(gamma=1) {E_U(1.0,('chord',al))[0]:.4f}")
