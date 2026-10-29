"""Certify the K = 6 (2 x 3 grid, Appendix B pieces, m -> infinity) value of calc/r123_s1_grid.py (Section 2.4).

What is maximised.  A term of the expansion assigns to every m-bit block of each of the three rectangles of the 2 x 3
grid (row pair {0,1} x column pair C, C in {01, 02, 12}) a piece type u in [-2, 2]; the block then compresses to
rho = sqrt(2+u) bits per bit with rows merged and kappa = sqrt(2-u) with columns merged (continuum model of Section 2.3).  The
algorithm picks per term the cheaper direction; the exponent is
    Phi(w) = sup over terms  min( R_w(rho_1, rho_2, rho_3), C_w(kappa_1, kappa_2, kappa_3) ),
with R_w = max(top dot product 3 + sum rho, row-table costs) and C_w = max(triangle Omega_w(2+kappa_C), column tables),
exactly as r123_s1_grid.dir_cost computes them (the bypass fraction beta and split phi on an 11 x 11 grid: a
restriction of the algorithm's choices, so each value is an achievable cost, i.e. an upper bound for that term).
For a rectangle whose blocks carry a mixture of types, the uniform-fraction bypass used by the model sees only the
mean rho and mean kappa; (mean rho, mean kappa) lies in the convex hull of the quarter circle rho^2 + kappa^2 = 4 and
is dominated componentwise by the arc point in its direction, so by monotonicity pure types are the worst case.

Certification.  R_w is non-decreasing in each rho (every strategy's step sizes are non-decreasing in rho and Omega_w
is non-decreasing in each argument when w is non-decreasing with w - k w' >= 1, checked below), C_w likewise in kappa.
On a box [lo, hi]^3 of u: rho <= sqrt(2+hi), kappa <= sqrt(2-lo), so  sup_box min(R, C) <= min(R(sqrt(2+hi)),
C(sqrt(2-lo))).  Branch and bound over [-2, 2]^3 (all boxes, no symmetry assumed) until the upper bound is within TOL
of the best evaluated profile.  Floating point: costs are sums/max/min of O(10) doubles, error << 1e-9.
Output: calc/r123_k6_certify_out.txt."""
import os, sys, heapq, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r122d_model as M
import r123_s1_grid as G

WS, VALS = G.WS, None
TOL = 2e-3
OUT = os.path.join(HERE, 'r123_k6_certify_out.txt')
lines = []


def log(s=''):
    print(s, flush=True); lines.append(s)


def R(rho, w):      # rows merged: top = 2 merged rows (dot product), each row of 3 nodes built sequentially
    return G.dir_cost('B', np.asarray(rho, float).reshape(1, 3), 2, 3, w)[0]


def C(kap, w):      # columns merged: top = triangle of 3 merged columns, each column of 2 nodes
    return G.dir_cost('B', np.asarray(kap, float).reshape(3, 1), 3, 2, w)[0]


def val(u, w, t=1.0):
    u = np.asarray(u, float)
    return min(R(np.minimum(2.0, t * np.sqrt(2 + u)), w), C(np.minimum(2.0, t * np.sqrt(2 - u)), w))


def ub(lo, hi, w, t=1.0):
    return min(R(np.minimum(2.0, t * np.sqrt(2 + hi)), w), C(np.minimum(2.0, t * np.sqrt(2 - lo)), w))


def bnb(w, tol=TOL, max_boxes=400000, t=1.0):
    val_t = lambda u, w: val(u, w, t)
    ub_t = lambda lo, hi, w: ub(lo, hi, w, t)
    lo0, hi0 = np.full(3, -2.0), np.full(3, 2.0)
    best_v, best_u = -np.inf, None
    for u0 in ([2, -0.8177, -0.8177], [-0.7199, -0.2258, -0.2258], [0, 0, 0]):   # old search optima as seeds
        v = val_t(u0, w)
        if v > best_v:
            best_v, best_u = v, np.array(u0, float)
    heap = [(-ub_t(lo0, hi0, w), 0, lo0, hi0)]
    cnt, n_eval = 1, 0
    pruned = -np.inf          # largest upper bound of a discarded box (discarded boxes are NOT below best_v)
    while heap:
        nub, _, lo, hi = heapq.heappop(heap)
        if -nub <= best_v + tol:
            heapq.heappush(heap, (nub, 0, lo, hi))
            break
        d = int(np.argmax(hi - lo)); mid = (lo[d] + hi[d]) / 2
        for a, b in ((lo[d], mid), (mid, hi[d])):
            l2, h2 = lo.copy(), hi.copy(); l2[d], h2[d] = a, b
            c = (l2 + h2) / 2
            v = val_t(c, w); n_eval += 1
            if v > best_v:
                best_v, best_u = v, c
            u2 = ub_t(l2, h2, w)
            if u2 > best_v + tol:
                cnt += 1
                heapq.heappush(heap, (-u2, cnt, l2, h2))
            else:
                pruned = max(pruned, u2)
        if cnt > max_boxes:
            break
    glob_ub = max([best_v, pruned] + [-x[0] for x in heap])
    return best_v, best_u, glob_ub, n_eval


def breakdown(u, w):
    u = np.asarray(u, float); rho, kap = np.sqrt(2 + u), np.sqrt(2 - u)
    top_r = 3 + rho.sum()
    Ed = {(0, 1): top_r}
    tr = G.dir_cost('B', rho.reshape(1, 3), 2, 3, w)[0]
    F = 2 + kap
    top_c = float(M.Om(F[0], F[1], F[2], w))
    tc = G.dir_cost('B', kap.reshape(3, 1), 3, 2, w)[0]
    return (f"rho = {np.round(rho, 4).tolist()}, kappa = {np.round(kap, 4).tolist()}; rows: top {top_r:.4f}, "
            f"max(top, tables) {tr:.4f}; columns: top {top_c:.4f}, max(top, tables) {tc:.4f}")


def log2_up(x):
    """log2 of a positive integer, rounded up by 1e-12 (exact integer shift, then one double log)."""
    from math import log2
    b = x.bit_length()
    if b <= 60:
        return log2(x) + 1e-12
    return (b - 60) + log2((x >> (b - 60)) + 1) + 1e-12


def appB_scale(m):
    """Exact App. B block numbers via prefix sums (same quantities as r122c_exponents.appB_block):
    h = least with p(h) >= 2^m; r_s = log2 sum_{i=0}^{2h} p(s+i), c_s = log2 sum_i p(2h-s-i), s in [-2h, 2h].
    Returns h, t_m = max_s |(min(r_s/m, 2), min(c_s/m, 2))| / 2, and rho_bar_m = max_s (min(r_s/m,2)+min(c_s/m,2))/2."""
    import r122c_exponents as X
    n = 64
    while True:
        p = X.partitions_upto(n)
        hs = [k for k in range(n + 1) if p[k] >= 2 ** m]
        if hs and 4 * hs[0] + 2 <= n:
            break
        n *= 2
    h = hs[0]
    pre = [0]
    for v in p:
        pre.append(pre[-1] + v)            # pre[k] = sum_{j < k} p(j)
    S = lambda a, b: pre[min(max(b + 1, 0), n + 1)] - pre[min(max(a, 0), n + 1)]   # sum_{a <= k <= b} p(k)
    tmax, rb = 0.0, 0.0
    for s in range(-2 * h, 2 * h + 1):
        r = min(log2_up(S(s, s + 2 * h)) / m, 2.0)
        c = min(log2_up(S(-s, 2 * h - s)) / m, 2.0)
        tmax = max(tmax, (r * r + c * c) ** 0.5 / 2); rb = max(rb, (r + c) / 2)
    return h, tmax, rb


def check_w(w, name):
    """Omega_w non-decreasing in each argument needs w non-decreasing and w(k) - k w'(k) >= 1 on [0, 1]."""
    k = np.linspace(1e-4, 1 - 1e-4, 20001)
    h = 1e-6
    wp = (w(k + h) - w(k - h)) / (2 * h)
    a = float(np.min(wp)); b = float(np.min(w(k) - k * wp))
    log(f"  w = {name}: min w' = {a:.4f} (>= 0 needed), min (w - k w') = {b:.4f} (>= 1 needed), w(1) = {float(w(1.0)):.4f}")
    return a >= -1e-6 and b >= 1 - 1e-6


def check_monotone(w, rng, n=300):
    bad = 0
    for _ in range(n):
        x = rng.uniform(0, 2, 3); y = x + rng.uniform(0, 0.5, 3) * (rng.uniform(size=3) < 0.6)
        bad += R(y, w) < R(x, w) - 1e-9
        bad += C(y, w) < C(x, w) - 1e-9
    return bad


if __name__ == '__main__':
    t0 = time.time()
    rng = np.random.default_rng(7)
    log('K = 6, 2 x 3 grid, Appendix B pieces (m -> infinity): certified sup over all terms of min(rows, columns)')
    log(f'(branch and bound over u in [-2,2]^3, box bound by monotonicity at the upper corner in rho/kappa; TOL {TOL})')
    log()
    log('Omega_w monotonicity conditions:')
    for wk in ('om2', 'current', 'vxxz'):
        ok = check_w(WS[wk], WS[wk].name)
        log(f'    -> {"holds" if ok else "FAILS"}')
    log()
    for wk in ('om2', 'current', 'vxxz'):
        w = WS[wk]
        bad = check_monotone(w, rng)
        tree = M.tree(6, w)[0]
        v_old = val([2, -0.8177, -0.8177] if wk == 'om2' else [-0.7199, -0.2258, -0.2258], w)
        best_v, best_u, U, n = bnb(w)
        log(f'{wk}: {w.name}')
        log(f'  random monotonicity spot check violations: {bad} / 600')
        log(f'  old search profile re-evaluated: {v_old:.4f}')
        log(f'  best profile found: {best_v:.5f} at u = {np.round(best_u, 4).tolist()} (e = {best_v / 15:.4f})')
        log(f'  CERTIFIED upper bound: Phi <= {U:.5f} (e <= {U / 15:.4f}); {n} evaluations')
        log(f'  breakdown at best profile: {breakdown(best_u, w)}')
        log(f'  best subset-merge tree under the same w: {tree:.4f}; width bound for all trees bw(6) = {M.bw(6)}')
        log(f'  margin to the tree: {tree - U:.4f}; to the width bound 8: {8 - U:.4f}')
        log()
    log('Finite block length m (exact partition numbers): every per-block point (min(r_s/m,2), min(c_s/m,2)) lies in')
    log('the disc of radius 2 t_m, so a rectangle\'s mean (rho, kappa) is dominated by t_m (sqrt(2+u), sqrt(2-u)) capped at 2;')
    log('terms are computed separately: + 3 log2(4h+1)/m for the 3 rectangles (per-block re-indexing 2^O(m) is N^o(1)).')
    for m in (128, 256, 512, 1024):
        t1 = time.time()
        h, tm, rb = appB_scale(m)
        ent = 3 * float(np.log2(4 * h + 1)) / m
        log(f'm={m}: h={h}, t_m={tm:.5f}, rho_bar_m={rb:.4f}, 3 log2(4h+1)/m = {ent:.4f}  ({time.time() - t1:.0f} s)')
        for wk in ('om2', 'current', 'vxxz'):
            w = WS[wk]
            bv, bu, U, n = bnb(w, t=tm)
            tree = M.tree(6, w)[0]
            log(f'   [{wk}] certified Phi_m <= {U:.4f} + {ent:.4f} = {U + ent:.4f} (e <= {(U + ent) / 15:.4f}); '
                f'tree {tree:.4f}: {"below" if U + ent < tree else "NOT below"}; width bound 8: '
                f'{"below" if U + ent < 8 else "not below"}')
        open(OUT, 'w').write('\n'.join(lines) + '\n')
    log()
    log(f'time {time.time() - t0:.1f} s')
    open(OUT, 'w').write('\n'.join(lines) + '\n')
