# Certified bounds of Section 2.4 and Appendix A (upper bounds on each construction's exponent, valid for the max over
# ALL term profiles).
# K = 9 (g = 3):
#   top: for every profile and direction, T = Omega_w(E1,E2,E3) <= (w(1)/2) top2(E) (top2 = sum of the two largest
#   edges; uses w <= w(1), w(1) >= 2); and min over directions of top2 <= 10 + 4 rho_bar (two largest row pairs and
#   column pairs share 4 rectangles; the other 2+2 contribute <= 2 each).  Choosing the direction with the smaller
#   top2: T <= (5 + 2 rho_bar) w(1).
#   row tables: a row table of that direction has size E_ij + E_ij' <= top2; its build cost is <= size + Delta, where
#   Delta = max over row profiles of (sub-table build cost - sub-table size) (naming cancels).  Certified on a grid:
#   blocks are ordered P (compress to 1) <= Pperp (to log_8 56) <= kept (x, y) [any option for the lower kind is an
#   option for the upper one at no smaller cost], so on a grid cell the cost is <= its value at the cell's top
#   corner and the size >= its value at the bottom corner.  Same for App. B pieces in rho.
#   => Phi <= max((5 + 2 rho_bar) w(1), 10 + 4 rho_bar + Delta).
# K = 16 (g = 4): for G, E, EM every row-pair edge lies in [10, 16]; min over directions of the min over the three
#   pairings of the cut <= (1/3)(sum E + sum F) <= 16 + 24 rho_bar; in that pairing each merge
#   <N^A, N^s, N^C> has s <= 16 <= 20 <= min(A, C), so it costs A + C + x (w(s/x) - 2), x = min(A, C), i.e. at most
#   the cut plus X(w) = max_{20 <= x <= 32} x (w(16/x) - 2)^+.  => Phi_top <= 16 + 24 rho_bar + X(w).  Tables: all-kept
#   row cost + naming 6 (exact evaluation), below the top.
import sys, itertools
import numpy as np
sys.path.insert(0, '.')
import r122d_model as M
import r122d_values as RV

L56 = M.L56
rhoE = float(M.rho_E(0.5))
RB = {'G': 11 / 6, 'E': rhoE, 'EM': (11 / 6 + rhoE) / 2, 'B': np.sqrt(2)}


def sub_cost_size(g, w, comps):
    """comps[s] = (kept k, Pperp z) per-rectangle means at step s+1 (P = 1 - k - z), or ('B', rho)."""
    groups = []
    for c in comps:
        if c[0] == 'B':
            groups.append((0.0, [(1.0, c[1])]))
        else:
            k, z = c
            groups.append((k, [(1 - k - z, 1.0), (z, L56)]))
    cost, size = M.row_table_cost(g, w, groups)
    return cost, size


def delta_family(w, n=21):
    """certified max over (k, z) per step (k + z <= 1) of cost - size, g = 3, via cell corners."""
    pts = np.linspace(0, 1, n)
    cells = [(i, j) for i in range(n - 1) for j in range(n - 1) if pts[i] + pts[j] <= 1 + 1e-12]
    cache = {}
    def cs(c1, c2):
        key = (c1, c2)
        if key not in cache:
            cache[key] = sub_cost_size(3, w, [c1, c2])
        return cache[key]
    best = -np.inf; arg = None
    for (i1, j1) in cells:
        for (i2, j2) in cells:
            hi1 = (pts[i1 + 1], min(pts[j1 + 1], 1 - pts[i1 + 1])); hi2 = (pts[i2 + 1], min(pts[j2 + 1], 1 - pts[i2 + 1]))
            # top corner must dominate: k_hi = k+, k_hi + z_hi = min(1, k+ + z+)
            hi1 = (pts[i1 + 1], min(1.0, pts[i1 + 1] + pts[j1 + 1]) - pts[i1 + 1])
            hi2 = (pts[i2 + 1], min(1.0, pts[i2 + 1] + pts[j2 + 1]) - pts[i2 + 1])
            lo1 = (pts[i1], pts[j1]); lo2 = (pts[i2], pts[j2])
            v = cs(hi1, hi2)[0] - cs(lo1, lo2)[1]
            if v > best:
                best, arg = v, (lo1, hi1, lo2, hi2)
    return best, arg


def tables_family_capped(w, caps, n=31):
    """certified max of (naming + sub-table build cost) over row profiles whose row table fits under cap (size of
    the row table = naming + sub-table size <= top2 <= cap in the direction with the smaller top2).  Cells of the
    (k, z) grid per step; naming of a step group <= n_s H(k + z)/3 (Jensen; each rectangle has one of k, z), max over
    the cell; admissible iff size at the bottom corner <= cap (naming >= 0)."""
    pts = np.linspace(0, 1, n)
    P = [(i, j) for i in range(n) for j in range(n) if pts[i] + pts[j] <= 1 + 1e-12]
    cache = {}
    def cs(c1, c2):
        key = (c1, c2)
        if key not in cache:
            cache[key] = sub_cost_size(3, w, [c1, c2])
        return cache[key]
    def hmax(lo, hi):
        lo = min(lo, 1.0); hi = min(hi, 1.0)
        return 1.0 if lo <= 0.5 <= hi else float(max(M.H(lo), M.H(hi)))
    cells = [(i, j) for (i, j) in P if i < n - 1 and j < n - 1]
    res = {}
    for cap in caps:
        res[cap] = _fam_one(cells, pts, cs, hmax, cap)
    return res


def _fam_one(cells, pts, cs, hmax, cap):
    best = -np.inf; arg = None
    for (i1, j1) in cells:
        lo1 = (pts[i1], pts[j1]); hi1 = (pts[i1 + 1], min(1.0, pts[i1 + 1] + pts[j1 + 1]) - pts[i1 + 1])
        for (i2, j2) in cells:
            lo2 = (pts[i2], pts[j2]); hi2 = (pts[i2 + 1], min(1.0, pts[i2 + 1] + pts[j2 + 1]) - pts[i2 + 1])
            if cs(lo1, lo2)[1] > cap:
                continue
            nam = 2 * hmax(sum(lo1), sum(hi1)) / 3 + 4 * hmax(sum(lo2), sum(hi2)) / 3
            # naming + sub-table size <= cap, so naming <= cap - size(bottom corner)
            nam = min(nam, cap - cs(lo1, lo2)[1])
            v = nam + cs(hi1, hi2)[0]
            if v > best:
                best, arg = v, (lo1, lo2)
    return best, arg


def tables_B_capped(w, cap, n=161):
    pts = np.linspace(0, 2, n)
    best = -np.inf; arg = None
    C = {}
    for i in range(n):
        for j in range(n):
            C[(i, j)] = sub_cost_size(3, w, [('B', pts[i]), ('B', pts[j])])
    for i in range(n - 1):
        for j in range(n - 1):
            if C[(i, j)][1] > cap:
                continue
            v = C[(i + 1, j + 1)][0]
            if v > best:
                best, arg = v, (pts[i], pts[j])
    return best, arg


def delta_B(w, n=81):
    pts = np.linspace(0, 2, n)
    cache = {}
    def cs(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = sub_cost_size(3, w, [('B', a), ('B', b)])
        return cache[(a, b)]
    best = -np.inf; arg = None
    for i in range(n - 1):
        for j in range(n - 1):
            v = cs(pts[i + 1], pts[j + 1])[0] - cs(pts[i], pts[j])[1]
            if v > best:
                best, arg = v, (pts[i], pts[j])
    return best, arg


def X16(w):
    x = np.linspace(20, 32, 1201)
    return float(np.max(np.maximum(0, x * (w(16 / x) - 2))))


def X16_general(w, Arng, srng, h=0.05):
    """max over A, C in Arng, s in srng of Omega_w(A, s, C) - (A + C) on a grid of step h, plus the Lipschitz margin
    3 h (Omega is 1-Lipschitz in each argument for these w: d/dM = 1, d/dm = w(k) - k w'(k) - 1 in [0, 1],
    d/ds = w'(k) <= 1; the difference is <= 2-Lipschitz per argument, half-step in 3 arguments)."""
    A = np.arange(Arng[0], Arng[1] + 1e-9, h); S_ = np.arange(srng[0], srng[1] + 1e-9, h)
    best = -np.inf
    for a in A:                       # chunked over A (memory)
        CC, SS = np.meshgrid(A, S_, indexing='ij')
        v = M.Om(a, SS, CC, w) - (a + CC)
        best = max(best, float(v.max()))
    return best + 3 * h


def certify_w(wk):
    WS, META = RV.load()
    w = WS[wk]
    out = []
    om = float(w(1.0))
    out.append(f'\n== [{wk}] w(1) = {om:.4f}, w(4/5) = {float(w(0.8)):.4f}')
    caps = {k: 10 + 4 * RB[k] for k in ('G', 'EM', 'E')}
    fam = tables_family_capped(w, sorted(set(caps.values())))
    for kind in ('G', 'EM', 'E', 'B'):
        rb = RB[kind]
        cap = 10 + 4 * rb
        top = (5 + 2 * rb) * om
        tab, arg = tables_B_capped(w, cap) if kind == 'B' else fam[cap]
        out.append(f'  K=9 {kind:2s}: certified Phi <= max(top (5+2 rho_bar) w(1) = {top:.4f}, tables (cap {cap:.4f}) '
                   f'{tab:.4f}) = {max(top, tab):.4f} (e <= {max(top, tab)/36:.4f}); width bound 18; worst table cell {arg}')
    xb = 8 * (om - 1)
    x = max(0.0, 20 * (float(w(0.8)) - 2))
    kept, size = M.row_table_cost(4, w, [(1.0, [])] * 3)
    out.append(f'  K=16 merge excess: X_B = 8 (w(1) - 1) = {xb:.4f} (grid h=0.25 + margin: '
               f'{X16_general(w, (8, 32), (4, 16), h=0.25):.4f}); X_GE = 20 (w(4/5) - 2)^+ = {x:.4f} '
               f'(grid: {X16_general(w, (20, 32), (10, 16), h=0.25):.4f})')
    out.append(f'  K=16 B : certified Phi <= max(16 + 24 sqrt2 + X_B, tables) = max({16 + 24*RB["B"] + xb:.4f}, '
               f'{kept:.4f}) (e <= {max(16 + 24*RB["B"] + xb, kept)/120:.4f}); width bound 60')
    for kind in ('G', 'EM', 'E'):
        rb = RB[kind]
        v = max(16 + 24 * rb + x, kept + 6)
        out.append(f'  K=16 {kind:2s}: certified Phi <= max(16 + 24 rho_bar + X_GE = {16 + 24*rb + x:.4f}, tables '
                   f'{kept + 6:.4f}) = {v:.4f} (e <= {v/120:.4f}); width bound 60')
    return '\n'.join(out)


if __name__ == '__main__':
    from multiprocessing import Pool
    print('rho_bar:', {k: round(float(v), 5) for k, v in RB.items()}, flush=True)
    with Pool(3) as pool:
        for txt in pool.imap(certify_w, ['om2', 'current', 'vxxz']):
            print(txt, flush=True)
