# App. B pieces on m-bit blocks at FINITE m: explicit certified upper bounds (no continuum model); Table 9, rows
# 'B, m = 256 / 512'.
# Per block type s in [-2h, 2h] (h = h_m least with p(h) >= 2^m; exact partition numbers, r122c_exponents.appB_block):
# rows cost r_s = log2 sum_{i=0}^{2h} p(s+i) bits, columns c_s = log2 sum_i p(2h-s-i) (paper (54), (55)); per lambda a
# rectangle whose blocks have types s_j costs mean_j min(r_s/m, 2) row-wise (raw (x, y) always works) and likewise
# column-wise.  rho_bar_m = max_s (min(r_s/m, 2) + min(c_s/m, 2)) / 2 replaces sqrt 2 in the certified bounds
# (the certified bounds of r122d_certify.py; the arguments only use rho <= 2 and rho + kappa <= 2 rho_bar per rectangle).
# Terms: (4h+1)^(lambda/m) per rectangle, each computed separately: + n_rect log2(4h+1)/m (n_rect = 9 at K = 9, 36 at
# K = 16).  Per-block re-indexing through the generic piece F_s costs 2^{O(m)} per block step: a constant for fixed
# m, so N^{o(1)} as lambda -> infinity.  K = 9 row tables re-certified at the cap 10 + 4 rho_bar_m.
# Resumable: one job per m in calc/r122d_bfinite_jobs.jsonl; summary in calc/r122d_bfinite_out.txt.
import sys
import numpy as np
sys.path.insert(0, '.')
import r122c_exponents as X
import r122d_certify as C
import r122d_values as RV
import r122d_model as M
from resumable import run_jobs

MS = [64, 128, 256, 512]


def job(m):
    h, rows = X.appB_block(m)
    rb = max((min(r, 2.0) + min(c, 2.0)) / 2 for _, r, c in rows)
    ent = float(np.log2(4 * h + 1) / m)
    WS, _ = RV.load()
    out = {'h': h, 'rho_bar': rb, 'ent': ent}
    for wk, w in WS.items():
        om = float(w(1.0))
        cap = 10 + 4 * rb
        tab, _ = C.tables_B_capped(w, cap, n=81)
        k9 = max((5 + 2 * rb) * om, float(tab)) + 9 * ent
        kept, _ = M.row_table_cost(4, w, [(1.0, [])] * 3)
        k16 = max(16 + 24 * rb + 8 * (om - 1), kept) + 36 * ent
        out[wk] = {'k9': k9, 'k9_top': (5 + 2 * rb) * om, 'k9_tab': float(tab), 'k16': k16}
    return out


if __name__ == '__main__':
    res = run_jobs(MS, job, 'r122d_bfinite_jobs.jsonl', processes=1)
    L = []
    for m in MS:
        r = res[str(m)]
        L.append(f"m={m}: h={r['h']}, rho_bar_m = {r['rho_bar']:.4f} (limit sqrt 2 = {np.sqrt(2):.4f}), "
                 f"log2(4h+1)/m = {r['ent']:.4f}")
        for wk in ('om2', 'current', 'vxxz'):
            d = r[wk]
            L.append(f"   [{wk}] K=9: certified <= max(top {d['k9_top']:.4f}, tables {d['k9_tab']:.4f}) + 9 x entropy = "
                     f"{d['k9']:.4f} (width bound 18: {'below' if d['k9'] < 18 else 'not below'}); K=16: certified <= "
                     f"{d['k16']:.4f} (width bound 60: {'below' if d['k16'] < 60 else 'not below'})")
    txt = '\n'.join(L)
    open('r122d_bfinite_out.txt', 'w').write(txt + '\n')
    print(txt)
