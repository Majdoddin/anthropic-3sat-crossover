# Last step of Table 9: substitute matrix bounds into the parametric results.  Reads values.yaml via r122d_values.load
# (omega_now, alpha_now, omega_rect_curve; asserted not superseded) and VXXZ's Table 1 (peer-reviewed fallback).
# Prints for K = 9 and K = 16, per construction: a certified LOWER bound on the construction's exponent (a profile
# the adversary can pick: the uniform one; at K = 9 also 10 + 4 rho_bar from the concentrated one) and the certified
# UPPER bound (r122d_certify.py; K = 9 tables from the resumable jobs of r122d_tables9.py), against the input
# floor 2/K, the tree width bound (every contraction tree; w-independent) and the best subset-merge tree under w (an
# upper bound from the same w).  e = exponent / C(K,2); SAT bar ~0.36 (an estimate).
# Output: calc/r122d_subst_out.txt (also printed).
import sys
import numpy as np
sys.path.insert(0, '.')
import r122d_model as M
import r122d_values as RV
import r122d_certify as C
import r122d_tables9 as T9
from resumable import run_jobs

SAT_BAR = 0.36
KINDS = ('G', 'EM', 'E', 'B')
RSYM = {'G': 11 / 6, 'E': float(M.rho_E(0.5)), 'EM': float(M.rho_E(0.5)), 'B': np.sqrt(2)}


def main():
    WS, (omega, alpha, expr) = RV.load(general=True)
    tab = run_jobs(['om2', 'current', 'vxxz'], T9.job, 'r122d_tables9_jobs.jsonl', processes=3)
    L = [f'values.yaml: omega_now = {omega}, alpha_now = {alpha}, omega_rect_curve = {expr}',
         'w: ' + '; '.join(f'{k} = {v.name}' for k, v in WS.items()),
         'rho_bar = ' + ', '.join(f'{k} {C.RB[k]:.5f}' for k in KINDS) + '; rho_sym = ' +
         ', '.join(f'{k} {RSYM[k]:.5f}' for k in KINDS)]
    for K, g in ((9, 3), (16, 4)):
        S = K * (K - 1) // 2
        L.append(f'\n===== K = {K}: C(K,2) = {S}; input floor {K-1} (e = {2/K:.4f}); tree width bound {M.bw(K)} '
                 f'(e = {M.bw(K)/S:.4f}, all trees); SAT bar e ~ {SAT_BAR}')
        for wk, w in WS.items():
            om = float(w(1.0)); w45 = float(w(0.8))
            tv = M.tree(K, w)[0]
            L.append(f'-- [{wk}] w(1/2) = {float(w(0.5)):.4f}, w(4/5) = {w45:.4f}, w(1) = {om:.4f}; best subset-merge '
                     f'tree {tv:.4f} (e = {tv/S:.4f})')
            base = wk.replace('+gen', '')
            for kind in KINDS:
                rb, rs = C.RB[kind], RSYM[kind]
                n = (g * (g - 1) // 2) ** 2
                uni = M.term_top(kind, np.full(n, 0.0 if kind == 'B' else 0.5), g, w)
                if g == 3:
                    lo = max(uni, 10 + 4 * rb)
                    d = tab[f'"{base}"'][kind]
                    tabv = d['tables'] if base == wk else np.nan    # tables certified for the three base w's
                    hi = max((5 + 2 * rb) * om, tabv) if base == wk else (5 + 2 * rb) * om
                    note = '' if base == wk else ' (top only; tables certified for the base w)'
                else:
                    lo = uni
                    X = 8 * (om - 1) if kind == 'B' else max(0.0, 20 * (w45 - 2))
                    kept, _ = M.row_table_cost(4, w, [(1.0, [])] * 3)
                    hi = max(16 + 24 * rb + X, kept + (0 if kind == 'B' else 6))
                    note = ''
                verdict_bw = 'below' if hi < M.bw(K) else ('not below' if lo >= M.bw(K) else 'undecided')
                verdict_tr = 'below' if hi < tv else ('not below' if lo >= tv else 'undecided')
                L.append(f'   {kind:2s}: certified [{lo:.4f}, {hi:.4f}] (e in [{lo/S:.4f}, {hi/S:.4f}]); vs width bound '
                         f'{verdict_bw}; vs best tree {verdict_tr}{note}')
    txt = '\n'.join(L)
    open('r122d_subst_out.txt', 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
