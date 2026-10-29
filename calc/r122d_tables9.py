# Certified K = 9 row-table bound of Table 9 (resumable, one job per w).  Same argument as r122d_certify.py, with the
# naming of a row table bounded by min(Jensen max, cap - sub-table size), since naming + sub-table size <= top2 <= cap
# in the direction with the smaller top2.  Results: calc/r122d_tables9_jobs.jsonl and calc/r122d_tables9_out.txt.
import sys
sys.path.insert(0, '.')
import r122d_certify as C
import r122d_values as RV
from resumable import run_jobs

KINDS = ('G', 'EM', 'E', 'B')


def job(wk):
    WS, _ = RV.load()
    w = WS[wk]
    caps = {k: 10 + 4 * C.RB[k] for k in KINDS}
    fam = C.tables_family_capped(w, sorted(set(caps[k] for k in ('G', 'EM', 'E'))))
    out = {}
    for k in KINDS:
        tab, arg = C.tables_B_capped(w, caps[k]) if k == 'B' else fam[caps[k]]
        top = (5 + 2 * C.RB[k]) * float(w(1.0))
        out[k] = {'top': top, 'tables': float(tab), 'cap': caps[k], 'phi': max(top, float(tab)), 'cell': str(arg)}
    return out


if __name__ == '__main__':
    res = run_jobs(['om2', 'current', 'vxxz'], job, 'r122d_tables9_jobs.jsonl', processes=3)
    with open('r122d_tables9_out.txt', 'w') as f:
        for wk in ('om2', 'current', 'vxxz'):
            r = res['"%s"' % wk]
            for k in KINDS:
                d = r[k]
                f.write(f"[{wk}] K=9 {k:2s}: certified Phi <= max(top {d['top']:.4f}, tables {d['tables']:.4f} "
                        f"(cap {d['cap']:.4f})) = {d['phi']:.4f} (e <= {d['phi']/36:.4f}); width bound 18; worst cell {d['cell']}\n")
    print(open('r122d_tables9_out.txt').read())
