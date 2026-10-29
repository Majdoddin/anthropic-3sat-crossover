# Targeted follow-ups to r122c_exponents.py (same cost model, imported); Section 2.3 of the paper.
#  (1) E (grouped by ranks) at omega = 2: which ingredients it needs (octonion factor? G factor?), optimal options.
#  (2) alpha thresholds of V (covering) and B (App. B pieces) by bisection.
#  (3) adversarial mixtures for B: random 3- and 4-point mixtures of piece types, omega = 2 and today.
#  (4) B at finite block length m with the exact partition numbers: max over pure types + 2-point mixtures, plus the
#      entropy bound log2(4h+1)/m.
#  (5) V today: the components at the worst point (cover count, table, derivation, contraction).
import sys
import numpy as np
sys.path.insert(0, '.')
from r122c_exponents import *
import io, contextlib

def sec1():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("(1) E at omega = 2, zeta = 1/2 (worst term):")
        for g, gf, lab in [(1.0, L56, "gamma=1, G factor allowed"), (9.0, L56, "no octonion factor (gamma=9), G factor allowed"),
                           (1.0, 9.0, "gamma=1, G factor forbidden"), (9.0, 9.0, "no octonion factor, no G factor")]:
            z = 0.5
            b = GR[:, None, None] * (1 - z); f = GR[None, :, None] * z; q = GR[None, None, :] * z
            O = 1 + b / 2 + f + q * L56 / 2; I = 1 + b + q
            c = mm(O, I, 'om2') + (1 - z - b) * g + (z - f - q) * gf
            c = np.where(f + q <= z + 1e-12, c, np.inf)
            j = np.unravel_index(np.argmin(c), c.shape)
            e = E_E(g, 'om2', gf)
            print(f"   {lab}: table cost {c[j]:.4f} with bypassed P-blocks {float(b[j[0],0,0]):.3f}, fused Pperp {float(f[0,j[1],0]):.3f},"
                  f" bypassed Pperp {float(q[0,0,j[2]]):.3f}, G-factor Pperp {z - float(f[0,j[1],0]) - float(q[0,0,j[2]]):.3f}; total E = {e[0]:.4f}")
        print("   hand formula (all P bypassed, f = 1/4, G factor 1/4): 1/3 + 2 + 1/2 + 1/2 + (1/4) L56 + ... =",
              round(1/3 + float(mm(1 + 0.25 + 0.25, 1.5, 'om2')) + 0.25 * L56, 4))
    return buf.getvalue()

def sec2():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("(2) alpha thresholds (chord model), bisection:")
        aV = bisect(lambda al: not (E_V(9.0, ('chord', al), AS_=np.linspace(0, 0.5, 101))[0] < 4 - 1e-9), 0.5, 0.6, it=12)
        aV1 = bisect(lambda al: not (E_V(1.0, ('chord', al), AS_=np.linspace(0, 0.5, 101))[0] < 4 - 1e-9), 0.5, 0.6, it=12)
        print(f"   V without octonion factor: gain iff alpha > {aV:.4f};  V with gamma = 1: alpha > {aV1:.4f}")
        def B_pure0(al): return cost_mix([1.0], [0.0], ('chord', al))
        aB = bisect(lambda al: not (B_pure0(al) < 4 - 1e-9), 0.5, 0.6, it=30)
        print(f"   B (limit m -> inf, worst type u = 0): gain iff alpha > {aB:.4f}; B at alpha = 0.501: {E_B(('chord', 0.501), n2=11)[0]:.5f} (2-point mixtures on an 11-point grid)")
    return buf.getvalue()

def sec3():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("(3) adversarial random mixtures for B:")
        rng = np.random.default_rng(1)
        for reg in ['om2', 'today']:
            best = (-1, None)
            for _ in range(500):
                n = rng.integers(3, 5); us = rng.uniform(-2, 2, n); w = rng.dirichlet(np.ones(n))
                v = cost_mix(w, us, reg)
                if v > best[0]: best = (v, (np.round(us, 2), np.round(w, 2)))
            # also mixtures concentrated near 0
            for _ in range(200):
                n = rng.integers(2, 5); us = rng.normal(0, 0.2, n).clip(-2, 2); w = rng.dirichlet(np.ones(n))
                v = cost_mix(w, us, reg)
                if v > best[0]: best = (v, (np.round(us, 2), np.round(w, 2)))
            print(f"   {reg}: worst random mixture {best[0]:.4f} at u={best[1][0]}, w={best[1][1]}; pure u=0 {cost_mix([1.0],[0.0],reg):.4f}")
    return buf.getvalue()

def sec4():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("(4) B at finite m (exact partition numbers):")
        for m in [64, 256]:
            h, rows = appB_block(m)
            ent = np.log2(4 * h + 1) / m
            # pure types (rows or columns, whichever cheaper) and 2-point mixtures on a subsample of types
            sel = rows[::max(1, len(rows) // 25)]
            for reg in ['om2', 'today']:
                worst = max(min(cost_mix_rows([1.0], [r], reg), cost_mix_rows([1.0], [c], reg)) for _, r, c in rows[::max(1, len(rows)//400)])
                for i1 in range(len(sel)):
                    for i2 in range(i1 + 1, len(sel)):
                        for w in (0.25, 0.5, 0.75):
                            _, r1, c1 = sel[i1]; _, r2, c2 = sel[i2]
                            v = min(cost_mix_rows([w, 1 - w], [r1, r2], reg), cost_mix_rows([w, 1 - w], [c1, c2], reg))
                            worst = max(worst, v)
                print(f"   m={m} (h={h}): {reg}: max cost {worst:.4f} + entropy bound {ent:.4f} = {worst + ent:.4f}")
    return buf.getvalue()

def sec5():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("(5) V today at the worst point, no octonion factor:")
        a = 0.5
        for d in [0.85, 0.9, 0.921, 0.95, 1.0]:
            cover = (H(a) - d * H(a / d)) / 3; Cd = float(C_oct(np.array([d]), 9.0, 'today')[0])
            beta = min(max(8 * a / 7, a), d); gst = max(beta * float(H(a / beta)) / 3 + a - beta, 0)
            print(f"   d={d}: cover {cover:.4f}, table C(d) {Cd:.4f}, derivation 3+d+g {3 + d + gst:.4f}, total {cover + max(Cd, 3 + d + gst):.4f};"
                  f" per-Q contraction H(a)/3+3+a = {1/3 + 3.5:.4f}")
        for a in [0.3, 0.4, 0.45]:
            v = E_V(9.0, 'today', AS_=np.array([a]))
            print(f"   a={a}: V {v[0]:.4f} at d={v[2]:.3f};  U(gamma = Alman-Li) at this a: {float(H(a)/3 + C_oct(np.array([a]), 1.5676, 'today')[0]):.4f}")
    return buf.getvalue()

def run_idx(i):
    fs = [f for n, f in sorted(globals().items()) if n.startswith('sec') and callable(f)]
    return fs[i]()

if __name__ == '__main__':
    from multiprocessing import Pool
    fs = [f for n, f in sorted(globals().items()) if n.startswith('sec') and callable(f)]
    with Pool(4) as pool:
        for r in pool.map(lambda_call, range(len(fs))) if False else pool.map(run_idx, range(len(fs))):
            print(r, end='', flush=True)
