# Exact check of the expansion and bypass of Theorem 2.7 (K = 4 at omega = 2): K = 4 clique (a, a' top; b, b' bottom),
# arbitrary integer node weights.
# W_a[z_aa', z_ab, x], W_a'[z_aa', z_a'b', y], W_b[z_bb', z_ab, u], W_b'[z_bb', z_a'b', v],
# x = z_a(b'), v = z_b'(a), y = z_a'(b), u = z_b(a').  Z = sum of W_a W_a' W_b W_b' [x = v][y = u].
# Checks:
#  - Z by brute force (lambda = 3) and by two different contraction orders (lambda = 3, 6);
#  - 2^t Z = sum over Q of Z_Q, with Z_Q computed from merged-row tables (index set Q and column-choice kernel);
#  - top table via the octonion structure tensor built from Cayley-Dickson through the isomorphism phi below;
#  - top table via the re-indexing route (row-choice block: x_s = k_s xor y_s, y_s becomes a summed index)
#    as ONE matrix product (the bypass of Theorem 2.7);
#  - negative controls: sigma(y,x) in place of sigma(x,y), sigma(u,v) in place of sigma(v,u).
# Exact int64 arithmetic (bounds checked).
import itertools, sys
import numpy as np
sys.path.insert(0, '.')

def sigma(p, q):
    P = (p & 1, (p >> 1) & 1, (p >> 2) & 1); Qb = (q & 1, (q >> 1) & 1, (q >> 2) & 1)
    phi = sum(P[i] * Qb[j] for i in range(3) for j in range(i, 3))
    phi += Qb[0] * P[1] * P[2] + P[0] * Qb[1] * P[2] + P[0] * P[1] * Qb[2]
    return -1 if phi % 2 else 1
S = np.array([[sigma(p, q) for q in range(8)] for p in range(8)], dtype=np.int64)
X8 = np.array([[p ^ q for q in range(8)] for p in range(8)])
def KR(order):  # K[x,y,k] = [x^y=k] s(x,y) with s chosen by order
    K = np.zeros((8, 8, 8), dtype=np.int64)
    for x in range(8):
        for y in range(8):
            K[x, y, x ^ y] = S[x, y] if order == 'xy' else S[y, x]
    return K
# column-choice kernel on (x,y,u,v): [x^y^u^v=0] s(x,u) s(v,y)
KC = np.zeros((8, 8, 8, 8), dtype=np.int64)
for x, y, u, v in itertools.product(range(8), repeat=4):
    if x ^ y ^ u ^ v == 0:
        KC[x, y, u, v] = S[x, u] * S[v, y]

# octonion structure tensor via Cayley-Dickson + isomorphism phi (found by a separate search, hard-coded here)
def qmul(a, b):
    a0, a1, a2, a3 = a; b0, b1, b2, b3 = b
    return (a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2, a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0)
def qc(a): return (a[0], -a[1], -a[2], -a[3])
def omul(X, Y):
    a, b, c, d = X[:4], X[4:], Y[:4], Y[4:]
    l = tuple(p - q for p, q in zip(qmul(a, c), qmul(qc(d), b))); r = tuple(p + q for p, q in zip(qmul(d, a), qmul(b, qc(c))))
    return l + r
phi = {0: (1, 0), 1: (1, 1), 2: (1, 2), 4: (1, 4), 3: (-1, 3), 5: (-1, 5), 6: (-1, 6), 7: (1, 7)}
Phi = np.zeros((8, 8), dtype=np.int64)  # column p = coordinates of phi(e_p) in CD basis
for p, (s, j) in phi.items(): Phi[j, p] = s
CDT = np.zeros((8, 8, 8), dtype=np.int64)  # f_j f_k = sum_m CDT[j,k,m] f_m
for j in range(8):
    for k in range(8):
        CDT[j, k, :] = omul(tuple(np.eye(8, dtype=int)[j]), tuple(np.eye(8, dtype=int)[k]))
# structure constants in the e-basis: e_p e_q = sum_r c[p,q,r] e_r ; Phi orthogonal signed permutation -> inverse = transpose
c = np.einsum('jp,kq,jkm,mr->pqr', Phi, Phi, CDT, Phi)
assert np.array_equal(c, KR('xy')), "octonion tensor != sigma kernel"
OCT = c

def fe(subs, *ops):
    # exact: every partial sum is an integer of absolute value < 2^53 (checked by the final bound below)
    r = np.einsum(subs, *[o.astype(np.float64) for o in ops], optimize='optimal')
    assert np.all(np.abs(r) < 2.0**52)
    return np.rint(r).astype(np.int64)

def blocks_split(arr, axis, t):  # replace axis of size 8^t by t axes of size 8 (most significant block first)
    sh = list(arr.shape); sh[axis:axis + 1] = [8] * t
    return arr.reshape(sh)

def run(lam, seed, wmax=4, top_variant='xy', bot_variant='vu'):
    t = lam // 3; N = 2 ** lam
    rng = np.random.default_rng(seed)
    Wa, Wa2, Wb, Wb2 = (rng.integers(-wmax, wmax + 1, size=(N, N, N), dtype=np.int64) for _ in range(4))
    # reference 1: merge rows (tree), P[zab, zab2, x, y]
    # exact: all partial sums are integers of absolute value < 2^53, so float64 BLAS is exact
    f = lambda A: A.astype(np.float64)
    ex = lambda A: np.rint(A).astype(np.int64)
    P = ex(np.tensordot(f(Wa), f(Wa2), axes=([0], [0])).transpose(0, 2, 1, 3))   # (a,x,b,y) -> (a,b,x,y)
    Pb = ex(np.tensordot(f(Wb), f(Wb2), axes=([0], [0])).transpose(0, 2, 1, 3))
    Z1 = int(np.einsum('abxy,abyx->', P, Pb))        # u = y, v = x
    # reference 2: merge a with b' (different order): sum over the diagonal x = v first
    M1 = ex(np.tensordot(f(Wa), f(Wb2), axes=([2], [2])))   # (s,a,r,b): (z_aa', z_ab, z_bb', z_a'b')
    M2 = ex(np.tensordot(f(Wa2), f(Wb), axes=([2], [2])))   # (s,b,r,a)
    Z2 = int(np.einsum('sarb,sbra->', M1, M2))
    assert Z1 == Z2
    if lam == 3:  # true brute force over all six edge strings
        Zb = 0
        for s in range(N):
            for r in range(N):
                Zb += int(np.einsum('ax,by,ay,bx->', Wa[s], Wa2[s], Wb[r], Wb2[r]))  # Wb[r,zab,u=y], Wb2[r,zab2,v=x]
        assert Zb == Z1
    # expansion over Q
    KRt = KR(top_variant)
    KRb = np.zeros((8, 8, 8), dtype=np.int64)  # bottom kernel on (u, v, k)
    for u in range(8):
        for v in range(8):
            KRb[u, v, u ^ v] = S[v, u] if bot_variant == 'vu' else S[u, v]
    Pt = P.reshape([N, N] + [8] * t + [8] * t)       # (zab, zab2, x_1..x_t, y_1..y_t)
    Pbt = Pb.reshape([N, N] + [8] * t + [8] * t)     # (zab, zab2, u_1..u_t, v_1..v_t)
    total = 0
    letters = 'cdefghijklmnopqrstuvwxyz'
    oct_ok = True; mm_ok = True
    for Qbits in range(2 ** t):
        Q = [s for s in range(t) if (Qbits >> s) & 1]
        # top table
        T = Pt.copy(); B = Pbt.copy()
        # contract row-choice blocks one at a time; keep axis bookkeeping by einsum strings
        def apply_rows(Tin, kern, use_oct=False):
            # Tin axes: A,B, x_1..x_t, y_1..y_t ; output axes: A,B, then per block: k (row) or (x,y) (col)
            xs = letters[:t]; ys = letters[t:2 * t]; ks = letters[2 * t:3 * t]
            ins = 'AB' + xs + ys
            ops = [Tin]; subs = [ins]; out = 'AB'
            for s in range(t):
                if s in Q:
                    out += xs[s] + ys[s]
                else:
                    ops.append(kern); subs.append(xs[s] + ys[s] + ks[s]); out += ks[s]
            return fe(','.join(subs) + '->' + out, *ops)
        T = apply_rows(Pt, KRt)
        if top_variant == 'xy':
            T_oct = apply_rows(Pt, OCT)
            oct_ok &= np.array_equal(T, T_oct)
        B = apply_rows(Pbt, KRb)
        # final contraction: apply the column kernel to B one Q-block at a time, then a dot product with T.
        # T and B axes: A, B, then per block s: k (row) or two axes (col), in block order.
        Bk = B.astype(np.float64)
        pos = 2
        for s in range(t):
            if s in Q:
                # axes pos, pos+1 of Bk are (u_s, v_s); replace by (x_s, y_s) via KC[x,y,u,v]
                Bk = np.tensordot(Bk, KC.astype(np.float64), axes=([pos, pos + 1], [2, 3]))  # new axes x,y at end
                Bk = np.moveaxis(Bk, [-2, -1], [pos, pos + 1])
                pos += 2
            else:
                pos += 1
        assert np.all(np.abs(Bk) < 2.0**52)
        ZQ = int(np.einsum('i,i->', T.ravel(), np.rint(Bk).astype(np.int64).ravel()))  # int64: |terms| < 2^33, < 2^25 terms
        total += ZQ
        # re-indexing route for the top table: row blocks split alternately into R1 (y inner, k on a side)
        # and R2 (x inner, k on a' side); the table is ONE matrix product over (z_aa', y_R1, x_R2).
        if top_variant == 'xy':
            R = [s for s in range(t) if s not in Q]; R1 = R[0::2]; R2 = R[1::2]
            Wat = Wa.reshape([N, N] + [8] * t)    # (z_aa', z_ab, x_1..x_t)
            Wa2t = Wa2.reshape([N, N] + [8] * t)  # (z_aa', z_a'b', y_1..y_t)
            # A'[z_aa', z_ab, per block: R1 -> (y_s, k_s) ; else x_s]; value sigma(k^y, y) W_a[..x_s = k^y..]
            Ap = Wat
            for s in R1:
                ax = 2 + s  # x_s axis position stays at 2+s if we insert new axes at the end and move later
            # build explicitly with index arrays (small sizes)
            def reindex_a(W):
                out = np.zeros(W.shape[:2] + tuple(64 if s in R1 else 8 for s in range(t)), dtype=np.int64)
                for idx in itertools.product(*[range(64) if s in R1 else range(8) for s in range(t)]):
                    src = []; sg = 1
                    for s, j in enumerate(idx):
                        if s in R1:
                            y, k = divmod(j, 8); x = k ^ y; src.append(x); sg *= S[x, y]
                        else:
                            src.append(j)
                    out[(slice(None), slice(None)) + idx] = sg * W[(slice(None), slice(None)) + tuple(src)]
                return out
            def reindex_b(W):
                out = np.zeros(W.shape[:2] + tuple(64 if s in R2 else 8 for s in range(t)), dtype=np.int64)
                for idx in itertools.product(*[range(64) if s in R2 else range(8) for s in range(t)]):
                    src = []; sg = 1
                    for s, j in enumerate(idx):
                        if s in R2:
                            x, k = divmod(j, 8); y = k ^ x; src.append(y); sg *= S[x, y]
                        else:
                            src.append(j)
                    out[(slice(None), slice(None)) + idx] = sg * W[(slice(None), slice(None)) + tuple(src)]
                return out
            A2 = reindex_a(Wat); B2 = reindex_b(Wa2t)
            # einsum: A2 axes (z, A, per block), B2 axes (z, B, per block). For R1: A2 has (y,k) packed (64),
            # B2 has y (8). For R2: A2 has x (8), B2 has (x,k) packed (64). For Q: A2 has x, B2 has y (both outer).
            A2u = A2.reshape(A2.shape[:2] + tuple(x for s in range(t) for x in ((8, 8) if s in R1 else (8,))))
            B2u = B2.reshape(B2.shape[:2] + tuple(x for s in range(t) for x in ((8, 8) if s in R2 else (8,))))
            sa = 'zA'; sb = 'zB'; out = 'AB'
            for s in range(t):
                l0, l1, l2 = letters[3 * s:3 * s + 3]
                if s in R1:
                    sa += l0 + l1; sb += l0; out += l1          # y inner, k outer (a side)
                elif s in R2:
                    sa += l0; sb += l0 + l1; out += l1          # x inner, k outer (a' side)
                else:
                    sa += l0; sb += l2; out += l0 + l2          # column block: x, y outer
            T_mm = fe(sa + ',' + sb + '->' + out, A2u, B2u)
            mm_ok &= np.array_equal(T_mm, T)
    return Z1, total, t, oct_ok, mm_ok

def job(args):
    lam, seed, tv, bv = args
    Z, tot, t, ok1, ok2 = run(lam, seed, top_variant=tv, bot_variant=bv)
    if tv == 'xy' and bv == 'vu':
        return (f"lambda={lam} seed={seed}: Z={Z}, sum_Q Z_Q={tot}, 2^t Z={2**t * Z}, equal={tot == 2**t * Z}, "
                f"top table via octonion tensor equal={ok1}, via one matrix product (re-indexing) equal={ok2}")
    return f"negative control lambda={lam} top={tv} bottom={bv}: sum_Q Z_Q == 2^t Z: {tot == 2**t * Z}"
if __name__ == '__main__':
    from multiprocessing import Pool
    import sys as _s
    jobs = [(3, 0, 'xy', 'vu'), (3, 1, 'xy', 'vu'), (3, 0, 'yx', 'vu'), (3, 0, 'xy', 'uv'),
            (6, 0, 'xy', 'vu'), (6, 1, 'xy', 'vu'), (6, 0, 'yx', 'vu'), (6, 0, 'xy', 'uv')]
    if len(_s.argv) > 1: jobs = [j for j in jobs if str(j[0]) == _s.argv[1]][:int(_s.argv[2])]
    with Pool(4) as pool:
        for line in pool.imap(job, jobs):
            print(line, flush=True)
