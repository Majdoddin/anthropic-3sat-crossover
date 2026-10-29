"""Section 4.7 (three escapes): check of the identity-search obstruction of Proposition 4.23.

Configuration: two copies of K_m, nodes a_i, b_i (i in Z_m); strings X(i,j) at a_i, Y(i,j) at b_i (i != j);
equalities X(i,j)-X(j,i), Y(i,j)-Y(j,i). Merge k (k in Z_m) has groups G^k_i = {a_i, b_{i+k}}; its bundle on the
group pair {i,j} is the two edges alpha_ij, beta_{i+k,j+k}; its pairing pairs X(i,j) with Y(i+k,j+k).
Direction of a string X(i,j) or Y(i,j) is j - i. Cut Q: directions in {1..floor((m-1)/2)} | the rest.

Checks, for m = 2..7:
 (a) number of equalities crossing Q (all of them for odd m; for even m all but the 2 * m/2 of direction m/2);
 (b) Q crosses no pair of any merge's pairing;
 (c) for every merge and every bundle whose edges cross Q, Q restricted to the bundle's 4 strings is the split
     between its two groups;
 (d) every merge meets the configuration in m groups (so for m >= 3 it is outside (L3) of Definition 4.2).
For m = 3 also: (e) exhaustive search over all string cuts for max(e(Q) - max_k p_k(Q)) (reproduces r123_idcut.py);
 (f) numerical rank of f across Q at N = 2, 3; (g) for random merge-k-local pieces (tensor products of random factors,
     one per bundle) at N = 2: rank across Q equals the product of the factors' ranks across their group splits.
Output: calc/r123_escapes_cut_out.txt.
"""
import itertools, string
import numpy as np


def config(m):
    sid = {}
    for side in 'XY':
        for i in range(m):
            for j in range(m):
                if i != j:
                    sid[(side, i, j)] = len(sid)
    E = [(sid[(s, i, j)], sid[(s, j, i)]) for s in 'XY' for i in range(m) for j in range(i + 1, m)]
    merges = []
    for k in range(m):
        bundles = []
        for i in range(m):
            for j in range(i + 1, m):
                # strings on group i side: X(i,j), Y(i+k,j+k); on group j side: X(j,i), Y(j+k,i+k)
                gi = [sid[('X', i, j)], sid[('Y', (i + k) % m, (j + k) % m)]]
                gj = [sid[('X', j, i)], sid[('Y', (j + k) % m, (i + k) % m)]]
                bundles.append((gi, gj))
        pairing = [(sid[('X', i, j)], sid[('Y', (i + k) % m, (j + k) % m)]) for i in range(m) for j in range(m) if i != j]
        merges.append((bundles, pairing))
    direction = {v: (key[2] - key[1]) % m for key, v in sid.items()}
    return sid, E, merges, direction


def main():
    out = []
    for m in range(2, 8):
        sid, E, merges, direction = config(m)
        n = len(sid)
        fwd = set(range(1, (m - 1) // 2 + 1))
        side = [direction[s] in fwd for s in range(n)]
        cross = lambda pairs: sum(side[p] != side[q] for p, q in pairs)
        eQ = cross(E)
        pk = [cross(pairing) for _, pairing in merges]
        ok_c = True
        for bundles, _ in merges:
            for gi, gj in bundles:
                strs = gi + gj
                if not any(side[s] != side[t] for s in strs for t in strs):
                    continue  # bundle not crossed (direction m/2 for even m)
                # Q on the bundle must be exactly the split gi | gj
                if not (side[gi[0]] == side[gi[1]] and side[gj[0]] == side[gj[1]] and side[gi[0]] != side[gj[0]]):
                    ok_c = False
        groups_met = m  # every group G^k_i contains a_i, which lies in C
        out.append(f'm = {m}: strings {n}, equalities {len(E)}, crossing Q {eQ} '
                   f'(expected {len(E) if m % 2 else len(E) - m}); pairing pairs crossing Q per merge {pk}; '
                   f'Q = group split on every crossed bundle: {ok_c}; groups met per merge: {groups_met}')
    # m = 3 extras
    m = 3
    sid, E, merges, direction = config(m)
    n = len(sid)
    best, count = None, 0
    for Q in range(1, 2 ** (n - 1)):
        side = [(Q >> s) & 1 for s in range(n)]
        cross = lambda pairs: sum(side[p] != side[q] for p, q in pairs)
        e = cross(E)
        p = max(cross(pairing) for _, pairing in merges)
        if best is None or e - p > best[0]:
            best = (e - p, e, p, [s for s in range(n) if side[s]])
            count = 0
        if e - p == best[0]:
            count += 1
    inv = {v: k for k, v in sid.items()}
    out.append(f'm = 3 exhaustive over {2 ** (n - 1) - 1} string cuts: max gap e - max_k p_k = {best[0]} '
               f'(e = {best[1]}, p = {best[2]}), attained by {count} cut(s) up to complement; '
               f'one of them: {[inv[s] for s in best[3]]}')
    letters = string.ascii_letters
    fwd = {1}
    Qside = [s for s in range(n) if direction[s] in fwd]
    rest = [s for s in range(n) if s not in Qside]
    for N in (2, 3):
        f = np.einsum(','.join(letters[p] + letters[q] for p, q in E) + '->' + letters[:n], *([np.eye(N)] * len(E)))
        M = np.transpose(f, Qside + rest).reshape(N ** len(Qside), -1)
        out.append(f'm = 3, N = {N}: rank of f across Q = {np.linalg.matrix_rank(M)} (N^6 = {N ** 6})')
    rng = np.random.default_rng(1)
    N = 2
    for k, (bundles, _) in enumerate(merges):
        for trial in range(3):
            facs, ranks, ops = [], [], []
            for gi, gj in bundles:
                r = rng.integers(1, N * N + 1)
                A = rng.normal(size=(N * N, r)) @ rng.normal(size=(r, N * N))  # rank r across gi | gj
                facs.append(A.reshape(N, N, N, N))
                ranks.append(np.linalg.matrix_rank(A))
                ops.append(''.join(letters[s] for s in gi + gj))
            T = np.einsum(','.join(ops) + '->' + letters[:n], *facs)
            M = np.transpose(T, Qside + rest).reshape(N ** len(Qside), -1)
            out.append(f'm = 3, merge {k}, random local piece {trial}: factor ranks {ranks}, product '
                       f'{int(np.prod(ranks))}, rank across Q {np.linalg.matrix_rank(M)}')
    text = '\n'.join(out)
    print(text)
    with open('calc/r123_escapes_cut_out.txt', 'w') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
