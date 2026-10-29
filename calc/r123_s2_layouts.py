"""Is the 2 x 3 grid the best place for ideal crossing pieces at K = 6 (Section 4.6, Table 12 of the paper)?  Random layouts: a set of
edge-disjoint crossings (pairs of disjoint edges (a,d), (b,c)) of K6, every other edge raw; ideal pieces (see
r123_s2_sizegame.py); value = max over mixtures (differential evolution + Powell; a lower estimate of the max) of the
min over all contraction trees of the largest table. Exploratory: the minimum over layouts of lower estimates is not
a certified bound. Output calc/r123_s2_layouts_out.txt (+ _jobs.jsonl)."""
import itertools, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from resumable import run_jobs
import r123_s2_sizegame as S

K = 6
EDGES = list(itertools.combinations(range(K), 2))


def layout(seed, m):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        free = set(EDGES); cross = []
        order = rng.permutation(len(EDGES))
        for i in order:
            e = EDGES[i]
            if e not in free:
                continue
            partners = [f for f in free if f != e and not set(f) & set(e)]
            if not partners:
                continue
            f = partners[rng.integers(len(partners))]
            free -= {e, f}
            (a, d), (b, c) = e, f
            cross.append((a, b, c, d))
            if len(cross) == m:
                break
        if len(cross) == m:
            raw = sorted(free)
            return raw, cross
    return None


def job(j):
    lay = layout(j['seed'], j['m'])
    if lay is None:
        return None
    raw, cross = lay
    G = S.Game(K, raw, cross)
    v, x = S.maxmin(G, 'ideal', j['seed'], 1)
    return {'value': v, 'cross': cross, 'profile': np.round(x, 3).tolist()}


if __name__ == '__main__':
    jobs = [{'seed': s, 'm': m} for m in range(1, 8) for s in range(40)]
    res = run_jobs(jobs, job, os.path.join(HERE, 'r123_s2_layouts_jobs.jsonl'), processes=4)
    rows = [(r['value'], j['m'], r['cross']) for j in jobs if (r := res[json.dumps(j, sort_keys=True)])]
    rows.sort()
    with open(os.path.join(HERE, 'r123_s2_layouts_out.txt'), 'w') as out:
        out.write('K = 6, ideal crossing pieces, random layouts: best 15 (value = search estimate of the max-min '
                  'table size; tree 8, 2 x 3 grid see r123_s2_sizegame_out.txt)\n')
        for v, m, c in rows[:15]:
            out.write(f'{v:.4f} (e = {v / 15:.4f})  m = {m}  crossings (a,b,c,d) {c}\n')
        for m in range(1, 8):
            vs = [v for v, mm, _ in rows if mm == m]
            if vs:
                out.write(f'm = {m}: {len(vs)} layouts, min {min(vs):.4f}, median {np.median(vs):.4f}\n')
    print(open(os.path.join(HERE, 'r123_s2_layouts_out.txt')).read())
