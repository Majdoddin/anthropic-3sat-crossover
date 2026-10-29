"""Per-level schedules of Table 10 (Appendix B.1 of the paper): the dynamic programme of r123_levers.run (free
schedule, tau = 1, overheads 3 zeta and 3/K, rho = 1/sqrt2) with the path traced, for density 1 and 4.26, brute-force
base (x0 = 7) and tree base (x0 = 8, e0 = 2.042994/8). Each path entry is (log2 K_l, e_l, log2 zeta_l, log2 s_l) of
one level; the final line of each case gives the top level. Prints to stdout."""
import sys, math
sys.path.insert(0, 'calc')
from r123_levers import s_min
def run(bar, rho=2**-0.5, x0=7, e0=0.5, L=14):
    zetas = [2.0 ** (-j / 2) for j in range(4, 36)]
    front = [(float(x0), e0, [])]
    for l in range(L):
        done = [st for st in front if st[1] < bar]
        if done:
            best = min(done, key=lambda t: t[0]); return l, best
        new = []
        for x, e, path in front:
            for z in zetas:
                s = s_min(x, z, 1.0)
                new.append((2*(x+math.log2(s)), (rho+3*z)*e+3*2.0**-x, path+[(round(x,1), round(e,4), round(math.log2(z),1), round(math.log2(s),1))]))
        new.sort(key=lambda t: (t[0], t[1]))
        front, be = [], float('inf')
        for st in new:
            if st[1] < be - 1e-12: front.append(st); be = st[1]
for D, x0, e0 in [(1.0,7,0.5),(4.26,7,0.5),(1.0,8,2.042994/8),(4.26,8,2.042994/8)]:
    bar = min(0.5283, 0.711/D)/6
    l, (x, e, path) = run(bar, x0=x0, e0=e0)
    print(D, x0, e0, 'levels', l, 'log2K', round(x), 'e', round(e,5)); print('  path (log2K, e, log2 zeta, log2 s):', path)
