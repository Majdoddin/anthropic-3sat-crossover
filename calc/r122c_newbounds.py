# The K = 4 constructions of r122c_exponents.py under the OpenAI bounds: omega <= 9/4, alpha > 0.465, omega(1,0.709,1) < 2.092 (companion paper, over C),
# and optionally Uznanski's alpha >= 1/2 (unverified). Upper bound on omega(1,k,1) = linear interpolation between these
# points (valid by convexity of omega(1,k,1)). Reuses the cost models of r122c_exponents.py unchanged.
import numpy as np, r122c_exponents as X
MODELS = {'OpenAI (alpha>0.465)': [(0.465, 2.0), (0.709, 2.092), (1.0, 2.25)],
          'OpenAI + Uznanski (alpha=1/2)': [(0.5, 2.0), (0.709, 2.092), (1.0, 2.25)]}
for name, pts in MODELS.items():
    X.ks, X.ws = map(np.array, zip(*pts)); X.OMEGA = 2.25
    tree = 2 * float(X.w1k1(0.5, 'today'))
    gAL = np.log(np.sqrt(8 ** (4 * 2.25 / 3) - 2 * (2 ** (2.25 / 3) - 1) * 8 ** (2 * 2.25 / 3))) / np.log(8)
    gU = X.bisect(lambda g: X.E_U(g, 'today')[0] < tree - 1e-9, 1.0, 2.0)
    v9 = X.E_V(9.0, 'today'); v1 = X.E_V(1.0, 'today')
    b = X.E_B('today', n2=21)
    print(f"{name}: tree {tree:.4f}; U at gamma=1 {X.E_U(1.0,'today')[0]:.4f}, U gains iff R~(O) < {8**gU:.3f} "
          f"(Alman-Li bound at omega=9/4: {8**gAL:.2f}); V no octonion {v9[0]:.4f} (a={v9[1]:.3f}, d={v9[2]:.3f}); "
          f"V gamma=1 {v1[0]:.4f}; E gamma=1 {X.E_E(1.0,'today')[0]:.4f}; B (m->inf) {b[0]:.4f}")
# without the companion's point omega(1,0.709,1) < 2.092 (only alpha > 0.465 and omega <= 9/4, both Lean-claimed)
X.ks, X.ws = np.array([0.465, 1.0]), np.array([2.0, 2.25]); tree = 2 * float(X.w1k1(0.5, 'today'))
v9 = X.E_V(9.0, 'today')
print(f"alpha>0.465 and 9/4 only: tree {tree:.4f}; V no octonion {v9[0]:.4f} (d={v9[2]:.3f}); B {X.E_B('today', n2=21)[0]:.4f}")
# The curve omega(1,k,1) = 2 (k <= 1/2), <= 1 + k + 1/(4k) (k >= 1/2): Amanov-van den Berg-Zuiddam arXiv 2610.30054
# Thm 1.2 / Cor. 4.17 (over every field), same formula as Uznanski Cor. 1.2; Lean claimed by Zhou-Wang (every assoc.
# ring) and by Uznanski (assuming Strassen's spectral theorem). Convex, so a dense table is a valid upper bound.
kk = np.linspace(0.5, 1.0, 501)
X.ks = np.concatenate([[0.0], kk]); X.ws = np.concatenate([[2.0], 1 + kk + 1 / (4 * kk)]); X.OMEGA = 2.25
tree = 2 * float(X.w1k1(0.5, 'today'))
gU = X.bisect(lambda g: X.E_U(g, 'today')[0] < tree - 1e-9, 1.0, 2.0)
v9 = X.E_V(9.0, 'today'); v1 = X.E_V(1.0, 'today')
print(f"AvdBZ curve 1+k+1/(4k): tree {tree:.4f}; V no octonion {v9[0]:.4f} (a={v9[1]:.3f}, d={v9[2]:.3f}); V gamma=1 {v1[0]:.4f}; "
      f"U gamma=1 {X.E_U(1.0,'today')[0]:.4f}, U gains iff R~(O) < {8**gU:.3f}; E gamma=1 {X.E_E(1.0,'today')[0]:.4f}; B {X.E_B('today', n2=21)[0]:.4f}")
