# Unification check: the one-level grid model of r122d_model.py run at g = 2 (K = 4), joint max-min over term
# profiles (numerical search, Powell), compared with the K = 4 values of Section 2.3 (r122c_exponents.py).
# Output r122_unify_g2_out.txt.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r122d_model as M, r122d_values as V
ws, _ = V.load()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r122_unify_g2_out.txt'), 'w') as out:
    for wn in ('om2', 'current'):
        for kind in ('B', 'G', 'E', 'U'):
            out.write(f'{wn:8s} {kind}: joint max-min exponent of N at K = 4: {M.maxmin_joint(kind, 2, ws[wn], starts=30)[0]:.4f}\n')
print(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r122_unify_g2_out.txt')).read())
