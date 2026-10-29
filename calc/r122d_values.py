# Build the w functions from values.yaml (latest entries, no superseded_by) and VXXZ's Table 1 (peer-reviewed
# fallback, copied in r122c_exponents.py).  Nothing about the current curve is hard-coded here.
import os, re, yaml, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r122d_model as M

YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'values.yaml')


def _num(s):
    return float(eval(str(s).split()[0], {}, {}))


def load(general=False):
    V = yaml.safe_load(open(YAML))
    for key in ('omega_now', 'alpha_now', 'omega_rect_curve'):
        assert 'superseded_by' not in V[key], f'{key} is superseded in values.yaml; update r122d_values.py'
    omega = _num(V['omega_now']['value']); alpha = _num(V['alpha_now']['value'])
    expr = V['omega_rect_curve']['value']
    py = re.sub(r'(\d)\s*k', r'\1*k', expr)          # "1/(4k)" -> "1/(4*k)"
    cur = M.w_curve(py, alpha, omega, f'current (values.yaml: alpha {alpha}, omega {omega}, w = {expr})')
    import r122c_exponents as X
    vx = M.w_table(X.T, X.OMEGA, f'VXXZ Table 1 chords (omega {X.OMEGA})')
    out = {'om2': M.w_const2(), 'current': cur, 'vxxz': vx}
    if general:
        out['current+gen'] = M.with_general(cur, M.avdbz_general, cur.name + ' + AvdBZ general-shape formula (r101b.md)')
    return out, (omega, alpha, expr)
