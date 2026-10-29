"""Explicit main-term threshold of OpenAI's Appendix D (Boolean tensor contraction) (Section 3.9 of the paper).

Model (derived from the text of OpenAI's Appendix D; see Section 3.9 of the paper):
  log2 T(s) ~ s/sqrt(lam) * ( Cw * M0/s + 1 ),   lam = ceil(log2 s)
  Cw = (1/ln2) * 2*pi*sqrt(v*pc/6) / sqrt(eps) * G(rho, gamma)
where per weight there are v*M/D mode variables, the call precision is
p_t ~ pc*M*D/u, u_d = eps*lam/gamma^d, children have <= rho*M wires, and
G = sum_d (rho*sqrt(gamma))^d (geometric sum over depths; u_d shrinks by gamma).
The '+1' is the leaf term 2^H, H = s/floor(sqrt(lam)).
Threshold: lam* = ((Cw*M0/s + 1)/bar)^2, bar = competitor exponent per unit of s.
Lower-order terms (poly factors, inherited precision, rounding) are dropped:
they only add cost, so lam* is a lower bound for the bound as proved.
"""
import math

LN2 = math.log(2)


def G(rho, gamma):
    q = rho * math.sqrt(gamma)
    assert q < 1
    return 1.0 / (1.0 - q)


def Cw(v, pc, eps, rho, gamma=2.0, use_G=True):
    nats = 2 * math.pi * math.sqrt(v * pc / 6.0)
    g = G(rho, gamma) if use_G else 1.0
    return nats / LN2 / math.sqrt(eps) * g


def eps_for_gamma(eps_halving, gamma):
    # constraint is on a_max = 2*u0/(1-1/gamma); halving gives a_max = 4*u0
    return eps_halving * 4.0 / (2.0 / (1.0 - 1.0 / gamma))


# instances: s = n+m, k = 3
dens = {
    1.0: dict(bar_per_s=0.5283 * 1.0 / 2.0,          # 2^{0.5283 m}, m = s/2
              M0s_written=6.0 / 2.0,                 # 2*(sum_v e(d_v)) = 6n = 3s (min over instances)
              M0s_tweak=3.0 / 2.0),                  # km = 3m = 1.5 s
    4.26: dict(bar_per_s=0.711 / 5.26,               # 2^{0.711 n}, n = s/5.26
               M0s_written=2 * (2 * 3 * 4.26 - 3) / 5.26,  # 2*(2*12.78n - 3n)/5.26n
               M0s_tweak=3 * 4.26 / 5.26),
}

variants = {
    # name: (v, pc, eps, rho, gamma, M0 key)
    "as written (eps=2^-17, 6M/D modes, p_t=8*(3M/u)*D, child 3M/32)":
        (6, 24, 2.0 ** -17, 3 / 32, 2.0, "M0s_written"),
    "as written, paper's own child bound M/8":
        (6, 24, 2.0 ** -17, 1 / 8, 2.0, "M0s_written"),
    "written constants, network km wires":
        (6, 24, 2.0 ** -17, 3 / 32, 2.0, "M0s_tweak"),
    "eps=1/960 only (Delta=s^0.9 kept)":
        (6, 24, 1 / 960, 3 / 32, 2.0, "M0s_written"),
    "tweak: eps=1/100 (Delta=2L), v=2, pc=6, child M/32, km wires":
        (2, 6, 1 / 100, 1 / 32, 2.0, "M0s_tweak"),
}

lines = []
P = lines.append
P("R123 OpenAI App. D threshold (main term).  lam = ceil(log2 s), s = n+m, k = 3")
P("")
for name, (v, pc, eps, rho, gamma, key) in variants.items():
    c = Cw(v, pc, eps, rho, gamma)
    c1 = Cw(v, pc, eps, rho, gamma, use_G=False)
    P(f"[{name}]")
    P(f"  per-root-wire constant Cw = {c:.1f} bits (depth-0 only: {c1:.1f}; G = {G(rho, gamma):.4f})")
    for D, d in dens.items():
        m0 = d[key]
        per_s = c * m0 + 1.0
        lam = (per_s / d["bar_per_s"]) ** 2
        lam0 = ((c1 * m0 + 1.0) / d["bar_per_s"]) ** 2
        P(f"  density {D}: M0/s = {m0:.3f}, bar/s = {d['bar_per_s']:.5f}, "
          f"coef = {per_s:.1f}/sqrt(lam)  ->  log2 s* >= {lam:.3e}  (depth-0 only {lam0:.3e})")
    P("")

# schedule gamma and r optimisation for the tweak
P("tweak, optimising r (rho = 2/r, eps = 1/(4(4 log2 r + 1)) for halving) and gamma:")
best = None
for r in [4, 8, 16, 32, 64, 128, 256]:
    rho = 2.0 / r
    eps_h = 1.0 / (4 * (4 * math.log2(r) + 1))
    for gamma in [1.5, 2, 3, 4, 5, 6, 8, 12, 16]:
        if rho * math.sqrt(gamma) >= 1:
            continue
        eps = eps_for_gamma(eps_h, gamma)
        c = Cw(2, 6, eps, rho, gamma)
        if best is None or c < best[0]:
            best = (c, r, gamma, eps)
    P(f"  r={r:4d}: eps(halving)={eps_h:.5f}, Cw(gamma=2)={Cw(2, 6, eps_h, rho, 2.0):.1f}")
c, r, gamma, eps = best
P(f"  best: r={r}, gamma={gamma}, eps={eps:.5f}, Cw={c:.1f}")
for D, d in dens.items():
    lam = ((c * d["M0s_tweak"] + 1.0) / d["bar_per_s"]) ** 2
    P(f"  density {D}: log2 s* >= {lam:.3e}")
P("")

# validity checks of the 'sufficiently large s' conditions as literally written
P("literal (D.5) condition u >= lam^{4/5} at the deepest active depth (2^d < (C sqrt lam)^{1/3}):")
eps = 2.0 ** -17
for C in [3.0, 8.578]:
    lam_req = (C ** (1 / 3) / eps) ** 30
    P(f"  C = {C}: needs lam >= (C^(1/3)/eps)^30 = 2^{math.log2(lam_req):.1f}")
P(f"  even depth 0 (u0 = eps*lam >= lam^0.8) needs lam >= eps^-5 = 2^{5 * 17}")
P("")

# depth profile at the as-written density-1 threshold
lam = 4.43e10
P(f"depth profile, as written, density 1, lam = {lam:.3e}:")
M = 3.0
d = 0
while M > 1 / math.floor(math.sqrt(lam)):
    u = eps * lam / 2 ** d
    share = 2 * math.pi * math.sqrt(24) * M / math.sqrt(u) / LN2
    P(f"  d={d}: M/s={M:.3e}, u={u:.3e}, bits/s * 1/sqrt(lam)-scaled contribution = {share * math.sqrt(lam):.1f}"
      f", inherited-precision ratio ~ (32/3)(1/2)/2^u = 2^-{u:.0f}")
    M *= 3 / 32
    d += 1
P(f"  leaf wires <= H = s/{math.floor(math.sqrt(lam))}")

out = "\n".join(lines)
print(out)
with open(__file__.replace(".py", "_out.txt"), "w") as f:
    f.write(out + "\n")
