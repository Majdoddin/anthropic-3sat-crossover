"""Rough threshold of OpenAI's main-text construction (OpenAI's paper, sec. 2.6-2.7) on 3-occurrence 3-CNF
(Section 3.9 of the paper).
Per-n exponent (2.11): E = (3R/Q) log2(2eQ/(3R)) [Taylor data] + 2 tau log2(2D+1) [measurements] + 2/r [one group],
with D = r^{2Q}, and the measurement count needs 2 D exp(-R^2/(8Q)) <= D tau/(3r), i.e. tau >= 6 r exp(-R^2/(8Q)).
Size constraint: the balanced schedules need 12 D^2 <= M ~ 3n (2.17); optimistically D <= n. So log2 n* >= 2Q log2 r
(optimistic) or 4Q log2 r + log2 4 (paper). Minimise log2 D subject to E <= target. Main term only; estimate."""
import math
def E(R, Q, r):
    tau = 6 * r * math.exp(-R * R / (8 * Q))
    return 3 * R / Q * math.log2(2 * math.e * Q / (3 * R)) + 2 * tau * (2 * Q * math.log2(r) + 1) + 2 / r
with open('calc/r123_openai_out.txt', 'w') as out:
    for name, target in (('3-occurrence class vs pathwidth 2^{n/3}', 1 / 3), ('density theorem via copying (N = 3m) vs 2^{0.5283 m}', 0.5283 / 3)):
        best = None
        for r in range(4, 400, 2):
            for lq in [x / 10 for x in range(20, 130)]:
                Q = 10 ** lq
                if best is not None and 2 * Q * math.log2(r) >= best[0]:
                    continue
                Emin = min(E(Q * 10 ** (-y / 20), Q, r) for y in range(16, 200))
                if Emin <= target:
                    best = (2 * Q * math.log2(r), r, Q)
        lD, r, Q = best
        out.write(f'{name}: target E < {target:.3f}: r = {r}, Q = {Q:.2e}: log2 D = {lD:.2e}; log2 n* >= {lD:.2e} (D <= n), {2 * lD + 2:.2e} (12 D^2 <= 3n)\n')
print(open('calc/r123_openai_out.txt').read())
