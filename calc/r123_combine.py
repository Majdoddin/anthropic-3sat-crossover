"""How the general-3-SAT levers combine (sparsification Delta, per-level rho, start x0 + top q); Section 3.9 and
Table 7 of the paper.
Uses r123_sparsify's high-precision e* and r123_general.threshold; main term, lower bounds on log2 n*.
Run with system python3 (needs mpmath). Output: calc/r123_combine_out.txt."""
import r123_sparsify as S, r123_general as G

bar = 0.4092
with open('r123_combine_out.txt', 'w') as out:
    for b in ('calabro_paper', 'const_2_noclamp'):
        S.use(b)
        es = S.e_star2(bar)
        for rho in (2 ** -0.5, 0.625, 0.5):
            for x0, q in ((18, 17), (7, 2.2)):
                l, xK, ln = G.threshold(es, rho, x0, q)
                out.write(f'{b:16s} e*={es:.2e} rho={rho:.3f} x0={x0} q={q}: levels {l}, log2 n* {ln:.2e}\n')
