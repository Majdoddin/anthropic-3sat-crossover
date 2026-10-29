# Lowered crossover points of Anthropic's subexponential 3-SAT algorithm: scripts and outputs

This repository holds the scripts behind the computed values of the paper *Lowered crossover points of Anthropic's
subexponential 3-SAT algorithm*
(Ruhollah Majdoddin, October 2026), each with the output it produced. The paper itself is [`paper.pdf`](paper.pdf), the
version submitted to arXiv on 30 October 2026. The paper refers to the scripts by the path
`calc/NAME.py` relative to the top of this repository. The table below lists, for every script, the items of the paper whose values
it computes or checks and the output file in which the values appear.

## How to run

The scripts were run with Python 3.13 and numpy 2.3.4, scipy 1.16.3 and PyYAML 6.0.3. Three scripts
(`r123_sparsify.py`, `r123_combine.py`, `r123_chr_check.py`) use mpmath 1.3.0 for high-precision arithmetic and need
neither numpy nor scipy. After cloning, a script is run from the top of the repository as

```
git clone https://github.com/Majdoddin/anthropic-3sat-crossover
cd anthropic-3sat-crossover
python3 calc/NAME.py
```

Four scripts write their files relative to the working directory and are run from inside `calc/`:

```
cd anthropic-3sat-crossover/calc
python3 r122d_tables9.py        # likewise r122d_subst.py, r122d_bfinite.py, r123_combine.py
```

No script needs arguments. `r122b_expansion.py` accepts an optional pair `LAMBDA COUNT` that restricts the run to the
first COUNT jobs at that lambda, and `r123_baseline.py` an optional path for its JSON output. Scripts marked "prints"
in the table write their result to standard output, which is redirected into the output file, for example

```
python3 calc/r122d_certify.py > calc/r122d_certify_out.txt
```

The others write their output file themselves. Some scripts run their jobs in a pool of 2 to 4 processes (column
"Proc."). Scripts built on `resumable.py` append every finished job to `calc/NAME_jobs.jsonl` and skip the jobs found
there when they are started again, so an interrupted run resumes where it stopped; a run from scratch starts without
that file. For `r123_idsearch.py` this job file is the output `r123_idsearch_out.jsonl` itself.

Helper modules, imported by the scripts and without output of their own: `r122d_model.py` (the one-level grid model
of Section 2.4 for any upper-bound curve w of omega(1,k,1)), `r122d_values.py` (builds the three curves of the paper:
omega = 2, the current curve from `values.yaml`, and the VXXZ table), and `resumable.py` (resumable job runs). Several
scripts in the table are also imported by others: `r122c_exponents.py` (the K = 4 cost model and the VXXZ table),
`r122d_certify.py`, `r122d_tables9.py`, `r123_levers.py` (the model of the recursion), `r123_general.py`,
`r123_sparsify.py`, `r123_s1_grid.py`, `r123_s2_sizegame.py`, `r123_trace_c.py` and `r123_idsearch.py`.

`values.yaml` holds the three literature values that the scripts read (through `r122d_values.py`): omega <= 9/4,
alpha >= 1/2 and omega(1,k,1) <= 1 + k + 1/(4k) for 1/2 <= k <= 1, each with its source.

## What the values are

The paper uses these labels. A *certified* value is an upper bound on the exponent of a construction over all term
profiles, from closed-form bounds and a certified grid (Section 2.4, "Two kinds of numbers occur"). A *search* value is
the largest value found by a numerical search over profiles, a lower estimate of the construction's exponent (same
place). A *computed* threshold comes from a numerical optimisation over admissible schedules and is exact for the
model; each reported schedule can be checked level by level (Section 3.2, "Status labels"). An *estimate* is obtained
from the cruder closed level formula of Anthropic's Theorem A(a) (same place). A *check* is a computation that verifies
a proved statement, or the finite part of a proof, exactly or on a stated range.
The paper rounds certified upper bounds up and the lower ends of certified intervals down, so a printed bound
can differ from the output in its last digit.

## Scripts

| Script | Paper items (kind of value) | Output | Proc. |
|---|---|---|---|
| `r122b_expansion.py` | Theorem 2.7: the expansion identity and the bypass, checked exactly at lambda = 3, 6 with negative controls (check) | `r122b_expansion_out.txt` (prints) | 4 |
| `r122c_exponents.py` | Section 2.3 and Table 2: covering hybrid 23/6 at omega = 2 and 4.0790 on the VXXZ curve; App. B pieces 2 + 4/(4 - sqrt2) = 3.547 and 4.0802, mixtures never worse than the pure type; best tree 4.0860 on the VXXZ curve; octonion-rank threshold 10.03; grouping counted by ranks 3.8179, floor 3.8012; Table 3, K = 4 (computed; closed forms where the paper says so) | `r122c_exponents_out.txt` (prints) | 1 |
| `r122c_followup.py` | Section 2.3: grouping counted by ranks 3.8179 and its floor 3.8012; App. B pieces at m = 64 and 256 (3.796, 3.631; 4.140 on the VXXZ curve, Table 2); random mixtures of piece types (computed) | `r122c_followup_out.txt` (prints) | 4 |
| `r122c_newbounds.py` | Table 2, current and OpenAI rows (3.9305, 3.9142; 4.0264, 3.9963, 3.9720); Section 2.3: octonion-rank threshold 12.34 on the current curve, Alman-Li bound 21.94; Section 1: about N^3.93 (computed) | `r122c_newbounds_out.txt` (prints) | 1 |
| `r122_unify_g2.py` | Section 4.6: 3.566 for App. B pieces at K = 4 in the grid model of Section 2.4 (search) | `r122_unify_g2_out.txt` | 1 |
| `r122d_certify.py` | Section 2.4 and Table 9: certified bounds at K = 16 (grouping by ranks 59.229, App. B pieces 57.95 and 59.95, G, EM) and the K = 9 tops (certified) | `r122d_certify_out.txt` (prints) | 3 |
| `r122d_tables9.py` | Table 9: certified K = 9 row-table bounds (G 17.721, EM 17.657, E 17.593 at omega = 2; all three curves) (certified); run from `calc/` | `r122d_tables9_out.txt` | 3 |
| `r122d_subst.py` | Table 9, Section 2.4 and Table 3, K = 9 and 16: certified intervals of G, EM, E and App. B pieces on the three curves; best trees found under each curve 18 / 20.25 / 21.344 and 60 / 64 / 65.376 (certified; the trees are upper bounds) ; run from `calc/` after `r122d_tables9.py` | `r122d_subst_out.txt` | 3 |
| `r122d_bfinite.py` | Table 9 rows "B, m = 256 / 512" and Section 2.4: 16.456, 16.187, 18.019 at K = 9; 59.718, 61.72 at K = 16 (certified); run from `calc/` | `r122d_bfinite_out.txt` | 1 |
| `r123_s1_grid.py` | Section 2.4 and Table 3, K = 6: App. B pieces 7.357 (omega = 2) and 8.014 (current), grouping counted by ranks 8.116; best trees 8 and 9, width bound 8; Table 2: tree 4 (search; the trees are the best found) | `r123_s1_grid_out.txt` | 3 |
| `r123_s1_k16.py` | Section 2.4 and Table 3, K = 16: uniform profile 49.94 with a tree on top; with the K = 4 construction on top >= 46.862 (omega = 2) and >= 48.883 (current) (search) | `r123_s1_k16_out.txt` | 4 |
| `r123_k6_certify.py` | Section 2.4 and Table 3, K = 6: Phi <= 7.360, 8.017, 8.442 on the three curves; best profiles 7.357, 8.014; at m = 128: 7.818, 8.492; the monotonicity condition w - k w' >= 1 (certified) | `r123_k6_certify_out.txt` | 1 |
| `r123_s2_sizegame.py` | Table 8, Remark 2.9, column "ideal pieces" of Table 3, Appendix C: size-game values at K = 4, 6, 9 (search; 6.725 is a grid upper bound) | `r123_s2_sizegame_out.txt` | 2 |
| `r123_s2_layouts.py` | Table 12 and Section 4.6: 280 random layouts at K = 6, best value 7.0 (search) | `r123_s2_layouts_out.txt` | 4 |
| `r123_laser_threecut_lb.py` | Section 4.6 after Proposition 4.22: the three-cut size game at K = 4 has value at least 3 over all two-piece families on a grid of step 1/4 (search on the grid, not a proof for all real parameters) | `r123_laser_threecut_lb_out.txt` (prints) | 1 |
| `r123_density.py` | Table 4, columns (P), (G), (X); bars (7); Table 5 remark (i) (2474); Table 7 rows group size, initial size, rho = 1/2; abstract 2^33858 (computed) | `r123_density_out.txt` | 3 |
| `r123_density_exactU.py` | bars (8): 0.1321, 0.0711, 0.0444, 0.0302 (computed) | `r123_density_exactU_out.txt` | 3 |
| `r123_base_tree.py` | Table 4, column (T); Lemma 3.7 and after (5323 against 5396); Table 7 row base case (computed) | `r123_base_tree_out.txt` | 3 |
| `r123_base_tree_exactU.py` | Table 4, column (U); Table 7 row exact edge-bit total (computed) | `r123_base_tree_exactU_out.txt` | 3 |
| `r123_cluster_q.py` | Proposition 3.9 and the text after it (5037, q = 2.033, 44154, C_0 budgets 2^57.6 to 2^93.0); Table 7 row top balancing (computed) | `r123_cluster_q_out.txt` | 3 |
| `r123_cluster_q_exactU.py` | Table 4, columns (C) and (U+C), with q = 2.10, 2.41 and C_0 <= 2^61 of the caption; C_0 <= 2^51.1 after Proposition 3.9; abstract 2^420 (computed) | `r123_cluster_q_exactU_out.txt` | 3 |
| `r123_maxsat_paper.py` | Table 6, column (P): the paper's schedule against the Max-3-SAT bar (computed) | `r123_maxsat_paper_out.txt` | 1 |
| `r123_base_tree_maxsat.py` | bars (10) and Table 6, column (U) (computed) | `r123_base_tree_maxsat_out.txt` | 3 |
| `r123_cluster_q_maxsat.py` | Table 6, columns (C) and (U+C); C_0 budget 2^48.8 (computed) | `r123_cluster_q_maxsat_out.txt` | 3 |
| `r123_levers.py` | Table 7, second block: bar 0.0867, sharper tails tau = 9, overhead constants, no balancing slack (computed; the levers are hypothetical); the model of Section 3.4, whose per-level schedules Table 10 (Appendix B.1) lists | `r123_levers_out.txt` | 2 |
| `r123_levers_paths.py` | Table 10 (Appendix B.1): the per-level schedules of the model of `r123_levers.py`, with log2 K_1 = 38.5 and 40.7 (computed) | `r123_levers_paths_out.txt` | 1 |
| `r123_schedule.py` | Remark 3.3: closed-form slack schedule against the optimum (2529 / 2474, 5258 / 5177, 22343 / 21996; gamma = 0.031, 0.026, 0.019) (computed) | `r123_schedule_out.txt` | 1 |
| `r123_baseline.py` | Section 3.3: 8046 at level 4, which reproduces the remark "log m > 8000" of Anthropic's paper; Table 7 row growth of K (378) (computed with the paper's schedule; the levers are hypothetical) | `r123_baseline_out.txt` (prints), `r123_baseline_out.json` | 1 |
| `r123_szeider_check.py` | Section 1, footnote 1: the model of Section 3.3 against exhaustive search, log2 m* >= 275616 at m = 5n (the value of Szeider's addendum) and 8046 at m = n (check) | `r123_szeider_check_out.txt` | 1 |
| `r123_general.py` | Table 5, rows G0, G5, G7; abstract 2^{2.2e12}; Section 3.2 (counting and randomized bars move the thresholds by one or two levels); Section 3.10 (U/n about 16300, e* = 2.5e-5, 31 levels) (estimate) | `r123_general_out.txt`, `r123_general_out.json` | 1 |
| `r123_general_stable.py` | Table 5, rows G1, G4; Section 3.10 (U/n about 2500, e* = 1.6e-4, 25 levels); Table 7, about 1400 bits per variable in row G7 (estimate) | `r123_general_stable_out.txt` (prints) | 1 |
| `r123_sparsify.py` | Table 7, row sparsification density (6.2e7, 1.6e7, 32x); text after Proposition 3.15 (U/n about 210, 4.7 bits); row G7 with Calabro's bound as quoted (estimate) | `r123_sparsify_out.txt`, `r123_sparsify_out.json` | 1 |
| `r123_combine.py` | Table 7, row rho (general) at rho = 1/sqrt2 and 1/2; Section 3.9 (rho -> 1/2 worth 2000x and 500x; five levels against three) (estimate); run from `calc/` | `r123_combine_out.txt` | 1 |
| `r123_openai.py` | Section 3.9, OpenAI's algorithm: log2 n* >= 1.5e7 and 8.5e7 (estimate) | `r123_openai_out.txt` | 1 |
| `r123_appD.py` | Section 3.9 and Section 1, OpenAI's Appendix D: C_w = 18535 bits per root wire, log2 s* about 4.4e10 (density one) and 1.4e12 (density 4.26); with the improvements justified but not proved, C_w = 106 to 134 and log2 s* about 3.7e5 to 5.9e5 and 3.7e6 to 5.8e6 (estimate) | `r123_appD_out.txt` | 1 |
| `r123_trace_c.py` | Lemma 3.4 and Table 11: the requirements on c on the exact range, maximum 27.198 at lambda = 1, c = 30 (check) | `r123_trace_c_out.txt` | 1 |
| `r123_chr_check.py` | Lemma 3.5: the constants 9.234, 0.5456, 3.79248 + 0.04277 = 3.8353 and 3.7007; Lemma 3.4 and Table 11 caption: at most 9.59 beyond the exact range (check) | `r123_chr_check_out.txt` | 1 |
| `r123_grid_singleblock.py` | Lemma 4.17, Theorem 4.18, Corollaries 4.15 and 4.19: rounding checks, delta < 8.5e-22, log2 m* >= 6.2e3 (check) | `r123_grid_singleblock_out.txt` | 1 |
| `r123_escapes_cut.py` | Proposition 4.23: the cut Q for two copies of K_n, n = 2..7, exhaustive search at n = 3, ranks 64 and 729 (check) | `r123_escapes_cut_out.txt` | 1 |
| `r123_escapes_window.py` | Proposition 4.24 and the text after it: bounds 5/8 and 2/3, power profiles, the free-fermion example (check) | `r123_escapes_window_out.txt` | 1 |
| `r123_idsearch.py` | Section 4.7 after Proposition 4.23: the App. H identity found at L = 1 on the rectangle; residuals 0.98 and 0.95 at L = 1, 2 on the configuration (search) | `r123_idsearch_out.jsonl` | 3 |
| `r123_idcut.py` | Section 4.7: at least N^6 = 64 merge-local product terms at N = 2, so 3L < 64 terms cannot suffice (check) | `r123_idcut_out.txt` (prints) | 1 |

The recorded outputs belong to version 1 of the paper on arXiv.

## Licence

The scripts are released under the MIT licence; see `LICENSE`. The licence does not cover the paper `paper.pdf`,
which is © 2026 Ruhollah Majdoddin, all rights reserved.
