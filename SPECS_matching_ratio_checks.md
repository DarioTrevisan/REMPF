# SPECS.md — Matching ratio diagnostics for p=2 vs p=4 optimizers

## Purpose

Extend the existing matching simulation/analysis code to distinguish the ratios that are directly comparable with the formal logarithmic-scale flow from the ratios that measure the internal kurtosis of a given optimizer.

The current plot appears to show

- `p_opt=2`: \(\mathbb E[M_4^{(2)}] / (\mathbb E[M_2^{(2)}])^2\), close to the Gaussian linear value \(2\);
- `p_opt=4`: likely \(\mathbb E[M_4^{(4)}] / (\mathbb E[M_2^{(4)}])^2\), close to \(1.61\).

The formal flow prediction for the quartic constant should instead be compared to the **cross-normalized** ratio

\[
R_{44}^{\rm cross}(n)
:=
\frac{\mathbb E[M_4^{(4)}]}{(\mathbb E[M_2^{(2)}])^2}.
\]

The flow predicts, in the present normalization,

\[
R_{22}(n)\to 2,
\qquad
R_{44}^{\rm cross}(n)\to a_\star\approx 1.80788,
\]

whereas it does **not** predict that

\[
R_{44}^{\rm self}(n)
:=
\frac{\mathbb E[M_4^{(4)}]}{(\mathbb E[M_2^{(4)}])^2}
\]

should converge to \(a_\star\). The self-normalized ratio can be significantly smaller if the \(p=4\) optimizer pays a larger quadratic cost to reduce quartic tails.

---

## Mathematical quantities to compute

For each system size `n`, each random seed/sample `s`, and each optimizer exponent `p_opt in {2,4}`, store the raw matching costs

\[
M_2^{(p_{\rm opt})}(n,s),
\qquad
M_4^{(p_{\rm opt})}(n,s).
\]

Here `M_p` must use the same convention for all optimizers. Prefer total cost divided by the number of points, i.e.

\[
M_p = \frac1n\sum_{i=1}^n |X_i-Y_{\sigma(i)}|^p,
\]

on the unit torus, unless the existing code uses another convention. If another convention is used, document it in the output metadata. All ratios below are invariant under some common rescalings but not all, so the convention must be explicit.

For each `n`, estimate the following sample means:

\[
\bar M_2^{(2)} = \mathbb E[M_2^{(2)}],
\qquad
\bar M_4^{(2)} = \mathbb E[M_4^{(2)}],
\]

\[
\bar M_2^{(4)} = \mathbb E[M_2^{(4)}],
\qquad
\bar M_4^{(4)} = \mathbb E[M_4^{(4)}].
\]

Then compute these diagnostic ratios:

### 1. Quadratic-optimizer Gaussian sanity ratio

\[
R_{22}(n)
:=
\frac{\bar M_4^{(2)}}{(\bar M_2^{(2)})^2}.
\]

Expected behavior:

\[
R_{22}(n)\to 2.
\]

This checks that the `p_opt=2` optimizer produces a nearly Gaussian/linear-Poisson field at the level of fourth over second moments.

### 2. Quartic optimizer self-normalized kurtosis ratio

\[
R_{44}^{\rm self}(n)
:=
\frac{\bar M_4^{(4)}}{(\bar M_2^{(4)})^2}.
\]

This is the ratio apparently shown by the blue curve in the current plot. It measures the kurtosis of the `p_opt=4` optimizer in its own quadratic scale. It is **not** the direct comparison to the formal quartic flow constant.

### 3. Quartic optimizer cross-normalized flow-comparison ratio

\[
R_{44}^{\rm cross}(n)
:=
\frac{\bar M_4^{(4)}}{(\bar M_2^{(2)})^2}.
\]

Expected formal-flow target:

\[
R_{44}^{\rm cross}(n)\to a_\star\approx 1.80788.
\]

This is the main ratio to compare against the logarithmic-scale flow prediction.

### 4. Quadratic inflation of the quartic optimizer

\[
Q_2(n)
:=
\frac{\bar M_2^{(4)}}{\bar M_2^{(2)}}.
\]

This explains the gap between self and cross ratios:

\[
R_{44}^{\rm cross}(n)
=
R_{44}^{\rm self}(n)\,Q_2(n)^2.
\]

This identity must be checked numerically as a consistency test. If `R44_self ≈ 1.61` and `R44_cross ≈ 1.80788`, then

\[
Q_2\approx \sqrt{1.80788/1.61}\approx1.06.
\]

So only a six percent increase in quadratic cost for the `p_opt=4` optimizer would reconcile the blue curve with the formal flow value.

### 5. Relative quartic improvement over the quadratic optimizer

\[
I_4(n)
:=
1-
\frac{\bar M_4^{(4)}}{\bar M_4^{(2)}}.
\]

This measures the direct fractional gain in quartic cost achieved by optimizing for `p=4` instead of `p=2`.

### 6. Optional: same-sample paired ratios

If the exact same point configurations are used for both `p_opt=2` and `p_opt=4`, additionally compute paired sample-level ratios:

\[
Q_{2,s}(n)=\frac{M_2^{(4)}(n,s)}{M_2^{(2)}(n,s)},
\]

\[
C_{44,s}(n)=\frac{M_4^{(4)}(n,s)}{(M_2^{(2)}(n,s))^2},
\]

and report their means and confidence intervals. These are not identical to ratios of means, but they can reduce variance and reveal sample-wise correlations. Keep them clearly labeled as paired diagnostics.

---

## Required input data schema

The analysis script should accept either CSV or Parquet files with at least these columns:

```text
n              integer system size
seed           integer or string sample id
p_opt          integer or float, values 2 and 4
M2             float, p=2 matching cost under this optimizer
M4             float, p=4 matching cost under this optimizer
status         optional string, e.g. ok/failed/timeout
runtime_sec    optional float
solver_gap     optional float
```

The data must contain both `p_opt=2` and `p_opt=4` entries for each `(n, seed)` whenever possible. If some seeds are missing one optimizer, the script should support both:

1. **paired mode**, retaining only seeds with both optimizers;
2. **unpaired mode**, using all available samples for each optimizer.

Default should be paired mode for cross diagnostics, because it is cleaner and reduces variance.

---

## Required scripts

### `analyze_matching_ratios.py`

Command-line interface:

```bash
python analyze_matching_ratios.py \
  --input results.csv \
  --outdir figures_ratio_checks \
  --paired true \
  --bootstrap 5000 \
  --alpha 0.05 \
  --target-a 1.80788
```

Required outputs:

```text
figures_ratio_checks/summary_by_n.csv
figures_ratio_checks/ratio_curves.pdf
figures_ratio_checks/ratio_curves.png
figures_ratio_checks/cross_vs_self.pdf
figures_ratio_checks/quadratic_inflation.pdf
figures_ratio_checks/finite_size_fits.pdf
figures_ratio_checks/report.md
```

Optional but useful:

```text
figures_ratio_checks/summary_by_n.parquet
figures_ratio_checks/bootstrap_samples.parquet
figures_ratio_checks/raw_qc.md
```

---

## Statistical estimators

For each `n`, compute means:

```python
mean_M2_p2 = mean(M2[p_opt == 2])
mean_M4_p2 = mean(M4[p_opt == 2])
mean_M2_p4 = mean(M2[p_opt == 4])
mean_M4_p4 = mean(M4[p_opt == 4])
```

Then compute:

```python
R22        = mean_M4_p2 / mean_M2_p2**2
R44_self   = mean_M4_p4 / mean_M2_p4**2
R44_cross  = mean_M4_p4 / mean_M2_p2**2
Q2         = mean_M2_p4 / mean_M2_p2
I4         = 1.0 - mean_M4_p4 / mean_M4_p2
identity_error = R44_cross - R44_self * Q2**2
```

`identity_error` should be numerically near machine precision if all ratios are computed from the same means. If not, the script should raise a warning.

---

## Confidence intervals

Use bootstrap confidence intervals because ratios of means have nontrivial covariance.

### Paired bootstrap

In paired mode, resample seeds with replacement within each fixed `n`. For each bootstrap sample:

1. keep both `p_opt=2` and `p_opt=4` rows for each selected seed;
2. recompute all means and ratios.

Use percentile intervals by default:

```python
ci_low  = percentile(samples, 100 * alpha/2)
ci_high = percentile(samples, 100 * (1-alpha/2))
```

Also compute bootstrap standard errors.

### Delta method optional cross-check

Optionally implement a delta-method variance estimate for ratios of means, mainly as a sanity check. Bootstrap intervals should be the plotted default.

---

## Required plots

### Plot 1: all main ratios versus `n`

File:

```text
ratio_curves.pdf
ratio_curves.png
```

X-axis: `n` on log2 scale.

Y-axis: ratio.

Curves:

- `R22 = E[M4^(2)] / E[M2^(2)]^2`
- `R44_self = E[M4^(4)] / E[M2^(4)]^2`
- `R44_cross = E[M4^(4)] / E[M2^(2)]^2`

Add horizontal reference lines:

- `2.0`, labeled `linear Gaussian ratio 2`;
- `a_star = 1.80788`, labeled `formal flow a*`.

Use 95% confidence intervals as error bars or shaded bands.

Interpretation:

- `R22` should approach `2`.
- `R44_cross`, not `R44_self`, is the relevant comparison to `a_star`.
- `R44_self` is expected to be lower if `Q2 > 1`.

### Plot 2: self versus cross ratio for the p=4 optimizer

File:

```text
cross_vs_self.pdf
```

Curves:

- `R44_self`
- `R44_cross`
- optionally `R44_self * Q2^2`, which should coincide with `R44_cross`

Add horizontal line at `a_star = 1.80788`.

This plot should make clear whether the previous blue curve was simply self-normalized.

### Plot 3: quadratic inflation

File:

```text
quadratic_inflation.pdf
```

Plot:

\[
Q_2(n)=\bar M_2^{(4)}/\bar M_2^{(2)}.
\]

Add horizontal line at `1`. If `Q2` is around `1.05` to `1.07`, then a self ratio near `1.61` is compatible with a cross ratio near `1.81`.

### Plot 4: quartic improvement

File:

```text
quartic_improvement.pdf
```

Plot:

\[
I_4(n)=1-\bar M_4^{(4)}/\bar M_4^{(2)}.
\]

This quantifies the direct relative gain in quartic matching cost obtained by optimizing the quartic objective.

### Plot 5: finite-size extrapolation

File:

```text
finite_size_fits.pdf
```

Use x-axis choices:

- `1 / log(n)`
- `1 / log(n)^2`
- optionally `1 / sqrt(log(n))`

Fit the large-size tail only, e.g. `n >= 256` and `n >= 512`, to models:

```text
R(n) = R_inf + c1 / log(n)
R(n) = R_inf + c1 / log(n) + c2 / log(n)^2
R(n) = R_inf + c1 / log(n)^alpha, with alpha fixed in {0.5, 1, 2}
```

Do not overinterpret the fits. The report should present them as diagnostic only. Include confidence intervals for `R_inf` from bootstrap resampling or weighted least squares using bootstrap standard errors.

---

## Report contents

Generate `report.md` with:

1. data summary:
   - number of samples per `n` and `p_opt`;
   - number of paired samples retained;
   - missing or failed runs;

2. table of main estimates:

```text
n, samples, R22, R22_ci_low, R22_ci_high,
R44_self, R44_self_ci_low, R44_self_ci_high,
R44_cross, R44_cross_ci_low, R44_cross_ci_high,
Q2, Q2_ci_low, Q2_ci_high,
I4, I4_ci_low, I4_ci_high
```

3. explicit diagnostic identity:

\[
R_{44}^{\rm cross}=R_{44}^{\rm self}Q_2^2.
\]

Print maximum absolute identity error.

4. comparison with formal targets:

- `R22` versus `2`;
- `R44_cross` versus `a_star = 1.80788`;
- `R44_self` explained as a different normalization.

5. finite-size fit summary:

- fitted `R_inf` values under different models and tail cutoffs;
- warnings if fit extrapolations are unstable.

6. conclusion paragraph automatically generated from the estimates, e.g.

```text
The self-normalized p=4 ratio is near ..., but the cross-normalized ratio is near ... .
The quadratic inflation Q2 is ... . Therefore the apparent discrepancy with a_star is / is not
explained by self-normalization.
```

---

## Quality-control checks

The script must verify:

1. all costs are positive;
2. for each `n`, both optimizers have enough samples;
3. paired seeds align correctly;
4. `identity_error = R44_cross - R44_self * Q2**2` is below `1e-10` when ratios are computed from the same sample means;
5. bootstrap samples do not produce NaNs or infinities;
6. confidence intervals shrink with sample size in a plausible way;
7. `R22` approaches approximately `2` for large `n` as a sanity check.

Warnings should be printed but should not stop execution unless data are invalid.

---

## Suggested implementation details

### Data loading

Use `pandas`. Accept `.csv`, `.parquet`, and `.feather`.

### Bootstrap

Use `numpy.random.default_rng(seed)` with a CLI option:

```bash
--rng-seed 12345
```

For each `n`, create a pivot table indexed by seed with columns:

```text
M2_p2, M4_p2, M2_p4, M4_p4
```

Drop rows with missing values in paired mode.

### Plotting

Use `matplotlib` only. Do not rely on seaborn.

Recommended style:

- `n` axis logarithmic base 2;
- show markers and lines;
- include error bars;
- label curves with mathematical names and optimizer meaning;
- include reference constants.

### Numerical precision

Use `float64`. Print ratios with at least 5 decimal digits.

Because finite-size corrections are expected to be slow, do not round intermediate quantities. Store full precision in `summary_by_n.csv`.

---

## Acceptance criteria

The implementation is acceptable if:

1. it produces all required output files from an existing raw results table;
2. the generated `summary_by_n.csv` contains all main ratios and confidence intervals;
3. `R44_cross = R44_self * Q2^2` is verified numerically;
4. the report clearly states whether the previous blue curve was self-normalized;
5. the plots make it visually clear whether `R44_cross` is closer to `a_star = 1.80788` than `R44_self`;
6. finite-size fits are included but explicitly marked as diagnostic, not conclusive.

---

## Optional extension: direct comparison with continuum constants

If the code also stores the unscaled raw costs on the unit torus, add estimates of the logarithmic constants:

\[
C_2^{(p_{\rm opt})}(n)
:=
\frac{\bar M_2^{(p_{\rm opt})}}{\log n},
\]

\[
C_4^{(p_{\rm opt})}(n)
:=
\frac{\bar M_4^{(p_{\rm opt})}}{(\log n)^2}.
\]

The precise normalization depends on whether `n` denotes number of points in the unit torus or side length. Therefore this extension must be disabled by default unless the normalization metadata are present.

---

## Minimal pseudocode

```python
def summarize_one_n(df_n, paired=True, B=5000, alpha=0.05, rng=None):
    wide = pivot_by_seed(df_n)
    if paired:
        wide = wide.dropna(subset=["M2_p2", "M4_p2", "M2_p4", "M4_p4"])

    def compute(block):
        m2p2 = block["M2_p2"].mean()
        m4p2 = block["M4_p2"].mean()
        m2p4 = block["M2_p4"].mean()
        m4p4 = block["M4_p4"].mean()
        R22 = m4p2 / m2p2**2
        R44_self = m4p4 / m2p4**2
        R44_cross = m4p4 / m2p2**2
        Q2 = m2p4 / m2p2
        I4 = 1.0 - m4p4 / m4p2
        return dict(
            mean_M2_p2=m2p2,
            mean_M4_p2=m4p2,
            mean_M2_p4=m2p4,
            mean_M4_p4=m4p4,
            R22=R22,
            R44_self=R44_self,
            R44_cross=R44_cross,
            Q2=Q2,
            I4=I4,
            identity_error=R44_cross - R44_self * Q2**2,
        )

    point = compute(wide)
    boot = []
    nrows = len(wide)
    for _ in range(B):
        idx = rng.integers(0, nrows, size=nrows)
        boot.append(compute(wide.iloc[idx]))
    return point, bootstrap_intervals(boot, alpha)
```

---

## Interpretation guide

After running, inspect these scenarios:

### Scenario A: self ratio low, cross ratio near `a_star`

If

```text
R44_self ≈ 1.60-1.65,
Q2 ≈ 1.05-1.07,
R44_cross ≈ 1.78-1.85,
```

then the plots are consistent with the formal flow. The previous discrepancy was due to normalization.

### Scenario B: both self and cross ratios near `1.61`

If

```text
Q2 ≈ 1,
R44_cross ≈ R44_self ≈ 1.61,
```

then the formal flow prediction `a_star ≈ 1.80788` is likely not describing the matching data, or the finite-size regime is still far from asymptotic.

### Scenario C: cross ratio above `a_star` but drifting downward

This may indicate slow `1/log(n)` corrections. Use finite-size fits, but do not overclaim.

### Scenario D: R22 not close to 2

If `R22` does not approach `2`, first check normalization, torus distance convention, solver accuracy, and whether `M4` is measured on the `p=2` optimizer rather than from a separate `p=4` run.
