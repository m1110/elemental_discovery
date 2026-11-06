# Audit of the first model, and what replaced it

The first version of this package (commit `47508db`) was a careful, well-documented model.
Every check it reported, though, was **in sample**: each law was scored on the same data it was
fitted to. For new elements the question is how well a law *predicts nuclei it has not seen*.
This audit re-tested every component out of sample. It found seven holes, and fixed each one
with mathematics that is simpler or more principled, not with more parameters.

All numbers are reproducible: `python -m pytest -q` checks them and
`python -m elemental_discovery report` regenerates [REPORT.md](REPORT.md).

## The test: fit the past, predict the future

The elements were discovered in order of increasing Z, so the honest test is to fit on every
nucleus below a cut in Z, predict every nucleus above it, and compare the errors with the error
bars the model claimed. A calibrated model has forward z-scores with rms ≈ 1.

## Holes found

| # | Component | Original claim | Out-of-sample finding | Fix |
|---|---|---|---|---|
| 1 | Q-alpha error bars (mass model + GP) | ±0.3–0.7 MeV for Z = 119–122 | Forward errors 1.2–1.8 MeV; z-rms **2.5–3.5** (error bars 2.5–3.5× too small). A stationary GP's variance saturates at its amplitude however far it extrapolates. | Add `(k·d)²` in quadrature, with d the distance to the nearest measured nucleus and k fitted by maximum likelihood on forward tests. z-rms → **1.02–1.09**. |
| 2 | Which shell hypothesis fits best | Z = 120 (in-sample Q-alpha rms 0.56 MeV) | Out of sample Z = 120 predicts heavier nuclei *worst* (forward rms 1.82 MeV, versus 1.23 for Z = 126). | Score hypotheses by forward predictive log-likelihood and use the tempered weights for model averaging. |
| 3 | Alpha half-life law | 6-parameter Viola–Seaborg, 0.36-decade scatter | Fitted on Z < 104 and asked to predict Z ≥ 104, it misses by **1.41 decades**. | 3-parameter universal decay law (WKB tunnelling exponent, linearised): **0.53 decades**, and better calibrated at every cut. |
| 4 | Fission half-lives | ±2 decades; shell slope k found by grid search over [1, 3.5] | The optimum sat on the grid edge. Fitted freely, **k = 0.2 ± 1.0**, consistent with zero: the calibration nuclei have shell energies of −5 to 0 MeV, while the island has −7 to −11 MeV. The island was assumed, not fitted. | Bayesian linear regression with an *explicit* prior k ~ N(2.25, 0.75²). Predictive sd on the island is now **3–7 decades**. |
| 5 | Null searches for 119/120 | Censored-Gaussian (Tobit) likelihood; 50Ti+249Bk "1.4σ tension" | A 90 % CL limit is a zero-count Poisson observation at a known exposure, not a censoring point. | Exact likelihood P(0) = E[exp(−L σ)] with L = ln 10 / σ_lim. Each reaction's own null exposure then updates its cross section, exactly once (hierarchical Bayes). 50Ti+249Bk drops from ~90 fb to ~23 fb. |
| 6 | Beam-time estimates | "3 beam days" for 48Ca + 254Es | Assumes a 0.5 mg/cm² target of an isotope that exists only in micrograms. | Availability-limited target density (254Es: 0.001 mg/cm²). |
| 7 | Point estimates of feasibility | Survival in flight and beam days at central values; anchoring by dominant mode only | Every one of these is biased optimistic. For a log-normal half-life, mean survival < survival at the median. The chain anchors only if *every* unknown link alpha-decays. | The **discovery calculus** below. |

Consequences for the headline results:

* "Element 122 decay energies differ by 5× the model uncertainty" becomes **≈ 3×** once the error
  bars are calibrated. A single chain narrows the field but does not settle it.
* "Natural occurrence ruled out" is **not supported by theory** once fission uncertainties are
  honest (P ≈ 0.4 that the longest-lived candidate survives Solar-System ages under the shell
  prior). The case against primordial superheavies rests on experimental null results.
* The island's *location* (N ≈ 184) is robust. Its *lifetimes* are prior-dominated.

## The replacement mathematics

### 1. One primitive: linear laws with predictive variance (`linear.py`)

Every extrapolating systematic (alpha, fission, cross section) has the form
y = x·β + ε with y a logarithm. Its honest uncertainty is the predictive standard deviation

```
sd(x)² = s² + xᵀ C x,     C = s² (XᵀX + s² Λ)⁻¹
```

This is small inside the data and grows automatically outside it. Λ is an optional Gaussian
prior precision, which turns a hidden assumption (a grid range) into a stated one with a
measurable effect. Leave-one-out residuals come in closed form, `e_i / (1 − h_ii)`.

### 2. Alpha decay: the universal decay law

```
log10 T = a·χ + b·ρ + c
χ = Z_α Z_d √(A_red/Q)
ρ = √(A_red Z_α Z_d (A_d^⅓ + A_α^⅓))
```

χ and ρ are the two terms of the WKB Coulomb-tunnelling exponent, so the three coefficients
have physical meaning. Fewer parameters and better extrapolation is the point of the exercise.

### 3. Q-alpha uncertainty that knows when it is extrapolating

```
sd_Q² = sd_GP² + (k·d)²,   k = argmax Σ_forward log N(e_i; 0, sd_GP,i² + k² d_i²)
```

The fitted k is 0.14–0.21 MeV per unit distance in the Z–N plane, depending on the hypothesis.

### 4. Null results as Poisson exposures

```
L = −ln(1 − CL) / σ_lim                      (exposure; 2.303/σ_lim at 90 % CL)
P(0 | θ) = E_ε[ exp(−L · 10^(c0 + c1 x + ε)) ]
p(ε_j | all) ∝ N(ε_j; 0, s²) · P(0 | θ, ε_j) / P(0 | θ)   (per-reaction update, no double counting)
```

### 5. The discovery calculus (`calculus.py`)

```
P_disc(D) = P_anchor · E_{σ,T}[ P_Poisson(≥ k ; λ(σ,T)·D) ]
λ        = σ · I_beam · n_target · ε · 2^(−t_flight/T)
P_anchor = Π_unknown links Φ( (log T_SF − log T_α) / √(sd_SF² + sd_α²) )
```

The model has three log-normal inputs, one Bernoulli and one Poisson, and the expectation is
evaluated by Gauss–Hermite quadrature. Its output is the number a planning committee actually
needs: *the probability that a campaign of D days yields k anchored chains*.
`python -m elemental_discovery discover 120` prints it for every route to an element.

## What remains open

* **Data.** 47 alpha emitters and 15 fission half-lives, compiled by hand, are thin above
  Z = 110. Hundreds of measured superheavy decays exist, and adding them from
  NUBASE2020/ENSDF is the single largest improvement available. It would shrink k in item 1
  and could constrain the fission shell slope directly.
* **Fission physics.** No simple law tried here generalises out of sample: fissility polynomials
  and a one-variable barrier law both give 2.4–4.7 decades leave-one-out. Saddle-point shell
  corrections are the missing physics.
* **Correlations.** The calculus treats the residue's half-life and P_anchor as independent;
  they share the Q-alpha error.
* **Evaporation channel.** One xn channel is chosen per reaction. A one-neutron shift can move a
  chain off the known nuclei (for example 299Uue under Z = 120, which never anchors), so the
  3n/4n mixture should be summed.

## Update (v0.3): evaluated data replace the hand-typed tables

The largest open item above was the data. Masses, half-lives, branching ratios and the list of
known nuclides now come from **NUBASE2020** (Kondev et al., *Chin. Phys. C* 45, 030001, 2021),
vendored as an unmodified extract in `elemental_discovery/nubase2020_subset.txt`.

| Dataset | v0.2 (hand-typed) | v0.3 (NUBASE2020) |
|---|---|---|
| Alpha emitters | 47 (16 with Z ≥ 104) | 463 (93 with Z ≥ 104) |
| Fission half-lives | 15, even-even only | 103, odd included |
| Known nuclides Z ≥ 100 | ranges, gaps ignored (293Og wrongly "known") | exact set of 177 |

What changed, and why it matters:

* **Fission shell slope is now measured, and it is weak.** k = 0.44 ± 0.34 (Z = 114 shells;
  0.35 ± 0.54 for Z = 126, 1.35 ± 0.71 for Z = 120), against v0.1's assumed [1, 3.5]. The data
  disfavour the prior centre (2.25) at about 2σ, so fission competes more strongly with alpha
  decay and fewer chains anchor to known nuclei. Discovery probabilities fall accordingly.
* **The odd-nucleon fission hindrance is fitted: 1.4 ± 0.4 decades** per odd nucleon type, not
  the assumed 3. v0.1 gave odd-odd nuclei ~3 decades too much fission stability.
* **Mass-model weighting.** With 463 Q-alpha rows at 0.15 MeV weight, the binding energies were
  swamped (rms 2.6 MeV). Weighting Q-alpha by the model's own scatter (0.7 MeV), chosen by
  forward test, predicts heavier nuclei better (1.27 vs 1.73 MeV) and restores the binding
  energies (~0.8–0.9 MeV).
* **Calibration holds.** The forward Q-alpha error falls to 0.75–1.12 MeV (from 1.23–1.82). The
  uncalibrated GP is still ~2× overconfident; the distance term restores z-rms ≈ 1.
* **Hypothesis ranking.** The Z = 126 variant now predicts best both in and out of sample. The
  weights are tempered to one effective observation per tested element, because isotopes of an
  element share the model's bias.

Still open: AME2020 `rct1` Q-values would give odd-nucleus Q-alpha directly (it was not
reachable from this environment); fusion cross sections remain hand-compiled; and fission
still lacks a law that generalises out of sample (≈ 2.4 decades leave-one-out).
