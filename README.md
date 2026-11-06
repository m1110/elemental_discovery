# elemental_discovery

A calibrated, uncertainty-aware model of **where undiscovered elements could
exist, how they could be made, and how their existence would be confirmed**.

The model starts from measured nuclear and atomic data. It fits physics-based
systematics to those data and then extrapolates beyond oganesson (Z = 118),
reporting how far each prediction lies from the data. The full results are in
**[docs/REPORT.md](docs/REPORT.md)**.

## What changed in v0.2: simpler, out-of-sample-validated mathematics

v0.1 checked every law against the data it was fitted to. v0.2 asks whether each law
**predicts nuclei it has not seen**: fit on everything below a cut in Z, predict everything
above it. That test exposed seven holes, documented in **[docs/AUDIT.md](docs/AUDIT.md)**.
The replacements are simpler or more principled, not bigger:

* **One primitive** (`linear.py`). Every systematic is a linear law in log space that
  reports its *predictive* standard deviation `sd² = s² + xᵀCx`, which grows automatically as
  it extrapolates. Assumptions enter as explicit Gaussian priors, not hidden grid ranges.
* **Alpha decay.** A 4-parameter WKB "universal decay law" (three tunnelling terms plus a
  fitted odd-nucleon hindrance) replaces the 6-parameter Viola–Seaborg form and extrapolates
  better (worst forward error 0.93 vs 1.11 decades on the evaluated data).
* **Q-alpha error bars.** These were 2–3.5× too small out of sample. A distance-to-data
  term, calibrated by forward tests, brings the forward z-rms to ≈ 1.
* **Fission.** A Bayesian law with an explicit shell-slope prior and a fitted odd-nucleon
  hindrance. It reports the several decades of uncertainty on the island instead of hiding them.
* **Null searches.** These are treated as zero-count Poisson exposures, not censored points,
  and each reaction is updated by its own failed search.
* **Discovery calculus** (`calculus.py`). This reports one probability that matters,
  `P(≥ k anchored chains in D beam days)`, with every uncertainty propagated through it.

## What it models

| Question | Module | Physics |
|---|---|---|
| How tightly bound is a nucleus? | `nuclear.py` | Liquid drop + Myers–Swiatecki shell correction with deformation, fitted to binding energies and alpha Q-values; Gaussian-process correction plus a forward-test-calibrated extrapolation error |
| Where is the next magic proton number? | `nuclear.py`, `validation.py` | The model is refitted under Z = 114, 120 and 126; each is scored by how well it *predicted* heavier nuclei (forward log-likelihood) |
| How does it decay, and how fast? | `decay.py` | Universal (WKB) alpha-decay law; Bayesian fission law with an explicit shell-slope prior; crude EC/β estimates; full decay chains |
| Where is the island of stability? | `survey.py` | Half-life maps for Z = 100–130 under each hypothesis |
| How could we make it? | `production.py` | Fusion Q values, barriers, excitation energy, xn channel; cross-section law fitted by an exact Poisson likelihood to measured cross sections *and* zero-event searches |
| What is the chance a campaign succeeds? | `calculus.py` | P(≥ k anchored chains in D days): log-normal cross section × flight survival × Poisson counting × probability that the chain anchors to known nuclei |
| How would we know it is real? | `confirmation.py`, `atomic.py` | Genetic decay-chain correlation, random-correlation background, Poisson significance, K X-ray Z identification (finite-nucleus Dirac solver), IUPAC/IUPAP criteria |
| Could it exist in nature? | `confirmation.py` | Survival in kilonovae, cosmic rays, and the Earth and meteorites |
| Where does the periodic table end? | `atomic.py` | Critical charge Z_cr ≈ 170 where the 1s level enters the Dirac sea; Madelung vs. relativistic configurations |

## Headline results

These are model outputs, not established facts. See the report for the uncertainties.
Since v0.3 every nuclear number is calibrated on the evaluated **NUBASE2020** table: 463 alpha
emitters, 103 fission half-lives and 177 known nuclides with Z ≥ 100 (v0.2 had 47, 15 and
hand-typed ranges).

* **Fission is less shell-stabilised than assumed.** With 103 evaluated half-lives the shell
  slope is *measured* (k = 0.44 ± 0.34 decades/MeV for shells at Z = 114), below the [1, 3.5]
  v0.1 assumed. The odd-nucleon hindrance, fitted for the first time, is 1.4 ± 0.4 decades per
  odd nucleon type, not the assumed 3.
* **Best routes, as one-campaign discovery probabilities** (300 days, 3 anchored chains,
  6 pµA, averaged over shell hypotheses): element 119 via 50Ti + 249Bk or 51V + 248Cm (≈ 0.6),
  element 120 via 50Ti + 251Cf (≈ 0.4). These are lower than in v0.2 because faster fission
  now competes with alpha decay, so chains anchor to known nuclei less often. Es-target routes
  are target-limited (≤ 0.2).
* **Separator flight losses.** Averaged over the half-life uncertainty, about half of 296Ubn
  atoms decay before reaching the detector if the shell is at Z = 114 (≈ 2 % under Z = 120/126).
* **Shell hypotheses.** The model variant with its shell at Z = 126 predicts heavier nuclei best,
  both in and out of sample. That is a statement about this model, not proof of a magic number.
  The most discriminating measurement is now element 120 (296Ubn: 1.6 MeV spread, ≈ 4σ).
* **Island of stability.** The longest lifetimes sit at N ≈ 183–184 under every hypothesis,
  7 neutrons beyond anything made. Their length is uncertain by ±3–4 decades.
* **Natural occurrence.** Survival over Solar-System ages is unlikely but not excluded by theory
  (P ≈ 0.2). The evidence against primordial superheavies is experimental.

## Usage

```bash
pip install -r requirements.txt
python -m elemental_discovery report             # regenerate docs/REPORT.md + figures
python -m elemental_discovery reactions 120      # how to make element 120
python -m elemental_discovery nucleus 120 296 --hypothesis Z=120   # decay chain + confirmation plan
python -m elemental_discovery element 119        # atomic structure, X-ray fingerprint
python -m elemental_discovery discover 120       # probability of discovery for every route
python -m pytest -q                              # validation tests against known physics
```

The first run spends about 20 s calibrating the Dirac X-ray model and about 30 s
running the mass-model forward tests. Both results are cached in
`~/.cache/elemental_discovery/`; set `ELEMENTAL_CACHE` to move it.

## Data and caveats

Masses, half-lives, branching ratios and the list of known nuclides come from **NUBASE2020**
(F.G. Kondev et al., *Chin. Phys. C* 45, 030001 (2021)). `elemental_discovery/nubase2020_subset.txt`
is an unmodified extract of the evaluated file (ground states with Z ≥ 80, plus light
calibration nuclei); `tools/extract_nubase.py` regenerates it from the full `nubase_4.mas20.txt`.
Superheavy masses are flagged `#` (systematic), but AME links them through measured decay
energies, so Q-alpha differences reproduce the published values (the tests check this). Odd
nuclei get an extra 0.25 MeV Q uncertainty, because their observed decays feed excited states
whose energies are estimated.

Fusion cross sections, upper limits and K X-ray energies are not in NUBASE. They were compiled
by hand from the discovery papers and are rounded, so check them before quoting any number.

This is a transparent, global model. It is not a substitute for
state-of-the-art microscopic calculations. Its value is that every
extrapolation carries an honest error bar and every assumption can be
inspected in a few hundred lines of code.
