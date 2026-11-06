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
* **Alpha decay.** A 3-parameter WKB "universal decay law" replaces the 6-parameter
  Viola–Seaborg form. Forward error is 0.53 decades instead of 1.41.
* **Q-alpha error bars.** These were 2.5–3.5× too small out of sample. A distance-to-data
  term, calibrated by forward tests, brings the forward z-rms to ≈ 1.
* **Fission.** Fitted freely, the data show *no* shell stabilisation (k = 0.2 ± 1.0), so the
  island of stability rests on a physics prior. That prior is now explicit, and its
  consequence, several decades of uncertainty on the island, is reported.
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

* **Best routes, as one-campaign discovery probabilities** (300 days, 3 anchored chains,
  6 pµA): element 120 via 50Ti + 251Cf (≈ 0.9), element 119 via 50Ti + 249Bk or 51V + 248Cm
  (≈ 0.75). The already-searched 50Ti + 249Cf drops to ≈ 0.6, and Es-target routes are
  target-limited (≈ 0.2 or less).
* **Separator flight losses.** Predicted element-120 half-lives are near the ~1 µs flight
  time. Averaged over their uncertainty, about half of 296Ubn atoms decay before reaching
  the detector if the shell is at Z = 114 (5–10 % under Z = 120/126).
* **Shell hypotheses.** In sample Z = 120 fits best. Out of sample it predicts worst, and
  Z = 126 predicts best. Element-122 chains discriminate at ≈ 3σ (not 5σ, once error bars are
  calibrated).
* **Island of stability.** The longest lifetimes sit at N ≈ 183–184 under every hypothesis,
  7 neutrons beyond anything made. Their *length* is prior-dominated (±5–7 decades).
* **Natural occurrence.** Theory alone can no longer rule it out. The evidence against
  primordial superheavies is experimental.

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

The calibration data in `elemental_discovery/data.py` (AME2020/NUBASE2020
masses and decay data, discovery-paper cross sections and limits, K X-ray
energies) were compiled by hand and rounded. They are fine for calibrating
phenomenological systematics, but check them against the evaluated tables
before quoting any number.

This is a transparent, global model. It is not a substitute for
state-of-the-art microscopic calculations. Its value is that every
extrapolation carries an honest error bar and every assumption can be
inspected in a few hundred lines of code.
