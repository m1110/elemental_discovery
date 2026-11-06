# elemental_discovery

A calibrated, uncertainty-aware model of **where undiscovered elements could
exist, how they could be made, and how their existence would be confirmed**.

The model starts from measured nuclear and atomic data. It fits physics-based
systematics to those data and then extrapolates beyond oganesson (Z = 118),
reporting how far each prediction lies from the data. The full results are in
**[docs/REPORT.md](docs/REPORT.md)**.

## What it models

| Question | Module | Physics |
|---|---|---|
| How tightly bound is a nucleus? | `nuclear.py` | Liquid drop + Myers–Swiatecki shell correction with deformation, fitted to binding energies and alpha Q-values; Gaussian-process correction with error bars that grow away from data |
| Where is the next magic proton number? | `nuclear.py`, `survey.py` | The model is refitted under Z = 114, 120 and 126 hypotheses and each is scored against superheavy data |
| How does it decay, and how fast? | `decay.py` | Refitted Viola–Seaborg alpha systematics, Swiatecki-type fission systematics, crude EC/β estimates, full decay chains |
| Where is the island of stability? | `survey.py` | Half-life maps for Z = 100–130 under each hypothesis |
| How could we make it? | `production.py` | Fusion Q values, barriers, excitation energy, xn channel; cross-section systematic fitted by censored likelihood to measured cross sections *and* published non-observations; beam-time calculator |
| How would we know it is real? | `confirmation.py`, `atomic.py` | Genetic decay-chain correlation, random-correlation background, Poisson significance, K X-ray Z identification (finite-nucleus Dirac solver), IUPAC/IUPAP criteria, separator flight-time losses |
| Could it exist in nature? | `confirmation.py` | Survival in kilonovae, cosmic rays, and the Earth and meteorites |
| Where does the periodic table end? | `atomic.py` | Critical charge Z_cr ≈ 170 where the 1s level enters the Dirac sea; Madelung vs. relativistic configurations |

## Headline results

These are model outputs, not established facts. See the report for the
uncertainties.

* **Island of stability.** Under all three shell hypotheses the longest
  half-lives sit at N ≈ 183–184, around Z ≈ 113–115. That is 6–7 neutrons beyond
  anything made so far, and stable-beam fusion can't get there.
* **Elements 119 and 120.** The best routes are 50Ti + 249Bk (~170 fb) and
  50Ti + 249Cf/251Cf (~125 fb), uncertain by about ×2.5. That works out to weeks
  to months of beam time per handful of atoms. The GSI 50Ti + 249Bk limit
  (65 fb) is in mild (~1.4σ) tension with this prediction.
* **Half-lives near 1 µs.** Predicted element-120 half-lives are close to the
  separator flight time, so a large fraction of atoms could decay before
  reaching the detector.
* **Confirmation.** Predicted chains from 119/120 reach known nuclei (Mc, Lv,
  Og), so genetic correlation is available. K X-rays near 194 keV (Z = 119) and
  199 keV (Z = 120) would identify Z directly.
* **Telling the hypotheses apart.** Element 122 decay energies differ by about
  3 MeV between the hypotheses, roughly 5× the model uncertainty.
* **Natural occurrence.** Survival on Earth or in meteorites is ruled out.
  Kilonova spectra are the most plausible natural channel.

## Usage

```bash
pip install -r requirements.txt
python -m elemental_discovery report             # regenerate docs/REPORT.md + figures
python -m elemental_discovery reactions 120      # how to make element 120
python -m elemental_discovery nucleus 120 296 --hypothesis Z=120   # decay chain + confirmation plan
python -m elemental_discovery element 119        # atomic structure, X-ray fingerprint
python -m pytest -q                              # validation tests against known physics
```

The first run spends about 20 s calibrating the Dirac X-ray model. The result
is cached in `~/.cache/elemental_discovery/`; set `ELEMENTAL_CACHE` to move it.

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
