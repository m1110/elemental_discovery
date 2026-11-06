"""Validation tests: each checks the models against physics we already know."""

import math

import numpy as np
import pytest

from elemental_discovery import atomic, confirmation, data, decay, nuclear, production, survey

HYPS = list(nuclear.PROTON_HYPOTHESES)


# --------------------------------------------------------------- mass model
@pytest.mark.parametrize("h", HYPS)
def test_mass_model_reproduces_binding_energies(h):
    m = nuclear.get_model(h)
    assert m.rms_binding < 1.5  # MeV, out of ~800-1800 MeV totals
    # 208Pb within 3 MeV of 1636.43
    assert abs(m.binding(82, 126) - 1636.43) < 3.0


@pytest.mark.parametrize("h", HYPS)
def test_gp_corrected_qalpha_matches_data(h):
    m = nuclear.get_model(h)
    res = [m.q_alpha(z, a - z) - q for z, a, q, _ in data.ALPHA_DECAYS]
    assert np.sqrt(np.mean(np.square(res))) < 0.2


def test_uncertainty_grows_away_from_data():
    m = nuclear.get_model("Z=114")
    _, near = m.q_alpha_with_uncertainty(118, 176)  # 294Og: measured
    _, far = m.q_alpha_with_uncertainty(126, 184)  # nothing measured nearby
    assert near < 0.2 < far


def test_doubly_magic_lead_is_spherical_and_shell_bound():
    m = nuclear.get_model("Z=114")
    assert m.shell_energy(82, 126) < -5
    assert m.deformation(82, 126) == 0.0


def test_beta_stability_valley_for_uranium():
    m = nuclear.get_model("Z=114")
    assert 142 <= m.beta_stable_n(92) <= 148  # 238U has N = 146


# --------------------------------------------------------------- decay
def test_viola_seaborg_calibration():
    _, rms = decay.viola_seaborg_fit()
    assert rms < 0.6
    # 238U: Q = 4.27 MeV -> 4.47 Gyr
    lt = decay.log10_t_alpha(92, 146, 4.270)
    assert abs(lt - math.log10(4.468e9 * data.SECONDS_PER_YEAR)) < 1.0


@pytest.mark.parametrize("h", HYPS)
def test_known_superheavy_decay_modes(h):
    assert decay.predict(118, 176, h).dominant_mode == "alpha"  # 294Og
    assert decay.predict(112, 172, h).dominant_mode == "SF"  # 284Cn


def test_decay_chain_from_element_120_reaches_known_nuclei():
    chain, ending = decay.decay_chain(120, 176, "Z=114")
    assert chain[0].name == "296Ubn"
    assert ending.startswith("reaches known")


# --------------------------------------------------------------- atomic
def test_dirac_point_nucleus_limit():
    # light element: finite size negligible, compare with Sommerfeld formula
    e = atomic.dirac_energy(40.0, -1, a=90.0)[0]
    exact = math.sqrt(1 - (40 * atomic.ALPHA) ** 2)
    assert abs(e - exact) < 1e-4


def test_kalpha_calibration_and_ordering():
    _, _, err = atomic.kalpha_screening()
    assert err < 0.01
    k = atomic.k_alpha1_kev([118, 119, 120])
    assert k[0] < k[1] < k[2]
    assert k[2] - k[1] > 3.0  # neighbouring Z resolvable with a Ge detector


def test_critical_charge_near_literature():
    zc = atomic.critical_charge()["Z_critical"]
    assert 165 < zc < 176


def test_configurations():
    assert atomic.madelung_configuration(26).endswith("4s2 3d6")  # Fe
    assert atomic.madelung_configuration(119).endswith("8s1")
    assert atomic.madelung_block(121) == "g"


def test_systematic_names():
    assert data.symbol(119) == "Uue" and data.symbol(120) == "Ubn" and data.symbol(118) == "Og"


# --------------------------------------------------------------- production
def test_cross_section_fit_reproduces_measurements():
    for zp, ap, zt, at, s, _ in data.CROSS_SECTIONS:
        lg, sig = production.log10_sigma_pb(zp, ap, zt, at)
        assert abs(lg - math.log10(s)) < 3 * sig


def test_cross_sections_fall_with_projectile_charge():
    s = [production.candidate_reactions(120)[i].sigma_pb()[0] for i in range(3)]
    assert s == sorted(s, reverse=True)
    ca = production.Reaction(20, 48, 98, 249, "249Cf").sigma_pb()[0]
    ti = production.Reaction(22, 50, 98, 249, "249Cf").sigma_pb()[0]
    assert ca > ti


def test_hot_fusion_excitation_energy():
    r = production.Reaction(20, 48, 98, 249, "249Cf")  # Og discovery reaction
    assert 25 < r.e_star_at_barrier() < 45
    assert r.best_xn() in (3, 4)


def test_beam_days_poisson():
    # 1 event/day expected: P(>=1) = 95 % needs -ln(0.05) = 3.0 days
    assert abs(production.beam_days(1.0, 1, 0.95) - 3.0) < 0.01


# --------------------------------------------------------------- confirmation
def test_poisson_significance():
    assert abs(confirmation.poisson_sf(1, 1.0) - (1 - math.exp(-1))) < 1e-12
    assert abs(confirmation.gaussian_sigma(0.0013499) - 3.0) < 0.01


def test_random_correlations_negligible_for_fast_chains():
    chain, _ = decay.decay_chain(120, 176)
    assert confirmation.random_chain_expectation(chain) < 1e-6


def test_no_primordial_superheavies():
    peak, _ = survey.island_of_stability("Z=114")
    out = confirmation.natural_search_feasibility(10 ** peak["log10_T_s"])
    earth = [r for r in out if "Earth" in r["environment"]][0]
    assert not earth["feasible"]


def test_island_lies_near_n184():
    for h in HYPS:
        peak, _ = survey.island_of_stability(h)
        assert 178 <= peak["N"] <= 186


# --------------------------------------------------------------- linear laws
def test_linear_law_predictive_sd_grows_with_extrapolation():
    from elemental_discovery.linear import LinearLaw
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 40)
    X = np.column_stack([np.ones_like(x), x])
    law = LinearLaw.fit(X, 2 + 3 * x + rng.normal(0, 0.1, 40))
    assert abs(law.coef[1] - 3) < 0.2
    assert law.sd([1, 0.5]) < law.sd([1, 3.0]) < law.sd([1, 10.0])
    # closed-form leave-one-out equals brute force
    brute = []
    for i in range(len(x)):
        m = np.ones(len(x), bool)
        m[i] = False
        c, *_ = np.linalg.lstsq(X[m], law.y[m], rcond=None)
        brute.append(X[i] @ c - law.y[i])
    assert np.allclose(law.loo_residuals(), brute)


def test_prior_pulls_unconstrained_coefficient():
    from elemental_discovery.linear import LinearLaw
    X = np.column_stack([np.ones(10), np.linspace(0, 1, 10), np.zeros(10)])  # 3rd unconstrained
    y = np.linspace(0, 1, 10) + np.random.default_rng(1).normal(0, 0.05, 10)
    law = LinearLaw.fit(X, y, prior_mean=[0, 0, 5.0], prior_sd=[np.inf, np.inf, 1.0])
    assert abs(law.coef[2] - 5.0) < 1e-6 and abs(law.coef_sd()[2] - 1.0) < 1e-6  # prior returned


# --------------------------------------------------------------- out of sample
def test_wkb_alpha_law_extrapolates_better_than_viola_seaborg():
    from elemental_discovery import validation
    vs, udl = validation.alpha_laws()
    assert udl["parameters"] < vs["parameters"]
    assert udl["forward_worst"] < vs["forward_worst"]


@pytest.mark.parametrize("h", HYPS)
def test_qalpha_errors_are_calibrated_out_of_sample(h):
    cal = nuclear.forward_calibration(h)
    assert cal["z_rms_gp_only"] > 2.0  # the uncalibrated GP was overconfident
    assert 0.8 < cal["z_rms_calibrated"] < 1.3


def test_qalpha_error_grows_with_distance():
    m = nuclear.get_model("Z=114")
    s = [m.q_alpha_with_uncertainty(z, n)[1] for z, n in ((118, 176), (120, 178), (122, 180), (126, 184))]
    assert s == sorted(s) and s[-1] > 3 * s[0]


def test_fission_shell_slope_is_prior_dominated():
    free, _ = decay.sf_law("Z=114", prior=False)
    post, _ = decay.sf_law("Z=114")
    assert abs(free.coef[3]) < 2 * free.coef_sd()[3]  # data alone: consistent with zero
    assert post.coef_sd()[3] < free.coef_sd()[3]
    m = nuclear.get_model("Z=114")
    _, sd_known = decay.log10_t_sf_with_sd(m, 112, 172)
    _, sd_island = decay.log10_t_sf_with_sd(m, 114, 184)
    assert sd_island > 2 * sd_known


# --------------------------------------------------------------- Poisson exposure
def test_exposure_from_limit():
    # 90 % CL with zero events <=> 2.303 expected events at the limit
    assert abs(production.exposure_from_limit(1.0) - math.log(10)) < 1e-12


def test_null_search_lowers_that_reaction_only():
    y, w = production.log10_sigma_samples(22, 50, 97, 249)  # searched at GSI, 0 events
    y0, w0 = production.log10_sigma_samples(22, 50, 97, 249, condition_on_nulls=False)
    assert w @ y < w0 @ y0 - 0.3
    y, w = production.log10_sigma_samples(23, 51, 96, 248)  # never searched
    y0, w0 = production.log10_sigma_samples(23, 51, 96, 248, condition_on_nulls=False)
    assert abs(w @ y - w0 @ y0) < 1e-9


# --------------------------------------------------------------- discovery calculus
def test_flight_survival_is_below_point_estimate():
    from elemental_discovery.calculus import flight_survival
    point = 2 ** -1.0  # T = t_flight exactly
    assert flight_survival(-6.0, 1.0, 1e-6) < point
    assert abs(flight_survival(-6.0, 1e-6, 1e-6) - point) < 1e-6


def test_discovery_probability_is_sane():
    from elemental_discovery.calculus import DiscoveryModel, Setup
    r = production.Reaction(22, 50, 98, 251, "251Cf")
    dm = DiscoveryModel(r, "Z=114", Setup())
    ps = [dm.p_discovery(d) for d in (1, 30, 300, 3000)]
    assert ps == sorted(ps) and 0 <= ps[0] < ps[-1] <= dm.p_anchor <= 1
    d = dm.days_for(0.3)
    assert abs(dm.p_discovery(d) - 0.3) < 1e-3


def test_einsteinium_routes_are_target_limited():
    from elemental_discovery.calculus import DiscoveryModel, Setup
    es = DiscoveryModel(production.Reaction(20, 48, 99, 254, "254Es"), "Z=114", Setup())
    bk = DiscoveryModel(production.Reaction(22, 50, 97, 249, "249Bk"), "Z=114", Setup())
    assert es.per_pb_day < bk.per_pb_day / 100


def test_hypothesis_weights_normalised():
    from elemental_discovery.calculus import hypothesis_weights
    w = hypothesis_weights()
    assert abs(sum(w.values()) - 1) < 1e-12 and all(v > 0 for v in w.values())
