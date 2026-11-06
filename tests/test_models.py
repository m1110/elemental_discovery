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
