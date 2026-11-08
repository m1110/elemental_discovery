"""Qualitative constraints the model must satisfy, each tied to an empirical finding."""

import pytest

from depthtime import (ARCHETYPES, NEUTRAL, ORDINARY, WAITING, MentalState,
                       depth_sweep, monte_carlo, readout, state_at_depth)
from depthtime.calibrate import fit_individual
from depthtime.model import Individual


def ro(depth, **ctx):
    return readout(state_at_depth(depth, **ctx))


def test_reference_is_unity():
    r = readout(ORDINARY)
    assert r.prospective == pytest.approx(1.0)
    assert r.retrospective == pytest.approx(1.0)
    assert r.passage == pytest.approx(1.0)


def test_state_validation():
    with pytest.raises(ValueError):
        MentalState(content=1.2, focus=0, awareness=0, self_ref=0)


def test_cognitive_load_shortens_prospective_lengthens_retrospective():
    # Block, Hancock & Zakay (2010) meta-analysis
    r = ro(0.0)
    assert r.prospective < 1.0
    assert r.retrospective > 1.0
    assert r.passage > 1.0


def test_waiting_drags_and_is_remembered_short():
    r = readout(WAITING)
    assert r.prospective > 1.0
    assert r.passage < 1.0
    assert r.retrospective < 1.0


def test_present_moment_awareness_lengthens_time():
    # Kramer et al. 2013; Droit-Volet et al. 2015; Wittmann & Schmidt 2014
    r = ro(0.5)
    assert r.prospective > 1.1
    assert r.passage < 1.0


def test_no_thought_produces_timelessness_not_speed():
    r = ro(1.0)
    assert r.timelessness > 0.85
    assert r.prospective < 0.2
    assert r.retrospective < 0.3


def test_depth_curve_is_nonmonotonic():
    sweep = depth_sweep(101)
    pro = [r.prospective for _, r in sweep]
    peak = max(range(len(pro)), key=pro.__getitem__)
    assert 0.35 <= sweep[peak][0] <= 0.65   # peak in present-moment zone
    assert pro[0] < pro[peak] > pro[-1]


def test_arousal_and_action_preparation_dilate_the_moment():
    # Droit-Volet & Meck 2007; Hagura et al. 2012
    calm = ro(0.25, movement=0.9, arousal=0.4)
    threat = ro(0.25, movement=0.9, arousal=0.95, action_prep=0.9)
    assert threat.prospective > calm.prospective
    assert threat.retrospective > calm.retrospective   # Stetson et al. 2007 memory account


def test_interoception_speeds_clock():
    s = state_at_depth(0.5)
    low = readout(s, Individual(interoception=0.5))
    high = readout(s, Individual(interoception=1.5))
    assert high.prospective > low.prospective


def test_movement_absorption_makes_time_fly():
    still = ro(0.25)
    moving = ro(0.25, movement=0.8, arousal=0.4)
    assert moving.prospective < still.prospective


def test_expertise_unlocks_timelessness_in_nondual_practice():
    nov = monte_carlo("nondual", ARCHETYPES["novice"], n=8)
    exp = monte_carlo("nondual", ARCHETYPES["expert"], n=8)
    assert exp["mean_timelessness"] > nov["mean_timelessness"] + 0.3
    assert exp["felt_retrospective_min"] < nov["felt_retrospective_min"]
    assert exp["time_wandering_frac"] < nov["time_wandering_frac"]


def test_holiday_paradox_in_bjj_and_parkour():
    for k in ("bjj", "parkour"):
        m = monte_carlo(k, ARCHETYPES["intermediate"], n=8)
        assert m["felt_prospective_min"] < m["clock_min"]
        assert m["felt_retrospective_min"] > m["clock_min"]


def test_open_monitoring_expert_feels_more_time_than_novice():
    nov = monte_carlo("open_monitoring", ARCHETYPES["novice"], n=8)
    exp = monte_carlo("open_monitoring", ARCHETYPES["expert"], n=8)
    assert exp["felt_prospective_min"] > nov["felt_prospective_min"]


def test_calibration_recovers_direction_of_experience():
    obs = [
        dict(practice="nondual", clock_min=30, prospective_min=None,
             retrospective_min=11, passage=None),
        dict(practice="open_monitoring", clock_min=30, prospective_min=39,
             retrospective_min=None, passage=-2),
    ]
    ind, loss = fit_individual(obs, n_random=12, n_refine=12)
    assert ind.experience > 0.6
