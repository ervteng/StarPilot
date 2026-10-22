"""Kia Stinger 2022 stopped-lead obstacle bias. Pure Python: needs no compiled modules (msgq/acados/params)."""
from types import SimpleNamespace

import pytest

from opendbc.car.hyundai.interface import CarInterface as HyundaiCarInterface
from opendbc.car.hyundai.values import CAR as HYUNDAI_CAR
from opendbc.car.honda.interface import CarInterface as HondaCarInterface
from opendbc.car.honda.values import CAR as HONDA_CAR
from opendbc.car.toyota.interface import CarInterface as ToyotaCarInterface
from opendbc.car.toyota.values import CAR as TOYOTA_CAR
from openpilot.selfdrive.controls.lib import longitudinal_vehicle_tunes as lvt
from openpilot.selfdrive.controls.lib.longitudinal_vehicle_tunes import (
  get_kia_stinger_2022_stopped_lead_obstacle_bias as stinger_bias,
  get_stopped_lead_obstacle_bias,
  get_stopped_lead_obstacle_biases,
)

T_FOLLOW = 1.45


def lead(d_rel, v_lead=0.0, radar=True, model_prob=0.99, y_rel=0.0, status=True):
  return SimpleNamespace(status=status, dRel=d_rel, vLead=v_lead, yRel=y_rel, radar=radar, modelProb=model_prob)


def mpc_gap(v, t=T_FOLLOW):
  return v * v / (2 * 2.5) + t * v + 6.0


def target_gap(v):
  return v * v / (2 * lvt.KIA_STINGER_2022_STOPPED_LEAD_TARGET_DECEL) + lvt.KIA_STINGER_2022_STOPPED_LEAD_FINAL_GAP_M


@pytest.fixture(scope="module")
def stinger():
  return HyundaiCarInterface.get_non_essential_params(HYUNDAI_CAR.KIA_STINGER_2022)


def bias(CP, d, v, enabled=True, **kw):
  return stinger_bias(CP, lead(d, **kw), v, t_follow=T_FOLLOW, enabled=enabled)


def test_mpc_constants_mirror_long_mpc():
  pytest.importorskip("openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc", exc_type=ImportError)
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
  assert long_mpc.COMFORT_BRAKE == lvt.KIA_STINGER_2022_STOPPED_LEAD_MPC_COMFORT_BRAKE
  assert long_mpc.STOP_DISTANCE == lvt.KIA_STINGER_2022_STOPPED_LEAD_MPC_STOP_DISTANCE


def test_tunables_stay_inside_data_range():
  assert 1.2 <= lvt.KIA_STINGER_2022_STOPPED_LEAD_TARGET_DECEL <= 1.6
  assert 3.5 <= lvt.KIA_STINGER_2022_STOPPED_LEAD_FINAL_GAP_M <= 4.5


def test_gate_requires_slot_and_fingerprint(stinger):
  assert bias(stinger, 40.0, 10.0, enabled=True) != 0.0
  assert bias(stinger, 40.0, 10.0, enabled=False) == 0.0
  for CP in (
    HyundaiCarInterface.get_non_essential_params(HYUNDAI_CAR.KIA_EV6),
    HyundaiCarInterface.get_non_essential_params(HYUNDAI_CAR.KIA_EV9),
    HondaCarInterface.get_non_essential_params(HONDA_CAR.HONDA_CRV_5G),
    ToyotaCarInterface.get_non_essential_params(TOYOTA_CAR.TOYOTA_PRIUS),
  ):
    assert stinger_bias(CP, lead(40.0), 10.0, t_follow=T_FOLLOW, enabled=True) == 0.0


def test_dispatcher_slot_a_and_other_cars_unchanged(stinger):
  ev6 = HyundaiCarInterface.get_non_essential_params(HYUNDAI_CAR.KIA_EV6)
  ev9 = HyundaiCarInterface.get_non_essential_params(HYUNDAI_CAR.KIA_EV9)
  prius = ToyotaCarInterface.get_non_essential_params(TOYOTA_CAR.TOYOTA_PRIUS)
  stopped = lead(6.8, v_lead=0.1)
  # slot A: Stinger gets nothing, in either mode
  for mode in ("acc", "blended"):
    assert get_stopped_lead_obstacle_bias(stinger, lead(40.0), 10.0, mode, T_FOLLOW, False) == 0.0
  # the flag does not leak to other cars
  assert get_stopped_lead_obstacle_bias(ev6, lead(40.0), 10.0, "acc", T_FOLLOW, True) == 0.0
  assert get_stopped_lead_obstacle_bias(ev9, stopped, 3.0, "blended", T_FOLLOW, True) == pytest.approx(1.5)
  assert get_stopped_lead_obstacle_bias(prius, lead(18.0, 0.2), 8.0, "acc", T_FOLLOW, True) > 0.0
  assert get_stopped_lead_obstacle_bias(prius, lead(18.0, 0.2), 8.0, "acc", T_FOLLOW, True) == \
         get_stopped_lead_obstacle_bias(prius, lead(18.0, 0.2), 8.0, "acc")
  # tune only in acc mode (the mode it was designed and simulated in)
  assert get_stopped_lead_obstacle_bias(stinger, lead(40.0), 10.0, "blended", T_FOLLOW, True) == 0.0
  assert get_stopped_lead_obstacle_bias(stinger, lead(40.0), 10.0, "acc", T_FOLLOW, True) != 0.0


@pytest.mark.parametrize("kw", [
  {"v_lead": 1.5},                       # moving lead
  {"y_rel": 2.0},                        # other lane
  {"radar": False, "model_prob": 0.85},  # weak vision lead
  {"status": False},
  {"d_rel": 90.0},                       # too far
])
def test_inactive_outside_activity_window(stinger, kw):
  args = {"d_rel": 40.0}
  args.update(kw)
  d = args.pop("d_rel")
  assert bias(stinger, d, 10.0, **args) == 0.0


def test_inactive_when_stopped_or_too_fast_or_not_closing(stinger):
  assert bias(stinger, 40.0, 0.0) == 0.0
  assert bias(stinger, 40.0, 23.0) == 0.0
  assert bias(stinger, 40.0, 0.1, v_lead=0.0) == 0.0  # closing < 0.15
  assert bias(stinger, 60.0, 10.0, radar=False, model_prob=0.95) != 0.0


@pytest.mark.parametrize("flag", ["model_should_stop", "red_light", "forcing_stop", "standstill"])
def test_planner_gates_suppress_bias(stinger, flag):
  leads = (lead(40.0), lead(50.0))
  active = get_stopped_lead_obstacle_biases(stinger, leads, 10.0, "acc", T_FOLLOW, stinger_stopped_lead_tune=True)
  assert all(b != 0.0 for b in active)
  suppressed = get_stopped_lead_obstacle_biases(
    stinger, leads, 10.0, "acc", T_FOLLOW, stinger_stopped_lead_tune=True, **{flag: True})
  assert suppressed == (0.0, 0.0)


@pytest.mark.parametrize("v", [3.0, 6.0, 10.0, 15.0, 20.0])
def test_steady_state_bias_matches_target_curve(stinger, v):
  d = 55.0  # well inside the full-strength window and far outside the safe distance
  b = bias(stinger, d, v)
  assert b == pytest.approx(min(target_gap(v) - mpc_gap(v), d - 0.5, lvt.KIA_STINGER_2022_STOPPED_LEAD_MAX_BIAS_M))
  if target_gap(v) - mpc_gap(v) <= 40.0:
    # with the bias, the MPC's keep-out gap (mpc_gap + bias) is the constant-decel curve
    assert mpc_gap(v) + b == pytest.approx(target_gap(v))


def test_bias_sign_convention(stinger):
  assert bias(stinger, 55.0, 5.0) < 0.0    # obstacle moved farther: closer, later stop
  assert bias(stinger, 55.0, 20.0) > 0.0   # obstacle moved closer: brake earlier
  # near standstill the keep-out gap collapses to ~g0 instead of the MPC's 6 m (obstacle = d - bias)
  v = 0.5
  assert mpc_gap(v) + bias(stinger, 55.0, v) == pytest.approx(target_gap(v))
  assert -bias(stinger, 55.0, v) < lvt.KIA_STINGER_2022_STOPPED_LEAD_MPC_STOP_DISTANCE - 3.5 + 1.0


def test_negative_bias_never_plans_inside_the_safe_decel_curve(stinger):
  for v in (1.0, 2.0, 4.0, 8.0, 12.0):
    for d in (5.0, 8.0, 12.0, 20.0, 40.0, 70.0):
      b = bias(stinger, d, v)
      if b < 0.0:
        safe_gap = v * v / (2 * lvt.KIA_STINGER_2022_STOPPED_LEAD_SAFE_DECEL) + lvt.KIA_STINGER_2022_STOPPED_LEAD_FINAL_GAP_M
        assert mpc_gap(v) + b >= safe_gap - 1e-9


def test_late_detection_never_weaker_than_baseline(stinger):
  """Obstacle position handed to the MPC is d_rel - bias (+ stopped-equivalent, ~0); baseline bias is 0."""
  for v in (3.0, 6.0, 9.0, 12.0, 15.0):
    safe_gap = v * v / (2 * lvt.KIA_STINGER_2022_STOPPED_LEAD_SAFE_DECEL) + lvt.KIA_STINGER_2022_STOPPED_LEAD_FINAL_GAP_M
    for d in (2.0, 4.0, 0.5 * safe_gap, safe_gap - 0.1, safe_gap):
      assert bias(stinger, d, v) >= 0.0, (v, d)
    # fully faded in only well beyond the safe distance
    assert bias(stinger, safe_gap + lvt.KIA_STINGER_2022_STOPPED_LEAD_LATE_FADE_M, v) <= 0.0 or v > 9.0
    # monotone: closer lead never has a more negative bias
    ds = [safe_gap + k for k in (0.0, 0.5, 1.0, 1.5, 3.0, 10.0)]
    bs = [bias(stinger, d, v) for d in ds]
    assert all(b2 <= b1 + 1e-9 for b1, b2 in zip(bs, bs[1:])) or all(b >= 0 for b in bs)


def test_positive_bias_is_capped_by_lead_distance(stinger):
  for d in (5.0, 20.0, 40.0, 79.0):
    assert bias(stinger, d, 20.0) <= max(d - 0.5, 0.0)
  assert bias(stinger, 79.9, 20.0) < bias(stinger, 55.0, 20.0)  # far-edge ramp
