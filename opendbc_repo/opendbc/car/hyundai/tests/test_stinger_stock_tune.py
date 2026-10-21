import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, gen_empty_fingerprint
from opendbc.car.structs import CarControl
from opendbc.car.hyundai import carcontroller as carcontroller_module
from opendbc.car.hyundai import interface as interface_module
from opendbc.car.hyundai.carcontroller import CarController, Ioniq6LongitudinalTuningState, update_ioniq_6_longitudinal_tuning
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, DBC, KIA_STINGER_2022_LONG_TUNING_TESTING_GROUND_ID, kia_stinger_2022_longitudinal_tuning
from openpilot.selfdrive.controls.lib import longitudinal_vehicle_tunes as tunes

LongCtrlState = CarControl.Actuators.LongControlState
SLOT = KIA_STINGER_2022_LONG_TUNING_TESTING_GROUND_ID

STINGER = SimpleNamespace(brand="hyundai", carFingerprint="KIA_STINGER_2022", openpilotLongitudinalControl=True)
GV70 = SimpleNamespace(brand="hyundai", carFingerprint="GENESIS_GV70_ELECTRIFIED_1ST_GEN")


def toggles():
  return SimpleNamespace(always_on_lateral_lkas=False, force_torque_controller=False, nnff=False, nnff_lite=False)


def fake_testing_ground(variant):
  return SimpleNamespace(use=lambda slot_id, v="B": slot_id == SLOT and v == variant)


def select_variant(monkeypatch, variant):
  tg = fake_testing_ground(variant)
  for module in (carcontroller_module, interface_module, tunes):
    monkeypatch.setattr(module, "testing_ground", tg)
  tunes._kia_stinger_2022_tune_latch.clear()


@pytest.fixture(autouse=True)
def reset_latch():
  tunes._kia_stinger_2022_tune_latch.clear()
  yield
  tunes._kia_stinger_2022_tune_latch.clear()


def make_cp(car=CAR.KIA_STINGER_2022, alpha_long=True):
  return CarInterface.get_params(car, gen_empty_fingerprint(), [], alpha_long, False, False, toggles())


class Harness:
  """Runs CarController.update on the plain CAN path and parses SCC12 / SCC14 from what was sent."""

  def __init__(self, car=CAR.KIA_STINGER_2022, alpha_long=True):
    self.CP = make_cp(car, alpha_long)
    self.controller = CarController(DBC[self.CP.carFingerprint], self.CP)
    self.controller._ecu_disable_checked = True  # do not read the shared EcuDisableFailed param
    self.parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("LKAS11", 0), ("CLU11", 0)], 0)
    self.scc_parser = CANParser(DBC[self.CP.carFingerprint][Bus.pt], [("SCC12", 0), ("SCC14", 0)], 0)

  def step(self, accel, state=LongCtrlState.pid, v_ego=20.0, a_ego=0.0, enabled=True, override=False,
           lead=None):
    cc = CarControl.new_message()
    cc.enabled = enabled
    cc.latActive = True
    cc.longActive = enabled
    cc.actuators.accel = accel
    cc.actuators.longControlState = state
    cc.cruiseControl.override = override
    cc.hudControl.setSpeed = 30.0
    cs = SimpleNamespace(
      out=SimpleNamespace(vEgo=v_ego, vEgoRaw=v_ego, aEgo=a_ego, steeringAngleDeg=0.0, standstill=False,
                          gasPressed=False, brakePressed=False, steeringTorque=0.0),
      lkas11=self.parser.vl["LKAS11"], clu11=self.parser.vl["CLU11"], is_metric=True,
      redneck_send_button=0, redneck_last_sent_button=0, main_cruise_on=True,
      openpilot_lead_visible=lead is not None, openpilot_lead_distance=lead or 0.0, openpilot_lead_rel_speed=-1.0 if lead is not None else 0.0,
    )
    actuators, sends = self.controller.update(cc.as_reader(), cs, 0, toggles())
    return actuators, sends

  def scc(self, sends):
    self.scc_parser.update([(0, [m for m in sends if m[0] in (1057, 905)])])
    return self.scc_parser.vl["SCC12"], self.scc_parser.vl["SCC14"]

  def run(self, accel, frames, **kwargs):
    """Runs full 100 Hz frames; returns the list of (aReqValue, JerkUpper, JerkLower) from every SCC pair sent."""
    out = []
    for _ in range(frames):
      _, sends = self.step(accel, **kwargs)
      scc12, scc14 = self.scc(sends) if any(m[0] == 1057 for m in sends) else (None, None)
      if scc12 is not None:
        out.append((scc12["aReqValue"], scc14["JerkUpperLimit"], scc14["JerkLowerLimit"], scc12["aReqRaw"]))
    return out


class TestGate:
  @pytest.mark.parametrize("b, c, expected", ((False, False, None), (True, False, "B"), (False, True, "C")))
  def test_stinger_variants(self, b, c, expected):
    assert kia_stinger_2022_longitudinal_tuning(CAR.KIA_STINGER_2022, True, b, c) == expected

  def test_other_car_or_no_openpilot_long(self):
    assert kia_stinger_2022_longitudinal_tuning(CAR.KIA_STINGER, True, True, True) is None
    assert kia_stinger_2022_longitudinal_tuning(CAR.GENESIS_GV70_ELECTRIFIED_1ST_GEN, True, True, False) is None
    assert kia_stinger_2022_longitudinal_tuning(CAR.KIA_STINGER_2022, False, True, False) is None
    assert kia_stinger_2022_longitudinal_tuning(CAR.KIA_STINGER_2022, False, False, True) is None

  @pytest.mark.parametrize("variant, expected", (("A", None), ("B", "B"), ("C", "C")))
  def test_controller_latches_variant_at_startup(self, monkeypatch, variant, expected):
    select_variant(monkeypatch, variant)
    h = Harness()
    assert h.controller._stinger_2022_tune == expected
    select_variant(monkeypatch, "C" if variant != "C" else "B")
    assert h.controller._stinger_2022_tune == expected

  def test_controller_without_openpilot_long_or_other_car_is_off(self, monkeypatch):
    select_variant(monkeypatch, "B")
    assert Harness(alpha_long=False).controller._stinger_2022_tune is None
    assert Harness(car=CAR.KIA_STINGER).controller._stinger_2022_tune is None

  def test_planner_gate_latches(self, monkeypatch):
    select_variant(monkeypatch, "B")
    assert tunes.get_kia_stinger_2022_stock_tune(STINGER) == "B"
    select_variant(monkeypatch, "A")
    tunes._kia_stinger_2022_tune_latch["tune"] = "B"
    assert tunes.get_kia_stinger_2022_stock_tune(STINGER) == "B"


class TestVariantB:
  def jerks(self, **kwargs):
    return tunes.get_hyundai_canfd_scc_jerk_limits(STINGER, kia_stinger_2022_tune=True, **kwargs)

  def test_default_limits(self):
    assert self.jerks() == (tunes.KIA_STINGER_2022_SCC_JERK_UPPER, tunes.KIA_STINGER_2022_SCC_JERK_LOWER)

  @pytest.mark.parametrize("v_ego, upper", ((0.0, 1.0), (0.5, 0.85), (1.0, 0.7), (2.0, 0.6), (3.0, 0.5), (20.0, 0.5)))
  def test_upper_jerk_follows_speed(self, v_ego, upper):
    assert self.jerks(v_ego=v_ego)[0] == pytest.approx(upper)

  @pytest.mark.parametrize("lead_visible, lead_distance, accel, lower", (
    (True, 30.0, 0.0, 2.5),
    (True, 60.0, 0.0, 2.5),
    (True, 75.0, 0.0, 1.5),
    (True, 90.0, 0.0, 0.5),
    (True, 150.0, 0.0, 0.5),
    (False, 0.0, 0.3, 0.5),
    (False, 0.0, -0.4, 0.5),
    (False, 0.0, -0.6, 2.5),
    (True, 150.0, -0.6, 2.5),
  ))
  def test_lower_jerk_follows_lead(self, lead_visible, lead_distance, accel, lower):
    assert self.jerks(lead_visible=lead_visible, lead_distance=lead_distance, accel=accel)[1] == pytest.approx(lower)

  def test_urgent_and_stopping_use_full_lower_jerk(self):
    assert self.jerks(accel=-1.0, lead_visible=False, lead_distance=0.0)[1] == 5.0
    assert self.jerks(accel=-2.0, lead_visible=True, lead_distance=150.0)[1] == 5.0
    assert self.jerks(stopping=True, accel=-0.2, lead_visible=False, lead_distance=0.0)[1] == 5.0
    assert self.jerks(accel=-0.99)[1] == 2.5

  def test_not_active_without_flag(self):
    assert tunes.get_hyundai_canfd_scc_jerk_limits(STINGER) is None
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, False, 1.0, 0.0) == 1.0

  def test_gv70_unchanged(self):
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70) == (1.5, 2.0)
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, True, 0.0) == (1.5, 5.0)
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, False, -1.0) == (1.5, 5.0)
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, False, 0.0, v_ego=0.0, lead_visible=False, lead_distance=0.0) == (1.5, 2.0)
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, False, 0.0, kia_stinger_2022_tune=True) == (1.5, 2.0)
    assert tunes.shape_hyundai_canfd_scc_accel(GV70, True, False, False, 1.0, 0.0) == pytest.approx(0.03)
    other = SimpleNamespace(brand="hyundai", carFingerprint="HYUNDAI_IONIQ_6")
    assert tunes.get_hyundai_canfd_scc_jerk_limits(other, kia_stinger_2022_tune=True) is None

  def test_shape_rate_matches_50hz_limits(self):
    kwargs = {"kia_stinger_2022_tune": True, "v_ego": 20.0, "lead_visible": False, "lead_distance": 0.0}
    accel = tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, False, 1.0, 0.0, **kwargs)
    assert accel == pytest.approx(0.5 / 50)
    accel = tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, False, -0.3, accel, **kwargs)
    assert accel == pytest.approx(0.01 - 0.5 / 50)
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, False, -1.5, 0.0, **kwargs) == pytest.approx(-0.1)
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, False, False, False, 1.0, 0.5, **kwargs) == 0.0
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, True, True, False, 1.0, 0.5, **kwargs) == 0.0


class TestVariantC:
  def run_update(self, accel_cmd, v_ego=20.0, a_ego=0.0, state=None, long_state=LongCtrlState.pid, stinger=True):
    state = state or Ioniq6LongitudinalTuningState()
    return update_ioniq_6_longitudinal_tuning(state, accel_cmd, v_ego, a_ego, long_state, True, kia_stinger_2022=stinger)

  @pytest.mark.parametrize("v_ego, cap", ((0.0, 1.0), (1.0, 0.7), (3.0, 0.5), (20.0, 0.5)))
  def test_upper_jerk_capped_after_min_floor(self, v_ego, cap):
    state = self.run_update(1.0, v_ego=v_ego)
    assert state.jerk_upper == pytest.approx(cap)
    state = self.run_update(0.01, v_ego=20.0)  # tiny error: the 0.6 floor must not lift it above 0.5
    assert state.jerk_upper <= 0.5 + 1e-9

  def test_lower_jerk_capped(self):
    state = self.run_update(-0.9, a_ego=-2.0, state=Ioniq6LongitudinalTuningState(accel_last=0.0))
    assert state.jerk_lower <= 2.5
    stock_like = self.run_update(-0.9, a_ego=-2.0, state=Ioniq6LongitudinalTuningState(accel_last=0.0), stinger=False)
    assert stock_like.jerk_lower > 2.5

  def test_urgent_braking_is_not_smoothed(self):
    state = self.run_update(-2.0, state=Ioniq6LongitudinalTuningState(accel_last=0.0))
    assert state.actual_accel == pytest.approx(-2.0)
    assert state.jerk_lower == 5.0

  def test_non_urgent_braking_is_ramped(self):
    state = self.run_update(-0.9, state=Ioniq6LongitudinalTuningState(accel_last=0.0))
    assert -0.9 < state.actual_accel < 0.0
    assert state.jerk_lower <= 2.5

  def test_ev_specific_pieces_disabled(self):
    # launch hold would force a positive floor at low speed
    state = self.run_update(0.1, v_ego=0.2, long_state=LongCtrlState.starting)
    assert not state.launch_active
    assert state.desired_accel == pytest.approx(0.1)
    # stop brake cap would clamp a hard stopping decel at very low speed
    state = self.run_update(-0.8, v_ego=0.1, long_state=LongCtrlState.stopping)
    assert state.stopping
    assert state.desired_accel == pytest.approx(-0.8)
    # without the flag both are active
    assert self.run_update(0.1, v_ego=0.2, long_state=LongCtrlState.starting, stinger=False).launch_active
    assert self.run_update(-0.8, v_ego=0.1, long_state=LongCtrlState.stopping, stinger=False).desired_accel > -0.8

  def test_restart_from_stop_jerk_tweak_disabled(self):
    def run(stinger):
      state = Ioniq6LongitudinalTuningState(long_control_state_last=LongCtrlState.stopping)
      return self.run_update(1.0, v_ego=0.4, state=state, long_state=LongCtrlState.pid, stinger=stinger)
    # the restart tweak caps jerk_upper at 3.6 * 1.2 = 4.3 at 0.4 m/s; the stock curve (1.0 .. 0.7) applies instead
    assert run(True).jerk_upper == pytest.approx(0.88)
    assert run(False).jerk_upper > 1.0

  @pytest.mark.parametrize("kwargs", (
    {"ev6_gt_line": True}, {"low_speed_stop_brake_cap": True}, {"ev9": True}, {},
  ))
  @pytest.mark.parametrize("v_ego, a_ego, accel_cmd, long_state", (
    (0.0, 0.0, 0.8, LongCtrlState.starting), (0.3, 0.1, -0.5, LongCtrlState.stopping),
    (10.0, -0.5, -1.5, LongCtrlState.pid), (25.0, 0.2, 0.4, LongCtrlState.pid), (1.0, 0.0, -2.5, LongCtrlState.pid),
  ))
  def test_other_cars_unchanged_by_new_kwarg(self, kwargs, v_ego, a_ego, accel_cmd, long_state):
    def run(**extra):
      state = Ioniq6LongitudinalTuningState(accel_last=-0.3, actual_accel=-0.3)
      return update_ioniq_6_longitudinal_tuning(state, accel_cmd, v_ego, a_ego, long_state, True, **kwargs, **extra)
    assert run() == run(kia_stinger_2022=False)


class TestCanOutput:
  def test_slot_a_and_other_car_send_previous_bytes(self, monkeypatch):
    select_variant(monkeypatch, "A")
    h = Harness()
    out = h.run(-0.6, 4)
    assert out
    for accel_value, jerk_upper, jerk_lower, accel_raw in out:
      assert accel_value == pytest.approx(-0.6, abs=0.01)
      assert accel_raw == pytest.approx(-0.6, abs=0.01)
      assert jerk_upper == pytest.approx(3.0)
      assert jerk_lower == pytest.approx(5.0)

    select_variant(monkeypatch, "B")
    other = Harness(car=CAR.KIA_STINGER)
    for accel_value, jerk_upper, jerk_lower, _ in other.run(-0.6, 4):
      assert accel_value == pytest.approx(-0.6, abs=0.01)
      assert (jerk_upper, jerk_lower) == (pytest.approx(3.0), pytest.approx(5.0))

  @pytest.mark.parametrize("car, variant", (
    (CAR.KIA_STINGER_2022, "A"), (CAR.KIA_STINGER, "A"), (CAR.KIA_STINGER, "B"), (CAR.KIA_STINGER, "C"),
  ))
  def test_slot_a_and_other_car_bytes_match_pre_change_golden(self, monkeypatch, variant, car):
    """Golden SCC11/12/14 bytes were captured from the commit before this change (823a5d5)."""
    select_variant(monkeypatch, variant)
    golden = json.loads((Path(__file__).parent / "stinger_slot_a_golden.json").read_text())[str(car)]
    h = Harness(car=car)
    seq = [(0.0, LongCtrlState.pid, 20.0), (0.7, LongCtrlState.pid, 20.0), (0.7, LongCtrlState.pid, 20.0),
           (-0.6, LongCtrlState.pid, 20.0), (-2.0, LongCtrlState.pid, 20.0), (-0.3, LongCtrlState.stopping, 0.2),
           (0.4, LongCtrlState.starting, 0.1), (0.9, LongCtrlState.pid, 5.0)]
    rows = []
    for _ in range(2):
      for accel, state, v_ego in seq:
        for _ in range(2):
          actuators, sends = h.step(accel, state=state, v_ego=v_ego, lead=40.0)
          rows.append([actuators.accel, [(a, bytes(d).hex()) for a, d, _ in sends if a in (1057, 905, 1056)]])
    assert json.loads(json.dumps(rows)) == golden

  def test_variant_b_scc_values(self, monkeypatch):
    select_variant(monkeypatch, "B")
    h = Harness()
    out = h.run(1.0, 20, v_ego=20.0)
    values = [o[0] for o in out]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(10 * 0.5 / 50, abs=0.01)
    for accel_value, jerk_upper, jerk_lower, accel_raw in out:
      assert jerk_upper == pytest.approx(0.5)
      assert jerk_lower == pytest.approx(0.5)  # no lead
      assert accel_raw == pytest.approx(accel_value, abs=1e-6)

  def test_variant_b_urgent_braking_sends_lower_jerk_5(self, monkeypatch):
    select_variant(monkeypatch, "B")
    h = Harness()
    h.run(0.0, 4)
    out = h.run(-3.0, 4)
    assert all(o[2] == pytest.approx(5.0) for o in out)
    assert out[-1][0] < -0.1  # rate limited at 5 m/s^3 / 50 Hz

  def test_variant_b_lead_changes_lower_jerk(self, monkeypatch):
    select_variant(monkeypatch, "B")
    h = Harness()
    out = h.run(-0.3, 4, lead=30.0)
    assert all(o[2] == pytest.approx(2.5) for o in out)

  def test_variant_c_scc_values_and_urgent_pass_through(self, monkeypatch):
    select_variant(monkeypatch, "C")
    h = Harness()
    out = h.run(0.5, 40, v_ego=20.0)
    assert 0.0 < out[-1][0] < 0.5
    assert all(o[1] <= 0.5 + 1e-6 for o in out)
    assert all(o[2] <= 2.5 + 1e-6 for o in out)
    out = h.run(-0.9, 6, v_ego=20.0)
    assert out[-1][0] > -0.9  # ramped, not passed through
    out = h.run(-2.5, 2, v_ego=20.0)
    assert out[-1][0] == pytest.approx(-2.5, abs=0.01)
    assert out[-1][2] == pytest.approx(5.0)

  def test_variant_c_resets_when_disengaged(self, monkeypatch):
    select_variant(monkeypatch, "C")
    h = Harness()
    h.run(-0.8, 40, v_ego=20.0)
    assert h.controller._ioniq_6_long_tuning.accel_last != 0.0
    h.run(0.0, 2, state=LongCtrlState.off, enabled=False)
    assert h.controller._ioniq_6_long_tuning.accel_last == 0.0
    out = h.run(0.2, 2, v_ego=20.0)
    assert abs(out[0][0]) < 0.1


class TestDelayAndPlanner:
  @pytest.mark.parametrize("variant, delay", (("A", 0.15), ("B", 0.5), ("C", 0.5)))
  def test_actuator_delay_only_for_b_and_c(self, monkeypatch, variant, delay):
    select_variant(monkeypatch, variant)
    CP = make_cp()
    default_delay = CP.longitudinalActuatorDelay
    CarInterface.apply_post_fingerprint_params(CP, CAR.KIA_STINGER_2022, gen_empty_fingerprint(), [])
    assert CP.longitudinalActuatorDelay == pytest.approx(delay if variant != "A" else default_delay)

  def test_actuator_delay_other_car_untouched(self, monkeypatch):
    select_variant(monkeypatch, "B")
    CP = make_cp(CAR.KIA_STINGER)
    default_delay = CP.longitudinalActuatorDelay
    CarInterface.apply_post_fingerprint_params(CP, CAR.KIA_STINGER, gen_empty_fingerprint(), [])
    assert CP.longitudinalActuatorDelay == default_delay

  @pytest.mark.parametrize("variant, scale, rates", (("A", 1.0, (0.0, 0.0)), ("B", 1.25, (2.0, 1.25)), ("C", 1.25, (2.0, 1.25))))
  def test_planner_entries_only_for_b_and_c(self, monkeypatch, variant, scale, rates):
    select_variant(monkeypatch, variant)
    assert tunes.get_lead_follow_jerk_scale(STINGER) == scale
    assert tunes.get_far_follow_output_slew_rates(STINGER) == rates

  def test_planner_entries_other_cars_untouched(self, monkeypatch):
    select_variant(monkeypatch, "B")
    other = SimpleNamespace(brand="hyundai", carFingerprint="KIA_STINGER", openpilotLongitudinalControl=True)
    assert tunes.get_lead_follow_jerk_scale(other) == 1.0
    assert tunes.get_far_follow_output_slew_rates(other) == (0.0, 0.0)
    assert tunes.get_lead_follow_jerk_scale(GV70) == 1.75
