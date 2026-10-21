import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, gen_empty_fingerprint
from opendbc.car.structs import CarControl
from opendbc.car.hyundai import carcontroller as carcontroller_module
from opendbc.car.hyundai import interface as interface_module
from opendbc.car.hyundai.carcontroller import CarController, IONIQ_6_CANFD_SCC_ACCEL_STEP, IONIQ_6_CANFD_SCC_DECEL_STEP, Ioniq6LongitudinalTuningState, update_ioniq_6_longitudinal_tuning
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
  for module in (carcontroller_module, interface_module):
    monkeypatch.setattr(module, "testing_ground", tg)


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


class TestVariantB:
  def jerks(self, **kwargs):
    return tunes.get_hyundai_canfd_scc_jerk_limits(STINGER, kia_stinger_2022_tune=True, **kwargs)

  def test_default_limits(self):
    assert self.jerks() == (tunes.KIA_STINGER_2022_SCC_JERK_UPPER, tunes.KIA_STINGER_2022_SCC_JERK_LOWER) == (3.0, 2.5)

  @pytest.mark.parametrize("jerk_upper", (1.0, 3.0))
  def test_upper_jerk_is_the_callers(self, jerk_upper):
    assert self.jerks(jerk_upper=jerk_upper)[0] == jerk_upper

  @pytest.mark.parametrize("lead_visible, lead_distance, accel, lower", (
    (True, 30.0, 0.0, 2.5),
    (True, 60.0, 0.0, 2.5),
    (True, 62.5, 0.0, 1.5),
    (True, 65.0, 0.0, 0.5),
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
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, False, 0.0, jerk_upper=3.0, lead_visible=False, lead_distance=0.0) == (1.5, 2.0)
    assert tunes.get_hyundai_canfd_scc_jerk_limits(GV70, False, 0.0, kia_stinger_2022_tune=True) == (1.5, 2.0)
    assert tunes.shape_hyundai_canfd_scc_accel(GV70, True, False, False, 1.0, 0.0) == pytest.approx(0.03)
    other = SimpleNamespace(brand="hyundai", carFingerprint="HYUNDAI_IONIQ_6")
    assert tunes.get_hyundai_canfd_scc_jerk_limits(other, kia_stinger_2022_tune=True) is None

  def test_shape_ramps_braking_only(self):
    kwargs = {"kia_stinger_2022_tune": True, "jerk_upper": 3.0, "lead_visible": False, "lead_distance": 0.0}
    shape = lambda accel, last, **kw: tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, False, accel, last, **{**kwargs, **kw})
    # rising passes through whatever the step (no upper clip)
    assert shape(1.0, 0.0) == 1.0
    assert shape(0.0, -0.8) == 0.0
    # braking ramps at the 50 Hz lower jerk: 0.5 m/s^3 without a lead, 2.5 with one
    assert shape(-0.3, 0.0) == pytest.approx(-0.5 / 50)
    assert shape(-0.3, 0.0, lead_visible=True, lead_distance=30.0) == pytest.approx(-2.5 / 50)
    # urgent braking is not shaped
    assert shape(-1.5, 0.0) == -1.5
    assert shape(-1.0, 0.0) == -1.0
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, True, False, True, -0.3, 0.0, **kwargs) == pytest.approx(-5.0 / 50)  # stopping
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, False, False, False, 1.0, 0.5, **kwargs) == 0.0
    assert tunes.shape_hyundai_canfd_scc_accel(STINGER, True, True, False, 1.0, 0.5, **kwargs) == 0.0

  def test_gv70_shape_still_clips_both_ways(self):
    assert tunes.shape_hyundai_canfd_scc_accel(GV70, True, False, False, -1.5, 0.0) == pytest.approx(-0.1)
    assert tunes.shape_hyundai_canfd_scc_accel(GV70, True, False, False, 1.0, 0.0) == pytest.approx(0.03)


class IoniqReference:
  """What a plain Ioniq 6 does in CarController.update, written out with update_ioniq_6_longitudinal_tuning defaults."""

  def __init__(self):
    self.state = Ioniq6LongitudinalTuningState()
    self.accel_last = 0.0  # last accel sent at 100 Hz; the Stinger sends at 50 Hz so the controller's differs
    self.frame = 0

  def step(self, accel_cmd, long_state, v_ego, a_ego, accel_last):
    if long_state in (LongCtrlState.starting, LongCtrlState.pid, LongCtrlState.stopping):
      if self.frame % 5 == 0:
        self.state = update_ioniq_6_longitudinal_tuning(self.state, accel_cmd, v_ego, a_ego, long_state, True)
      smoothed = accel_cmd >= self.state.actual_accel or self.state.launch_active or self.state.stopping
      if not smoothed:
        accel = float(np.clip(accel_cmd, accel_last - IONIQ_6_CANFD_SCC_DECEL_STEP, accel_last + IONIQ_6_CANFD_SCC_ACCEL_STEP))
        self.state.desired_accel = accel_cmd
        self.state.actual_accel = accel
        self.state.accel_last = accel
        self.state.jerk_upper = 3.0
        self.state.jerk_lower = 5.0
        self.state.launch_active = False
        self.state.long_control_state_last = long_state
      else:
        accel = self.state.actual_accel
      self.frame += 1
      return accel, smoothed
    self.frame += 1
    return None, False


class TestVariantC:
  SEQUENCE = (
    [(0.0, LongCtrlState.pid, 20.0, 0.0)] * 6 + [(0.7, LongCtrlState.pid, 20.0, 0.0)] * 10 +
    [(-0.3, LongCtrlState.pid, 20.0, 0.2)] * 10 + [(-0.9, LongCtrlState.pid, 20.0, -0.3)] * 10 +
    [(-2.5, LongCtrlState.pid, 20.0, -1.0)] * 10 + [(0.2, LongCtrlState.pid, 18.0, -1.5)] * 10 +
    [(-0.5, LongCtrlState.pid, 6.0, -0.5)] * 10 + [(-0.4, LongCtrlState.stopping, 0.3, -0.3)] * 10 +
    [(-0.8, LongCtrlState.stopping, 0.1, -0.3)] * 10 + [(0.5, LongCtrlState.starting, 0.1, 0.0)] * 10 +
    [(0.9, LongCtrlState.pid, 1.0, 0.4)] * 10 + [(-0.1, LongCtrlState.pid, 3.0, 0.5)] * 10 +
    [(0.0, LongCtrlState.off, 3.0, 0.0)] * 4 + [(-1.2, LongCtrlState.pid, 15.0, 0.0)] * 10
  )

  def gate_active(self, monkeypatch, variant, long_state=LongCtrlState.pid, alpha_long=True):
    select_variant(monkeypatch, variant)
    h = Harness(alpha_long=alpha_long)
    h.run(-0.8, 10, state=long_state, v_ego=20.0)
    return h.controller._ioniq_6_long_tuning != Ioniq6LongitudinalTuningState()

  def test_gate_on_only_for_c_with_active_long_state(self, monkeypatch):
    assert self.gate_active(monkeypatch, "C")
    assert not self.gate_active(monkeypatch, "C", long_state=LongCtrlState.off)
    assert not self.gate_active(monkeypatch, "A")
    assert not self.gate_active(monkeypatch, "B")
    assert not self.gate_active(monkeypatch, "C", alpha_long=False)

  def test_matches_plain_ioniq_6(self, monkeypatch):
    select_variant(monkeypatch, "C")
    h = Harness()
    ref = IoniqReference()
    for accel_cmd, long_state, v_ego, a_ego in self.SEQUENCE:
      accel_last = h.controller.accel_last
      actuators, sends = h.step(accel_cmd, state=long_state, v_ego=v_ego, a_ego=a_ego)
      f32 = lambda x: float(np.float32(x))  # the controller reads accel from a capnp float32
      ref_accel, smoothed = ref.step(f32(accel_cmd), long_state, v_ego, a_ego, accel_last)
      assert h.controller._ioniq_6_long_tuning == ref.state
      if ref_accel is not None:
        assert actuators.accel == pytest.approx(ref_accel)
      if any(m[0] == 1057 for m in sends):
        scc12, scc14 = h.scc(sends)
        if ref_accel is not None:
          assert scc12["aReqValue"] == pytest.approx(ref_accel, abs=0.01)
          upper, lower = (ref.state.jerk_upper, ref.state.jerk_lower) if smoothed else \
            (3.0 if long_state == LongCtrlState.pid else 1.0, 5.0)
          assert scc14["JerkUpperLimit"] == pytest.approx(upper, abs=0.1)
          assert scc14["JerkLowerLimit"] == pytest.approx(lower, abs=0.1)

  def test_scc14_carries_state_jerks_while_smoothing(self, monkeypatch):
    select_variant(monkeypatch, "C")
    h = Harness()
    h.run(0.0, 10, v_ego=20.0)
    seen = set()
    for _ in range(30):
      _, sends = h.step(-0.9, v_ego=20.0, a_ego=-0.3)
      if any(m[0] == 1057 for m in sends):
        _, scc14 = h.scc(sends)
        tuning = h.controller._ioniq_6_long_tuning
        assert scc14["JerkLowerLimit"] == pytest.approx(tuning.jerk_lower, abs=0.1)
        assert scc14["JerkUpperLimit"] == pytest.approx(tuning.jerk_upper, abs=0.1)
        seen.add(round(tuning.jerk_lower, 1))
    assert len(seen) > 1 or seen != {5.0}

  def test_ev_pieces_are_the_ioniq_ones(self, monkeypatch):
    # stop brake cap and launch hold of the Ioniq 6 are active (no ev6_gt_line / ev9 / low-speed cap variants)
    select_variant(monkeypatch, "C")
    h = Harness()
    h.run(-0.8, 10, state=LongCtrlState.stopping, v_ego=0.1)
    assert h.controller._ioniq_6_long_tuning.stopping
    ref = update_ioniq_6_longitudinal_tuning(Ioniq6LongitudinalTuningState(), -0.8, 0.1, 0.0, LongCtrlState.stopping, True)
    assert ref.desired_accel > -0.8  # the Ioniq stop brake cap clamps it
    assert h.controller._ioniq_6_long_tuning.desired_accel == pytest.approx(ref.desired_accel)


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
    for accel_value, jerk_upper, jerk_lower, accel_raw in out[2:]:
      assert accel_value == pytest.approx(1.0, abs=0.01)  # rising accel is not ramped
      assert jerk_upper == pytest.approx(3.0)
      assert jerk_lower == pytest.approx(0.5)  # no lead
      assert accel_raw == pytest.approx(accel_value, abs=1e-6)

  def test_variant_b_braking_ramps_and_urgent_passes_through(self, monkeypatch):
    select_variant(monkeypatch, "B")
    h = Harness()
    h.run(0.0, 4)
    out = h.run(-0.9, 4)
    assert -0.9 < out[-1][0] < 0.0
    assert all(o[2] == pytest.approx(2.5) for o in h.run(-0.9, 2, lead=30.0))
    out = h.run(-3.0, 4)
    assert all(o[2] == pytest.approx(5.0) for o in out)
    assert out[-1][0] == pytest.approx(-3.0, abs=0.01)

  def test_variant_b_lead_changes_lower_jerk(self, monkeypatch):
    select_variant(monkeypatch, "B")
    h = Harness()
    out = h.run(-0.3, 4, lead=30.0)
    assert all(o[2] == pytest.approx(2.5) for o in out)


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

  @pytest.mark.parametrize("variant", ("A", "B", "C"))
  def test_no_planner_entries(self, monkeypatch, variant):
    select_variant(monkeypatch, variant)
    assert tunes.get_lead_follow_jerk_scale(STINGER) == 1.0
    assert tunes.get_far_follow_output_slew_rates(STINGER) == (0.0, 0.0)
    assert tunes.get_lead_follow_jerk_scale(GV70) == 1.75
