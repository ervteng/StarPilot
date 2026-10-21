from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint
from opendbc.car.structs import CarControl, CarParams
from opendbc.car.hyundai import carcontroller, hyundaican, interface
from opendbc.car.hyundai.carcontroller import CarController
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.longitudinal_tuning import ACTUATOR_DELAY, URGENT_ACCEL, UPDATE_FRAMES, LongitudinalTuner, \
                                                    LongitudinalTuningMode, stinger_long_tuning_mode
from opendbc.car.hyundai.values import CAR, DBC, STINGER_LONG_TUNING_TESTING_GROUND_ID

LongCtrlState = CarControl.Actuators.LongControlState
STINGER_SLOT = STINGER_LONG_TUNING_TESTING_GROUND_ID


def fake_testing_ground(slot, variant):
  # same contract as starpilot's testing_ground.use: is this slot active with this variant?
  return SimpleNamespace(use=lambda slot_id, v="B": (slot_id, v) == (slot, variant))


def startup_params(candidate):
  # get_car applies post-fingerprint params right after get_params
  CP = CarInterface.get_params(candidate, gen_empty_fingerprint(), [], True, False, False, None)
  CarInterface.apply_post_fingerprint_params(CP, candidate, gen_empty_fingerprint(), [])
  return CP


def cc(accel, state=LongCtrlState.pid, long_active=True):
  return SimpleNamespace(longActive=long_active, actuators=SimpleNamespace(accel=accel, longControlState=state))


def run(tuner, accel, v_ego=25.0, a_ego=0.0, updates=1, **kw):
  for _ in range(updates):
    tuner.update(cc(accel, **kw), v_ego, a_ego)
  return tuner


@pytest.mark.parametrize("fingerprint, op_long, selection, mode", [
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "A"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "B"), LongitudinalTuningMode.DYNAMIC),
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "C"), LongitudinalTuningMode.PREDICTIVE),
  (CAR.KIA_STINGER_2022, False, (STINGER_SLOT, "C"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER_2022, True, ("5", "B"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER, True, (STINGER_SLOT, "B"), LongitudinalTuningMode.OFF),
  (CAR.HYUNDAI_SONATA, True, (STINGER_SLOT, "C"), LongitudinalTuningMode.OFF),
])
def test_stinger_long_tuning_mode(fingerprint, op_long, selection, mode):
  assert stinger_long_tuning_mode(fingerprint, op_long, fake_testing_ground(*selection)) == mode


@pytest.mark.parametrize("variant, delay", [("A", None), ("B", ACTUATOR_DELAY), ("C", ACTUATOR_DELAY)])
def test_actuator_delay_follows_testing_ground(monkeypatch, variant, delay):
  monkeypatch.setattr(interface, "testing_ground", fake_testing_ground(STINGER_SLOT, variant))
  stinger = startup_params(CAR.KIA_STINGER_2022)
  sonata = startup_params(CAR.HYUNDAI_SONATA)

  monkeypatch.setattr(interface, "testing_ground", fake_testing_ground("1", "A"))
  baseline = startup_params(CAR.KIA_STINGER_2022)
  sonata_baseline = startup_params(CAR.HYUNDAI_SONATA)
  assert stinger.longitudinalActuatorDelay == pytest.approx(delay if delay is not None else baseline.longitudinalActuatorDelay)
  assert sonata.longitudinalActuatorDelay == pytest.approx(sonata_baseline.longitudinalActuatorDelay)


@pytest.mark.parametrize("candidate, variant, enabled", [
  (CAR.KIA_STINGER_2022, "B", True), (CAR.KIA_STINGER_2022, "C", True),
  (CAR.KIA_STINGER_2022, "A", False), (CAR.HYUNDAI_SONATA, "C", False),
])
def test_controller_reads_mode_at_startup(monkeypatch, candidate, variant, enabled):
  monkeypatch.setattr(carcontroller, "testing_ground", fake_testing_ground(STINGER_SLOT, variant))
  CP = startup_params(candidate)
  controller = CarController(DBC[CP.carFingerprint], CP)
  assert controller._long_tuner.enabled == enabled

  # changing the Testing Ground mid-drive doesn't change the running controller
  monkeypatch.setattr(carcontroller, "testing_ground", fake_testing_ground(STINGER_SLOT, "A"))
  assert controller._long_tuner.enabled == enabled


def test_dynamic_uses_speed_based_limits():
  tuner = run(LongitudinalTuner(LongitudinalTuningMode.DYNAMIC, radar_available=True), -0.5, v_ego=25.0, a_ego=-0.5)
  assert tuner.jerk_upper == pytest.approx(2.0)
  assert tuner.jerk_lower == pytest.approx(3.0)
  assert 0.0 < tuner.comfort_band_lower <= 0.04

  slow = run(LongitudinalTuner(LongitudinalTuningMode.DYNAMIC, radar_available=True), -0.5, v_ego=0.0)
  assert slow.jerk_lower == pytest.approx(3.5)


def test_predictive_limits_follow_the_accel_error():
  small = run(LongitudinalTuner(LongitudinalTuningMode.PREDICTIVE, radar_available=True), -0.05, v_ego=25.0)
  large = run(LongitudinalTuner(LongitudinalTuningMode.PREDICTIVE, radar_available=True), -1.5, v_ego=25.0)
  assert small.jerk_lower == pytest.approx(0.5)  # MIN_JERK
  assert small.jerk_lower < large.jerk_lower <= 3.0


def test_gentle_braking_is_rate_limited():
  tuner = run(LongitudinalTuner(LongitudinalTuningMode.DYNAMIC, radar_available=True), -1.5, v_ego=25.0)
  step = 3.0 * 0.01 * UPDATE_FRAMES
  assert tuner.desired_accel == pytest.approx(-1.5)
  assert tuner.actual_accel == pytest.approx(-step)


@pytest.mark.parametrize("mode", [LongitudinalTuningMode.DYNAMIC, LongitudinalTuningMode.PREDICTIVE])
def test_urgent_braking_is_not_delayed(mode):
  tuner = run(LongitudinalTuner(mode, radar_available=True), URGENT_ACCEL - 1.0, v_ego=25.0)
  assert tuner.actual_accel == pytest.approx(URGENT_ACCEL - 1.0)
  assert tuner.jerk_lower == pytest.approx(5.0)


def test_without_radar_accel_passes_through():
  tuner = run(LongitudinalTuner(LongitudinalTuningMode.PREDICTIVE, radar_available=False), -1.5, v_ego=25.0)
  assert tuner.actual_accel == pytest.approx(-1.5)
  assert tuner.jerk_lower == pytest.approx(5.0)


def test_inactive_resets_outputs():
  tuner = run(LongitudinalTuner(LongitudinalTuningMode.PREDICTIVE, radar_available=True), -1.0)
  run(tuner, -1.0, long_active=False)
  assert (tuner.actual_accel, tuner.desired_accel, tuner.jerk_upper, tuner.jerk_lower) == (0.0, 0.0, 0.0, 0.0)
  assert (tuner.comfort_band_upper, tuner.comfort_band_lower) == (0.0, 0.0)


def test_stop_request_waits_about_a_second():
  tuner = run(LongitudinalTuner(LongitudinalTuningMode.DYNAMIC, radar_available=True), -0.5, v_ego=1.0)
  run(tuner, -0.5, v_ego=0.2, state=LongCtrlState.stopping, updates=20)
  assert not tuner.stopping
  run(tuner, -0.5, v_ego=0.0, state=LongCtrlState.stopping, updates=2)
  assert tuner.stopping
  assert tuner.desired_accel == 0.0


def test_acc_commands_carry_the_tuning():
  CP = CarParams.new_message()
  CP.carFingerprint = CAR.KIA_STINGER_2022
  packer = CANPacker(DBC[CP.carFingerprint][Bus.pt])
  parser = CANParser(DBC[CP.carFingerprint][Bus.pt], [("SCC12", 0), ("SCC14", 0)], 0)
  tuning = SimpleNamespace(desired_accel=-1.2, jerk_upper=2.0, jerk_lower=3.0, comfort_band_upper=0.02, comfort_band_lower=0.04)

  msgs = hyundaican.create_acc_commands(packer, enabled=True, accel=-0.6, upper_jerk=3.0, idx=3,
                                        hud_control=SimpleNamespace(leadDistanceBars=3, leadVisible=False), set_speed=42,
                                        stopping=False, long_override=False, use_fca=False, CP=CP, tuning=tuning)
  parser.update([(1, msgs)])

  assert parser.vl["SCC12"]["aReqRaw"] == pytest.approx(-1.2, abs=0.01)
  assert parser.vl["SCC12"]["aReqValue"] == pytest.approx(-0.6, abs=0.01)
  assert parser.vl["SCC14"]["JerkUpperLimit"] == pytest.approx(2.0, abs=0.1)
  assert parser.vl["SCC14"]["JerkLowerLimit"] == pytest.approx(3.0, abs=0.1)
  assert parser.vl["SCC14"]["ComfortBandUpper"] == pytest.approx(0.02, abs=0.01)
  assert parser.vl["SCC14"]["ComfortBandLower"] == pytest.approx(0.04, abs=0.01)
