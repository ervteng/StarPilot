from types import SimpleNamespace

import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, DT_CTRL, gen_empty_fingerprint
from opendbc.car.structs import CarControl, CarParams
from opendbc.car.hyundai import carcontroller, hyundaican, interface
from opendbc.car.hyundai.carcontroller import CarController, KiaStingerLongitudinalTuningState, update_kia_stinger_longitudinal_tuning
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.longitudinal_tuning import ACTUATOR_DELAY, LongitudinalTuningMode, stinger_long_tuning_mode
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


def tune(accel, v_ego=20.0, state=None, steps=1, long_state=LongCtrlState.pid, active=True, override=False,
         lead_visible=False, lead_distance=0.0):
  state = state or KiaStingerLongitudinalTuningState()
  for _ in range(steps):
    state = update_kia_stinger_longitudinal_tuning(state, accel, v_ego, long_state, active, override, lead_visible, lead_distance)
  return state


@pytest.mark.parametrize("fingerprint, op_long, selection, mode", [
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "A"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "B"), LongitudinalTuningMode.DYNAMIC),
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "C"), LongitudinalTuningMode.PREDICTIVE),
  (CAR.KIA_STINGER_2022, True, (STINGER_SLOT, "D"), LongitudinalTuningMode.STOCK_LIKE),
  (CAR.KIA_STINGER_2022, False, (STINGER_SLOT, "D"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER_2022, True, ("5", "D"), LongitudinalTuningMode.OFF),
  (CAR.KIA_STINGER, True, (STINGER_SLOT, "D"), LongitudinalTuningMode.OFF),
  (CAR.HYUNDAI_SONATA, True, (STINGER_SLOT, "D"), LongitudinalTuningMode.OFF),
])
def test_mode_mapping(fingerprint, op_long, selection, mode):
  assert stinger_long_tuning_mode(fingerprint, op_long, fake_testing_ground(*selection)) == mode


def test_actuator_delay_for_stock_like(monkeypatch):
  monkeypatch.setattr(interface, "testing_ground", fake_testing_ground(STINGER_SLOT, "D"))
  assert startup_params(CAR.KIA_STINGER_2022).longitudinalActuatorDelay == pytest.approx(ACTUATOR_DELAY)
  assert ACTUATOR_DELAY == pytest.approx(0.5)


@pytest.mark.parametrize("candidate, variant, stock_like", [
  (CAR.KIA_STINGER_2022, "D", True), (CAR.KIA_STINGER_2022, "C", False),
  (CAR.KIA_STINGER_2022, "A", False), (CAR.HYUNDAI_SONATA, "D", False),
])
def test_controller_reads_mode_at_startup(monkeypatch, candidate, variant, stock_like):
  monkeypatch.setattr(carcontroller, "testing_ground", fake_testing_ground(STINGER_SLOT, variant))
  CP = startup_params(candidate)
  controller = CarController(DBC[CP.carFingerprint], CP)
  assert controller._stinger_stock_like == stock_like
  assert not controller._long_tuner.enabled or not stock_like  # sunnypilot tuner stays off for D

  # changing the Testing Ground mid-drive doesn't change the running controller
  monkeypatch.setattr(carcontroller, "testing_ground", fake_testing_ground(STINGER_SLOT, "A"))
  assert controller._stinger_stock_like == stock_like


@pytest.mark.parametrize("v_ego, jerk_upper", [(0.0, 1.0), (0.5, 0.85), (1.0, 0.7), (2.0, 0.6), (3.0, 0.5), (25.0, 0.5)])
def test_jerk_upper_vs_speed(v_ego, jerk_upper):
  assert tune(0.5, v_ego=v_ego).jerk_upper == pytest.approx(jerk_upper)


@pytest.mark.parametrize("accel, visible, distance, jerk_lower", [
  (-0.1, True, 30.0, 2.5),    # close lead
  (-0.1, True, 59.0, 2.5),
  (0.2, True, 100.0, 0.5),    # far lead
  (0.2, True, 90.0, 0.5),
  (-0.1, False, 0.0, 0.5),    # no lead
  (-0.3, False, 0.0, 0.5),
  (-0.5, False, 0.0, 2.5),    # openpilot braking without a lead
  (-1.0, False, 0.0, 2.5),
  (-1.0, True, 200.0, 2.5),
  (-0.1, True, 75.0, 1.5),    # lead distance interpolation
  (-0.4, False, 0.0, 1.5),    # accel interpolation
  (-0.4, True, 75.0, 1.5),
  (-0.45, True, 75.0, 2.0),   # max of the two criteria
])
def test_jerk_lower(accel, visible, distance, jerk_lower):
  assert tune(accel, lead_visible=visible, lead_distance=distance).jerk_lower == pytest.approx(jerk_lower)


def test_jerk_lower_is_continuous():
  def lower(accel, distance):
    return tune(accel, lead_visible=True, lead_distance=distance).jerk_lower
  assert abs(lower(-0.1, 59.99) - lower(-0.1, 60.01)) < 0.01
  assert abs(lower(-0.1, 89.99) - lower(-0.1, 90.01)) < 0.01
  assert abs(tune(-0.499).jerk_lower - tune(-0.501).jerk_lower) < 0.02
  assert abs(tune(-0.299).jerk_lower - tune(-0.301).jerk_lower) < 0.02


@pytest.mark.parametrize("accel", [-2.0, -2.5, -3.5])
def test_urgent_braking_is_not_delayed(accel):
  state = tune(accel, lead_visible=True, lead_distance=150.0)
  assert state.actual_accel == accel
  assert state.accel_last == accel
  assert state.jerk_lower == 5.0


def test_just_above_urgent_is_rate_limited():
  state = tune(-1.99, lead_visible=True, lead_distance=20.0)
  assert state.jerk_lower == pytest.approx(2.5)
  assert state.actual_accel == pytest.approx(-2.5 * DT_CTRL)


def test_ramp_rates():
  up = tune(1.0, v_ego=20.0, steps=10)
  assert up.actual_accel == pytest.approx(0.5 * DT_CTRL * 10)
  assert up.desired_accel == 1.0
  down = tune(-1.0, lead_visible=True, lead_distance=30.0, steps=10)
  assert down.actual_accel == pytest.approx(-2.5 * DT_CTRL * 10)
  # converges and stays at the command
  assert tune(0.3, steps=100).actual_accel == pytest.approx(0.3)
  # ramp never overshoots
  assert tune(-0.05, state=tune(-0.04, steps=1), steps=5).actual_accel >= -0.05


@pytest.mark.parametrize("kwargs, expected_upper", [
  (dict(active=False), 3.0),  # pid state: openpilot's 3.0
  (dict(override=True), 3.0),  # pid state: openpilot's 3.0
  (dict(long_state=LongCtrlState.stopping), 1.0),
  (dict(long_state=LongCtrlState.off), 1.0),
])
def test_pass_through_and_accel_last_reset(kwargs, expected_upper):
  state = tune(0.4, steps=50)  # ramping up, actual well below the command
  assert state.actual_accel < 0.4
  state = tune(-0.8, state=state, **kwargs)
  assert state.actual_accel == -0.8 and state.desired_accel == -0.8 and state.accel_last == -0.8
  assert state.jerk_upper == expected_upper
  assert state.jerk_lower == 5.0
  assert (state.comfort_band_upper, state.comfort_band_lower) == (0.0, 0.0)

  # re-enabling continues from the pass-through value instead of jumping
  resumed = tune(0.4, state=state, v_ego=20.0)
  assert resumed.actual_accel == pytest.approx(-0.8 + 0.5 * DT_CTRL)


def test_starting_state_is_tuned():
  state = tune(0.8, v_ego=0.0, long_state=LongCtrlState.starting, steps=10)
  assert state.jerk_upper == pytest.approx(1.0)
  assert state.actual_accel == pytest.approx(1.0 * DT_CTRL * 10)


@pytest.mark.parametrize("actual, upper, lower", [
  (0.0, 0.0, 0.0225), (0.1, 0.005, 0.009), (0.3, 0.040, 0.0), (0.5, 0.068, 0.0), (0.7, 0.073, 0.0),
  (0.9, 0.055, 0.0), (1.1, 0.070, 0.0), (0.2, 0.0225, 0.0),
  (-0.1, 0.0, 0.036), (-0.3, 0.0, 0.063), (-0.5, 0.0, 0.074), (-0.7, 0.0, 0.082), (-0.9, 0.0, 0.099), (-1.1, 0.0, 0.080),
  (-1.9, 0.0, 0.080),
])
def test_comfort_bands(actual, upper, lower):
  state = KiaStingerLongitudinalTuningState(accel_last=actual)
  state = tune(actual, state=state, lead_visible=True, lead_distance=10.0, v_ego=20.0)
  assert state.actual_accel == pytest.approx(actual)
  assert state.comfort_band_upper == pytest.approx(upper, abs=1e-6)
  assert state.comfort_band_lower == pytest.approx(lower, abs=1e-6)


def make_parser(candidate):
  return CANParser(DBC[candidate][Bus.pt], [("SCC11", 0), ("SCC12", 0), ("SCC14", 0)], 0)


def acc_values(candidate, tuning, accel=-0.6):
  CP = CarParams.new_message()
  CP.carFingerprint = candidate
  packer = CANPacker(DBC[candidate][Bus.pt])
  parser = make_parser(candidate)
  msgs = hyundaican.create_acc_commands(packer, enabled=True, accel=accel, upper_jerk=3.0, idx=3,
                                        hud_control=SimpleNamespace(leadDistanceBars=3, leadVisible=False), set_speed=42,
                                        stopping=False, long_override=False, use_fca=False, CP=CP, tuning=tuning)
  parser.update([(1, msgs)])
  return parser.vl["SCC12"], parser.vl["SCC14"]


def test_acc_commands_carry_the_tuning():
  state = KiaStingerLongitudinalTuningState(desired_accel=-1.2, jerk_upper=0.5, jerk_lower=2.5,
                                            comfort_band_upper=0.04, comfort_band_lower=0.07)
  scc12, scc14 = acc_values(CAR.KIA_STINGER_2022, state)
  assert scc12["aReqRaw"] == pytest.approx(-1.2, abs=0.01)
  assert scc12["aReqValue"] == pytest.approx(-0.6, abs=0.01)
  assert scc14["JerkUpperLimit"] == pytest.approx(0.5, abs=0.1)
  assert scc14["JerkLowerLimit"] == pytest.approx(2.5, abs=0.1)
  assert scc14["ComfortBandUpper"] == pytest.approx(0.04, abs=0.01)
  assert scc14["ComfortBandLower"] == pytest.approx(0.07, abs=0.01)


def test_pass_through_state_matches_untuned_messages():
  state = tune(-0.6, long_state=LongCtrlState.pid, active=False)
  assert acc_values(CAR.KIA_STINGER_2022, state, accel=-0.6) == acc_values(CAR.KIA_STINGER_2022, None, accel=-0.6)


LKAS11 = dict.fromkeys([
  "CF_Lkas_LdwsActivemode", "CF_Lkas_LdwsSysState", "CF_Lkas_SysWarning", "CF_Lkas_LdwsLHWarning", "CF_Lkas_LdwsRHWarning",
  "CF_Lkas_HbaLamp", "CF_Lkas_FcwBasReq", "CF_Lkas_HbaSysState", "CF_Lkas_FcwOpt", "CF_Lkas_HbaOpt", "CF_Lkas_FcwSysState",
  "CF_Lkas_FcwCollisionWarning", "CF_Lkas_FusionState", "CF_Lkas_FcwOpt_USM", "CF_Lkas_LdwsOpt_USM"], 0)


def stub_inputs():
  CS = SimpleNamespace(
    out=SimpleNamespace(vEgo=20.0, vEgoRaw=20.0, aEgo=0.0, steeringAngleDeg=0.0, steeringTorque=0.0, standstill=False,
                        gasPressed=False, brakePressed=False, gearShifter=0),
    is_metric=False, lkas11=LKAS11, lkas12={}, clu11={}, openpilot_lead_visible=True, openpilot_lead_distance=40.0,
    openpilot_lead_rel_speed=-1.0, main_cruise_on=True, main_cruise_tracking=False, msg_364={},
  )
  actuators = CarControl.Actuators.new_message(accel=0.8, longControlState=LongCtrlState.pid).as_reader()
  CC = SimpleNamespace(
    enabled=True, latActive=False, longActive=True,
    actuators=actuators,
    cruiseControl=SimpleNamespace(override=False, cancel=False, resume=False),
    hudControl=SimpleNamespace(setSpeed=30.0, leadDistanceBars=3, leadVisible=True, leftLaneVisible=False,
                               rightLaneVisible=False, visualAlert=0, leftLaneDepart=False, rightLaneDepart=False),
  )
  return CS, CC


def controller_scc(monkeypatch, candidate, variant, frames=1):
  monkeypatch.setattr(carcontroller, "testing_ground", fake_testing_ground(STINGER_SLOT, variant))
  monkeypatch.setattr(interface, "testing_ground", fake_testing_ground(STINGER_SLOT, variant))
  CP = startup_params(candidate)
  controller = CarController(DBC[CP.carFingerprint], CP)
  controller.long_active_ecu = True
  controller._ecu_disable_checked = True
  controller.ecu_disable_failed = False
  parser = make_parser(candidate)
  CS, CC = stub_inputs()
  for _ in range(frames):
    _, sends = controller.update(CC, CS, 0, SimpleNamespace())
    parser.update([(1, sends)])
  return parser.vl["SCC12"], parser.vl["SCC14"]


def test_controller_sends_stock_like_values_on_stinger(monkeypatch):
  scc12, scc14 = controller_scc(monkeypatch, CAR.KIA_STINGER_2022, "D", frames=10)
  assert scc12["aReqRaw"] == pytest.approx(0.8, abs=0.01)
  assert 0.0 < scc12["aReqValue"] < 0.8  # ramping toward aReqRaw
  assert scc14["JerkUpperLimit"] == pytest.approx(0.5, abs=0.1)
  assert scc14["JerkLowerLimit"] == pytest.approx(2.5, abs=0.1)  # lead at 40 m


@pytest.mark.parametrize("candidate, variant", [(CAR.KIA_STINGER_2022, "A"), (CAR.HYUNDAI_SONATA, "D")])
def test_other_setups_send_the_previous_values(monkeypatch, candidate, variant):
  scc12, scc14 = controller_scc(monkeypatch, candidate, variant, frames=10)
  assert scc12["aReqRaw"] == pytest.approx(0.8, abs=0.01)
  assert scc12["aReqValue"] == pytest.approx(0.8, abs=0.01)
  assert scc14["JerkUpperLimit"] == pytest.approx(3.0, abs=0.1)
  assert scc14["JerkLowerLimit"] == pytest.approx(5.0, abs=0.1)
  assert scc14["ComfortBandUpper"] == 0.0 and scc14["ComfortBandLower"] == 0.0
