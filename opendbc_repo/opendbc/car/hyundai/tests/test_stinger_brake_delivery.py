import numpy as np
import pytest

from opendbc.car import gen_empty_fingerprint, structs
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.stinger_brake_delivery import (MAX_EXTRA, SHORTFALL, REQUEST_KNOTS, StingerBrakeDelivery, compensate,
                                                       expected_shortfall, qualified)
from opendbc.car.hyundai.values import CAR, CarControllerParams

LongControlState = structs.CarControl.Actuators.LongControlState
REQUESTS = np.arange(-3.5, 2.01, 0.01)
SPEEDS = np.arange(0.0, 20.01, 0.1)


def params(car=CAR.KIA_STINGER_2022, alpha=True):
  fp = gen_empty_fingerprint()
  fp[0][0x391] = 8
  fp[0][0x38d] = 8
  return CarInterface.get_params(car, fp, [], alpha, False, False)


def test_shortfall_table_is_never_negative_and_matches_knots():
  assert min(SHORTFALL) >= 0.0 and len(SHORTFALL) == len(REQUEST_KNOTS)


def test_never_brakes_less_than_asked_and_bounded():
  for speed in SPEEDS:
    out = np.array([compensate(float(r), float(speed)) for r in REQUESTS])
    extra = REQUESTS - out
    assert extra.min() >= -1e-9 and extra.max() <= MAX_EXTRA + 1e-9
    assert out.min() >= CarControllerParams.ACCEL_MIN


def test_untouched_regimes():
  for speed in SPEEDS:
    assert compensate(0.4, float(speed)) == 0.4 and compensate(-0.2, float(speed)) == -0.2
    if speed >= 4.5 or speed <= 0.5:
      assert all(compensate(float(r), float(speed)) == float(r) for r in REQUESTS)
  for speed in np.arange(1.0, 3.5, 0.25):
    assert compensate(-1.6, float(speed)) == -1.6


def test_delivered_decel_lands_on_the_plan_in_the_corrected_band():
  for speed in (1.0, 2.0, 3.5):
    for request in (-0.3, -0.5, -0.7, -0.9, -1.2):
      command = compensate(request, speed)
      delivered = command + expected_shortfall(speed, command)
      assert delivered == pytest.approx(request, abs=0.01)


def test_continuous_and_amplifies_request_steps_by_at_most_one_and_a_half():
  for speed in (0.8, 1.0, 2.0, 3.5, 4.0):
    out = np.array([compensate(float(r), speed) for r in REQUESTS])
    assert (np.diff(out) / np.diff(REQUESTS)).max() <= 1.5 + 1e-6
  for request in (-0.45, -0.7, -1.0, -1.2):
    out = np.array([compensate(request, float(s)) for s in SPEEDS])
    assert np.abs(np.diff(out)).max() <= 0.05


@pytest.mark.parametrize('active,override,state', [
  (False, False, LongControlState.pid), (True, True, LongControlState.pid),
  (True, False, LongControlState.off), (True, False, LongControlState.starting)])
def test_identity_when_not_actively_braking_under_control(active, override, state):
  assert StingerBrakeDelivery().update(-0.8, 2.0, state, active, override) == -0.8


def test_active_pid_and_stopping_are_compensated():
  for state in (LongControlState.pid, LongControlState.stopping):
    assert StingerBrakeDelivery().update(-0.8, 2.0, state, True, False) < -0.8


def test_qualification_is_stinger_with_openpilot_long_only():
  assert qualified(params())
  assert not qualified(params(alpha=False))
  assert not qualified(params(CAR.GENESIS_G70_2020))
  assert not qualified(params(CAR.HYUNDAI_ELANTRA_2021))


def test_controller_applies_it_to_the_sent_request_only_for_the_stinger():
  def sent(car, accel=-0.8):
    cp = params(car)
    ci = CarInterface(cp)
    ci.update([])
    cc = structs.CarControl(enabled=True, latActive=True, longActive=True)
    cc.actuators.accel = accel
    cc.actuators.longControlState = LongControlState.pid
    ci.CS.out.vEgo = 2.0
    actuators, _ = ci.CC.update(cc.as_reader(), ci.CS, 10_000_000_000)
    return actuators.accel
  assert sent(CAR.KIA_STINGER_2022) < -0.8
  assert sent(CAR.GENESIS_G70_2020) == pytest.approx(-0.8)
