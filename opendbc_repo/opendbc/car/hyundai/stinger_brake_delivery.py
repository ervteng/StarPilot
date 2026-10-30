"""Stinger brake-delivery compensation for openpilot longitudinal.

At crawl and city speeds the Stinger brakes less than SCC12 asks for. Measured on openpilot-long drives
00000003--0f3f0eb0f2 and 00000004--a5fd1905e8 (0.5 s windows with a steady request, 0.2 s actuation lag, engaged, no pedals):
  - request -0.45..-1.0 m/s^2 at 1-3.5 m/s: aEgo is 0.15-0.40 m/s^2 weaker than asked (median about 0.25), both drives
  - request <= -1.5, or speed >= 4.5 m/s: the car delivers the request, or slightly more
The two drives disagree cell by cell, so the table stays at or under the smaller of the two single-drive fits
(analysis/stinger_brake_delivery_fit.py): 0.22 at -1.0, 0.17 at -0.7. It is zero from -0.2 up, and from -1.5 down where the car
already delivers. The entry ramp (-0.2 -> -0.7) is held to slope 1/3 so the exact inverse amplifies a request step by at most 1.5x:
logged braking requests at 1-4.5 m/s step by <= 0.1 m/s^2 per 50 ms (p99), so the added jerk stays <= 0.05 m/s^2 per 50 ms.
The request is lowered by the shortfall the car is expected to add back, so the delivered decel lands on the planned one.
Acceleration requests, stop holds (v <= 0.5 m/s), pedal override and non-pid/stopping control states pass through unchanged.
"""
import numpy as np

from opendbc.car import structs
from opendbc.car.hyundai.values import CAR, CarControllerParams

LongControlState = structs.CarControl.Actuators.LongControlState

SPEED_KNOTS = (0.5, 1.0, 3.5, 4.5, 6.0)
SPEED_SCALE = (0.0, 1.0, 1.0, 0.0, 0.0)
REQUEST_KNOTS = (-3.5, -1.5, -1.0, -0.7, -0.2, 0.0)
SHORTFALL = (0.0, 0.0, 0.22, 0.165, 0.0, 0.0)
MAX_EXTRA = 0.35
ACTIVE_STATES = (LongControlState.pid, LongControlState.stopping)


def expected_shortfall(speed: float, request: float) -> float:
  """Decel the car is expected to leave undelivered when asked for `request` (<= 0) at `speed`."""
  return float(np.interp(speed, SPEED_KNOTS, SPEED_SCALE) * np.interp(request, REQUEST_KNOTS, SHORTFALL))


def compensate(request: float, speed: float) -> float:
  """Command whose expected delivery (command + shortfall) equals `request`: exact piecewise-linear inverse."""
  scale = float(np.interp(speed, SPEED_KNOTS, SPEED_SCALE))
  if scale <= 0.0 or not REQUEST_KNOTS[0] < request < REQUEST_KNOTS[-1]:
    return request
  delivered = [knot + scale * short for knot, short in zip(REQUEST_KNOTS, SHORTFALL, strict=True)]
  command = float(np.interp(request, delivered, REQUEST_KNOTS))
  command = min(request, max(command, request - MAX_EXTRA))
  return float(max(command, CarControllerParams.ACCEL_MIN))


def qualified(cp) -> bool:
  return cp.carFingerprint == CAR.KIA_STINGER_2022 and bool(cp.openpilotLongitudinalControl)


class StingerBrakeDelivery:
  def update(self, accel: float, speed: float, state, active: bool, override: bool) -> float:
    if not active or override or state not in ACTIVE_STATES:
      return accel
    return compensate(accel, speed)
