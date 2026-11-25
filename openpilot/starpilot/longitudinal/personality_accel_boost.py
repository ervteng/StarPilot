"""Opt-in acceleration offset per driving personality, ported from commaai/openpilot ch3-model.

Default off. The offset is added to the model (e2e) target, so it applies in Experimental mode.
The numbers are comma's ch3-model values. They have not been measured on this car.
"""

import numpy as np

from openpilot.cereal import log
from openpilot.starpilot.saved_source import read_saved

KEY = 'PersonalityAccelBoost'
ACCEL_BOOST_RELAXED = 0.0       # m/s^2
ACCEL_BOOST_STANDARD = 0.05     # m/s^2
ACCEL_BOOST_AGGRESSIVE = 0.15   # m/s^2
SPEED_BP = (5.0, 10.0)          # m/s: no boost below 5 m/s, full boost above 10 m/s
ACCEL_BP = (-1.0, -0.5)         # m/s^2: the boost fades out as the target falls below -0.5 m/s^2


def personality_boost_max(personality) -> float:
  if personality == log.LongitudinalPersonality.aggressive:
    return ACCEL_BOOST_AGGRESSIVE
  if personality == log.LongitudinalPersonality.standard:
    return ACCEL_BOOST_STANDARD
  return ACCEL_BOOST_RELAXED


def personality_accel_boost(v_ego: float, accel_target: float, personality) -> float:
  speed_scale = float(np.interp(v_ego, SPEED_BP, [0.0, 1.0]))
  accel_scale = float(np.interp(accel_target, ACCEL_BP, [0.0, 1.0]))
  return personality_boost_max(personality) * speed_scale * accel_scale


class PersonalityAccelBoostPreferences:
  def __init__(self, params):
    self.params = params
    self.enabled = False
    self.next_ns = 0
    self.last_ns = 0

  def sample(self, now_ns):
    if now_ns < self.last_ns:
      self.next_ns = 0
    self.last_ns = now_ns
    if now_ns >= self.next_ns:
      self.next_ns = now_ns + 1_000_000_000
      try:
        raw, readable = read_saved(self.params, KEY, 16)
        self.enabled = readable and raw == b'1'
      except (OSError, RuntimeError, TypeError, ValueError):
        self.enabled = False
    return self.enabled
