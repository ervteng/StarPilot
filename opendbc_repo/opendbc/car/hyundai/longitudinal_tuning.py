"""
Hyundai longitudinal tuning: jerk-limited accel, speed-based or predictive SCC jerk limits and comfort bands.

Based on sunnypilot's Hyundai longitudinal tuning (opendbc/sunnypilot/car/hyundai/longitudinal), MIT licensed:
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
"""
from dataclasses import dataclass

import numpy as np

from opendbc.car import DT_CTRL, rate_limit, structs
from opendbc.car.hyundai.values import CAR, CarControllerParams, STINGER_LONG_TUNING_TESTING_GROUND_ID

LongCtrlState = structs.CarControl.Actuators.LongControlState


class LongitudinalTuningMode:
  OFF = 0
  DYNAMIC = 1
  PREDICTIVE = 2
  STOCK_LIKE = 3  # state machine lives in carcontroller (update_kia_stinger_longitudinal_tuning)


UPDATE_FRAMES = 5  # update at 20 Hz; the integrator steps assume this
ACTUATOR_DELAY = 0.5  # s, planner actuator delay while tuning is on (the integrator adds lag)
MIN_JERK = 0.5
JERK_LIMIT = 4.0
LOOKAHEAD_JERK_BP = [2., 5., 20.]
LOOKAHEAD_JERK_UPPER_V = [0.3, 0.45, 0.6]
LOOKAHEAD_JERK_LOWER_V = [0.3, 0.45, 0.6]
# Below this request the tuning steps aside: stock lower jerk and no rate limit, so hard braking isn't delayed.
# Replaying an FCW and a highway jam brake, rate-limiting those requests cost 4-18 m of gap.
URGENT_ACCEL = -2.0  # m/s^2
URGENT_JERK_LOWER = 5.0  # m/s^3, what openpilot sends without the tuning


def stinger_long_tuning_mode(car_fingerprint, openpilot_longitudinal: bool, testing_ground) -> int:
  """Mode from the Galaxy Testing Ground slot (A off, B dynamic, C predictive, D stock-like); Stinger 2022 with openpilot long only."""
  if car_fingerprint != CAR.KIA_STINGER_2022 or not openpilot_longitudinal:
    return LongitudinalTuningMode.OFF
  if testing_ground.use(STINGER_LONG_TUNING_TESTING_GROUND_ID, "D"):
    return LongitudinalTuningMode.STOCK_LIKE
  if testing_ground.use(STINGER_LONG_TUNING_TESTING_GROUND_ID, "C"):
    return LongitudinalTuningMode.PREDICTIVE
  if testing_ground.use(STINGER_LONG_TUNING_TESTING_GROUND_ID, "B"):
    return LongitudinalTuningMode.DYNAMIC
  return LongitudinalTuningMode.OFF


def jerk_limited_integrator(desired_accel: float, last_accel: float, jerk_upper: float, jerk_lower: float) -> float:
  step = (jerk_upper if desired_accel >= last_accel else jerk_lower) * DT_CTRL * UPDATE_FRAMES
  return rate_limit(desired_accel, last_accel, -step, step)


@dataclass
class LongitudinalTuner:
  mode: int = LongitudinalTuningMode.OFF
  radar_available: bool = False

  accel_cmd: float = 0.0
  desired_accel: float = 0.0
  actual_accel: float = 0.0
  accel_last: float = 0.0
  jerk_upper: float = 0.0
  jerk_lower: float = 0.0
  comfort_band_upper: float = 0.0
  comfort_band_lower: float = 0.0
  stopping: bool = False
  stopping_count: int = 0
  long_control_state_last: int = LongCtrlState.off

  @property
  def enabled(self) -> bool:
    return self.mode in (LongitudinalTuningMode.DYNAMIC, LongitudinalTuningMode.PREDICTIVE)

  @property
  def predictive(self) -> bool:
    return self.mode == LongitudinalTuningMode.PREDICTIVE

  def _update_stopping(self, long_control_state) -> None:
    if long_control_state != LongCtrlState.stopping:
      self.stopping = False
      self.stopping_count = 0
      return
    if self.long_control_state_last == LongCtrlState.off:
      self.stopping = True
      return
    # hold off StopReq for ~1 s of stopping state so the jerk-limited accel can settle first
    if self.stopping_count > 1 / (DT_CTRL * UPDATE_FRAMES):
      self.stopping = True
    self.stopping_count += 1

  def _speed_jerk_limits(self, v_ego: float, long_control_state) -> tuple[float, float]:
    # Speed-based caps after ISO 15622:2018
    upper = float(np.interp(v_ego, [0.0, 5.0, 20.0], [2.0, 3.0, 2.0])) if long_control_state == LongCtrlState.pid else 0.5
    lower_v = [5.0, 3.5, 3.0] if self.predictive else [3.5, 3.5, 3.0]
    return upper, float(np.interp(v_ego, [0.0, 5.0, 20.0], lower_v))

  def _lookahead_jerk(self, accel_error: float, v_ego: float) -> tuple[float, float]:
    # Jerk needed to reach the target accel within a speed-dependent lookahead window
    t_upper = float(np.interp(v_ego, LOOKAHEAD_JERK_BP, LOOKAHEAD_JERK_UPPER_V))
    t_lower = float(np.interp(v_ego, LOOKAHEAD_JERK_BP, LOOKAHEAD_JERK_LOWER_V))
    return (float(np.clip(accel_error / t_upper, -JERK_LIMIT, JERK_LIMIT)),
            float(np.clip(accel_error / t_lower, -JERK_LIMIT, JERK_LIMIT)))

  def _update_jerk(self, long_active: bool, v_ego: float, long_control_state) -> None:
    upper_cap, lower_cap = self._speed_jerk_limits(v_ego, long_control_state)
    if self.predictive:
      j_upper, j_lower = self._lookahead_jerk(self.accel_cmd - self.accel_last, v_ego)
      self.jerk_upper = min(max(j_upper, MIN_JERK), upper_cap)
      self.jerk_lower = min(max(-j_lower, MIN_JERK), lower_cap) if self.radar_available else 5.0
    else:
      self.jerk_upper = upper_cap
      self.jerk_lower = lower_cap
    if self.accel_cmd <= URGENT_ACCEL:
      self.jerk_lower = URGENT_JERK_LOWER
    if not long_active:
      self.jerk_upper = self.jerk_lower = 0.0

  def _update_accel(self, long_active: bool) -> None:
    if not long_active:
      self.desired_accel = self.actual_accel = self.accel_last = 0.0
      return
    self.desired_accel = 0.0 if self.stopping else float(np.clip(self.accel_cmd, CarControllerParams.ACCEL_MIN,
                                                                 CarControllerParams.ACCEL_MAX))
    if self.radar_available and self.desired_accel > URGENT_ACCEL:
      self.actual_accel = jerk_limited_integrator(self.desired_accel, self.accel_last, self.jerk_upper, self.jerk_lower)
    else:
      self.actual_accel = self.desired_accel
    self.accel_last = self.actual_accel

  def _update_comfort_band(self, long_active: bool, a_ego: float) -> None:
    if not long_active:
      self.comfort_band_upper = self.comfort_band_lower = 0.0
    elif self.predictive:
      bands = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
      self.comfort_band_upper = float(np.interp(a_ego, [0.0, 0.3, 0.6, 0.9, 1.2, 2.0], bands))
      self.comfort_band_lower = float(np.interp(a_ego, [-3.5, -2.5, -1.5, -1.0, -0.5, -0.05], bands[::-1]))
    else:
      self.comfort_band_upper = float(np.interp(a_ego, [0.0, 2.0], [0.0, 0.04]))
      self.comfort_band_lower = float(np.interp(a_ego, [-3.5, -0.05], [0.04, 0.0]))

  def update(self, CC: structs.CarControl, v_ego: float, a_ego: float) -> None:
    long_control_state = CC.actuators.longControlState
    self.accel_cmd = CC.actuators.accel
    self._update_stopping(long_control_state)
    self._update_jerk(CC.longActive, v_ego, long_control_state)
    self._update_accel(CC.longActive)
    self._update_comfort_band(CC.longActive, a_ego)
    self.long_control_state_last = long_control_state
