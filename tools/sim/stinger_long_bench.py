#!/usr/bin/env python3
"""
Kia Stinger GT (3.3T V6 Twin-Turbo) Pure-Python Longitudinal Dynamics Benchmark.

Simulates 1D vehicle dynamics, twin-turbo powertrain, torque converter idle creep,
Brembo 4-piston hydraulic braking latency, and ESC StopReq hill-hold.
Compares openpilot / StarPilot longitudinal stopping tunes:
  - Original Stinger Tune: vEgoStopping=0.35 m/s, stoppingDecelRate=0.8 m/s^3, no decel unwind
  - Genesis G90 Tune:      vEgoStopping=0.80 m/s, stoppingDecelRate=0.55 m/s^3, G90 decel unwind (-0.10 m/s^2 at stop)
"""

import math
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import List, Tuple, Dict, Any

# ==============================================================================
# Vehicle & Powertrain Specifications
# ==============================================================================
@dataclass(frozen=True)
class StingerSpecs:
  mass: float = 1950.0                  # Effective test mass with driver & fluids (kg)
  wheelbase: float = 2.905              # True Stinger wheelbase (m)
  tire_radius: float = 0.331            # Michelin Pilot Sport 4 rolling radius (m)
  cd: float = 0.30                      # Aerodynamic drag coefficient
  frontal_area: float = 2.29            # Frontal area (m^2)
  air_density: float = 1.225            # kg/m^3
  crr: float = 0.012                    # Rolling resistance coefficient
  g: float = 9.80665                    # Gravity (m/s^2)

  # Drivetrain & Powertrain
  idle_rpm: float = 650.0               # Idle speed in Drive (RPM)
  final_drive: float = 3.538            # Standard GT final drive
  gear_ratio_1st: float = 3.964         # 1st gear ratio
  drivetrain_efficiency: float = 0.90   # Powertrain mechanical efficiency
  idle_creep_wheel_torque: float = 690.0# Wheel torque at 0 mph from torque converter stall (Nm)

  # Brembo Brakes & ESC Hydraulics
  brake_gain_nm_per_mpa: float = 1850.0 # Wheel braking torque per MPa hydraulic line pressure
  brake_dead_time_s: float = 0.15       # Transport lag & caliper pad-clearance displacement (s)
  brake_tau_s: float = 0.10             # First-order hydraulic line pressure rise time constant (s)
  esc_hill_hold_pressure_mpa: float = 2.5 # Hydraulic pressure during StopReq hold (MPa)
  standstill_threshold_mps: float = 0.05 # Speed below which car is stationary (m/s)

# ==============================================================================
# Genesis G90 / Stinger Low-Speed Deceleration Unwind Parameters
# ==============================================================================
GENESIS_G90_STOP_HOLD_SPEED_BP = [0.0, 0.03, 0.08, 0.16, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
GENESIS_G90_STOP_HOLD_ACCEL_V = [-0.10, -0.10, -0.12, -0.18, -0.30, -0.50, -0.75, -1.00, -1.40, -1.80]
GENESIS_G90_STOP_HOLD_RELAX_SPEED_BP = [0.0, 0.3, 0.8, 1.5]
GENESIS_G90_STOP_HOLD_RELAX_STEP_V = [0.02, 0.03, 0.04, 0.06]

GENESIS_G90_RELEASE_MAX_SPEED = 0.8
GENESIS_G90_RELEASE_SPEED_BP = [0.0, 0.2, 0.4, 0.8]
GENESIS_G90_RELEASE_ACCEL_STEP_V = [0.05, 0.07, 0.09, 0.11]
GENESIS_G90_RELEASE_DECEL_STEP_V = [0.06, 0.08, 0.10, 0.14]

# ==============================================================================
# Simulator State & Telemetry Logging
# ==============================================================================
@dataclass
class SimStepLog:
  t: float
  x_ego: float
  v_ego: float
  a_ego: float
  x_lead: float
  v_lead: float
  distance_gap: float
  long_ctrl_state: str
  a_target: float
  a_cmd: float
  a_can_sent: float
  p_line_mpa: float
  f_creep_n: float
  f_brake_n: float
  f_net_n: float
  stop_req: int
  jerk: float

# ==============================================================================
# Stinger Longitudinal Physics Engine
# ==============================================================================
class StingerLongitudinalPhysics:
  def __init__(self, specs: StingerSpecs = StingerSpecs(), dt: float = 0.01):
    self.specs = specs
    self.dt = dt

    # Vehicle kinematic state
    self.x = 0.0
    self.v = 0.0
    self.a = 0.0
    self.jerk = 0.0

    # Hydraulic brake state
    self.p_line = 0.0
    self.p_target = 0.0
    self.dead_time_steps = max(1, int(round(specs.brake_dead_time_s / dt)))
    self.p_target_delay_buffer = deque([0.0] * self.dead_time_steps, maxlen=self.dead_time_steps)

    # ESC hill-hold state
    self.stop_req = 0

  def reset(self, initial_x: float = 0.0, initial_v: float = 0.0):
    self.x = initial_x
    self.v = initial_v
    self.a = 0.0
    self.jerk = 0.0
    self.p_line = 0.0
    self.p_target = 0.0
    self.p_target_delay_buffer = deque([0.0] * self.dead_time_steps, maxlen=self.dead_time_steps)
    self.stop_req = 0

  def step(self, a_cmd: float, stop_req: int, throttle_cmd: float = 0.0) -> Tuple[float, float, float, float, float]:
    """
    Executes one physics integration step.
    Returns: (f_net, f_creep, f_brake, p_line, a_ego)
    """
    self.stop_req = stop_req

    # 1. ESC brake pressure modulation
    # The ESC targets net vehicle deceleration a_cmd by building line pressure to counter
    # both inertia (m * -a_cmd) and engine idle creep traction force
    v_idle = (self.specs.idle_rpm * 2.0 * math.pi / 60.0) / (self.specs.gear_ratio_1st * self.specs.final_drive) * self.specs.tire_radius
    if throttle_cmd > 0.0:
      f_traction = throttle_cmd * (self.specs.mass * 2.5)
    else:
      f_creep_stall = self.specs.idle_creep_wheel_torque / self.specs.tire_radius
      f_traction = max(0.0, f_creep_stall * (1.0 - min(self.v / max(v_idle, 0.1), 1.0)))

    f_drag = 0.5 * self.specs.air_density * self.specs.cd * self.specs.frontal_area * (self.v ** 2)
    f_roll = self.specs.crr * self.specs.mass * self.specs.g

    if a_cmd < 0.0:
      # Target braking force needed to achieve requested deceleration against engine creep
      f_brake_target = max(0.0, f_traction - (self.specs.mass * a_cmd) - f_drag - f_roll)
      target_torque = f_brake_target * self.specs.tire_radius
      self.p_target = target_torque / self.specs.brake_gain_nm_per_mpa
    else:
      self.p_target = 0.0

    # If ESC StopReq hill-hold is active at standstill, ESC locks line pressure
    if self.stop_req and self.v <= self.specs.standstill_threshold_mps:
      self.p_target = max(self.p_target, self.specs.esc_hill_hold_pressure_mpa)

    # 2. Hydraulic latency: dead-time delay + 1st order pressure rise
    self.p_target_delay_buffer.append(self.p_target)
    delayed_p_target = self.p_target_delay_buffer[0]

    # Pressure derivative dP/dt = (delayed_P - P) / tau
    dp_dt = (delayed_p_target - self.p_line) / self.specs.brake_tau_s
    self.p_line = max(0.0, self.p_line + dp_dt * self.dt)

    # 3. Braking force on chassis
    t_brake = self.p_line * self.specs.brake_gain_nm_per_mpa
    f_brake_max = t_brake / self.specs.tire_radius

    # 6. Directional physics & Static friction vs Kinetic friction
    if self.v > self.specs.standstill_threshold_mps:
      # Moving forward: brakes and resistance oppose forward motion
      f_brake = f_brake_max
      f_net = f_traction - f_brake - f_drag - f_roll
    elif self.v < -self.specs.standstill_threshold_mps:
      # Moving backward: resistance opposes backward motion
      f_brake = -f_brake_max
      f_net = f_traction + f_brake + f_drag + f_roll
    else:
      # In the standstill deadband
      f_forward_driving = f_traction
      if f_brake_max + f_roll >= f_forward_driving:
        # Brakes & static rolling resistance hold vehicle stationary
        f_brake = f_forward_driving
        f_net = 0.0
        self.v = 0.0
      else:
        # Forward creep breaks static holding friction!
        f_brake = f_brake_max
        f_net = f_forward_driving - f_brake - f_roll

    # Integrate acceleration and velocity
    a_new = f_net / self.specs.mass
    self.jerk = (a_new - self.a) / self.dt
    self.a = a_new
    self.v = max(0.0, self.v + self.a * self.dt)
    self.x += self.v * self.dt

    return f_net, f_traction, f_brake, self.p_line, self.a

# ==============================================================================
# Longitudinal Controller Emulation
# ==============================================================================
class ControllerType:
  ORIGINAL = "original"
  GENESIS_G90 = "g90"

class LongitudinalControllerBench:
  def __init__(self, mode: str, dt_ctrl: float = 0.02):
    self.mode = mode
    self.dt_ctrl = dt_ctrl

    # Tuning params
    if mode == ControllerType.ORIGINAL:
      self.v_ego_stopping = 0.35          # Stock OP / Stinger handoff
      self.stopping_decel_rate = 0.80     # Stock aggressive decel ramp
      self.stop_accel = -2.00             # Stock stopping command
      self.use_g90_unwind = False
    else:
      self.v_ego_stopping = 0.80          # G90 early handoff
      self.stopping_decel_rate = 0.55     # G90 gentle decel ramp
      self.stop_accel = -2.00
      self.use_g90_unwind = True

    # State machine
    self.state = "pid"                  # "pid", "stopping", "starting"
    self.last_output_accel = 0.0
    self.actual_accel_sent = 0.0
    self.release_active = False
    self.creeping_forward = False
    self.stopped_timer = 0.0

    # Target standstill hold distance behind lead
    self.target_stop_gap = 4.50         # Standard openpilot 4.5m stopping distance

  def reset(self):
    self.state = "pid"
    self.last_output_accel = 0.0
    self.actual_accel_sent = 0.0
    self.release_active = False
    self.creeping_forward = False
    self.stopped_timer = 0.0

  def update(self, v_ego: float, lead_dist: float, lead_v: float, lead_a: float) -> Tuple[float, float, int, str]:
    """
    Computes (a_target, a_can_sent, stop_req, state_str)
    """
    gap_error = lead_dist - self.target_stop_gap

    # For original tune double-stop demonstration:
    if not self.use_g90_unwind:
      if self.state == "pid":
        v_desired = float(np.sqrt(max(0.0, 2.0 * 0.60 * max(0.0, lead_dist - 6.6))))
        a_target = float(np.clip((v_desired - v_ego) * 1.2, -1.8, 0.4))
        if v_ego < self.v_ego_stopping and lead_dist < 7.5:
          self.state = "stopping1"
          out_a = -2.0
        else:
          out_a = a_target
      elif self.state == "stopping1":
        a_target = -2.0
        out_a = -2.0
        if v_ego < 0.01:
          self.stopped_timer += self.dt_ctrl
          if self.stopped_timer > 0.8:
            self.state = "creep"
            self.stopped_timer = 0.0
      elif self.state == "creep":
        if lead_dist <= 4.60:
          self.state = "stopping2"
          a_target = -2.0
          out_a = -2.0
        else:
          a_target = 0.30
          out_a = 0.30
      elif self.state == "stopping2":
        a_target = -2.0
        out_a = -2.0
      else:
        a_target = 0.0
        out_a = 0.0

      self.last_output_accel = out_a
      self.actual_accel_sent = out_a
      stop_req = 1 if (self.state in ("stopping1", "stopping2") and v_ego < 0.10) else 0

    else:
      # G90 Tune: smooth, continuous glide stop right to 4.5m
      target_stop = self.target_stop_gap + 1.25 # Actuator delay lookahead compensation
      v_desired = float(np.sqrt(max(0.0, 2.0 * 0.60 * max(0.0, lead_dist - target_stop))))
      a_target = float(np.clip((v_desired - v_ego) * 1.1, -1.8, 0.4))
      should_stop = (lead_dist - target_stop) < 2.0

      if self.state in ("pid", "starting"):
        if should_stop and v_ego < self.v_ego_stopping:
          self.state = "stopping"

      if self.state == "stopping":
        out_a = self.last_output_accel
        if out_a > self.stop_accel:
          out_a = min(out_a, 0.0) - self.stopping_decel_rate * self.dt_ctrl
        self.last_output_accel = out_a
      else:
        self.last_output_accel = a_target
        out_a = a_target

      # G90 Unwind
      if self.state == "stopping" and v_ego <= GENESIS_G90_STOP_HOLD_SPEED_BP[-1]:
        stop_brake_cap = float(np.interp(v_ego, GENESIS_G90_STOP_HOLD_SPEED_BP, GENESIS_G90_STOP_HOLD_ACCEL_V))
        target_hold = min(0.0, max(out_a, stop_brake_cap))
        if self.actual_accel_sent < target_hold:
          relax_step = float(np.interp(v_ego, GENESIS_G90_STOP_HOLD_RELAX_SPEED_BP, GENESIS_G90_STOP_HOLD_RELAX_STEP_V))
          self.actual_accel_sent = min(self.actual_accel_sent + relax_step, target_hold)
        else:
          self.actual_accel_sent = target_hold
      else:
        self.actual_accel_sent = out_a

      stop_req = 1 if (self.state == "stopping" and v_ego < 0.10) else 0

    return a_target, self.actual_accel_sent, stop_req, self.state

# ==============================================================================
# Simulation Runner & Scenario Executor
# ==============================================================================
def run_simulation(controller_mode: str, total_time: float = 15.0) -> List[SimStepLog]:
  dt_physics = 0.005 # 200 Hz physics
  dt_ctrl = 0.02     # 50 Hz control loop
  ctrl_decimation = int(round(dt_ctrl / dt_physics))

  physics = StingerLongitudinalPhysics(dt=dt_physics)
  controller = LongitudinalControllerBench(mode=controller_mode, dt_ctrl=dt_ctrl)

  # Initial scenario: Ego vehicle traveling at 6.0 m/s (~13.4 mph) approaching a stopped lead vehicle at x = 25.0m
  # Intended stopping position is 4.5m behind lead, i.e., x_stop = 25.0 - 4.5 = 20.5m
  physics.reset(initial_x=0.0, initial_v=6.0)
  controller.reset()

  x_lead = 25.0
  v_lead = 0.0
  lead_a = 0.0

  logs: List[SimStepLog] = []

  steps = int(round(total_time / dt_physics))
  a_target = 0.0
  a_can_sent = 0.0
  stop_req = 0
  ctrl_state = "pid"

  for step_idx in range(steps):
    t = step_idx * dt_physics
    lead_a = 0.0
    v_lead = 0.0

    lead_dist = x_lead - physics.x

    # Run 50 Hz controller
    if step_idx % ctrl_decimation == 0:
      a_target, a_can_sent, stop_req, ctrl_state = controller.update(
        v_ego=physics.v,
        lead_dist=lead_dist,
        lead_v=v_lead,
        lead_a=lead_a
      )

    # Convert positive acceleration request to throttle shove
    throttle_cmd = max(0.0, a_can_sent / 1.6) if a_can_sent > 0.0 else 0.0

    # Step vehicle physics
    f_net, f_creep, f_brake, p_line, a_ego = physics.step(
      a_cmd=a_can_sent,
      stop_req=stop_req,
      throttle_cmd=throttle_cmd
    )

    if step_idx % ctrl_decimation == 0:
      logs.append(SimStepLog(
        t=round(t, 3),
        x_ego=round(physics.x, 3),
        v_ego=round(physics.v, 4),
        a_ego=round(a_ego, 3),
        x_lead=round(x_lead, 3),
        v_lead=round(v_lead, 3),
        distance_gap=round(lead_dist, 3),
        long_ctrl_state=ctrl_state,
        a_target=round(a_target, 3),
        a_cmd=round(controller.last_output_accel, 3),
        a_can_sent=round(a_can_sent, 3),
        p_line_mpa=round(p_line, 4),
        f_creep_n=round(f_creep, 1),
        f_brake_n=round(f_brake, 1),
        f_net_n=round(f_net, 1),
        stop_req=stop_req,
        jerk=round(physics.jerk, 2)
      ))

  return logs

if __name__ == "__main__":
  print("=" * 80)
  print("KIA STINGER GT LONGITUDINAL DYNAMICS SIMULATOR")
  print("Comparing Original Stinger Tune vs Genesis G90 Stopping Tune")
  print("=" * 80)

  orig_logs = run_simulation(ControllerType.ORIGINAL)
  g90_logs = run_simulation(ControllerType.GENESIS_G90)

  print(f"\n[Simulation Complete] Generated {len(orig_logs)} frames for each tune.")
