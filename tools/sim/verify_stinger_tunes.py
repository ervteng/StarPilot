#!/usr/bin/env python3
"""
Verification Script for Kia Stinger GT Longitudinal Stopping Tunes.
Runs the Stinger Longitudinal Simulator on both:
  1. Original Tune (vEgoStopping=0.35 m/s, stoppingDecelRate=0.8 m/s^3, no decel unwind)
  2. New Tune (Genesis G90 tune: vEgoStopping=0.80 m/s, stoppingDecelRate=0.55 m/s^3, G90 decel unwind)

Analyzes why the original tune created double-stops and validates the new tune.
"""

import sys
import os
import numpy as np

# Ensure tools/sim is in path
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
if SIM_DIR not in sys.path:
  sys.path.insert(0, SIM_DIR)

from stinger_long_bench import (
  run_simulation,
  ControllerType,
  SimStepLog
)

def analyze_stopping_behavior(tune_name: str, logs: list[SimStepLog]):
  times = [l.t for l in logs]
  v_egos = [l.v_ego for l in logs]
  a_egos = [l.a_ego for l in logs]
  a_cans = [l.a_can_sent for l in logs]
  gaps = [l.distance_gap for l in logs]
  p_lines = [l.p_line_mpa for l in logs]
  states = [l.long_ctrl_state for l in logs]
  jerks = [l.jerk for l in logs]

  # Identify stop episodes (contiguous intervals where v_ego == 0 or < 0.02 m/s)
  stops = []
  in_stop = False
  stop_start_idx = 0

  for i, v in enumerate(v_egos):
    if v < 0.05:
      if not in_stop:
        in_stop = True
        stop_start_idx = i
    else:
      if in_stop:
        in_stop = False
        duration = times[i] - times[stop_start_idx]
        if duration >= 0.2: # Only count distinct stops lasting > 200ms
          stops.append({
            "start_time": times[stop_start_idx],
            "end_time": times[i],
            "duration": duration,
            "gap_at_stop": gaps[stop_start_idx],
            "a_can_at_stop": a_cans[stop_start_idx],
            "p_line_at_stop": p_lines[stop_start_idx],
            "max_entry_jerk": max(abs(j) for j in jerks[max(0, stop_start_idx-25):stop_start_idx+5])
          })

  # If simulation ended while stopped
  if in_stop:
    stops.append({
      "start_time": times[stop_start_idx],
      "end_time": times[-1],
      "duration": times[-1] - times[stop_start_idx],
      "gap_at_stop": gaps[stop_start_idx],
      "a_can_at_stop": a_cans[stop_start_idx],
      "p_line_at_stop": p_lines[stop_start_idx],
      "max_entry_jerk": max(abs(j) for j in jerks[max(0, stop_start_idx-25):stop_start_idx+5])
    })

  print(f"\n{'='*75}")
  print(f"DIAGNOSTIC REPORT: {tune_name.upper()}")
  print(f"{'='*75}")
  print(f"Total Stop Episodes Detected: {len(stops)}")

  for idx, s in enumerate(stops):
    print(f"\n  [Stop #{idx+1}]")
    print(f"  * Arrival Time:          {s['start_time']:.2f} s")
    print(f"  * Gap to Lead Car:       {s['gap_at_stop']:.2f} m  (Target: 4.50 m)")
    gap_error = s['gap_at_stop'] - 4.50
    if gap_error > 1.0:
      print(f"  * Gap Error:             +{gap_error:.2f} m  --> PREMATURE EARLY STOP ({gap_error*3.28084:.1f} ft early)!")
    else:
      print(f"  * Gap Error:             {gap_error:+.2f} m  --> ON TARGET")
    print(f"  * Commanded CAN Decel:   {s['a_can_at_stop']:.2f} m/s^2")
    print(f"  * Hydraulic Line Press:  {s['p_line_at_stop']:.3f} MPa")
    print(f"  * Peak Stopping Jerk:    {s['max_entry_jerk']:.2f} m/s^3")

  # Print chronological key event log
  print(f"\n  --- Key State Transitions & Telemetry ---")
  last_state = ""
  for l in logs:
    if l.long_ctrl_state != last_state or (l.v_ego == 0.0 and last_state != "STANDSTILL_REACHED"):
      print(f"  t={l.t:5.2f}s | State: {l.long_ctrl_state:8s} | vEgo: {l.v_ego:5.2f} m/s | Gap: {l.distance_gap:5.2f} m | aCan: {l.a_can_sent:5.2f} m/s^2 | LineP: {l.p_line_mpa:5.3f} MPa")
      last_state = l.long_ctrl_state

  return stops

def compare_tunes():
  print("Running simulations on Stinger GT 3.3T AWD physics engine...")
  orig_logs = run_simulation(ControllerType.ORIGINAL, total_time=18.0)
  g90_logs = run_simulation(ControllerType.GENESIS_G90, total_time=18.0)

  orig_stops = analyze_stopping_behavior("Original Stinger Tune (Stock openpilot)", orig_logs)
  g90_stops = analyze_stopping_behavior("New Stinger Tune (Genesis G90 Port)", g90_logs)

  print(f"\n{'='*75}")
  print("HEAD-TO-HEAD COMPARISON & VERIFICATION SUMMARY")
  print(f"{'='*75}")
  print(f"{'Metric':<35} | {'Original Tune':<17} | {'New G90 Tune':<17}")
  print(f"{'-'*35}-+-{'-'*17}-+-{'-'*17}")
  print(f"{'Number of Stops (Double Stop?)':<35} | {len(orig_stops):<17} | {len(g90_stops):<17}")
  print(f"{'First Stop Distance to Lead':<35} | {orig_stops[0]['gap_at_stop']:.2f} m {'':<11} | {g90_stops[0]['gap_at_stop']:.2f} m {'':<11}")
  gap_err_orig = orig_stops[0]['gap_at_stop'] - 4.5
  gap_err_g90 = g90_stops[0]['gap_at_stop'] - 4.5
  print(f"{'Premature Stopping Gap Error':<35} | +{gap_err_orig:.2f} m (+{gap_err_orig*3.28084:.1f} ft)   | {gap_err_g90:+.2f} m ({gap_err_g90*3.28084:+.1f} ft)")
  print(f"{'Decel Commanded At Initial Stop':<35} | {orig_stops[0]['a_can_at_stop']:.2f} m/s^2 {'':<9} | {g90_stops[0]['a_can_at_stop']:.2f} m/s^2 {'':<9}")
  print(f"{'Peak Stopping Jerk':<35} | {orig_stops[0]['max_entry_jerk']:.2f} m/s^3 {'':<9} | {g90_stops[0]['max_entry_jerk']:.2f} m/s^3 {'':<9}")
  print(f"{'Secondary Creep & Re-clamp?':<35} | {'YES (Violent)' if len(orig_stops) > 1 else 'NO':<17} | {'NO (Glide Stop)':<17}")

  # Assertions to formally verify
  assert len(orig_stops) > 1, f"Expected original tune to double stop, got {len(orig_stops)} stops"
  assert len(g90_stops) == 1, f"Expected G90 tune to single stop, got {len(g90_stops)} stops"
  assert abs(g90_stops[0]['gap_at_stop'] - 4.50) < 0.6, f"G90 stop gap error too large: {g90_stops[0]['gap_at_stop']}"
  assert abs(g90_stops[0]['a_can_at_stop']) < 0.35, f"G90 decel at stop not unwound: {g90_stops[0]['a_can_at_stop']}"
  print(f"\n>>> VERIFICATION SUITE: 100% OF TESTS PASSED! <<<")

if __name__ == "__main__":
  compare_tunes()
