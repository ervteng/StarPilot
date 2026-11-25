from openpilot.cereal import log
from openpilot.starpilot.longitudinal.personality_accel_boost import (
  ACCEL_BOOST_AGGRESSIVE, ACCEL_BOOST_STANDARD, PersonalityAccelBoostPreferences, personality_accel_boost,
)

P = log.LongitudinalPersonality


def test_boost_is_full_at_speed_and_zero_at_low_speed():
  assert personality_accel_boost(0.0, 0.0, P.aggressive) == 0.0
  assert personality_accel_boost(5.0, 0.0, P.aggressive) == 0.0
  assert abs(personality_accel_boost(10.0, 0.0, P.aggressive) - ACCEL_BOOST_AGGRESSIVE) < 1e-9
  assert abs(personality_accel_boost(20.0, 0.0, P.standard) - ACCEL_BOOST_STANDARD) < 1e-9


def test_boost_fades_out_as_target_falls_and_relaxed_is_zero():
  assert personality_accel_boost(20.0, -1.0, P.aggressive) == 0.0
  assert 0.0 < personality_accel_boost(20.0, -0.75, P.aggressive) < ACCEL_BOOST_AGGRESSIVE
  assert personality_accel_boost(20.0, 0.0, P.relaxed) == 0.0


class _Params:
  def __init__(self, value):
    self.value = value

  def get(self, key, *args, **kwargs):
    return self.value


def test_preference_is_default_off_and_samples_saved_value(monkeypatch):
  import openpilot.starpilot.longitudinal.personality_accel_boost as mod
  prefs = PersonalityAccelBoostPreferences(_Params(None))
  monkeypatch.setattr(mod, 'read_saved', lambda params, key, n: (None, True))
  assert prefs.sample(0) is False
  monkeypatch.setattr(mod, 'read_saved', lambda params, key, n: (b'1', True))
  assert prefs.sample(2_000_000_000) is True
