"""Kia Stinger 2022 with openpilot long reports AEB as on, so the dash hides the AEB-disabled light."""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.hyundai.hyundaican import create_acc_commands, create_acc_opt
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.values import CAR, DBC


def aeb_signals(car):
  cp = CarInterface.get_params(car, gen_empty_fingerprint(), [], True, False, False)
  dbc = DBC[cp.carFingerprint][Bus.pt]
  hud = structs.CarControl.HUDControl()
  msgs = create_acc_commands(CANPacker(dbc), True, 0., 1., 7, hud, 20., False, False, True, cp)
  msgs += create_acc_opt(CANPacker(dbc), cp)
  parser = CANParser(dbc, [("FCA11", 0), ("FCA12", 0)], 0)
  parser.update([(1_000_000_000, msgs)])
  return parser.vl["FCA11"]["FCA_Status"], parser.vl["FCA12"]["FCA_USM"]


class TestStingerAebLight(unittest.TestCase):
  def test_stinger_2022_reports_aeb_on(self):
    self.assertEqual(aeb_signals(CAR.KIA_STINGER_2022), (2, 2))

  def test_other_cars_keep_aeb_disabled(self):
    for car in (CAR.KIA_STINGER, CAR.GENESIS_G90):
      with self.subTest(car=car):
        self.assertEqual(aeb_signals(car), (1, 1))


if __name__ == "__main__":
  unittest.main()
