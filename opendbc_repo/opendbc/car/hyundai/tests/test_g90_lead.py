"""Source-derived G90 lead hysteresis and existing CAN encoder fields."""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.hyundai.g90_lead import G90LeadState, LeadObservation, eligible
from opendbc.car.hyundai.hyundaican import create_acc_commands
from opendbc.car.hyundai.interface import CarInterface
from opendbc.car.hyundai.tests.test_g90_longitudinal import params
from opendbc.car.hyundai.values import CAR, DBC


def observation(distance=26., relative=-.5, present=True, drive=1):
  return LeadObservation(drive, 100, 101, 102, present, distance, relative)


class TestG90Lead(unittest.TestCase):
  def test_visibility_gap_and_relative_state_counts_controller_updates(self):
    state = G90LeadState()
    for _ in range(49):
      lead = state.update(observation(), False)
    self.assertFalse(lead.lead_visible)
    self.assertEqual(lead.object_gap, 0)
    lead = state.update(observation(), False)
    self.assertTrue(lead.lead_visible)
    self.assertEqual((lead.object_gap, lead.object_rel_gap), (4, 2))
    for _ in range(49):
      lead = state.update(observation(0, 0, False), False)
    self.assertTrue(lead.lead_visible)
    self.assertFalse(state.update(observation(0, 0, False), False).lead_visible)
    state = G90LeadState()
    for frame in range(50):
      lead = state.update(observation(15 if frame % 2 else 26), False)
    self.assertEqual(lead.object_gap, 2)  # Original non-current gaps share one counter.
    self.assertEqual(state.update(observation(.05, -1), False).lead_distance, 20.)
    self.assertEqual(state.update(observation(999, -999), False).lead_rel_speed, -170.)
    self.assertEqual(state.update(observation(26, -.2), False).object_rel_gap, 1)

  def test_optional_payload_preserves_sibling_and_accel_aeb_bytes(self):
    for car, alpha in ((CAR.GENESIS_G90, True), (CAR.GENESIS_G90, False), (CAR.GENESIS_G80, True)):
      cp = params(car, alpha)
      hud = structs.CarControl.HUDControl(leadVisible=True, leadDistanceBars=2)
      state = G90LeadState()
      for _ in range(50):
        lead = state.update(observation(), False)

      def packed(payload, cp=cp, hud=hud):
        packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])
        return create_acc_commands(packer, True, .2, 1., 7, hud, 70., False, False, True, cp, lead_data=payload)
      baseline, changed = packed(None), packed(lead)
      if car != CAR.GENESIS_G90 or not alpha:
        self.assertEqual(baseline, changed)
      else:
        for old, new in zip(baseline, changed, strict=True):
          if old[0] in (0x421, 0x38d):
            self.assertEqual(old, new)
        parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [('FCA11', 0), ('SCC12', 0)], 0)
        parser.update([(1_000_000_000, changed)])
        self.assertEqual(parser.vl['FCA11']['FCA_Status'], 1)
        non_fca = create_acc_commands(CANPacker(DBC[cp.carFingerprint][Bus.pt]), True, .2, 1., 7,
                                      hud, 70., False, False, False, cp, lead_data=lead)
        parser.update([(1_010_000_000, non_fca)])
        self.assertEqual(parser.vl['SCC12']['AEB_Status'], 1)


class TestStingerLead(unittest.TestCase):
  def test_eligibility_is_g90_and_stinger_2022_with_openpilot_longitudinal(self):
    expected = {(CAR.GENESIS_G90, True), (CAR.KIA_STINGER_2022, True)}
    for car in (CAR.GENESIS_G90, CAR.KIA_STINGER_2022, CAR.KIA_STINGER, CAR.GENESIS_G80, CAR.HYUNDAI_SONATA):
      for alpha in (False, True):
        with self.subTest(car=car, alpha=alpha):
          self.assertEqual(eligible(params(car, alpha)), (car, alpha) in expected)

  def test_stinger_only_gets_lead_state_not_g90_longitudinal_policy(self):
    ci = CarInterface(params(CAR.KIA_STINGER_2022, True))
    self.assertIsNotNone(ci.CC.g90_lead_state)
    self.assertIsNone(ci.CC.g90_longitudinal)
    self.assertIsNone(CarInterface(params(CAR.KIA_STINGER_2022, False)).CC.g90_lead_state)

  def test_stinger_scc11_scc14_carry_real_lead(self):
    cp = params(CAR.KIA_STINGER_2022, True)
    hud = structs.CarControl.HUDControl(leadVisible=True, leadDistanceBars=2)
    state = G90LeadState()
    for _ in range(50):
      lead = state.update(observation(26., -.5), False)

    def packed(payload):
      packer = CANPacker(DBC[cp.carFingerprint][Bus.pt])
      return create_acc_commands(packer, True, .2, 1., 7, hud, 70., False, False, True, cp, lead_data=payload)

    parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [('SCC11', 0), ('SCC14', 0)], 0)
    parser.update([(1_000_000_000, packed(lead))])
    self.assertEqual([parser.vl['SCC11'][k] for k in ('ObjValid', 'ACC_ObjStatus', 'ACC_ObjDist', 'ACC_ObjRelSpd')], [1, 1, 26, -.5])
    self.assertEqual((parser.vl['SCC14']['ObjGap'], parser.vl['SCC14']['ObjDistStat']), (4, 2))
    # Without lead data the fixed placeholder is still sent.
    parser.update([(1_010_000_000, packed(None))])
    self.assertEqual([parser.vl['SCC11'][k] for k in ('ObjValid', 'ACC_ObjStatus', 'ACC_ObjDist', 'ACC_ObjRelSpd')], [1, 1, 1, 0])
    self.assertEqual(parser.vl['SCC14']['ObjGap'], 2)
    # Stock-longitudinal Stinger ignores lead data entirely.
    stock = params(CAR.KIA_STINGER_2022, False)
    self.assertEqual(create_acc_commands(CANPacker(DBC[stock.carFingerprint][Bus.pt]), True, .2, 1., 7, hud, 70., False, False, True, stock,
                                         lead_data=lead),
                     create_acc_commands(CANPacker(DBC[stock.carFingerprint][Bus.pt]), True, .2, 1., 7, hud, 70., False, False, True, stock))
