import math
from dataclasses import dataclass, replace

from opendbc.can import CANParser
from opendbc.can.dbc import DBC as DBCReader
from opendbc.can.parser import get_raw_value
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.hyundai.values import CAR, DBC, HyundaiFlags, HYUNDAI_MANDO_FRONT_RADAR_DBC, HYUNDAI_MRREVO14F_RADAR_DBC, \
                                       HYUNDAI_MRR30_RADAR_DBC, HYUNDAI_MRR35_RADAR_DBC
from openpilot.common.swaglog import cloudlog

RADAR_START_ADDR = 0x500
RADAR_MSG_COUNT = 32
G90_RADAR_MSG_COUNT = 64
MRREVO14F_RADAR_START_ADDR = 0x602
MRREVO14F_RADAR_MSG_COUNT = 16
MRR30_RADAR_START_ADDR = 0x210
MRR30_RADAR_MSG_COUNT = 16
MRR35_RADAR_START_ADDR = 0x3A5
MRR35_RADAR_MSG_COUNT = 32
GV70_RADAR_START_ADDR = 0x210
GV70_RADAR_MSG_COUNT = 16
GV70_RADAR_DBC = "hyundai_radar_210_21f_generated"

STINGER_2022_RADAR_BUS = 1
STINGER_2022_RADAR_START_ADDR = 0x238
STINGER_2022_RADAR_SLOT_COUNT = 10
STINGER_2022_RADAR_MSG_COUNT = 30
STINGER_2022_MAX_TRIPLET_SPAN_NS = 20_000_000
STINGER_2022_MAX_OBJECT_AGE_NS = 100_000_000


def _field(dat: bytes, start: int, size: int) -> int:
  return (int.from_bytes(dat, "big") >> (64 - start - size)) & ((1 << size) - 1)


@dataclass(frozen=True)
class Stinger2022CandidateObject:
  slot: int
  object_id: int
  cycle: int
  type_raw: int
  distance: float
  lateral_raw: float
  absolute_speed: float

  def relative_speed(self, ego_speed: float) -> float:
    if not math.isfinite(ego_speed):
      raise ValueError("ego speed must be finite")
    return self.absolute_speed - ego_speed


CandidateObject = Stinger2022CandidateObject


def decode_stinger_2022_triplet(slot: int, payloads: tuple[bytes, ...]) -> Stinger2022CandidateObject | None:
  if not 0 <= slot < STINGER_2022_RADAR_SLOT_COUNT or len(payloads) != 3 or any(len(dat) != 8 for dat in payloads):
    return None
  a, b, c = payloads
  cycle = _field(a, 44, 2)
  object_id = _field(a, 56, 7)
  if cycle != _field(b, 0, 2) or cycle != _field(c, 0, 2):
    return None
  if object_id != _field(b, 57, 7) or object_id != _field(c, 43, 7):
    return None
  distance_raw = _field(a, 0, 13)
  if object_id == 0 or distance_raw == 0:
    return None

  return Stinger2022CandidateObject(
    slot=slot, object_id=object_id, cycle=cycle, type_raw=_field(a, 13, 3),
    distance=distance_raw * 0.05, lateral_raw=_field(a, 16, 12) * 0.05 - 102.4,
    absolute_speed=_field(a, 32, 12) * 0.05 - 100.0,
  )


decode_triplet = decode_stinger_2022_triplet


class Stinger2022ObjectDecoder:
  def __init__(self):
    self.frames: dict[tuple[int, int], tuple[int, bytes]] = {}

  def update(self, timestamp_ns: int, address: int, dat: bytes, bus: int) -> Stinger2022CandidateObject | None:
    if bus != STINGER_2022_RADAR_BUS or not STINGER_2022_RADAR_START_ADDR <= address < STINGER_2022_RADAR_START_ADDR + STINGER_2022_RADAR_MSG_COUNT or len(dat) != 8:
      return None
    slot, part = divmod(address - STINGER_2022_RADAR_START_ADDR, 3)
    self.frames[slot, part] = (timestamp_ns, dat)
    if part != 2 or any((slot, i) not in self.frames for i in range(3)):
      return None
    triple = [self.frames[slot, i] for i in range(3)]
    times = [t for t, _ in triple]
    if max(times) != timestamp_ns or max(times) - min(times) > STINGER_2022_MAX_TRIPLET_SPAN_NS:
      return None

    # Consume the complete cycle so repeated final frames cannot reuse old parts.
    for i in range(3):
      del self.frames[slot, i]
    return decode_stinger_2022_triplet(slot, tuple(dat for _, dat in triple))


StingerObjectDecoder = Stinger2022ObjectDecoder



@dataclass(frozen=True)
class RadarTrackConfig:
  start_addr: int
  msg_count: int
  radar_type: str
  bus: int = 1
  frequency: int = 50
  parser_msg_count: int | None = None
  expected_length: int | None = None
  dbc_name: str | None = None

  @property
  def can_parser_msg_count(self) -> int:
    return self.parser_msg_count if self.parser_msg_count is not None else self.msg_count


RADAR_TRACK_CONFIGS = {
  HYUNDAI_MANDO_FRONT_RADAR_DBC: RadarTrackConfig(RADAR_START_ADDR, RADAR_MSG_COUNT, "mando"),
  HYUNDAI_MRREVO14F_RADAR_DBC: RadarTrackConfig(MRREVO14F_RADAR_START_ADDR, MRREVO14F_RADAR_MSG_COUNT, "mrrevo14f"),
  HYUNDAI_MRR30_RADAR_DBC: RadarTrackConfig(MRR30_RADAR_START_ADDR, MRR30_RADAR_MSG_COUNT, "mrr30", bus=0, expected_length=32),
  HYUNDAI_MRR35_RADAR_DBC: RadarTrackConfig(MRR35_RADAR_START_ADDR, MRR35_RADAR_MSG_COUNT, "mrr35", bus=0, frequency=20, expected_length=24),
}

# POC for parsing corner radars: https://github.com/commaai/openpilot/pull/24221/


def get_radar_track_config(car_fingerprint, flags: int = 0) -> RadarTrackConfig | None:
  if (car_fingerprint == CAR.HYUNDAI_PALISADE_2023 and
      flags & HyundaiFlags.CAN_CANFD_BLENDED and flags & HyundaiFlags.CANFD_LKA_STEERING):
    return RadarTrackConfig(GV70_RADAR_START_ADDR, GV70_RADAR_MSG_COUNT, "palisade_210", bus=0,
                            frequency=20, expected_length=32, dbc_name=GV70_RADAR_DBC)

  if car_fingerprint == CAR.GENESIS_GV70_ELECTRIFIED_1ST_GEN:
    return RadarTrackConfig(GV70_RADAR_START_ADDR, GV70_RADAR_MSG_COUNT, "gv70_210", bus=0,
                            frequency=20, expected_length=32, dbc_name=GV70_RADAR_DBC)

  if car_fingerprint == CAR.KIA_STINGER_2022:
    return RadarTrackConfig(STINGER_2022_RADAR_START_ADDR, STINGER_2022_RADAR_MSG_COUNT, "stinger_2022",
                            bus=STINGER_2022_RADAR_BUS, frequency=50, parser_msg_count=0,
                            expected_length=8)

  radar_dbc = DBC[car_fingerprint].get(Bus.radar)
  if car_fingerprint == CAR.GENESIS_G90 and radar_dbc == HYUNDAI_MANDO_FRONT_RADAR_DBC:
    return RadarTrackConfig(RADAR_START_ADDR, G90_RADAR_MSG_COUNT, "mando", parser_msg_count=RADAR_MSG_COUNT)

  radar_config = RADAR_TRACK_CONFIGS.get(radar_dbc)
  if radar_config is None:
    return None

  if car_fingerprint == CAR.HYUNDAI_IONIQ_6 and flags & HyundaiFlags.CANFD_CAMERA_SCC:
    return replace(radar_config, bus=1)

  return radar_config


def radar_tracks_available(radar_config: RadarTrackConfig | None, fingerprint) -> bool:
  if radar_config is None:
    return False

  if radar_config.radar_type in ("gv70_210", "palisade_210", "stinger_2022"):
    return all(fingerprint[radar_config.bus].get(addr) == radar_config.expected_length
               for addr in range(radar_config.start_addr, radar_config.start_addr + radar_config.msg_count))

  msg_len = fingerprint[radar_config.bus].get(radar_config.start_addr)
  if msg_len is None:
    return False

  return radar_config.expected_length is None or msg_len == radar_config.expected_length


def get_radar_can_parser(CP, radar_config):
  if radar_config is None:
    return None

  if radar_config.radar_type == "stinger_2022":
    messages = [("WHL_SPD11", 50)]
    return CANParser(DBC[CP.carFingerprint][Bus.pt], messages, 0)

  messages = [(f"RADAR_TRACK_{addr:x}", radar_config.frequency)
              for addr in range(radar_config.start_addr, radar_config.start_addr + radar_config.can_parser_msg_count)]
  dbc_name = radar_config.dbc_name or DBC[CP.carFingerprint][Bus.radar]
  return CANParser(dbc_name, messages, radar_config.bus)



class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.radar_config = get_radar_track_config(CP.carFingerprint, CP.flags)
    self.updated_messages = set()
    self.track_id = 0
    self.stinger_2022_radar = self.radar_config is not None and self.radar_config.radar_type == "stinger_2022"
    if self.stinger_2022_radar:
      self.trigger_msg = STINGER_2022_RADAR_START_ADDR + STINGER_2022_RADAR_MSG_COUNT - 1
      self.stinger_2022_decoder = Stinger2022ObjectDecoder()
      self.stinger_2022_objects: dict[int, tuple[int, Stinger2022CandidateObject]] = {}
      self.stinger_2022_last_seen_ns: dict[int, int] = {}
      self.latest_can_ts_ns = 0
      self.v_ego = 0.0
    else:
      self.trigger_msg = (self.radar_config.start_addr + self.radar_config.can_parser_msg_count - 1
                          if self.radar_config is not None else RADAR_START_ADDR)
    self.g90_extended_mando = (CP.carFingerprint == CAR.GENESIS_G90 and self.radar_config is not None and
                               self.radar_config.msg_count > self.radar_config.can_parser_msg_count)
    self.g90_mando_signals = []
    if self.g90_extended_mando:
      radar_dbc = DBCReader(DBC[CP.carFingerprint][Bus.radar])
      self.g90_mando_signals = list(radar_dbc.addr_to_msg[RADAR_START_ADDR].sigs.values())

    self.radar_off_can = CP.radarUnavailable
    # Probe whether radar tracks still exist on the Ioniq 6 while OP long is active,
    # without changing planner behavior yet.
    self.ioniq_6_radar_probe = CP.carFingerprint == CAR.HYUNDAI_IONIQ_6 and CP.openpilotLongitudinalControl and self.radar_off_can
    self.ioniq_6_radar_probe_logged = False
    self.ioniq_6_radar_probe_updates = 0
    self.rcp = get_radar_can_parser(CP, self.radar_config)

    # Precompute (addr, "RADAR_TRACK_xxx") pairs once. _update runs on the
    # CAN-driven card loop (core 4, shared with controlsd/selfdrived), so avoid
    # rebuilding 32 f-strings per radar frame.
    self.track_addrs: list[tuple[int, str]] = []
    if self.radar_config is not None and not self.stinger_2022_radar:
      self.track_addrs = [(addr, f"RADAR_TRACK_{addr:x}")
                          for addr in range(self.radar_config.start_addr,
                                            self.radar_config.start_addr + self.radar_config.can_parser_msg_count)]

  def update(self, can_strings):
    if self.ioniq_6_radar_probe and self.rcp is not None and not self.ioniq_6_radar_probe_logged:
      vls = self.rcp.update(can_strings)
      self.updated_messages.update(vls)
      self.ioniq_6_radar_probe_updates += 1

      if self.trigger_msg in self.updated_messages:
        rr = self._update(self.updated_messages)
        cloudlog.warning(f"Ioniq 6 radar probe: saw {len(rr.points)} radar tracks with radarUnavailable forced on")
        self.updated_messages.clear()
        self.ioniq_6_radar_probe_logged = True
      elif self.ioniq_6_radar_probe_updates >= 500:
        cloudlog.warning("Ioniq 6 radar probe: no radar track frames observed after startup")
        self.ioniq_6_radar_probe_logged = True
        self.updated_messages.clear()

    if self.radar_off_can or (self.rcp is None):
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)
    if self.g90_extended_mando:
      self._update_g90_extended_mando_tracks(can_strings)
    elif self.stinger_2022_radar:
      self._update_stinger_2022_tracks(can_strings)

    if self.trigger_msg not in self.updated_messages:
      return None

    rr = self._update_stinger_2022(self.updated_messages) if self.stinger_2022_radar else self._update(self.updated_messages)
    self.updated_messages.clear()

    return rr


  def _decode_g90_mando_values(self, dat: bytes):
    vals = {}
    for sig in self.g90_mando_signals:
      raw = get_raw_value(dat, sig)
      if sig.is_signed:
        raw -= ((raw >> (sig.size - 1)) & 1) * (1 << sig.size)
      vals[sig.name] = raw * sig.factor + sig.offset
    return vals

  def _update_g90_extended_mando_tracks(self, can_strings):
    if self.radar_config is None:
      return

    start_addr = self.radar_config.start_addr + self.radar_config.can_parser_msg_count
    end_addr = self.radar_config.start_addr + self.radar_config.msg_count

    for _, frames in can_strings:
      for address, dat, src in frames:
        if src != self.radar_config.bus or not (start_addr <= address < end_addr) or len(dat) < 8:
          continue

        self.updated_messages.add(address)
        msg = self._decode_g90_mando_values(dat)
        valid = msg["STATE"] in (3, 4)
        if valid:
          if address not in self.pts:
            self.pts[address] = structs.RadarData.RadarPoint()
            self.pts[address].trackId = self.track_id
            self.track_id += 1

          azimuth = math.radians(msg["AZIMUTH"])
          self.pts[address].measured = True
          self.pts[address].dRel = math.cos(azimuth) * msg["LONG_DIST"]
          self.pts[address].yRel = 0.5 * -math.sin(azimuth) * msg["LONG_DIST"]
          self.pts[address].vRel = msg["REL_SPEED"]
          self.pts[address].aRel = msg["REL_ACCEL"]
          self.pts[address].yvRel = float("nan")
        elif address in self.pts:
          del self.pts[address]

  def _update_stinger_2022_tracks(self, can_strings):
    if self.radar_config is None:
      return

    if "WHL_SPD11" in self.rcp.vl:
      whl = self.rcp.vl["WHL_SPD11"]
      self.v_ego = 0.5 * (whl["WHL_SPD_FL"] + whl["WHL_SPD_FR"]) * CV.KPH_TO_MS

    start_addr = self.radar_config.start_addr
    end_addr = start_addr + self.radar_config.msg_count

    for timestamp_ns, frames in can_strings:
      if timestamp_ns > self.latest_can_ts_ns:
        self.latest_can_ts_ns = timestamp_ns
      for address, dat, src in frames:
        if src != self.radar_config.bus or not (start_addr <= address < end_addr) or len(dat) != 8:
          continue

        self.updated_messages.add(address)
        obj = self.stinger_2022_decoder.update(timestamp_ns, address, bytes(dat), src)
        if obj is not None:
          self.stinger_2022_objects[obj.object_id] = (timestamp_ns, obj)

  def _update_stinger_2022(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    now_ns = self.latest_can_ts_ns


    for pt in self.pts.values():
      pt.measured = False

    for object_id, (ts, obj) in self.stinger_2022_objects.items():
      pt = self.pts.get(object_id)
      if pt is None:
        pt = structs.RadarData.RadarPoint()
        pt.trackId = self.track_id
        self.track_id += 1
        self.pts[object_id] = pt

      pt.measured = True
      pt.dRel = obj.distance
      pt.yRel = obj.lateral_raw
      pt.vRel = obj.relative_speed(self.v_ego)
      pt.aRel = float("nan")
      pt.yvRel = float("nan")
      self.stinger_2022_last_seen_ns[object_id] = ts

    self.stinger_2022_objects.clear()

    if now_ns > 0:
      for oid in list(self.pts.keys()):
        if now_ns - self.stinger_2022_last_seen_ns.get(oid, 0) > STINGER_2022_MAX_OBJECT_AGE_NS:
          del self.pts[oid]
          self.stinger_2022_last_seen_ns.pop(oid, None)

    ret.points = list(self.pts.values())
    return ret

  def _update(self, updated_messages):
    ret = structs.RadarData()
    if self.rcp is None:
      return ret

    if not self.rcp.can_valid:
      ret.errors.canError = True

    if self.radar_config is None:
      return ret

    radar_type = self.radar_config.radar_type
    vl = self.rcp.vl

    for addr, track_name in self.track_addrs:
      msg = vl[track_name]

      if radar_type == "mrr30":
        for i in ("1", "2"):
          track_key = addr * 2 + int(i) - 1
          if track_key not in self.pts:
            self.pts[track_key] = structs.RadarData.RadarPoint()
            self.pts[track_key].trackId = self.track_id
            self.track_id += 1

          valid = msg[f"{i}_STATE"] in (3, 4)
          if valid:
            pt = self.pts[track_key]
            pt.measured = True
            pt.dRel = msg[f"{i}_LONG_DIST"]
            pt.yRel = msg[f"{i}_LAT_DIST"]
            pt.vRel = msg[f"{i}_REL_SPEED"]
            pt.aRel = float("nan")
            pt.yvRel = float("nan")
          else:
            del self.pts[track_key]
        continue

      if radar_type in ("gv70_210", "palisade_210"):
        for i in ("1", "2"):
          track_key = addr * 2 + int(i) - 1
          valid = msg[f"{i}_STATE"] in (3, 4)
          if valid:
            pt = self.pts.get(track_key)
            if pt is None:
              pt = structs.RadarData.RadarPoint()
              pt.trackId = self.track_id
              self.track_id += 1
              self.pts[track_key] = pt
            pt.measured = True
            pt.dRel = msg[f"{i}_LONG_DIST"]
            pt.yRel = msg[f"{i}_LAT_DIST"]
            pt.vRel = msg[f"{i}_REL_SPEED"]
            pt.aRel = msg[f"{i}_REL_ACCEL"]
            pt.yvRel = msg[f"{i}_REL_LAT_SPEED"]
          elif track_key in self.pts:
            del self.pts[track_key]
        continue

      if radar_type == "mrrevo14f":
        for i in ("1", "2"):
          track_key = addr * 2 + int(i) - 1
          valid = msg[f"{i}_DISTANCE"] != 255.75
          if valid:
            pt = self.pts.get(track_key)
            if pt is None:
              pt = structs.RadarData.RadarPoint()
              pt.trackId = self.track_id
              self.track_id += 1
              self.pts[track_key] = pt
            pt.measured = True
            pt.dRel = msg[f"{i}_DISTANCE"]
            pt.yRel = msg[f"{i}_LATERAL"]
            pt.vRel = msg[f"{i}_SPEED"]
            pt.aRel = float("nan")
            pt.yvRel = float("nan")
          elif track_key in self.pts:
            del self.pts[track_key]
        continue

      if radar_type == "mrr35":
        # Most of the 32 channels are empty each frame. Only allocate a point
        # when the channel is valid; drop it otherwise. Avoids the per-frame
        # alloc-then-delete churn on the ~27 idle channels.
        if msg["STATE"] in (3, 4):
          pt = self.pts.get(addr)
          if pt is None:
            pt = structs.RadarData.RadarPoint()
            pt.trackId = self.track_id
            self.track_id += 1
            self.pts[addr] = pt
          pt.measured = True
          pt.dRel = msg["LONG_DIST"]
          pt.yRel = msg["LAT_DIST"]
          pt.vRel = msg["REL_SPEED"]
          pt.aRel = msg["REL_ACCEL"]
          pt.yvRel = float("nan")
        elif addr in self.pts:
          del self.pts[addr]
        continue

      if addr not in self.pts:
        self.pts[addr] = structs.RadarData.RadarPoint()
        self.pts[addr].trackId = self.track_id
        self.track_id += 1

      valid = msg['STATE'] in (3, 4)
      if valid:
        azimuth = math.radians(msg['AZIMUTH'])
        self.pts[addr].measured = True
        self.pts[addr].dRel = math.cos(azimuth) * msg['LONG_DIST']
        self.pts[addr].yRel = 0.5 * -math.sin(azimuth) * msg['LONG_DIST']
        self.pts[addr].vRel = msg['REL_SPEED']
        self.pts[addr].aRel = msg['REL_ACCEL']
        self.pts[addr].yvRel = float('nan')

      else:
        del self.pts[addr]

    ret.points = list(self.pts.values())
    return ret
