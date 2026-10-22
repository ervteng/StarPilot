import sys
import types
from types import SimpleNamespace

# Mock C-extensions and modules not available in x86 dev environment
if "cereal.messaging" not in sys.modules:
  try:
    import cereal.messaging  # noqa: F401
  except ImportError:
    class DummyMaster:
      pass

    msg_mod = types.ModuleType("cereal.messaging")
    msg_mod.SubMaster = DummyMaster
    msg_mod.PubMaster = DummyMaster
    sys.modules["cereal.messaging"] = msg_mod

if "openpilot.selfdrive.locationd.calibrationd" not in sys.modules:
  try:
    import openpilot.selfdrive.locationd.calibrationd
  except ImportError:
    calib_mod = types.ModuleType("openpilot.selfdrive.locationd.calibrationd")
    calib_mod.HEIGHT_INIT = [1.22]
    sys.modules["openpilot.selfdrive.locationd.calibrationd"] = calib_mod

if "openpilot.selfdrive.ui.ui_state" not in sys.modules:
  try:
    import openpilot.selfdrive.ui.ui_state  # noqa: F401
  except ImportError:
    ui_state_mod = types.ModuleType("openpilot.selfdrive.ui.ui_state")
    ui_state_mod.ui_state = SimpleNamespace(
      sm=SimpleNamespace(valid={}),
      status=0,
      always_on_lateral_active=False,
      is_metric=False,
      starpilot_toggles={},
      started_frame=0,
    )
    ui_state_mod.UIStatus = SimpleNamespace(DISENGAGED=0, ENGAGED=1, OVERRIDE=2)
    sys.modules["openpilot.selfdrive.ui.ui_state"] = ui_state_mod

from unittest.mock import MagicMock

import numpy as np
import pytest

from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer


@pytest.fixture
def renderer():
  r = object.__new__(ModelRenderer)
  r._car_space_transform = np.array([
    [580.0, -480.0, 0.0],
    [400.0, 0.0, -480.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)
  r._transform_dirty = False
  r._clip_region = SimpleNamespace(x=-500, y=-500, width=3160, height=2080)
  r._get_active_leads = list
  return r


def test_batched_projection_matches_single_line_projection(renderer):
  rng = np.random.default_rng(42)
  lines = [
    np.column_stack((np.linspace(0, 100, 33), rng.normal(0, 3, 33), rng.normal(0, 0.5, 33))).astype(np.float32)
    for _ in range(6)
  ]
  widths = [0.1, 0.15, 0.15, 0.1, 0.05, 0.05]
  max_idx = 30
  max_distance = 90.0

  # Batched call
  batched_polys = renderer._map_lines_to_polygons(lines, widths, 0.0, max_idx, max_distance)

  # Individual calls via _map_line_to_polygon
  individual_polys = [
    renderer._map_line_to_polygon(line, w, 0.0, max_idx, max_distance)
    for line, w in zip(lines, widths, strict=True)
  ]

  assert len(batched_polys) == len(lines)
  for b_poly, i_poly in zip(batched_polys, individual_polys, strict=True):
    assert b_poly.dtype == np.float32
    assert b_poly.shape == i_poly.shape
    np.testing.assert_allclose(b_poly, i_poly, rtol=1e-5, atol=1e-3)


def test_empty_lines_and_zero_lengths(renderer):
  lines = [
    np.empty((0, 3), dtype=np.float32),
    np.array([[10, 1, 0], [20, 2, 0]], dtype=np.float32),
    np.empty((0, 3), dtype=np.float32),
  ]
  widths = [0.1, 0.2, 0.1]
  polys = renderer._map_lines_to_polygons(lines, widths, 0.0, 10, 100.0)

  assert len(polys) == 3
  assert polys[0].shape == (0, 2)
  assert polys[1].shape[0] > 0
  assert polys[2].shape == (0, 2)


def test_lead_vehicle_clipping_parity(renderer):
  lead_mock = SimpleNamespace(dRel=30.0, yRel=0.0, status=True)
  renderer._get_active_leads = lambda: [("ego", lead_mock)]

  rng = np.random.default_rng(123)
  lines = [
    np.column_stack((np.linspace(0, 80, 50), rng.normal(0, 1, 50), np.zeros(50))).astype(np.float32)
    for _ in range(4)
  ]
  widths = [0.1] * 4

  batched_polys = renderer._map_lines_to_polygons(lines, widths, 0.0, 45, 80.0, clip_by_lead=True)
  unclipped_polys = renderer._map_lines_to_polygons(lines, widths, 0.0, 45, 80.0, clip_by_lead=False)
  individual_polys = [
    renderer._map_line_to_polygon(l, w, 0.0, 45, 80.0, clip_by_lead=True)
    for l, w in zip(lines, widths, strict=True)
  ]

  # Parity: batched output must match individual output
  for b_poly, i_poly in zip(batched_polys, individual_polys, strict=True):
    assert b_poly.shape == i_poly.shape
    np.testing.assert_allclose(b_poly, i_poly, rtol=1e-5, atol=1e-3)

  # Functional guarantee: lines MUST be truncated before the lead vehicle at 30.0m - 1.5m
  for b_poly, u_poly in zip(batched_polys, unclipped_polys, strict=True):
    assert b_poly.shape[0] < u_poly.shape[0]


def test_allow_invert_hill_geometry(renderer):
  x = np.linspace(5, 100, 40, dtype=np.float32)
  y = np.zeros(40, dtype=np.float32)
  z = np.sin(np.linspace(0, np.pi, 40)).astype(np.float32) * 5.0
  line = np.column_stack((x, y, z))

  poly_no_invert = renderer._map_lines_to_polygons([line], [0.2], 0.0, 39, 100.0, allow_invert=False)[0]
  poly_invert = renderer._map_lines_to_polygons([line], [0.2], 0.0, 39, 100.0, allow_invert=True)[0]
  individual_no_invert = renderer._map_line_to_polygon(line, 0.2, 0.0, 39, 100.0, allow_invert=False)

  # Parity check
  assert poly_no_invert.shape == individual_no_invert.shape
  np.testing.assert_allclose(poly_no_invert, individual_no_invert, rtol=1e-5, atol=1e-3)

  # Functional guarantee: allow_invert=False MUST discard downward reverse-slope points
  assert poly_no_invert.shape[0] < poly_invert.shape[0]


def test_clipping_region_bounds_parity(renderer):
  renderer._clip_region = SimpleNamespace(x=100, y=100, width=500, height=500)
  rng = np.random.default_rng(999)
  lines = [
    np.column_stack((np.linspace(1, 150, 60), rng.uniform(-10, 10, 60), np.zeros(60))).astype(np.float32)
    for _ in range(3)
  ]
  widths = [0.1, 0.2, 0.3]

  batched = renderer._map_lines_to_polygons(lines, widths, 0.0, 55, 150.0)
  individual = [renderer._map_line_to_polygon(l, w, 0.0, 55, 150.0) for l, w in zip(lines, widths, strict=True)]

  for b, i in zip(batched, individual, strict=True):
    assert b.shape == i.shape
    np.testing.assert_allclose(b, i, rtol=1e-5, atol=1e-3)


def test_radar_update_decoupled_from_model_regeneration(monkeypatch):
  from unittest.mock import MagicMock
  import pyray as rl
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelPoints

  r = object.__new__(ModelRenderer)
  r._path = ModelPoints()
  r._path.raw_points = np.zeros((10, 3), dtype=np.float32)
  r._transform_dirty = False
  r._started_frame = 0
  r._should_render_lead_indicator = lambda rs: True
  r._update_model = MagicMock()
  r._update_leads = MagicMock()
  r._update_adjacent_leads = MagicMock()
  r._draw_lane_lines = MagicMock()
  r._draw_path = MagicMock()
  r._draw_lead_indicator = MagicMock()
  r._draw_radar_tracks = MagicMock()
  r._update_raw_points = MagicMock()
  r._params = SimpleNamespace(get_bool=lambda *args, **kwargs: False)

  from openpilot.selfdrive.ui.ui_state import ui_state
  class MockSM:
    recv_frame = {"liveCalibration": 1, "modelV2": 1}
    updated = {"carParams": False, "modelV2": False, "radarState": True}
    valid = {"radarState": True, "starpilotRadarState": False}
    def __getitem__(self, k):
      return SimpleNamespace(openpilotLongitudinalControl=False, experimentalMode=False, leadOne=None, position=None, height=[1.22])

  mock_sm = MockSM()
  monkeypatch.setattr(ui_state, "sm", mock_sm)
  monkeypatch.setattr(ui_state, "started_frame", 0)

  # Frame 1: Only radarState updated. _update_model must NOT run, but _update_leads MUST run.
  r._render(rl.Rectangle(0, 0, 100, 100))
  assert not r._update_model.called, "Model geometry should not reproject on radarState alone"
  assert r._update_leads.called, "Lead indicators should update on radarState"

  # Frame 2: modelV2 updated. _update_model MUST run.
  mock_sm.updated["modelV2"] = True
  mock_sm.updated["radarState"] = False
  r._update_model.reset_mock()
  r._render(rl.Rectangle(0, 0, 100, 100))
  assert r._update_model.called, "Model geometry should reproject when modelV2 updates"



def _lead_renderer(renderer):
  renderer._path = SimpleNamespace(raw_points=np.column_stack((np.arange(0, 100, 1.0), np.zeros(100), np.arange(100) * 0.01)))
  renderer._path_offset_z = 0.0
  renderer._rect = SimpleNamespace(x=0, y=0, width=1000, height=800)
  renderer._map_to_screen = MagicMock(return_value=(500.0, 400.0))
  renderer._update_lead_vehicle = MagicMock(return_value="lead")
  return renderer


def test_leads_are_placed_in_camera_frame(renderer):
  r = _lead_renderer(renderer)
  lead = SimpleNamespace(status=True, dRel=30.0, yRel=1.0, vRel=-2.0)
  radar_state = SimpleNamespace(leadOne=lead, leadTwo=SimpleNamespace(status=False))
  path_x = r._path.raw_points[:, 0]

  r._update_leads(radar_state, path_x)

  x_cam = 30.0 + RADAR_TO_CAMERA
  assert r._map_to_screen.call_args.args[0] == pytest.approx(x_cam)
  # z is looked up at the camera-frame x (index 31, not 30)
  assert r._map_to_screen.call_args.args[2] == pytest.approx(r._path.raw_points[31, 2])
  # lead size/alpha logic keeps the radar-frame distance
  assert r._update_lead_vehicle.call_args.args[0] == 30.0


def test_adjacent_leads_are_placed_in_camera_frame(renderer):
  r = _lead_renderer(renderer)
  lead = SimpleNamespace(status=True, dRel=20.0, yRel=3.0, vRel=0.0)
  srs = SimpleNamespace(leadLeft=lead, leadRight=SimpleNamespace(status=False))

  r._update_adjacent_leads(srs, r._path.raw_points[:, 0])

  assert r._map_to_screen.call_args.args[0] == pytest.approx(20.0 + RADAR_TO_CAMERA)
  # displayed/effective distance stays radar frame
  assert r._update_lead_vehicle.call_args.args[0] == pytest.approx(23.0)


def test_lead_clip_box_uses_camera_frame(renderer):
  lead = SimpleNamespace(dRel=30.0, yRel=0.0, status=True)
  points = np.column_stack((np.linspace(0, 60, 61), np.zeros(61), np.zeros(61))).astype(np.float32)

  clipped = renderer._clip_line_by_lead(points, [("ego", lead)])

  # line ends 1.5 m behind the lead's camera-frame x
  assert clipped[-1, 0] == pytest.approx(30.0 + RADAR_TO_CAMERA - 1.5)


def test_radar_dots_project_camera_frame_distance(renderer, monkeypatch):
  import openpilot.selfdrive.ui.onroad.model_renderer as mr

  class Stop(Exception):
    pass

  captured = {}

  def fake_project(d_rel, *args):
    captured["d_rel"] = d_rel.copy()
    raise Stop

  points = [SimpleNamespace(dRel=10.0, yRel=0.5), SimpleNamespace(dRel=40.0, yRel=-1.0)]
  monkeypatch.setattr(mr, "project_radar_points", fake_project)
  monkeypatch.setattr(mr, "ui_state", SimpleNamespace(sm=_FakeSm(points)))
  renderer._params = SimpleNamespace(get_bool=lambda k: True)
  renderer._path = SimpleNamespace(raw_points=np.zeros((5, 3)))
  renderer._rect = SimpleNamespace(x=0, y=0, width=1000, height=800)
  renderer._path_offset_z = 0.0
  renderer._radar_path_generation = 0
  renderer._radar_transform_generation = 0
  renderer._radar_projection_key = None

  with pytest.raises(Stop):
    renderer._draw_radar_tracks()

  np.testing.assert_allclose(captured["d_rel"], [10.0 + RADAR_TO_CAMERA, 40.0 + RADAR_TO_CAMERA])


class _FakeSm:
  def __init__(self, points):
    self.valid = {"liveTracks": True}
    self.recv_frame = {"liveTracks": 1}
    self._msg = SimpleNamespace(points=points)

  def __getitem__(self, key):
    return self._msg
