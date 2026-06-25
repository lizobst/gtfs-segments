"""
Tests for gtfs-segments.

Uses small synthetic shapes so tests don't depend on a real GTFS feed.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from shapely.geometry import LineString

from gtfs_segments import (
    ShapeRuler,
    project_stops_sequential,
    segment_route,
    segment_shape,
    segment_feed,
)


# ============================================================
# Fixtures: synthetic GTFS data
# ============================================================

def _make_shapes(shape_id, coords):
    rows = []
    for i, (lat, lon) in enumerate(coords):
        rows.append({
            'shape_id': shape_id,
            'shape_pt_lat': lat,
            'shape_pt_lon': lon,
            'shape_pt_sequence': i + 1,
        })
    return pd.DataFrame(rows)


def _make_stops(stop_list):
    return pd.DataFrame(stop_list, columns=['stop_id', 'stop_name', 'stop_lat', 'stop_lon'])


def _make_trips(shape_id, trip_id='trip_1', route_id='route_1', direction_id=0):
    return pd.DataFrame([{
        'trip_id': trip_id,
        'route_id': route_id,
        'shape_id': shape_id,
        'direction_id': direction_id,
    }])


def _make_stop_times(trip_id, stop_ids):
    rows = []
    for i, sid in enumerate(stop_ids):
        rows.append({
            'trip_id': trip_id,
            'stop_id': sid,
            'stop_sequence': i + 1,
            'arrival_time': f'{6 + i}:00:00',
            'departure_time': f'{6 + i}:01:00',
        })
    return pd.DataFrame(rows)


# --- Straight corridor ---

@pytest.fixture
def straight_data():
    coords = [(29.40 + i * 0.002, -98.49) for i in range(20)]
    shapes = _make_shapes(1, coords)
    stops = _make_stops([
        (100, 'Stop A', 29.402, -98.4901),
        (101, 'Stop B', 29.410, -98.4899),
        (102, 'Stop C', 29.418, -98.4902),
        (103, 'Stop D', 29.426, -98.4898),
        (104, 'Stop E', 29.434, -98.4901),
    ])
    trips = _make_trips(1)
    stop_times = _make_stop_times('trip_1', [100, 101, 102, 103, 104])
    return shapes, trips, stops, stop_times


# --- Loop route ---

@pytest.fixture
def loop_data():
    """
    Route goes north on one street, loops east, returns south on a
    parallel street ~40m away. Stop 200 sits near both legs.
    """
    coords = []
    # Northbound on -98.490
    for i in range(12):
        coords.append((29.418 + i * 0.001, -98.490))
    # Loop east
    for angle in np.linspace(0, np.pi, 10):
        lat = 29.430 + 0.003 * np.sin(angle)
        lon = -98.490 + 0.0004 * (1 - np.cos(angle)) / 2
        coords.append((lat, lon))
    # Southbound on -98.4896 (parallel, ~40m east)
    for i in range(12):
        coords.append((29.430 - i * 0.001, -98.4896))

    shapes = _make_shapes(2, coords)
    stops = _make_stops([
        (200, 'Loop Base', 29.4201, -98.4898),       # between both legs
        (201, 'North Pre-Loop', 29.428, -98.4899),
        (202, 'Loop Top', 29.433, -98.4899),
        (203, 'Loop Descend', 29.431, -98.4898),      # on the curve
        (204, 'South Return', 29.425, -98.4897),
        (205, 'Loop End', 29.4201, -98.4897),
    ])
    trips = _make_trips(2)
    stop_times = _make_stop_times('trip_1', [200, 201, 202, 203, 204, 205])
    return shapes, trips, stops, stop_times


# --- Near-side/far-side pair ---

@pytest.fixture
def nearside_data():
    coords = [(29.40, -98.49 + i * 0.0003) for i in range(20)]
    shapes = _make_shapes(3, coords)
    stops = _make_stops([
        (300, 'Before Turn', 29.4001, -98.4895),
        (301, 'Near Side', 29.4001, -98.4880),
        (302, 'Far Side', 29.4001, -98.48795),
        (303, 'After Turn', 29.4001, -98.4865),
    ])
    trips = _make_trips(3)
    stop_times = _make_stop_times('trip_1', [300, 301, 302, 303])
    return shapes, trips, stops, stop_times


# --- Lollipop route ---

@pytest.fixture
def lollipop_data():
    coords = []
    for i in range(15):
        coords.append((29.45, -98.52 + i * 0.0015))
    center_lat, center_lon = 29.45, -98.4975
    for angle in np.linspace(0, 2 * np.pi, 20):
        coords.append((
            center_lat + 0.003 * np.sin(angle),
            center_lon + 0.003 * np.cos(angle),
        ))
    for i in range(15):
        coords.append((29.45, -98.4975 - i * 0.0015))

    shapes = _make_shapes(4, coords)
    stops = _make_stops([
        (400, 'West Start', 29.4501, -98.5195),
        (401, 'Mid Outbound', 29.4499, -98.5100),
        (402, 'Pre-Loop', 29.4501, -98.5005),
        (403, 'Loop North', 29.4530, -98.4975),
        (404, 'Loop East', 29.4500, -98.4945),
        (405, 'Mid Return', 29.4501, -98.5100),
        (406, 'West End', 29.4499, -98.5195),
    ])
    trips = _make_trips(4)
    stop_times = _make_stop_times('trip_1', [400, 401, 402, 403, 404, 405, 406])
    return shapes, trips, stops, stop_times


# ============================================================
# ShapeRuler tests
# ============================================================

class TestShapeRuler:

    def test_basic_construction(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        assert ruler.shape_id == 1
        assert ruler.length > 0
        assert len(ruler.vertices) == 20

    def test_auto_crs_detection(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        assert ruler.metric_crs == "EPSG:32614"

    def test_explicit_crs(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1, metric_crs="EPSG:32615")
        assert ruler.metric_crs == "EPSG:32615"

    def test_invalid_shape_id(self, straight_data):
        shapes, _, _, _ = straight_data
        with pytest.raises(ValueError, match="not found"):
            ShapeRuler(shapes, 999)

    def test_cumulative_distance_monotonic(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        dists = ruler.vertices['dist_cum'].values
        assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))

    def test_cumulative_distance_starts_at_zero(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        assert ruler.vertices['dist_cum'].iloc[0] == 0.0

    def test_length_matches_cumulative(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        assert abs(ruler.length - ruler.vertices['dist_cum'].max()) < 1.0

    def test_project_basic(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        ).to_crs(ruler.metric_crs)
        result = ruler.project(gdf.geometry.iloc[0])
        assert result['dist'] >= 0
        assert result['offset'] < 50
        assert result['snapped'] is not None

    def test_project_after_constraint(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        ).to_crs(ruler.metric_crs)
        result = ruler.project(gdf.geometry.iloc[0], after=2000)
        assert result['dist'] >= 2000

    def test_find_candidates_single(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        ).to_crs(ruler.metric_crs)
        candidates = ruler.find_candidates(gdf.geometry.iloc[2])
        assert len(candidates) == 1

    def test_find_candidates_loop(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            stops, geometry=gpd.points_from_xy(stops.stop_lon, stops.stop_lat),
            crs="EPSG:4326"
        ).to_crs(ruler.metric_crs)
        base_stop = gdf[gdf['stop_id'] == 200].geometry.iloc[0]
        candidates = ruler.find_candidates(base_stop)
        assert len(candidates) >= 2

    def test_slice_returns_linestring(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        geom = ruler.slice(100, 500)
        assert isinstance(geom, LineString)
        assert geom.length > 0

    def test_slice_includes_inner_vertices(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        geom = ruler.slice(0, ruler.length)
        assert len(geom.coords) > 2

    def test_slice_degenerate_returns_none(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        assert ruler.slice(500, 500) is None
        assert ruler.slice(500, 400) is None

    def test_repr(self, straight_data):
        shapes, _, _, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        r = repr(ruler)
        assert "shape_id=1" in r
        assert "length=" in r
        assert "vertices=" in r


# ============================================================
# Projection tests
# ============================================================

class TestProjection:

    def test_straight_monotonic(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        dists = projected['shape_dist_traveled'].values
        assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))

    def test_straight_all_stops_projected(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        assert len(projected) == 5

    def test_straight_low_offsets(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        assert projected['offset_m'].max() < 30

    def test_loop_monotonic(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        dists = projected['shape_dist_traveled'].values
        assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))

    def test_loop_base_projects_to_entry(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        base_dist = projected[projected['stop_id'] == 200]['shape_dist_traveled'].iloc[0]
        assert base_dist / ruler.length < 0.10

    def test_loop_no_collapsed_stops(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        dists = projected['shape_dist_traveled'].values
        diffs = np.diff(dists)
        assert all(d > 0 for d in diffs)

    def test_loop_offsets_reasonable(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        assert projected['offset_m'].max() < 50

    def test_lollipop_return_stops_correct(self, lollipop_data):
        shapes, _, stops, _ = lollipop_data
        ruler = ShapeRuler(shapes, 4)
        projected = project_stops_sequential(
            ruler, stops, [400, 401, 402, 403, 404, 405, 406])
        dists = projected['shape_dist_traveled'].values
        assert all(dists[i] <= dists[i + 1] for i in range(len(dists) - 1))
        mid_dist = projected[projected['stop_id'] == 405]['shape_dist_traveled'].iloc[0]
        assert mid_dist / ruler.length > 0.50

    def test_lollipop_end_near_full_length(self, lollipop_data):
        shapes, _, stops, _ = lollipop_data
        ruler = ShapeRuler(shapes, 4)
        projected = project_stops_sequential(
            ruler, stops, [400, 401, 402, 403, 404, 405, 406])
        last_dist = projected['shape_dist_traveled'].iloc[-1]
        assert last_dist / ruler.length > 0.85

    def test_missing_stop_skipped(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 999, 101, 102])
        assert len(projected) == 3
        assert 999 not in projected['stop_id'].values

    def test_n_candidates_populated(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        assert 'n_candidates' in projected.columns
        base_cands = projected[projected['stop_id'] == 200]['n_candidates'].iloc[0]
        assert base_cands >= 2


# ============================================================
# Segmentation tests
# ============================================================

class TestSegmentation:

    def test_segment_count(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        assert len(segments) == 4

    def test_segment_has_shape_id(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        assert 'shape_id' in segments.columns
        assert (segments['shape_id'] == 1).all()

    def test_segments_are_linestrings(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        for geom in segments.geometry:
            assert isinstance(geom, LineString)

    def test_segments_output_crs(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        assert segments.crs.to_epsg() == 4326

    def test_segment_distances_positive(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        assert (segments['segment_dist_m'] > 0).all()

    def test_segment_distances_sum(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        total = segments['segment_dist_m'].sum()
        first = projected['shape_dist_traveled'].min()
        last = projected['shape_dist_traveled'].max()
        assert abs(total - (last - first)) < 1.0

    def test_segment_from_to_chain(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected)
        for i in range(len(segments) - 1):
            assert segments.iloc[i]['to_stop_id'] == segments.iloc[i + 1]['from_stop_id']

    def test_nearside_farside_produces_segment(self, nearside_data):
        shapes, _, stops, _ = nearside_data
        ruler = ShapeRuler(shapes, 3)
        projected = project_stops_sequential(
            ruler, stops, [300, 301, 302, 303])
        segments = segment_route(ruler, projected)
        assert len(segments) == 3
        short = segments[
            (segments['from_stop_id'] == 301) & (segments['to_stop_id'] == 302)]
        assert len(short) == 1
        assert short.iloc[0]['segment_dist_m'] > 0

    def test_loop_segments_follow_shape(self, loop_data):
        shapes, _, stops, _ = loop_data
        ruler = ShapeRuler(shapes, 2)
        projected = project_stops_sequential(
            ruler, stops, [200, 201, 202, 203, 204, 205])
        segments = segment_route(ruler, projected)
        max_coords = max(len(s.geometry.coords) for _, s in segments.iterrows())
        assert max_coords > 3

    def test_custom_output_crs(self, straight_data):
        shapes, _, stops, _ = straight_data
        ruler = ShapeRuler(shapes, 1)
        projected = project_stops_sequential(
            ruler, stops, [100, 101, 102, 103, 104])
        segments = segment_route(ruler, projected, output_crs="EPSG:32614")
        assert segments.crs.to_epsg() == 32614


# ============================================================
# High-level API tests
# ============================================================

class TestSegmentShape:

    def test_returns_expected_keys(self, straight_data):
        shapes, trips, stops, stop_times = straight_data
        result = segment_shape(shapes, trips, stops, stop_times, shape_id=1)
        assert 'segments' in result
        assert 'projected_stops' in result
        assert 'ruler' in result
        assert 'diagnostics' in result

    def test_diagnostics_fields(self, straight_data):
        shapes, trips, stops, stop_times = straight_data
        result = segment_shape(shapes, trips, stops, stop_times, shape_id=1)
        diag = result['diagnostics']
        assert diag['shape_id'] == 1
        assert diag['n_stops'] == 5
        assert diag['n_segments'] == 4
        assert diag['max_offset_m'] >= 0
        assert diag['n_high_offset'] >= 0
        assert diag['n_degenerate'] >= 0

    def test_invalid_shape_raises(self, straight_data):
        shapes, trips, stops, stop_times = straight_data
        with pytest.raises(ValueError):
            segment_shape(shapes, trips, stops, stop_times, shape_id=999)


class TestSegmentFeed:

    def test_feed_from_directory(self, tmp_path, straight_data):
        shapes, trips, stops, stop_times = straight_data
        shapes.to_csv(tmp_path / 'shapes.txt', index=False)
        trips.to_csv(tmp_path / 'trips.txt', index=False)
        stops.to_csv(tmp_path / 'stops.txt', index=False)
        stop_times.to_csv(tmp_path / 'stop_times.txt', index=False)

        result = segment_feed(str(tmp_path), verbose=False)
        assert len(result['segments']) == 4
        assert len(result['failures']) == 0
        assert len(result['diagnostics']) == 1

    def test_feed_missing_file(self, tmp_path):
        (tmp_path / 'shapes.txt').write_text('shape_id\n')
        with pytest.raises(FileNotFoundError):
            segment_feed(str(tmp_path), verbose=False)

    def test_feed_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            segment_feed("/nonexistent/path", verbose=False)

    def test_feed_shape_id_filter(self, tmp_path, straight_data, loop_data):
        s1, t1, st1, stm1 = straight_data
        s2, t2, st2, stm2 = loop_data
        t2 = t2.copy()
        t2['trip_id'] = 'trip_2'

        shapes = pd.concat([s1, s2], ignore_index=True)
        trips = pd.concat([t1, t2], ignore_index=True)
        stops = pd.concat([st1, st2], ignore_index=True).drop_duplicates('stop_id')
        stm2_fixed = _make_stop_times('trip_2', [200, 201, 202, 203, 204, 205])
        stop_times = pd.concat([stm1, stm2_fixed], ignore_index=True)

        shapes.to_csv(tmp_path / 'shapes.txt', index=False)
        trips.to_csv(tmp_path / 'trips.txt', index=False)
        stops.to_csv(tmp_path / 'stops.txt', index=False)
        stop_times.to_csv(tmp_path / 'stop_times.txt', index=False)

        result = segment_feed(str(tmp_path), shape_ids=[1], verbose=False)
        assert len(result['diagnostics']) == 1
        assert result['diagnostics'].iloc[0]['shape_id'] == 1

    def test_feed_captures_failures(self, tmp_path, straight_data):
        shapes, trips, stops, stop_times = straight_data
        shapes.to_csv(tmp_path / 'shapes.txt', index=False)
        trips.to_csv(tmp_path / 'trips.txt', index=False)
        stops.to_csv(tmp_path / 'stops.txt', index=False)
        stop_times.to_csv(tmp_path / 'stop_times.txt', index=False)

        result = segment_feed(str(tmp_path), shape_ids=[1, 999], verbose=False)
        assert len(result['diagnostics']) == 1
        assert len(result['failures']) == 1
        assert result['failures'].iloc[0]['shape_id'] == 999
