# gtfs-route-segments

Segment GTFS shapes into stop-to-stop route segments that follow the original road geometry.

Most GTFS tools use Shapely's `.project()` to snap stops to shapes, which breaks on routes with loops, lollipop turnarounds, and out-and-back corridors. This library uses sequential projection with candidate-aware selection to handle complex route geometries correctly.

## Install

```bash
pip install gtfs-route-segments
```

For interactive map visualization:

```bash
pip install gtfs-segments[viz]
```

## Quick Start

### Segment an entire feed

```python
from gtfs_segments import segment_feed

result = segment_feed("path/to/gtfs")

segments = result['segments']       # GeoDataFrame of all stop-to-stop segments
diagnostics = result['diagnostics'] # per-shape stats (offsets, degenerates)
failures = result['failures']       # any shapes that failed
```

### Segment a single shape

```python
import pandas as pd
from gtfs_segments import segment_shape

shapes = pd.read_csv("gtfs/shapes.txt")
trips = pd.read_csv("gtfs/trips.txt")
stops = pd.read_csv("gtfs/stops.txt")
stop_times = pd.read_csv("gtfs/stop_times.txt")

result = segment_shape(shapes, trips, stops, stop_times, shape_id=232864)

segments = result['segments']
projected = result['projected_stops']
ruler = result['ruler']
diag = result['diagnostics']
```

### Low-level control

```python
from gtfs_segments import ShapeRuler, project_stops_sequential, segment_route

ruler = ShapeRuler(shapes, shape_id=232864)

# Get stop visit order from stop_times
trip_id = trips[trips['shape_id'] == 232864]['trip_id'].iloc[0]
st = stop_times[stop_times['trip_id'] == trip_id].sort_values('stop_sequence')
stop_sequence = st['stop_id'].tolist()
route_stops = stops[stops['stop_id'].isin(stop_sequence)].copy()

# Project stops onto shape and build segments
projected = project_stops_sequential(ruler, route_stops, stop_sequence)
segments = segment_route(ruler, projected)
```

## How It Works

### The Problem

Shapely's `.project()` finds the globally nearest point on a LineString. On a route that doubles back on itself, a stop at the base of a loop has two valid projection points — the entry and the exit. `.project()` picks whichever is geometrically closer, often the exit, which causes every stop inside the loop to collapse to the same distance.

### The Solution

1. **ShapeRuler** converts the shape to a metric CRS, builds a cumulative distance ruler, and indexes every segment for fast searching.

2. **find_candidates** scans every segment and collects all positions where a stop projects within tolerance (default 50m). A stop at a loop base gets candidates at both entry and exit.

3. **project_stops_sequential** processes stops in visit order (from `stop_times`), enforcing monotonic distances. For each stop, it picks the **earliest** candidate with a reasonable offset — biasing toward forward progress along the route rather than jumping ahead to a later occurrence.

4. **segment_route** slices the shape between consecutive stop distances, including all original vertices so segments follow the actual road geometry.

### CRS Auto-Detection

If no CRS is provided, the library auto-detects the UTM zone from the median longitude of the shape coordinates. Override with `metric_crs="EPSG:32614"` if needed.

## Diagnostics

The `diagnostics` DataFrame flags potential issues:

- **max_offset_m**: Largest perpendicular distance between a stop and the shape. Over 30m usually means the stop is set back from the road (transit centers, park-and-rides) or assigned to the wrong shape.
- **n_high_offset**: Count of stops with offset > 30m.
- **n_degenerate**: Count of segments shorter than 5m. Usually near-side/far-side stop pairs at intersections.

## Requirements

- Python ≥ 3.9
- geopandas ≥ 0.12
- shapely ≥ 2.0
- pandas ≥ 1.5
- numpy ≥ 1.23

## License

MIT
