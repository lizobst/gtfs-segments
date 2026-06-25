"""
gtfs-segments: Segment GTFS shapes into stop-to-stop route segments.

Handles complex route geometries including loops, lollipop turnarounds,
and out-and-back corridors using sequential projection with
candidate-aware selection.

Quick start:
    from gtfs_segments import segment_feed
    result = segment_feed("path/to/gtfs")
    segments = result['segments']  # GeoDataFrame of all segments

For single shapes:
    from gtfs_segments import segment_shape
    result = segment_shape(shapes_df, trips_df, stops_df, stop_times_df,
                           shape_id=232864)

For low-level control:
    from gtfs_segments import ShapeRuler, project_stops_sequential, segment_route
"""

from .ruler import ShapeRuler
from .projection import project_stops_sequential
from .segmentation import segment_route
from .feed import segment_feed, segment_shape

__version__ = "0.1.0"

__all__ = [
    "ShapeRuler",
    "project_stops_sequential",
    "segment_route",
    "segment_feed",
    "segment_shape",
]
