"""
Feed-level processing: load a GTFS directory and segment all shapes.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path

from .ruler import ShapeRuler
from .projection import project_stops_sequential
from .segmentation import segment_route


def _load_gtfs(gtfs_path):
    """Load required GTFS tables from a directory."""
    p = Path(gtfs_path)
    if not p.is_dir():
        raise FileNotFoundError(f"GTFS directory not found: {gtfs_path}")

    required = ['shapes.txt', 'trips.txt', 'stops.txt', 'stop_times.txt']
    for f in required:
        if not (p / f).exists():
            raise FileNotFoundError(f"Missing {f} in {gtfs_path}")

    return {
        'shapes': pd.read_csv(p / 'shapes.txt'),
        'trips': pd.read_csv(p / 'trips.txt'),
        'stops': pd.read_csv(p / 'stops.txt'),
        'stop_times': pd.read_csv(p / 'stop_times.txt'),
    }


def segment_shape(shapes_df, trips_df, stops_df, stop_times_df,
                  shape_id, metric_crs=None, tolerance_m=50):
    """
    Segment a single shape into stop-to-stop segments.

    Parameters
    ----------
    shapes_df, trips_df, stops_df, stop_times_df : DataFrame
        GTFS tables
    shape_id : int or str
        Target shape ID
    metric_crs : str or None
        Projected CRS. Auto-detects UTM zone if None.
    tolerance_m : float
        Max perpendicular distance for candidate projection (default 50m)

    Returns
    -------
    dict with:
        - segments : GeoDataFrame of stop-to-stop segments (WGS84)
        - projected_stops : GeoDataFrame of projected stop positions
        - ruler : ShapeRuler instance
        - diagnostics : dict with summary stats
    """
    ruler = ShapeRuler(shapes_df, shape_id, metric_crs=metric_crs)

    # Find a trip that uses this shape
    matching_trips = trips_df[trips_df['shape_id'] == shape_id]
    if matching_trips.empty:
        raise ValueError(f"No trips found for shape {shape_id}")

    trip_id = matching_trips['trip_id'].iloc[0]

    # Get stop sequence from stop_times
    st = (stop_times_df[stop_times_df['trip_id'] == trip_id]
          .sort_values('stop_sequence'))
    stop_sequence = st['stop_id'].tolist()

    if not stop_sequence:
        raise ValueError(
            f"No stops found for trip {trip_id} (shape {shape_id})")

    # Filter to stops in this trip
    route_stops = stops_df[stops_df['stop_id'].isin(stop_sequence)].copy()

    # Project and segment
    projected = project_stops_sequential(
        ruler, route_stops, stop_sequence, tolerance_m=tolerance_m)
    segments = segment_route(ruler, projected)

    # Diagnostics
    diagnostics = {
        'shape_id': shape_id,
        'length_m': ruler.length,
        'n_stops': len(projected),
        'n_segments': len(segments),
        'max_offset_m': projected['offset_m'].max() if len(projected) > 0 else 0,
        'n_high_offset': int((projected['offset_m'] > 30).sum()) if len(projected) > 0 else 0,
        'n_degenerate': int((segments['segment_dist_m'] < 5).sum()) if len(segments) > 0 else 0,
    }

    return {
        'segments': segments,
        'projected_stops': projected,
        'ruler': ruler,
        'diagnostics': diagnostics,
    }


def segment_feed(gtfs_path, shape_ids=None, metric_crs=None,
                 tolerance_m=50, verbose=True):
    """
    Segment an entire GTFS feed into stop-to-stop segments.

    Parameters
    ----------
    gtfs_path : str or Path
        Path to directory containing GTFS .txt files
    shape_ids : list or None
        Specific shape IDs to process. If None, processes all shapes.
    metric_crs : str or None
        Projected CRS. Auto-detects UTM zone if None.
    tolerance_m : float
        Max perpendicular distance for candidate projection (default 50m)
    verbose : bool
        Print progress (default True)

    Returns
    -------
    dict with:
        - segments : GeoDataFrame of all segments across all shapes
        - diagnostics : DataFrame with per-shape summary stats
        - failures : DataFrame of shapes that failed with error messages
    """
    gtfs = _load_gtfs(gtfs_path)
    shapes_df = gtfs['shapes']
    trips_df = gtfs['trips']
    stops_df = gtfs['stops']
    stop_times_df = gtfs['stop_times']

    # Determine which shapes to process
    if shape_ids is None:
        # Get shapes that have at least one trip
        shape_ids = (trips_df[trips_df['shape_id'].isin(
                        shapes_df['shape_id'].unique())]
                     ['shape_id'].unique().tolist())

    all_segments = []
    all_diagnostics = []
    failures = []

    for i, shape_id in enumerate(shape_ids):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processing shape {i + 1}/{len(shape_ids)}...")

        try:
            result = segment_shape(
                shapes_df, trips_df, stops_df, stop_times_df,
                shape_id, metric_crs=metric_crs, tolerance_m=tolerance_m)

            if len(result['segments']) > 0:
                all_segments.append(result['segments'])
            all_diagnostics.append(result['diagnostics'])

        except Exception as e:
            failures.append({'shape_id': shape_id, 'error': str(e)})

    # Combine results
    if all_segments:
        combined_segments = gpd.GeoDataFrame(
            pd.concat(all_segments, ignore_index=True),
            crs="EPSG:4326"
        )
    else:
        combined_segments = gpd.GeoDataFrame()

    diagnostics_df = pd.DataFrame(all_diagnostics)
    failures_df = pd.DataFrame(failures)

    if verbose:
        n = len(diagnostics_df)
        n_fail = len(failures_df)
        n_high = (diagnostics_df['n_high_offset'] > 0).sum() if n > 0 else 0
        n_degen = (diagnostics_df['n_degenerate'] > 0).sum() if n > 0 else 0
        n_segs = len(combined_segments)
        print(f"  Processed {n} shapes → {n_segs} segments")
        if n_fail > 0:
            print(f"  Failed: {n_fail}")
        if n_high > 0:
            print(f"  High offset (>30m): {n_high} shapes")
        if n_degen > 0:
            print(f"  Degenerate (<5m): {n_degen} shapes")

    return {
        'segments': combined_segments,
        'diagnostics': diagnostics_df,
        'failures': failures_df,
    }
