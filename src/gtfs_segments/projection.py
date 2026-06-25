"""
Sequential stop-to-shape projection with candidate-aware loop handling.
"""

import geopandas as gpd


def project_stops_sequential(ruler, stops_df, stop_sequence, tolerance_m=50):
    """
    Project stops onto a shape in visit order with monotonic enforcement.

    For each stop, finds all candidate projection points within tolerance,
    then picks the earliest one with a reasonable offset. This handles
    loops where a stop could match the entry or exit of a loop.

    Parameters
    ----------
    ruler : ShapeRuler
        Built for the target shape
    stops_df : DataFrame
        Stop locations with columns: stop_id, stop_name, stop_lat, stop_lon
    stop_sequence : list
        stop_ids in visit order (from stop_times sorted by stop_sequence)
    tolerance_m : float
        Max perpendicular distance for candidate detection (default 50m)

    Returns
    -------
    GeoDataFrame in metric CRS with columns:
        stop_id, stop_name, shape_dist_traveled, offset_m,
        snapped_geom, original_geom, n_candidates
    """
    gdf = gpd.GeoDataFrame(
        stops_df.copy(),
        geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
        crs="EPSG:4326"
    ).to_crs(ruler.metric_crs)

    results = []
    min_dist = 0.0

    for stop_id in stop_sequence:
        row = gdf[gdf['stop_id'] == stop_id]
        if row.empty:
            continue
        row = row.iloc[0]

        candidates = ruler.find_candidates(
            row.geometry, after=min_dist, tolerance_m=tolerance_m)

        if candidates:
            best_offset = min(c['offset'] for c in candidates)

            # Accept earliest candidate whose offset is within 2x of best
            # or within 15m absolute, whichever is more generous.
            threshold = max(best_offset * 2, 15)

            chosen = None
            for c in candidates:  # sorted by dist
                if c['offset'] <= threshold:
                    chosen = c
                    break

            if chosen is None:
                chosen = min(candidates, key=lambda c: c['offset'])
        else:
            chosen = ruler.project(row.geometry, after=min_dist)

        snapped = ruler.line.interpolate(chosen['dist'])

        results.append({
            'stop_id': stop_id,
            'stop_name': row.get('stop_name', ''),
            'shape_dist_traveled': chosen['dist'],
            'offset_m': chosen['offset'],
            'snapped_geom': snapped,
            'original_geom': row.geometry,
            'n_candidates': len(candidates) if candidates else 0,
        })

        min_dist = chosen['dist']

    return gpd.GeoDataFrame(
        results, geometry='snapped_geom', crs=ruler.metric_crs)
