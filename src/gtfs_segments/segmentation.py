"""
Segment builder: cuts a shape into stop-to-stop segments following
the original road geometry.
"""

import geopandas as gpd


def segment_route(ruler, projected_stops, output_crs="EPSG:4326"):
    """
    Build stop-to-stop segments following the shape geometry.

    Parameters
    ----------
    ruler : ShapeRuler
    projected_stops : GeoDataFrame
        Output of project_stops_sequential
    output_crs : str
        CRS for output (default WGS84)

    Returns
    -------
    GeoDataFrame with columns:
        segment_idx, from_stop_id, to_stop_id, from_stop_name,
        to_stop_name, segment_dist_m, geometry
    """
    sorted_stops = (projected_stops
                    .sort_values('shape_dist_traveled')
                    .reset_index(drop=True))
    segments = []

    for i in range(len(sorted_stops) - 1):
        sa = sorted_stops.iloc[i]
        sb = sorted_stops.iloc[i + 1]

        geom = ruler.slice(sa['shape_dist_traveled'],
                           sb['shape_dist_traveled'])
        if geom is None:
            continue

        segments.append({
            'segment_idx': i,
            'shape_id': ruler.shape_id,
            'from_stop_id': sa['stop_id'],
            'to_stop_id': sb['stop_id'],
            'from_stop_name': sa.get('stop_name', ''),
            'to_stop_name': sb.get('stop_name', ''),
            'segment_dist_m': (sb['shape_dist_traveled'] -
                               sa['shape_dist_traveled']),
            'geometry': geom,
        })

    if not segments:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(segments, crs=ruler.metric_crs).to_crs(output_crs)
