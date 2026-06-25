"""
Shape ruler: wraps a GTFS shape as a metric LineString with cumulative
distance, providing sequential projection and segment slicing.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import LineString


def _utm_crs_from_lon(lon):
    """Derive UTM EPSG code from a longitude value."""
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if zone <= 60 else "EPSG:32660"


class ShapeRuler:
    """
    Wraps a GTFS shape as a metric LineString with cumulative distance.

    Parameters
    ----------
    shapes_df : DataFrame
        Full shapes.txt with columns: shape_id, shape_pt_sequence,
        shape_pt_lat, shape_pt_lon
    shape_id : int or str
        Target shape to build the ruler for
    metric_crs : str or None
        Projected CRS for distance calculations. If None, auto-detects
        the UTM zone from the shape's median longitude.
    """

    def __init__(self, shapes_df, shape_id, metric_crs=None):
        df = (shapes_df[shapes_df['shape_id'] == shape_id]
              .sort_values('shape_pt_sequence')
              .copy())
        if df.empty:
            raise ValueError(f"Shape {shape_id} not found in shapes_df")

        self.shape_id = shape_id

        # Auto-detect UTM zone if no CRS provided
        if metric_crs is None:
            median_lon = df['shape_pt_lon'].median()
            metric_crs = _utm_crs_from_lon(median_lon)
        self.metric_crs = metric_crs

        # Build metric GeoDataFrame of shape vertices
        self.vertices = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.shape_pt_lon, df.shape_pt_lat),
            crs="EPSG:4326"
        ).to_crs(metric_crs).reset_index(drop=True)

        # Cumulative distance at each vertex
        self.vertices['dist_cum'] = (
            self.vertices.geometry
            .distance(self.vertices.geometry.shift())
            .fillna(0)
            .cumsum()
        )

        # Full line in metric CRS
        self.line = LineString(self.vertices.geometry.tolist())
        self.length = self.line.length

        # Pre-build segment index for fast windowed search
        self._seg_starts = []
        self._seg_lines = []
        for i in range(len(self.vertices) - 1):
            self._seg_starts.append(self.vertices['dist_cum'].iloc[i])
            self._seg_lines.append(
                LineString([self.vertices.geometry.iloc[i],
                            self.vertices.geometry.iloc[i + 1]])
            )
        self._seg_starts = np.array(self._seg_starts)

    def project(self, point_geom, after=0.0):
        """
        Project a point onto the shape, searching only after a minimum
        distance. Returns dict with dist, offset, snapped.
        """
        best_dist = None
        best_offset = float('inf')

        for seg_start, seg_line in zip(self._seg_starts, self._seg_lines):
            seg_end = seg_start + seg_line.length
            if seg_end < after:
                continue
            local_proj = seg_line.project(point_geom)
            global_proj = max(seg_start + local_proj, after)
            snapped = self.line.interpolate(global_proj)
            offset = point_geom.distance(snapped)
            if offset < best_offset:
                best_offset = offset
                best_dist = global_proj

        if best_dist is None:
            best_dist = self.length
            best_offset = point_geom.distance(
                self.line.interpolate(self.length))

        snapped = self.line.interpolate(best_dist)
        return {'dist': best_dist, 'offset': best_offset, 'snapped': snapped}

    def find_candidates(self, point_geom, after=0.0, tolerance_m=50):
        """
        Find all distinct positions where a point projects within tolerance.

        A stop at the base of a loop will have candidates at both the entry
        and exit. Nearby candidates (within 30m along shape) are merged.

        Returns list of dict {dist, offset}, sorted by distance.
        """
        raw = []
        for seg_start, seg_line in zip(self._seg_starts, self._seg_lines):
            seg_end = seg_start + seg_line.length
            if seg_end < after:
                continue
            local_proj = seg_line.project(point_geom)
            global_proj = max(seg_start + local_proj, after)
            snapped = self.line.interpolate(global_proj)
            offset = point_geom.distance(snapped)
            if offset <= tolerance_m:
                raw.append({'dist': global_proj, 'offset': offset})

        if not raw:
            return []

        # Merge candidates within 30m (keep best offset)
        raw.sort(key=lambda c: c['dist'])
        merged = [raw[0]]
        for c in raw[1:]:
            if c['dist'] - merged[-1]['dist'] > 30:
                merged.append(c)
            elif c['offset'] < merged[-1]['offset']:
                merged[-1] = c
        return merged

    def slice(self, d_start, d_end):
        """
        Extract the shape geometry between two distances.
        Returns LineString in metric CRS, or None if degenerate.
        """
        if d_end <= d_start:
            return None

        p1 = self.line.interpolate(d_start)
        p2 = self.line.interpolate(d_end)

        mask = ((self.vertices['dist_cum'] > d_start) &
                (self.vertices['dist_cum'] < d_end))
        inner = self.vertices[mask].geometry.tolist()

        coords = ([(p1.x, p1.y)] +
                  [(pt.x, pt.y) for pt in inner] +
                  [(p2.x, p2.y)])

        return LineString(coords) if len(coords) >= 2 else None

    def __repr__(self):
        return (f"ShapeRuler(shape_id={self.shape_id}, "
                f"length={self.length:.0f}m, "
                f"vertices={len(self.vertices)}, "
                f"crs={self.metric_crs})")
