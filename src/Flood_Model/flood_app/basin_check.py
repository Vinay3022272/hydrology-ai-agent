"""
Module 2 — Basin Boundary Check
Check whether a given lat/lon point lies inside the Mahanadi Basin.
"""

import geopandas as gpd
from shapely.geometry import Point
from functools import lru_cache

from config import BASIN_SHP_PATH


@lru_cache(maxsize=1)
def _load_basin_polygon():
    """Load the basin shapefile and reproject to EPSG:4326 for lat/lon checks."""
    gdf = gpd.read_file(BASIN_SHP_PATH)
    gdf = gdf.to_crs(epsg=4326)
    # Dissolve all sub-polygons into a single geometry
    return gdf.dissolve().geometry.iloc[0]


def is_inside_basin(lat: float, lon: float) -> bool:
    """
    Check whether a point (lat, lon) is inside the Mahanadi Basin.

    Parameters
    ----------
    lat : float
        Latitude (WGS84).
    lon : float
        Longitude (WGS84).

    Returns
    -------
    bool
        True if the point falls within the basin boundary.
    """
    basin_polygon = _load_basin_polygon()
    point = Point(lon, lat)  # Shapely uses (x=lon, y=lat)
    return basin_polygon.contains(point)
