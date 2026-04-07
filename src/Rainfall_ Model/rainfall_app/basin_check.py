"""
Module 2 - Basin Boundary Check
Check whether a given lat/lon point lies inside the Mahanadi Basin.
"""

from functools import lru_cache

import geopandas as gpd
from shapely.geometry import Point

from config import BASIN_SHP_PATH


@lru_cache(maxsize=1)
def _load_basin_polygon():
    """Load basin polygon reprojected to EPSG:4326."""
    gdf = gpd.read_file(BASIN_SHP_PATH)
    gdf = gdf.to_crs(epsg=4326)
    return gdf.dissolve().geometry.iloc[0]


def is_inside_basin(lat: float, lon: float) -> bool:
    """Return True when point is inside basin polygon."""
    basin_polygon = _load_basin_polygon()
    point = Point(lon, lat)
    return basin_polygon.contains(point)
