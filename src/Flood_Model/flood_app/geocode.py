"""
Module 1 — Geocoding
Convert place names to latitude/longitude candidates using Nominatim.
"""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


def search_place(place_name: str, max_results: int = 5) -> list[dict]:
    """
    Search for a place name and return candidate locations.

    Parameters
    ----------
    place_name : str
        The place name to search, e.g. "Cuttack".
    max_results : int
        Maximum number of candidate results to return.

    Returns
    -------
    list[dict]
        Each dict has keys: 'name', 'lat', 'lon'.
        Returns empty list if no results or an error occurs.
    """
    geolocator = Nominatim(
        user_agent="mahanadi_flood_app",
        timeout=10,
    )

    # Try the exact query first, then with regional hints for the Mahanadi Basin
    queries = [
        place_name,
        f"{place_name}, Odisha, India",
        f"{place_name}, Chhattisgarh, India",
    ]

    for query in queries:
        try:
            locations = geolocator.geocode(
                query,
                exactly_one=False,
                limit=max_results,
                country_codes="in",  # restrict to India
            )
        except (GeocoderTimedOut, GeocoderServiceError):
            continue

        if locations:
            return [
                {
                    "name": loc.address,
                    "lat": loc.latitude,
                    "lon": loc.longitude,
                }
                for loc in locations
            ]

    return []

