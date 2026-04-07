"""
Module 1 - Geocoding
Convert place names to latitude/longitude candidates using Nominatim.
"""

from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim


def search_place(place_name: str, max_results: int = 5) -> list[dict]:
    """Search for place name and return candidate locations."""
    geolocator = Nominatim(
        user_agent="mahanadi_rainfall_app",
        timeout=10,
    )

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
                country_codes="in",
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
