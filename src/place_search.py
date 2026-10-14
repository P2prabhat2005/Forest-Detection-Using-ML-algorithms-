"""Submitted place search for the local MVP; no autocomplete or bulk queries."""

import json
import os
import threading
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .firecube_live import JsonCache, LIVE_CACHE_DIR


class PlaceSearchError(RuntimeError):
    pass


class NominatimSearch:
    # All Streamlit sessions share this lock and request timer.
    _lock = threading.Lock()
    _last_request = 0.0

    def __init__(self, cache=None):
        self.endpoint = os.environ.get("FIRE_WARNING_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
        self.cache = cache or JsonCache(LIVE_CACHE_DIR / "places", ttl_seconds=7 * 86400)

    def search(self, query):
        query = " ".join(query.split())
        if not query or len(query) > 200:
            raise PlaceSearchError("Enter a place name between 1 and 200 characters.")
        url = self.endpoint + "?" + urlencode({"q": query, "format": "jsonv2", "limit": 5, "addressdetails": 1})
        with self._lock:
            cached = self.cache.get(url)
            if cached is not None:
                return cached
            delay = 1.1 - (time.monotonic() - NominatimSearch._last_request)
            if delay > 0:
                time.sleep(delay)
            NominatimSearch._last_request = time.monotonic()
            try:
                request = Request(url, headers={"User-Agent": "ForestFireEarlyWarningMVP/1.0 (local Streamlit place search)", "Accept-Language": "en"})
                with urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                places = [{"name": p["display_name"], "latitude": float(p["lat"]), "longitude": float(p["lon"])} for p in payload]
            except Exception as exc:
                raise PlaceSearchError(f"Place search unavailable. Try again later or enter coordinates. ({exc})") from exc
            self.cache.put(url, places)
            return places
