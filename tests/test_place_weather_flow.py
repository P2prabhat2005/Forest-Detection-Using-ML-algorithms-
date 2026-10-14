"""Focused failure-path tests; real provider checks are run separately."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.firecube_live import JsonCache, OpenMeteoWeatherProvider
from src.place_search import NominatimSearch, PlaceSearchError
from src.production_model import ProductionPredictionService


class PlaceWeatherTests(unittest.TestCase):
    def test_empty_search_does_not_contact_api(self):
        with patch("src.place_search.urlopen") as request:
            with self.assertRaises(PlaceSearchError):
                NominatimSearch().search("  ")
            request.assert_not_called()

    def test_api_error_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.place_search.urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(PlaceSearchError, "Place search unavailable"):
                    NominatimSearch(JsonCache(Path(directory))).search("Athens")

    def test_current_weather_failure_returns_status(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.firecube_live.urlopen", side_effect=OSError("offline")):
                result = OpenMeteoWeatherProvider(JsonCache(Path(directory))).fetch_current(38, 23.7)
        self.assertFalse(result["available"])
        self.assertIn("offline", result["message"])

    def test_entire_weather_history_failure_withholds_score(self):
        names = ("max_t2m", "max_d2m", "min_rh", "max_tp", "max_sp", "max_wind_speed")
        service = ProductionPredictionService(None, None)
        with patch.object(service, "predict", return_value={"missing_features": [f"{n}__mean10" for n in names]}):
            with self.assertRaisesRegex(RuntimeError, "prediction withheld"):
                service.predict_recent(38, 23.7)


if __name__ == "__main__":
    unittest.main()
