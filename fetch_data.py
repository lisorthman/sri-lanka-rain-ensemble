"""
Download daily historical weather for 15 Sri Lankan cities from the
Open-Meteo Historical Weather API (ERA5 reanalysis, free, no API key).
https://open-meteo.com/en/docs/historical-weather-api

Output: data/sri_lanka_weather.csv
"""
import csv
import json
import os
import time
import urllib.request

CITIES = {
    # city: (latitude, longitude, district, province, climate_zone)
    "Colombo":      (6.9271, 79.8612, "Colombo",      "Western",       "Wet"),
    "Gampaha":      (7.0873, 79.9990, "Gampaha",      "Western",       "Wet"),
    "Galle":        (6.0535, 80.2210, "Galle",        "Southern",      "Wet"),
    "Ratnapura":    (6.6828, 80.3992, "Ratnapura",    "Sabaragamuwa",  "Wet"),
    "Kandy":        (7.2906, 80.6337, "Kandy",        "Central",       "Wet"),
    "Nuwara Eliya": (6.9497, 80.7891, "Nuwara Eliya", "Central",       "Wet"),
    "Badulla":      (6.9934, 81.0550, "Badulla",      "Uva",           "Intermediate"),
    "Kurunegala":   (7.4863, 80.3623, "Kurunegala",   "North Western", "Intermediate"),
    "Puttalam":     (8.0362, 79.8283, "Puttalam",     "North Western", "Dry"),
    "Anuradhapura": (8.3114, 80.4037, "Anuradhapura", "North Central", "Dry"),
    "Polonnaruwa":  (7.9403, 81.0188, "Polonnaruwa",  "North Central", "Dry"),
    "Trincomalee":  (8.5874, 81.2152, "Trincomalee",  "Eastern",       "Dry"),
    "Batticaloa":   (7.7310, 81.6747, "Batticaloa",   "Eastern",       "Dry"),
    "Hambantota":   (6.1241, 81.1185, "Hambantota",   "Southern",      "Dry"),
    "Jaffna":       (9.6615, 80.0255, "Jaffna",       "Northern",      "Dry"),
}

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_mean", "precipitation_sum", "rain_sum",
    "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant", "shortwave_radiation_sum",
    "et0_fao_evapotranspiration", "relative_humidity_2m_mean",
    "pressure_msl_mean", "cloud_cover_mean", "dew_point_2m_mean",
]

START, END = "2015-01-01", "2025-12-31"
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch(city, lat, lon):
    q = (f"{URL}?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
         f"&daily={','.join(DAILY_VARS)}&timezone=Asia%2FColombo")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(q, timeout=120) as r:
                return json.load(r)["daily"]
        except Exception as e:  # rate limit / transient errors
            print(f"  {city}: retry {attempt + 1} ({e})")
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"failed to download {city}")


def main():
    os.makedirs("data", exist_ok=True)
    out = "data/sri_lanka_weather.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "city", "district", "province", "climate_zone",
                    "latitude", "longitude"] + DAILY_VARS)
        for city, (lat, lon, dist, prov, zone) in CITIES.items():
            print("Downloading", city)
            d = fetch(city, lat, lon)
            for i, day in enumerate(d["time"]):
                w.writerow([day, city, dist, prov, zone, lat, lon] +
                           [d[v][i] for v in DAILY_VARS])
            time.sleep(3)
    print("Saved", out)


if __name__ == "__main__":
    main()
