"""
scripts/check_s1_availability.py

Query the Copernicus Data Space OData API to count available Sentinel-1 GRD
pre-event acquisitions for each AvalCD event.

Does NOT download anything. Prints scene counts grouped by relative orbit.

Usage:
    python scripts/check_s1_availability.py \
        --user <copernicus_email> \
        --password <copernicus_password> \
        [--window-days 60]

Requirements:
    pip install requests
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import requests

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# AvalCD events — bounding boxes approximate scene centres
# (lat_min, lat_max, lon_min, lon_max)
EVENTS = [
    ("Livigno_20240403",  "20240403",  46.3, 46.7,  10.0, 10.5),
    ("Livigno_20250129",  "20250129",  46.3, 46.7,  10.0, 10.5),
    ("Livigno_20250318",  "20250318",  46.3, 46.7,  10.0, 10.5),
    ("Nuuk_20160413",     "20160413",  64.0, 65.0, -52.0, -50.0),
    ("Nuuk_20210411",     "20210411",  64.0, 65.0, -52.0, -50.0),
    ("Pish_20230221",     "20230221",  38.0, 39.5,  71.0,  73.0),
    ("Tromso_20241220",   "20241220",  69.4, 69.9,  18.5,  20.0),
]


def get_token(user: str, password: str) -> str:
    resp = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user,
            "password": password,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def footprint_wkt(lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float) -> str:
    return (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )


def query_scenes(
    token: str,
    name: str,
    event_date_str: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    window_days: int,
) -> None:
    event_date = datetime.strptime(event_date_str, "%Y%m%d")
    start = event_date - timedelta(days=window_days)
    end   = event_date - timedelta(days=1)

    wkt = footprint_wkt(lat_min, lat_max, lon_min, lon_max)

    filter_str = (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'GRD')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
        f" and ContentDate/Start gt {start.strftime('%Y-%m-%dT00:00:00.000Z')}"
        f" and ContentDate/Start lt {end.strftime('%Y-%m-%dT23:59:59.000Z')}"
    )

    headers = {"Authorization": f"Bearer {token}"}
    all_products = []
    url = ODATA_URL
    params = {"$filter": filter_str, "$orderby": "ContentDate/Start asc", "$top": 100}

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        all_products.extend(data.get("value", []))
        next_link = data.get("@odata.nextLink")
        url = next_link if next_link else None
        params = {}  # params are embedded in nextLink

    print(f"\n{name} ({event_date_str}): {len(all_products)} GRD scenes  "
          f"[{start.date()} → {end.date()}]")

    if not all_products:
        print("  No scenes found.")
        return

    # Group by relative orbit
    from collections import defaultdict
    by_orbit: dict[str, list] = defaultdict(list)
    for p in all_products:
        orbit = "unknown"
        for attr in p.get("Attributes", []):
            if attr.get("Name") == "relativeOrbitNumber":
                orbit = str(attr.get("Value", "unknown"))
                break
        by_orbit[orbit].append(p)

    for orbit, scenes in sorted(by_orbit.items(), key=lambda x: -len(x[1])):
        dates = sorted(s["ContentDate"]["Start"][:10] for s in scenes)
        platforms = {}
        for s in scenes:
            plat = s.get("Name", "")[:3]  # S1A or S1B
            platforms[plat] = platforms.get(plat, 0) + 1
        print(f"  Orbit {orbit:>4}: {len(scenes):2d} scenes | {platforms} | {dates}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",        required=True)
    parser.add_argument("--password",    required=True)
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--event",       default=None)
    args = parser.parse_args()

    print("Authenticating with Copernicus Data Space...")
    token = get_token(args.user, args.password)
    print("OK.")

    print(f"\nQuerying archive — {args.window_days}-day pre-event window, GRD only.")
    print("=" * 70)

    events = EVENTS
    if args.event:
        events = [e for e in EVENTS if e[0] == args.event]

    for name, date_str, lat_min, lat_max, lon_min, lon_max in events:
        try:
            query_scenes(token, name, date_str,
                         lat_min, lat_max, lon_min, lon_max,
                         args.window_days)
        except Exception as exc:
            print(f"\n{name}: ERROR — {exc}")

    print("\n" + "=" * 70)
    print("Done. No data downloaded.")


if __name__ == "__main__":
    main()
