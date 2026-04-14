"""
scripts/check_s1_availability.py

Query the Copernicus Data Space (sentinelsat) to count available Sentinel-1 GRD
pre-event acquisitions for each AvalCD event on the same relative orbit.

Does NOT download anything. Prints a table of scene counts and lists scene IDs.

Usage:
    python scripts/check_s1_availability.py \
        --user <copernicus_username> \
        --password <copernicus_password> \
        [--window-days 60]

Requirements:
    pip install sentinelsat

Output example:
    Event                  | Date       | RelOrbit | Window  | Scenes | IDs
    Livigno_20240403       | 2024-04-03 | 168      | 60 days | 5      | S1A_...
    ...
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from sentinelsat import SentinelAPI


# ---------------------------------------------------------------------------
# AvalCD events — bounding boxes are approximate scene centres, generous
# enough to guarantee the orbit is covered. Adjust if needed.
# ---------------------------------------------------------------------------
EVENTS = [
    # (name, event_date, lat_min, lat_max, lon_min, lon_max)
    ("Livigno_20240403",  "20240403",  46.3, 46.7,  10.0, 10.5),
    ("Livigno_20250129",  "20250129",  46.3, 46.7,  10.0, 10.5),
    ("Livigno_20250318",  "20250318",  46.3, 46.7,  10.0, 10.5),  # val scene
    ("Nuuk_20160413",     "20160413",  64.0, 65.0, -52.0, -50.0),
    ("Nuuk_20210411",     "20210411",  64.0, 65.0, -52.0, -50.0),
    ("Pish_20230221",     "20230221",  38.0, 39.5,  71.0,  73.0),
    ("Tromso_20241220",   "20241220",  69.4, 69.9,  18.5,  20.0),
]


def footprint_wkt(lat_min: float, lat_max: float,
                  lon_min: float, lon_max: float) -> str:
    return (
        f"POLYGON(({lon_min} {lat_min}, {lon_max} {lat_min}, "
        f"{lon_max} {lat_max}, {lon_min} {lat_max}, {lon_min} {lat_min}))"
    )


def check_event(
    api: SentinelAPI,
    name: str,
    event_date_str: str,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    window_days: int,
) -> None:
    event_date = datetime.strptime(event_date_str, "%Y%m%d")
    window_start = event_date - timedelta(days=window_days)
    # Exclude event date itself — we want pre-event only
    window_end = event_date - timedelta(days=1)

    footprint = footprint_wkt(lat_min, lat_max, lon_min, lon_max)

    products = api.query(
        area=footprint,
        date=(window_start, window_end),
        platformname="Sentinel-1",
        producttype="GRD",
    )

    if not products:
        print(f"\n{name} ({event_date_str}): NO SCENES FOUND in {window_days}-day window")
        return

    df = api.to_dataframe(products)

    # Group by relative orbit to show orbit distribution
    if "relativeorbitnumber" in df.columns:
        orbit_counts = df["relativeorbitnumber"].value_counts()
        print(f"\n{name} ({event_date_str}): {len(df)} scenes total, "
              f"{window_days}-day window [{window_start.date()} → {window_end.date()}]")
        print(f"  Relative orbit distribution:")
        for orbit, count in orbit_counts.items():
            scenes_on_orbit = df[df["relativeorbitnumber"] == orbit]
            dates = sorted(scenes_on_orbit["beginposition"].dt.date.astype(str).tolist())
            platforms = scenes_on_orbit["platformname"].value_counts().to_dict()
            print(f"    Orbit {orbit:>4}: {count} scenes  |  {platforms}  |  dates: {dates}")
    else:
        print(f"\n{name} ({event_date_str}): {len(df)} scenes (orbit info unavailable)")
        for pid, row in df.iterrows():
            print(f"  {row.get('title', pid)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check S1 GRD pre-event availability.")
    parser.add_argument("--user",         required=True, help="Copernicus Data Space username")
    parser.add_argument("--password",     required=True, help="Copernicus Data Space password")
    parser.add_argument("--window-days",  type=int, default=60,
                        help="Days before event to search (default: 60)")
    parser.add_argument("--event",        default=None,
                        help="Run for a single event name only (optional)")
    args = parser.parse_args()

    api = SentinelAPI(
        args.user,
        args.password,
        "https://apihub.copernicus.eu/apihub",
    )

    events = EVENTS
    if args.event:
        events = [e for e in EVENTS if e[0] == args.event]
        if not events:
            print(f"Event '{args.event}' not found. Available: {[e[0] for e in EVENTS]}")
            return

    print(f"Querying Copernicus archive — {args.window_days}-day pre-event window, GRD only.")
    print("=" * 70)

    for name, date_str, lat_min, lat_max, lon_min, lon_max in events:
        try:
            check_event(api, name, date_str,
                        lat_min, lat_max, lon_min, lon_max,
                        args.window_days)
        except Exception as exc:
            print(f"\n{name}: ERROR — {exc}")

    print("\n" + "=" * 70)
    print("Done. No data downloaded.")


if __name__ == "__main__":
    main()
