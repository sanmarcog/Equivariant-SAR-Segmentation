"""
scripts/download_tromso_preevents.py

Download a small set of Sentinel-1 GRD pre-event scenes for Tromsø_20241220
for the multi-temporal stacking toy experiment.

Downloads up to --n-scenes scenes spaced evenly across the 30-day pre-event
window, saves as .zip files to --out-dir.

Usage:
    python scripts/download_tromso_preevents.py \
        --user gsanmarco91@gmail.com \
        --password '!j6yrUurwfQuysG' \
        --out-dir data/tromso_preevents \
        --n-scenes 6

Requirements:
    pip install requests tqdm
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from tqdm import tqdm

ODATA_URL      = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL   = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# Tromsø AOI and event
EVENT_DATE  = "20241220"
LAT_MIN, LAT_MAX = 69.4, 69.9
LON_MIN, LON_MAX = 18.5, 20.0
WINDOW_DAYS = 30


def get_token(user: str, password: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={"client_id": "cdse-public", "grant_type": "password",
              "username": user, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def footprint_wkt(lat_min, lat_max, lon_min, lon_max) -> str:
    return (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )


def query_scenes(token: str) -> list[dict]:
    event_date = datetime.strptime(EVENT_DATE, "%Y%m%d")
    start = event_date - timedelta(days=WINDOW_DAYS)
    end   = event_date - timedelta(days=1)
    wkt   = footprint_wkt(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    filter_str = (
        f"Collection/Name eq 'SENTINEL-1'"
        f" and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq 'GRD')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
        f" and ContentDate/Start gt {start.strftime('%Y-%m-%dT00:00:00.000Z')}"
        f" and ContentDate/Start lt {end.strftime('%Y-%m-%dT23:59:59.000Z')}"
    )

    headers = {"Authorization": f"Bearer {token}"}
    products = []
    url = ODATA_URL
    params = {"$filter": filter_str, "$orderby": "ContentDate/Start asc", "$top": 100}

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        products.extend(data.get("value", []))
        url = data.get("@odata.nextLink")
        params = {}

    return products


def select_evenly(products: list[dict], n: int) -> list[dict]:
    """Pick n scenes evenly spaced by index through the full list."""
    if len(products) <= n:
        return products
    indices = [round(i * (len(products) - 1) / (n - 1)) for i in range(n)]
    return [products[i] for i in indices]


def download_scene(product: dict, out_dir: Path, user: str, password: str) -> None:
    pid   = product["Id"]
    name  = product["Name"]
    url   = f"{DOWNLOAD_URL}({pid})/$value"
    dest  = out_dir / f"{name}.zip"

    if dest.exists():
        print(f"  Already exists, skipping: {dest.name}")
        return

    print(f"  Downloading: {name}")

    # Refresh token immediately before each download to avoid expiry
    token = get_token(user, password)
    headers = {"Authorization": f"Bearer {token}"}

    stream_resp = requests.get(url, headers=headers, stream=True, timeout=300)
    stream_resp.raise_for_status()
    total = int(stream_resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in stream_resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"  Saved: {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user",      required=True)
    parser.add_argument("--password",  required=True)
    parser.add_argument("--out-dir",   default="data/tromso_preevents")
    parser.add_argument("--n-scenes",  type=int, default=6,
                        help="Number of scenes to download (evenly spaced, default 6)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Authenticating...")
    token = get_token(args.user, args.password)
    print("OK.\n")

    print(f"Querying Tromsø pre-event scenes ({WINDOW_DAYS}-day window)...")
    products = query_scenes(token)
    print(f"Found {len(products)} scenes total.")

    selected = select_evenly(products, args.n_scenes)
    print(f"Selected {len(selected)} evenly-spaced scenes for download:\n")
    for p in selected:
        date = p["ContentDate"]["Start"][:10]
        print(f"  {p['Name']}  ({date})")

    total_size_gb = sum(p.get("ContentLength", 0) for p in selected) / 1e9
    print(f"\nEstimated download size: {total_size_gb:.1f} GB")
    print(f"Output directory: {out_dir.resolve()}\n")

    for i, product in enumerate(selected, 1):
        print(f"\n[{i}/{len(selected)}]")
        try:
            download_scene(product, out_dir, args.user, args.password)
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping")
            time.sleep(5)

    print("\nDone.")


if __name__ == "__main__":
    main()
