from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests


API_KEY = os.getenv("ODDS_API_KEY", "").strip()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "exports" / "historical_odds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"
BASE_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds"

START_SEASON = 2024
END_SEASON = 2025   # start with one season first

SNAPSHOT_HOUR_UTC = 16


def season_window(season: int) -> tuple[date, date]:
    return date(season, 9, 1), date(season + 1, 2, 14)


def iter_days(start_d: date, end_d: date):
    cur = start_d
    while cur <= end_d:
        yield datetime(cur.year, cur.month, cur.day, SNAPSHOT_HOUR_UTC, 0, 0, tzinfo=timezone.utc)
        cur += timedelta(days=1)


def fetch_snapshot(dt: datetime) -> dict:
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "date": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def flatten_payload(payload: dict, requested_dt: datetime) -> list[dict]:
    rows = []

    for event in payload.get("data", []):
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "requested_snapshot": requested_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "snapshot_timestamp": payload.get("timestamp"),
                        "event_id": event.get("id"),
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "book_key": bookmaker.get("key"),
                        "book_title": bookmaker.get("title"),
                        "market_key": market.get("key"),
                        "outcome_name": outcome.get("name"),
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                    })
    return rows


def pull_season(season: int) -> pd.DataFrame:
    start_d, end_d = season_window(season)
    rows = []

    print(f"Pulling season {season}: {start_d} to {end_d}")

    for i, dt in enumerate(iter_days(start_d, end_d), start=1):
        try:
            payload = fetch_snapshot(dt)
            flat = flatten_payload(payload, dt)
            rows.extend(flat)
            print(f"[{i:03d}] {dt.strftime('%Y-%m-%d')} rows={len(flat)}")
        except Exception as e:
            print(f"[{i:03d}] {dt.strftime('%Y-%m-%d')} ERROR: {e}")

        time.sleep(0.2)

    return pd.DataFrame(rows)


def main():
    if not API_KEY:
        raise ValueError("ODDS_API_KEY is not set.")

    all_frames = []

    for season in range(START_SEASON, END_SEASON + 1):
        df = pull_season(season)
        out_path = OUT_DIR / f"nfl_historical_odds_{season}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} rows={len(df):,}")
        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUT_DIR / "nfl_historical_odds_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Saved: {combined_path} rows={len(combined):,}")


if __name__ == "__main__":
    main()