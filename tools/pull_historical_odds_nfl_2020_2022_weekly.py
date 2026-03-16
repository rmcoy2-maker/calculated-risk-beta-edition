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

START_SEASON = 2020
END_SEASON = 2022

# Tuesday = 1 in Python weekday()
TARGET_WEEKDAY = 1

# Midday-ish UTC snapshot
SNAPSHOT_HOUR_UTC = 16
SNAPSHOT_MINUTE_UTC = 0
SNAPSHOT_SECOND_UTC = 0


def season_window(season: int) -> tuple[date, date]:
    return date(season, 9, 1), date(season + 1, 2, 14)


def iter_target_weekdays(start_d: date, end_d: date, target_weekday: int):
    cur = start_d
    while cur <= end_d:
        if cur.weekday() == target_weekday:
            yield datetime(
                cur.year,
                cur.month,
                cur.day,
                SNAPSHOT_HOUR_UTC,
                SNAPSHOT_MINUTE_UTC,
                SNAPSHOT_SECOND_UTC,
                tzinfo=timezone.utc,
            )
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

    snapshot_timestamp = payload.get("timestamp")
    previous_timestamp = payload.get("previous_timestamp")
    next_timestamp = payload.get("next_timestamp")

    for event in payload.get("data", []):
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "requested_snapshot": requested_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "snapshot_timestamp": snapshot_timestamp,
                        "previous_timestamp": previous_timestamp,
                        "next_timestamp": next_timestamp,
                        "event_id": event.get("id"),
                        "sport_key": event.get("sport_key"),
                        "sport_title": event.get("sport_title"),
                        "commence_time": event.get("commence_time"),
                        "home_team": event.get("home_team"),
                        "away_team": event.get("away_team"),
                        "book_key": bookmaker.get("key"),
                        "book_title": bookmaker.get("title"),
                        "book_last_update": bookmaker.get("last_update"),
                        "market_key": market.get("key"),
                        "market_last_update": market.get("last_update"),
                        "outcome_name": outcome.get("name"),
                        "outcome_price": outcome.get("price"),
                        "outcome_point": outcome.get("point"),
                        "outcome_description": outcome.get("description"),
                    })
    return rows


def pull_season(season: int) -> pd.DataFrame:
    start_d, end_d = season_window(season)
    rows: list[dict] = []

    print(f"\nPulling season {season}: {start_d} to {end_d}")
    snapshots = list(iter_target_weekdays(start_d, end_d, TARGET_WEEKDAY))
    print(f"Weekly snapshots scheduled: {len(snapshots)}")

    for i, dt in enumerate(snapshots, start=1):
        try:
            payload = fetch_snapshot(dt)
            flat = flatten_payload(payload, dt)
            rows.extend(flat)
            print(f"[{i:02d}/{len(snapshots)}] {dt.strftime('%Y-%m-%d')} rows={len(flat)}")
        except Exception as e:
            print(f"[{i:02d}/{len(snapshots)}] {dt.strftime('%Y-%m-%d')} ERROR: {e}")

        time.sleep(0.2)

    return pd.DataFrame(rows)


def main():
    if not API_KEY:
        raise ValueError("ODDS_API_KEY is not set in the current PowerShell session.")

    all_frames = []

    for season in range(START_SEASON, END_SEASON + 1):
        df = pull_season(season)

        out_path = OUT_DIR / f"nfl_historical_odds_{season}_weekly.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} rows={len(df):,}")

        all_frames.append(df)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = OUT_DIR / "nfl_historical_odds_2020_2022_weekly_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined: {combined_path} rows={len(combined):,}")


if __name__ == "__main__":
    main()