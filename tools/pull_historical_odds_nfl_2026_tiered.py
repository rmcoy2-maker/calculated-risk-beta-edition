from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

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

EVENTS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
HISTORICAL_ODDS_URL = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds"

SEASON_YEAR = 2026

# Pull strategy
WEEKLY_TARGET_WEEKDAY = 1   # Tuesday
WEEKLY_HOUR_UTC = 16
DAILY_HOUR_UTC = 16

# Final window before kickoff
FINAL_HOURLY_WINDOW_HOURS = 12
HOURLY_STEP_HOURS = 1

SLEEP_SECONDS = 0.2


def utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_events() -> list[dict]:
    params = {
        "apiKey": API_KEY,
    }
    r = requests.get(EVENTS_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_historical_snapshot(snapshot_dt: datetime) -> dict:
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "date": utc_iso(snapshot_dt),
    }
    r = requests.get(HISTORICAL_ODDS_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def season_filter(events: list[dict], season_year: int) -> list[dict]:
    """
    Keep events commencing between Aug 1 of season year and Feb 20 of following year.
    """
    start = datetime(season_year, 8, 1, tzinfo=timezone.utc)
    end = datetime(season_year + 1, 2, 20, 23, 59, 59, tzinfo=timezone.utc)

    out = []
    for ev in events:
        ct = ev.get("commence_time")
        if not ct:
            continue
        dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        if start <= dt <= end:
            out.append(ev)
    return out


def build_snapshots_for_event(commence_dt: datetime) -> set[datetime]:
    """
    Tiered cadence:
    - weekly until 7 days before kickoff
    - daily from 7 days to 1 day before kickoff
    - hourly in final 12 hours
    """
    snapshots: set[datetime] = set()

    weekly_start = commence_dt - timedelta(days=60)
    daily_start = commence_dt - timedelta(days=7)
    hourly_start = commence_dt - timedelta(hours=FINAL_HOURLY_WINDOW_HOURS)

    # Weekly zone
    cur = weekly_start
    while cur < daily_start:
        cur_day = datetime(cur.year, cur.month, cur.day, WEEKLY_HOUR_UTC, 0, 0, tzinfo=timezone.utc)
        if cur_day.weekday() == WEEKLY_TARGET_WEEKDAY and cur_day < daily_start:
            snapshots.add(cur_day)
        cur += timedelta(days=1)

    # Daily zone
    cur = daily_start
    while cur < hourly_start:
        cur_day = datetime(cur.year, cur.month, cur.day, DAILY_HOUR_UTC, 0, 0, tzinfo=timezone.utc)
        if cur_day < hourly_start:
            snapshots.add(cur_day)
        cur += timedelta(days=1)

    # Hourly zone
    cur = hourly_start
    while cur < commence_dt:
        cur_hour = datetime(cur.year, cur.month, cur.day, cur.hour, 0, 0, tzinfo=timezone.utc)
        if cur_hour < commence_dt:
            snapshots.add(cur_hour)
        cur += timedelta(hours=HOURLY_STEP_HOURS)

    return snapshots


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
                        "requested_snapshot": utc_iso(requested_dt),
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


def build_snapshot_plan(events: list[dict]) -> pd.DataFrame:
    plan_rows = []

    for ev in events:
        event_id = ev.get("id")
        commence_time = ev.get("commence_time")
        home_team = ev.get("home_team")
        away_team = ev.get("away_team")

        if not commence_time:
            continue

        commence_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        snapshots = sorted(build_snapshots_for_event(commence_dt))

        for snap in snapshots:
            delta = commence_dt - snap
            hours_to_kickoff = delta.total_seconds() / 3600.0

            if hours_to_kickoff > 24 * 7:
                cadence = "weekly"
            elif hours_to_kickoff > 24:
                cadence = "daily"
            else:
                cadence = "hourly"

            plan_rows.append({
                "event_id": event_id,
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": commence_time,
                "requested_snapshot": utc_iso(snap),
                "cadence": cadence,
                "hours_to_kickoff": round(hours_to_kickoff, 2),
            })

    plan = pd.DataFrame(plan_rows)
    return plan.sort_values(["commence_time", "requested_snapshot"], kind="stable")


def dedupe_snapshot_plan(plan: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple games can share the same requested snapshot.
    We only need to call each unique timestamp once.
    """
    unique_snaps = (
        plan[["requested_snapshot", "cadence"]]
        .drop_duplicates()
        .sort_values("requested_snapshot", kind="stable")
        .reset_index(drop=True)
    )
    return unique_snaps


def main():
    if not API_KEY:
        raise ValueError("ODDS_API_KEY is not set in the current PowerShell session.")

    print("Fetching 2026 NFL events...")
    events = fetch_events()
    events = season_filter(events, SEASON_YEAR)
    print(f"Events kept for {SEASON_YEAR}: {len(events)}")

    if not events:
        print("No events found for target season window.")
        return

    print("Building snapshot plan...")
    plan = build_snapshot_plan(events)
    plan_out = OUT_DIR / f"nfl_{SEASON_YEAR}_snapshot_plan.csv"
    plan.to_csv(plan_out, index=False)
    print(f"Saved plan: {plan_out} rows={len(plan):,}")

    unique_snaps = dedupe_snapshot_plan(plan)
    unique_out = OUT_DIR / f"nfl_{SEASON_YEAR}_unique_snapshots.csv"
    unique_snaps.to_csv(unique_out, index=False)
    print(f"Saved unique snapshots: {unique_out} rows={len(unique_snaps):,}")

    all_rows = []

    print("Pulling historical snapshots...")
    for i, row in unique_snaps.iterrows():
        snap_str = row["requested_snapshot"]
        snap_dt = datetime.fromisoformat(snap_str.replace("Z", "+00:00"))

        try:
            payload = fetch_historical_snapshot(snap_dt)
            flat = flatten_payload(payload, snap_dt)
            all_rows.extend(flat)
            print(f"[{i+1:04d}/{len(unique_snaps):04d}] {snap_str} cadence={row['cadence']} rows={len(flat)}")
        except Exception as e:
            print(f"[{i+1:04d}/{len(unique_snaps):04d}] {snap_str} ERROR: {e}")

        time.sleep(SLEEP_SECONDS)

    raw_df = pd.DataFrame(all_rows)
    raw_out = OUT_DIR / f"nfl_historical_odds_{SEASON_YEAR}_tiered_raw.csv"
    raw_df.to_csv(raw_out, index=False)
    print(f"Saved raw odds: {raw_out} rows={len(raw_df):,}")

    print("Done.")


if __name__ == "__main__":
    main()