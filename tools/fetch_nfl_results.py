import io
from datetime import date
from typing import Iterable, Optional
import pandas as pd
import requests
from pathlib import Path
ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / 'exports'
EXPORTS.mkdir(parents=True, exist_ok=True)
RESULTS = EXPORTS / 'results.csv'
URL_CANDIDATES = ['https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules/sched_{season}.csv', 'https://raw.githubusercontent.com/nflverse/nflfastR-data/master/data/schedules/schedules_{season}.csv', 'https://raw.githubusercontent.com/nflverse/nflverse-data/master/schedules/sched_{season}.csv', 'https://raw.githubusercontent.com/nflverse/nflverse-data/master/schedules/schedules_{season}.csv']

def fetch_schedule_csv(season: int) -> pd.DataFrame:
    for url_tmpl in URL_CANDIDATES:
        url = url_tmpl.format(season=season)
        r = requests.get(url, timeout=30)
        if r.ok and r.text.strip():
            df = pd.read_csv(io.StringIO(r.text))
            df['__source__'] = url
            return df
    raise RuntimeError(f'Could not fetch schedule for {season} from nflverse.')

def build_results_from_schedule(df: pd.DataFrame, *, weeks: Optional[Iterable[int]]=None, since: Optional[str]=None) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    week = cols.get('week')
    gameday = cols.get('gameday') or cols.get('game_date') or cols.get('game_day') or cols.get('date')
    home = cols.get('home_team') or cols.get('home')
    away = cols.get('away_team') or cols.get('away')
    hs = cols.get('home_score') or cols.get('score_home') or cols.get('home_points')
    as_ = cols.get('away_score') or cols.get('score_away') or cols.get('away_points')
    finished = cols.get('game_finished') or cols.get('result') or cols.get('status')
    if not all([gameday, home, away, hs, as_]):
        raise ValueError('Schedule CSV missing required columns (need date/gameday, home_team, away_team, home_score, away_score)')
    out = df.copy()
    if weeks and week in out.columns:
        out = out[out[week].isin(list(weeks))]
    if finished and finished in out.columns:
        out = out[out[finished].astype(str).str.lower().isin(['true', 't', '1', 'final', 'post'])]
    out = out[pd.to_numeric(out[hs], errors='coerce').notna() & pd.to_numeric(out[as_], errors='coerce').notna()]
    if since:
        out_date = pd.to_datetime(out[gameday], errors='coerce').dt.date
        out = out[out_date >= pd.to_datetime(since).date()]
    out['game_date'] = pd.to_datetime(out[gameday], errors='coerce').dt.date
    out['home'] = out[home].astype(str).str.upper()
    out['away'] = out[away].astype(str).str.upper()
    out['home_score'] = pd.to_numeric(out[hs], errors='coerce').astype('Int64')
    out['away_score'] = pd.to_numeric(out[as_], errors='coerce').astype('Int64')
    out['game_id'] = out.apply(lambda r: f"{r['game_date']}-{r['away']}@{r['home']}", axis=1)
    out['total_points'] = out['home_score'].astype(float) + out['away_score'].astype(float)
    return out[['game_id', 'home_score', 'away_score', 'home', 'away', 'total_points']].dropna(subset=['home_score', 'away_score'])

def upsert_results(results_df: pd.DataFrame) -> int:
    if RESULTS.exists() and RESULTS.stat().st_size > 0:
        cur = pd.read_csv(RESULTS, encoding='utf-8-sig')
    else:
        cur = pd.DataFrame(columns=['game_id', 'home_score', 'away_score', 'home', 'away', 'total_points'])

    def norm_gid(g: str) -> str:
        return (g or '').strip().upper()
    cur['_key'] = cur['game_id'].map(norm_gid)
    results_df['_key'] = results_df['game_id'].map(norm_gid)
    merged = cur[~cur['_key'].isin(results_df['_key'])].drop(columns=['_key'], errors='ignore')
    merged = pd.concat([merged, results_df.drop(columns=['_key'], errors='ignore')], ignore_index=True)
    merged.to_csv(RESULTS, index=False, encoding='utf-8-sig')
    return len(results_df)

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', nargs='*', type=int, help='Seasons to fetch (e.g., 2024 2025)')
    ap.add_argument('--weeks', nargs='*', type=int, help='Limit to specific weeks')
    ap.add_argument('--since', type=str, help='Only include games on/after this date (YYYY-MM-DD)')
    args = ap.parse_args(argv)
    if not args.seasons and (not args.since):
        y = date.today().year
        args.seasons = [y - 1, y]
    total = 0
    for season in args.seasons or []:
        df = fetch_schedule_csv(season)
        out = build_results_from_schedule(df, weeks=args.weeks, since=args.since)
        total += upsert_results(out)
        print(f'Season {season}: upserted {len(out)} games')
    if args.since and (not args.seasons):
        y = pd.to_datetime(args.since).year
        for season in range(y, date.today().year + 1):
            df = fetch_schedule_csv(season)
            out = build_results_from_schedule(df, since=args.since)
            total += upsert_results(out)
            print(f'Season {season}: upserted {len(out)} games since {args.since}')
    print(f'Done. Wrote/updated: {RESULTS}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())