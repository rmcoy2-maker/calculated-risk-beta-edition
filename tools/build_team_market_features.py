from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------------
# Project / file discovery
# ------------------------------
def repo_root() -> Path:
    here = Path.cwd()
    if (here / 'streamlit_app.py').exists():
        return here
    for p in [here] + list(here.parents):
        if (p / 'streamlit_app.py').exists():
            return p
    return here


def find_input(exports_dir: Path) -> Path:
    candidates = [
        exports_dir / 'games_master_with_market_probs.csv',
        exports_dir / 'fort_knox_market_joined_moneyline.csv',
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        'Could not find games_master_with_market_probs.csv or fort_knox_market_joined_moneyline.csv in exports/'
    )


# ------------------------------
# Flexible column helpers
# ------------------------------
def pick_col(df: pd.DataFrame, options: list[str], required: bool = True) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt in df.columns:
            return opt
        key = str(opt).lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise KeyError(f'Missing required column. Tried: {options}. Found columns: {list(df.columns)}')
    return None


def pick_series(df: pd.DataFrame, options: list[str]) -> pd.Series:
    col = pick_col(df, options, required=False)
    if col is None:
        return pd.Series([None] * len(df), index=df.index, dtype='object')
    return df[col]



def split_matchup_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype('string').fillna('').str.strip()

    away = pd.Series([None] * len(s), index=s.index, dtype='object')
    home = pd.Series([None] * len(s), index=s.index, dtype='object')

    mask_at = s.str.contains(r'\s@\s', regex=True, na=False)
    if mask_at.any():
        parts = s[mask_at].str.split(r'\s@\s', n=1, expand=True)
        away.loc[mask_at] = parts[0]
        home.loc[mask_at] = parts[1]

    mask_vs = s.str.contains(r'\svs\.?\s', regex=True, na=False)
    if mask_vs.any():
        parts = s[mask_vs].str.split(r'\svs\.?\s', n=1, expand=True)
        away.loc[mask_vs] = parts[0]
        home.loc[mask_vs] = parts[1]

    return home, away



def ensure_home_away_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    home = pick_series(out, [
        'home', 'home_team', 'team_home',
        'HomeTeam', 'Home_Team',
        'homeName', 'Home', 'HOME',
        'host', 'home_name', 'team1_home', 'homeTeam',
    ])
    away = pick_series(out, [
        'away', 'away_team', 'team_away',
        'AwayTeam', 'Away_Team',
        'awayName', 'Away', 'AWAY',
        'visitor', 'away_name', 'team1_away', 'awayTeam',
    ])

    if home.isna().all() or (home.astype('string').fillna('').str.strip() == '').all():
        cand = pick_series(out, ['team2', 'Team2', 'team_b', 'opponent2'])
        if not cand.isna().all():
            home = cand

    if away.isna().all() or (away.astype('string').fillna('').str.strip() == '').all():
        cand = pick_series(out, ['team1', 'Team1', 'team_a', 'opponent1'])
        if not cand.isna().all():
            away = cand

    if home.isna().all() and away.isna().all():
        winner = pick_series(out, ['winner', 'winning_team'])
        loser = pick_series(out, ['loser', 'losing_team'])
        if not winner.isna().all() and not loser.isna().all():
            home = winner
            away = loser

    if home.isna().all() and away.isna().all():
        matchup = pick_series(out, ['matchup', 'game', 'fixture', 'teams', 'event'])
        if not matchup.isna().all():
            parsed_home, parsed_away = split_matchup_column(matchup)
            home = parsed_home
            away = parsed_away

    out['home'] = home.astype('string').str.strip()
    out['away'] = away.astype('string').str.strip()

    out.loc[out['home'].isin(['', 'nan', 'None', '<NA>']), 'home'] = pd.NA
    out.loc[out['away'].isin(['', 'nan', 'None', '<NA>']), 'away'] = pd.NA

    return out



def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['home_score'] = pd.to_numeric(
        pick_series(out, [
            'home_score', 'HomeScore', 'Home_Score',
            'score_home', 'final_home', 'home_points', 'home_pts',
        ]),
        errors='coerce',
    )
    out['away_score'] = pd.to_numeric(
        pick_series(out, [
            'away_score', 'AwayScore', 'Away_Score',
            'score_away', 'final_away', 'away_points', 'away_pts',
        ]),
        errors='coerce',
    )
    return out



def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = pick_col(out, ['_DateISO', '_dateiso', 'game_date', 'Date', 'date', 'gamedate'])
    out['game_date_norm'] = pd.to_datetime(out[date_col], errors='coerce')
    return out



def ensure_market_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out['market_prob_home'] = pd.to_numeric(
        pick_series(out, [
            'market_prob_home', 'home_market_prob', 'implied_prob_home',
            'prob_home', 'home_win_prob', 'home_implied_prob',
        ]),
        errors='coerce',
    )
    out['market_prob_away'] = pd.to_numeric(
        pick_series(out, [
            'market_prob_away', 'away_market_prob', 'implied_prob_away',
            'prob_away', 'away_win_prob', 'away_implied_prob',
        ]),
        errors='coerce',
    )

    if out['market_prob_home'].isna().all() or out['market_prob_away'].isna().all():
        raise KeyError(
            'Could not find usable market probability columns. '
            'Expected one of market_prob_home/home_market_prob/implied_prob_home and '
            'market_prob_away/away_market_prob/implied_prob_away.'
        )

    return out



def normalize_team(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)


# ------------------------------
# Feature engineering
# ------------------------------
def add_pre_game_team_features(df: pd.DataFrame) -> pd.DataFrame:
    game_id_col = pick_col(df, ['game_id', 'GameID', 'id'], required=False)
    season_col = pick_col(df, ['Season', 'season'], required=False)
    week_col = pick_col(df, ['Week', 'week'], required=False)

    work = df.copy()
    work = ensure_home_away_columns(work)
    work = ensure_score_columns(work)
    work = ensure_date_column(work)
    work = ensure_market_prob_columns(work)

    work['home'] = normalize_team(work['home'])
    work['away'] = normalize_team(work['away'])

    # Keep only rows with enough information to build historical features
    work = work.copy()
    work = work.loc[
        work['home'].notna()
        & work['away'].notna()
        & work['game_date_norm'].notna()
    ].reset_index(drop=True)

    sort_cols = ['game_date_norm']
    if season_col:
        sort_cols.append(season_col)
    if week_col:
        sort_cols.append(week_col)
    if game_id_col:
        sort_cols.append(game_id_col)

    work = work.sort_values(sort_cols).reset_index(drop=True)
    work['_row_id'] = np.arange(len(work))

    home_score = pd.to_numeric(work['home_score'], errors='coerce')
    away_score = pd.to_numeric(work['away_score'], errors='coerce')

    # Only final scored games should contribute actual outcomes.
    # Feature rows for future games will still receive prior-history features.
    work['_home_win'] = np.where(
        home_score.notna() & away_score.notna(),
        (home_score > away_score).astype(int),
        np.nan,
    )
    work['_away_win'] = np.where(
        home_score.notna() & away_score.notna(),
        (away_score > home_score).astype(int),
        np.nan,
    )

    home_long = pd.DataFrame({
        '_row_id': work['_row_id'],
        'team': work['home'],
        'opp': work['away'],
        'is_home': 1,
        'game_date': work['game_date_norm'],
        'market_expected': work['market_prob_home'],
        'actual_win': work['_home_win'],
    })
    away_long = pd.DataFrame({
        '_row_id': work['_row_id'],
        'team': work['away'],
        'opp': work['home'],
        'is_home': 0,
        'game_date': work['game_date_norm'],
        'market_expected': work['market_prob_away'],
        'actual_win': work['_away_win'],
    })

    long = pd.concat([home_long, away_long], ignore_index=True)
    long = long.sort_values(['team', 'game_date', '_row_id']).reset_index(drop=True)

    # Games with missing final score should not advance the historical record.
    long['is_scored_game'] = long['actual_win'].notna().astype(int)
    long['game_diff'] = long['actual_win'] - long['market_expected']
    long.loc[long['is_scored_game'] == 0, 'game_diff'] = np.nan

    g = long.groupby('team', sort=False)

    # Count only prior scored games.
    long['tm_games_before'] = g['is_scored_game'].cumsum() - long['is_scored_game']
    long['tm_actual_wins_before'] = g['actual_win'].transform(lambda s: s.fillna(0).cumsum()) - long['actual_win'].fillna(0)
    long['tm_expected_wins_before'] = g['market_expected'].transform(lambda s: s.where(long.loc[s.index, 'is_scored_game'] == 1, 0).cumsum())
    long['tm_expected_wins_before'] = long['tm_expected_wins_before'] - np.where(long['is_scored_game'] == 1, long['market_expected'], 0)

    long['tm_diff_before'] = long['tm_actual_wins_before'] - long['tm_expected_wins_before']
    long['tm_diff_per_game_before'] = np.where(
        long['tm_games_before'] > 0,
        long['tm_diff_before'] / long['tm_games_before'],
        np.nan,
    )
    long['tm_signal_strength_before'] = np.where(
        long['tm_games_before'] > 0,
        np.abs(long['tm_diff_per_game_before']) * np.sqrt(long['tm_games_before']),
        np.nan,
    )

    shifted_diff = g['game_diff'].shift(1)
    long['tm_diff_last4_before'] = (
        shifted_diff.groupby(long['team']).rolling(4, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    long['tm_diff_last8_before'] = (
        shifted_diff.groupby(long['team']).rolling(8, min_periods=1).sum().reset_index(level=0, drop=True)
    )

    long['tm_tier_before'] = 'Average'
    long.loc[long['tm_diff_per_game_before'] >= 0.30, 'tm_tier_before'] = 'Elite Overperformer'
    long.loc[
        (long['tm_diff_per_game_before'] >= 0.10) & (long['tm_diff_per_game_before'] < 0.30),
        'tm_tier_before',
    ] = 'Slight Overperformer'
    long.loc[
        (long['tm_diff_per_game_before'] <= -0.10) & (long['tm_diff_per_game_before'] > -0.30),
        'tm_tier_before',
    ] = 'Slight Underperformer'
    long.loc[long['tm_diff_per_game_before'] <= -0.30, 'tm_tier_before'] = 'Poor Underperformer'

    long['tm_regression_trap_before'] = (
        (long['tm_diff_before'] > 0) &
        (long['tm_diff_last4_before'].fillna(0) < 0)
    ).astype(int)

    home_feats = long.loc[long['is_home'] == 1, [
        '_row_id', 'team', 'opp',
        'tm_games_before', 'tm_actual_wins_before', 'tm_expected_wins_before',
        'tm_diff_before', 'tm_diff_per_game_before', 'tm_signal_strength_before',
        'tm_diff_last4_before', 'tm_diff_last8_before', 'tm_tier_before',
        'tm_regression_trap_before',
    ]].rename(columns={
        'team': 'home_team_norm',
        'opp': 'away_team_norm',
        'tm_games_before': 'home_tm_games_before',
        'tm_actual_wins_before': 'home_tm_actual_wins_before',
        'tm_expected_wins_before': 'home_tm_expected_wins_before',
        'tm_diff_before': 'home_tm_diff_before',
        'tm_diff_per_game_before': 'home_tm_diff_per_game_before',
        'tm_signal_strength_before': 'home_tm_signal_strength_before',
        'tm_diff_last4_before': 'home_tm_diff_last4_before',
        'tm_diff_last8_before': 'home_tm_diff_last8_before',
        'tm_tier_before': 'home_tm_tier_before',
        'tm_regression_trap_before': 'home_tm_regression_trap_before',
    })

    away_feats = long.loc[long['is_home'] == 0, [
        '_row_id', 'team', 'opp',
        'tm_games_before', 'tm_actual_wins_before', 'tm_expected_wins_before',
        'tm_diff_before', 'tm_diff_per_game_before', 'tm_signal_strength_before',
        'tm_diff_last4_before', 'tm_diff_last8_before', 'tm_tier_before',
        'tm_regression_trap_before',
    ]].rename(columns={
        'team': 'away_team_norm',
        'opp': 'home_team_norm',
        'tm_games_before': 'away_tm_games_before',
        'tm_actual_wins_before': 'away_tm_actual_wins_before',
        'tm_expected_wins_before': 'away_tm_expected_wins_before',
        'tm_diff_before': 'away_tm_diff_before',
        'tm_diff_per_game_before': 'away_tm_diff_per_game_before',
        'tm_signal_strength_before': 'away_tm_signal_strength_before',
        'tm_diff_last4_before': 'away_tm_diff_last4_before',
        'tm_diff_last8_before': 'away_tm_diff_last8_before',
        'tm_tier_before': 'away_tm_tier_before',
        'tm_regression_trap_before': 'away_tm_regression_trap_before',
    })

    merged = work.merge(home_feats, on='_row_id', how='left').merge(away_feats, on='_row_id', how='left')

    merged['net_tm_diff_before'] = merged['home_tm_diff_before'] - merged['away_tm_diff_before']
    merged['net_tm_diff_per_game_before'] = merged['home_tm_diff_per_game_before'] - merged['away_tm_diff_per_game_before']
    merged['net_tm_diff_last4_before'] = merged['home_tm_diff_last4_before'] - merged['away_tm_diff_last4_before']
    merged['net_tm_diff_last8_before'] = merged['home_tm_diff_last8_before'] - merged['away_tm_diff_last8_before']
    merged['net_tm_signal_strength_before'] = merged['home_tm_signal_strength_before'] - merged['away_tm_signal_strength_before']
    merged['either_tm_regression_trap_before'] = (
        merged['home_tm_regression_trap_before'].fillna(0).astype(int)
        | merged['away_tm_regression_trap_before'].fillna(0).astype(int)
    ).astype(int)

    drop_cols = ['_row_id', '_home_win', '_away_win']
    return merged.drop(columns=[c for c in drop_cols if c in merged.columns])


if __name__ == '__main__':
    root = repo_root()
    exports = root / 'exports'
    input_path = find_input(exports)
    df = pd.read_csv(input_path, low_memory=False, encoding='utf-8-sig')

    print('Input columns found:')
    print(df.columns.tolist())

    out = add_pre_game_team_features(df)
    out_path = exports / 'games_master_with_team_market_features.csv'
    out.to_csv(out_path, index=False)

    print(f'Input:  {input_path}')
    print(f'Output: {out_path}')
    print(f'Rows:   {len(out):,}')
    print('Added columns:')
    for c in [
        'home_tm_diff_before', 'away_tm_diff_before', 'net_tm_diff_before',
        'home_tm_diff_per_game_before', 'away_tm_diff_per_game_before', 'net_tm_diff_per_game_before',
        'home_tm_diff_last4_before', 'away_tm_diff_last4_before', 'net_tm_diff_last4_before',
        'home_tm_diff_last8_before', 'away_tm_diff_last8_before', 'net_tm_diff_last8_before',
        'home_tm_signal_strength_before', 'away_tm_signal_strength_before', 'net_tm_signal_strength_before',
        'home_tm_tier_before', 'away_tm_tier_before', 'either_tm_regression_trap_before',
    ]:
        if c in out.columns:
            print(f'  - {c}')
