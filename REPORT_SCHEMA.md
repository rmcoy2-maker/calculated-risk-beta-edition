# Calculated Risk canonical weekly file schema

Use one stable file family per week:

- `cr_games_{season}_w{week}.csv`
- `cr_edges_{season}_w{week}.csv`
- `cr_parlay_scores_{season}_w{week}.csv`
- `cr_markets_{season}_w{week}.csv`
- `cr_scores_{season}_w{week}.csv`

Also keep rolling aliases for the app:

- `cr_games_latest.csv`
- `cr_edges_latest.csv`
- `cr_parlay_scores_latest.csv`
- `cr_markets_latest.csv`
- `cr_scores_latest.csv`

## Required columns

### `cr_games`
- `season`
- `week`
- `game_date`
- `game_id`
- `away_team`
- `home_team`

Recommended:
- `spread_close`
- `total_close`
- `ml_home`
- `ml_away`
- `stadium`
- `weather_temperature`
- `weather_wind_mph`
- `weather_detail`

### `cr_edges`
- `season`
- `week`
- `game_id`
- `market`
- `side`
- `odds`
- `p_win`
- `ev`

Recommended:
- `line`
- `book`
- `edge_pct`
- `notes`

### `cr_parlay_scores`
- `season`
- `week`
- `game_id`
- `market`
- `side`
- `legs`
- `parlay_proba`
- `ev`

Recommended:
- `parlay_score`
- `dec_comb`
- `book`

### `cr_markets`
- `season`
- `week`
- `game_id`
- `market`
- `side`
- `line`
- `odds`

Recommended:
- `book`
- `player_name`
- `prop_type`
- `captured_at`

### `cr_scores`
- `season`
- `week`
- `game_id`
- `away_team`
- `home_team`
- `away_score`
- `home_score`

## Naming rules

- Prefix every weekly file with `cr_`
- Use lowercase snake case
- Keep week as `w11`, `w12`, etc.
- Prefer one canonical `game_id` everywhere
- Use the same team abbreviations across all files

## Suggested pipeline

1. build `cr_games`
2. build `cr_markets`
3. build `cr_edges`
4. build `cr_parlay_scores`
5. build `cr_scores`
6. run `tools/generate_all_editions.py`
