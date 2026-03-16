import pandas as pd
import numpy as np

BETS_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\backtest_bets_moneyline.csv"
MARKET_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\historical_odds\nfl_open_mid_close_odds.csv"
OUT_ALL_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\fort_knox_market_joined_moneyline.csv"
OUT_FK_PATH = r"C:\Projects\calculated-risk-beta-edition\exports\fort_knox_top3_market_joined_moneyline.csv"

TEAM_MAP = {
    "49ers": "San Francisco 49ers",
    "Bears": "Chicago Bears",
    "Bengals": "Cincinnati Bengals",
    "Bills": "Buffalo Bills",
    "Broncos": "Denver Broncos",
    "Browns": "Cleveland Browns",
    "Buccaneers": "Tampa Bay Buccaneers",
    "Cardinals": "Arizona Cardinals",
    "Chargers": "Los Angeles Chargers",
    "Chiefs": "Kansas City Chiefs",
    "Colts": "Indianapolis Colts",
    "Commanders": "Washington Commanders",
    "Cowboys": "Dallas Cowboys",
    "Dolphins": "Miami Dolphins",
    "Eagles": "Philadelphia Eagles",
    "Falcons": "Atlanta Falcons",
    "Giants": "New York Giants",
    "Jaguars": "Jacksonville Jaguars",
    "Jets": "New York Jets",
    "Lions": "Detroit Lions",
    "Packers": "Green Bay Packers",
    "Panthers": "Carolina Panthers",
    "Patriots": "New England Patriots",
    "Raiders": "Las Vegas Raiders",
    "Rams": "Los Angeles Rams",
    "Ravens": "Baltimore Ravens",
    "Saints": "New Orleans Saints",
    "Seahawks": "Seattle Seahawks",
    "Steelers": "Pittsburgh Steelers",
    "Texans": "Houston Texans",
    "Titans": "Tennessee Titans",
    "Vikings": "Minnesota Vikings",
}


def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return np.nan


def prob_to_american(p):
    if pd.isna(p) or p <= 0 or p >= 1:
        return np.nan
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def main():
    print("Loading bets...")
    bets = pd.read_csv(BETS_PATH)

    print("Loading market open/mid/close file...")
    market = pd.read_csv(MARKET_PATH)

    print("Filtering moneyline market...")
    market = market[market["market_key"] == "h2h"].copy()

    print("Normalizing dates...")
    bets["game_date"] = pd.to_datetime(bets["game_date"]).dt.date
    market["commence_date"] = pd.to_datetime(market["commence_time"]).dt.date

    print("Building selected team from home/away flag...")
    bets["selected_team"] = np.where(
        bets["side"].astype(str).str.lower() == "home",
        bets["home_team"],
        bets["away_team"]
    )

    print("Mapping team nicknames to full names...")
    bets["home_team_full"] = bets["home_team"].map(TEAM_MAP)
    bets["away_team_full"] = bets["away_team"].map(TEAM_MAP)
    bets["selected_team_full"] = bets["selected_team"].map(TEAM_MAP)

    missing_home = bets["home_team_full"].isna().sum()
    missing_away = bets["away_team_full"].isna().sum()
    missing_sel = bets["selected_team_full"].isna().sum()

    print("Missing team mappings:")
    print("home_team_full:", missing_home)
    print("away_team_full:", missing_away)
    print("selected_team_full:", missing_sel)

    market["home_team"] = market["home_team"].astype(str).str.strip()
    market["away_team"] = market["away_team"].astype(str).str.strip()
    market["team"] = market["outcome_name"].astype(str).str.strip()

    print("Joining bets to market rows...")
    merged = bets.merge(
        market,
        left_on=["game_date", "home_team_full", "away_team_full", "selected_team_full"],
        right_on=["commence_date", "home_team", "away_team", "team"],
        how="left",
        suffixes=("_bet", "_market")
    )

    print("Raw merged rows:", len(merged))
    print("Matched market rows:", merged["close_price"].notna().sum())

    print("Converting prices to implied probabilities...")
    for src_col, dst_col in [
        ("open_price", "open_prob"),
        ("mid_price", "mid_prob"),
        ("close_price", "close_prob"),
        ("closing_odds", "bet_close_prob"),
    ]:
        merged[dst_col] = merged[src_col].apply(american_to_prob)

    print("Collapsing to one row per bet using consensus across books...")
    bet_key = [
        "game_id",
        "Season",
        "Week",
        "game_date",
        "home_team_bet",
        "away_team_bet",
        "selected_team",
    ]

    collapsed = merged.groupby(bet_key, as_index=False).agg(
        profit=("profit", "first"),
        actual_win=("actual_win", "first"),
        model_prob=("model_prob", "first"),
        edge=("edge", "first"),
        closing_odds=("closing_odds", "first"),
        implied_prob=("implied_prob", "first"),
        books_used=("books_used", "first"),
        selected_team_full=("selected_team_full", "first"),
        open_prob=("open_prob", "mean"),
        mid_prob=("mid_prob", "mean"),
        close_prob=("close_prob", "mean"),
        open_price=("open_price", "median"),
        mid_price=("mid_price", "median"),
        close_price=("close_price", "median"),
        mid_diff_seconds=("mid_diff_seconds", "mean"),
        match_rows=("team", "count"),
    )

    collapsed["mid_diff_hours"] = collapsed["mid_diff_seconds"] / 3600.0
    collapsed["mid_valid_24h"] = collapsed["mid_diff_hours"] <= 24
    collapsed["mid_valid_36h"] = collapsed["mid_diff_hours"] <= 36
    collapsed["mid_valid_48h"] = collapsed["mid_diff_hours"] <= 48

    print("Computing model edge vs market states...")
    collapsed["model_edge_vs_open"] = collapsed["model_prob"] - collapsed["open_prob"]
    collapsed["model_edge_vs_mid"] = collapsed["model_prob"] - collapsed["mid_prob"]
    collapsed["model_edge_vs_close"] = collapsed["model_prob"] - collapsed["close_prob"]

    print("Computing fair moneyline from model probability...")
    collapsed["true_moneyline"] = collapsed["model_prob"].apply(prob_to_american)

    print("Computing market movement...")
    collapsed["open_to_mid_prob_move"] = collapsed["mid_prob"] - collapsed["open_prob"]
    collapsed["mid_to_close_prob_move"] = collapsed["close_prob"] - collapsed["mid_prob"]
    collapsed["open_to_close_prob_move"] = collapsed["close_prob"] - collapsed["open_prob"]

    print("Ranking weekly edges...")
    collapsed["edge_rank"] = collapsed.groupby(["Season", "Week"])["edge"].rank(
        method="first",
        ascending=False
    )
    collapsed["hit"] = (collapsed["profit"] > 0).astype(int)

    print("Creating Fort Knox Top 3 subset...")
    fort_knox = collapsed[collapsed["edge_rank"] <= 3].copy()

    print("Saving outputs...")
    collapsed.to_csv(OUT_ALL_PATH, index=False)
    fort_knox.to_csv(OUT_FK_PATH, index=False)

    print(f"Saved all bets: {OUT_ALL_PATH}")
    print(f"Saved Top 3 Fort Knox bets: {OUT_FK_PATH}")

    print("\nOverall summary")
    print("Unique bets:", len(collapsed))
    print("Matched consensus close prob:", collapsed["close_prob"].notna().sum())
    print("Avg model edge vs open:", collapsed["model_edge_vs_open"].mean())
    print("Avg model edge vs mid:", collapsed["model_edge_vs_mid"].mean())
    print("Avg model edge vs close:", collapsed["model_edge_vs_close"].mean())
    print("Positive edge vs close %:", (collapsed["model_edge_vs_close"] > 0).mean())
    print("Avg open->mid prob move:", collapsed["open_to_mid_prob_move"].mean())
    print("Avg mid->close prob move:", collapsed["mid_to_close_prob_move"].mean())
    print("Avg open->close prob move:", collapsed["open_to_close_prob_move"].mean())

    print("\nMID validity")
    print("<=24h:", collapsed["mid_valid_24h"].mean())
    print("<=36h:", collapsed["mid_valid_36h"].mean())
    print("<=48h:", collapsed["mid_valid_48h"].mean())

    print("\nFort Knox Top 3 summary")
    print("Bets:", len(fort_knox))
    if len(fort_knox):
        print("Hit rate:", fort_knox["hit"].mean())
        print("ROI:", fort_knox["profit"].sum() / len(fort_knox))
        print("Avg model edge vs open:", fort_knox["model_edge_vs_open"].mean())
        print("Avg model edge vs mid:", fort_knox["model_edge_vs_mid"].mean())
        print("Avg model edge vs close:", fort_knox["model_edge_vs_close"].mean())
        print("Positive edge vs close %:", (fort_knox["model_edge_vs_close"] > 0).mean())
        print("Avg open->mid prob move:", fort_knox["open_to_mid_prob_move"].mean())
        print("Avg mid->close prob move:", fort_knox["mid_to_close_prob_move"].mean())
        print("Avg open->close prob move:", fort_knox["open_to_close_prob_move"].mean())
        print("MID valid <=36h %:", fort_knox["mid_valid_36h"].mean())

    print("\nTop 10 rows:")
    cols = [
        "game_id",
        "Season",
        "Week",
        "selected_team",
        "model_prob",
        "edge",
        "closing_odds",
        "true_moneyline",
        "open_price",
        "mid_price",
        "close_price",
        "model_edge_vs_open",
        "model_edge_vs_mid",
        "model_edge_vs_close",
        "edge_rank",
        "profit",
    ]
    print(fort_knox[cols].head(10))


if __name__ == "__main__":
    main()