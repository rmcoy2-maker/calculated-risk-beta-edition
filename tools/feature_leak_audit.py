from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES = PROJECT_ROOT / "exports" / "model_features.csv"
GAMES = PROJECT_ROOT / "exports" / "games_master.csv"


def main():

    feats = pd.read_csv(FEATURES)
    games = pd.read_csv(GAMES)

    print("\nFEATURE DATASET SHAPE")
    print(feats.shape)

    print("\nFEATURE SAMPLE")
    print(feats.head())

    # -----------------------------
    # Merge game dates
    # -----------------------------
    games_small = games[["game_id", "game_date"]]

    df = feats.merge(games_small, on="game_id", how="left")

    df["game_date"] = pd.to_datetime(df["game_date"])

    # -----------------------------
    # Detect suspicious columns
    # -----------------------------
    suspicious_cols = [
        c for c in df.columns
        if any(x in c.lower() for x in [
            "score",
            "result",
            "win",
            "loss",
            "margin",
            "points_for",
            "points_against"
        ])
    ]

    print("\nPOSSIBLE LEAKAGE COLUMNS")
    print(suspicious_cols)

    # -----------------------------
    # Check if any columns perfectly predict winner
    # -----------------------------
    if "home_win" in df.columns:

        print("\nCHECKING CORRELATION WITH RESULT")

        corr = (
            df.select_dtypes(include=["number"])
            .corr()["home_win"]
            .sort_values(ascending=False)
        )

        print(corr.head(20))

    # -----------------------------
    # Check duplicate game rows
    # -----------------------------
    dupes = df["game_id"].value_counts()

    print("\nMAX ROWS PER GAME")
    print(dupes.max())

    print("\nROWS PER GAME DISTRIBUTION")
    print(dupes.value_counts().sort_index())

    # -----------------------------
    # Check missing values
    # -----------------------------
    print("\nMISSING VALUES (TOP 20)")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nLEAK AUDIT COMPLETE")


if __name__ == "__main__":
    main()
