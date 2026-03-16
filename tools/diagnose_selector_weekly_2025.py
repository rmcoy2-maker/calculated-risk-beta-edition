import pandas as pd

INPUT_FILE = "analysis_out/uncertainty_selector_bets.csv"


def main():

    df = pd.read_csv(INPUT_FILE, low_memory=False)

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

    df_2025 = df[df["Season"] == 2025].copy()

    weekly = (
        df_2025
        .groupby("Week")
        .agg(
            bets=("profit", "size"),
            wins=("actual_win", "sum"),
            hit_rate=("actual_win", "mean"),
            roi=("profit", "mean"),
            avg_market_prob=("market_prob", "mean"),
            avg_edge_governed=("edge_prob_governed", "mean"),
            avg_close_edge=("model_edge_vs_close", "mean"),
        )
        .reset_index()
        .sort_values("Week")
    )

    print("\n2025 WEEKLY SELECTOR PERFORMANCE\n")
    print(weekly.to_string(index=False))

    weekly.to_csv(
        "analysis_out/2025_weekly_selector_diagnostics.csv",
        index=False
    )

    print("\nSaved:")
    print("analysis_out/2025_weekly_selector_diagnostics.csv")


if __name__ == "__main__":
    main()