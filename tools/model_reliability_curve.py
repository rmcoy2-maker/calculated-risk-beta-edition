import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = "analysis_out/true_line_board_uncertainty.csv"
OUTPUT_TABLE = "analysis_out/model_reliability_table.csv"


def main():
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    required = ["model_prob", "actual_win"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["actual_win"] = pd.to_numeric(df["actual_win"], errors="coerce")

    df = df.dropna(subset=["model_prob", "actual_win"]).copy()

    df = df[(df["model_prob"] >= 0.40) & (df["model_prob"] <= 0.90)].copy()

    df["prob_bucket"] = pd.cut(
        df["model_prob"],
        bins=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        labels=["0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90"],
        include_lowest=True,
    )

    cal = (
        df.groupby("prob_bucket", observed=False)
          .agg(
              bets=("actual_win", "size"),
              predicted=("model_prob", "mean"),
              actual=("actual_win", "mean"),
          )
          .reset_index()
    )

    print("\nCALIBRATION TABLE\n")
    print(cal.to_string(index=False))

    cal.to_csv(OUTPUT_TABLE, index=False)
    print(f"\nSaved: {OUTPUT_TABLE}")

    plot_df = cal.dropna(subset=["predicted", "actual"]).copy()

    plt.figure(figsize=(7, 7))
    plt.plot(plot_df["predicted"], plot_df["actual"], marker="o")
    plt.plot([0.40, 0.90], [0.40, 0.90], linestyle="--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Win Rate")
    plt.title("Model Reliability Curve")
    plt.xlim(0.40, 0.90)
    plt.ylim(0.40, 0.90)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()