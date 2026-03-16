import pandas as pd

df = pd.read_csv("analysis_out/true_line/true_line_board.csv")

print("\nCorrelation checks:\n")

print("model_prob vs close_prob:",
      df["model_prob"].corr(df["close_prob"]))

print("true_prob_cal vs close_prob:",
      df["true_prob_cal"].corr(df["close_prob"]))

print("true_prob_shrunk_40 vs close_prob:",
      df["true_prob_shrunk_40"].corr(df["close_prob"]))