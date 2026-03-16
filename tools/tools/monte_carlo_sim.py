import numpy as np


def simulate_game(home_mean, away_mean, spread=None, total=None, sims=50000):

    home_mean = float(home_mean)
    away_mean = float(away_mean)

    home_scores = np.random.poisson(lam=max(home_mean, 0.01), size=sims)
    away_scores = np.random.poisson(lam=max(away_mean, 0.01), size=sims)

    margins = home_scores - away_scores
    totals = home_scores + away_scores

    results = {}

    results["home_win_prob"] = float(np.mean(margins > 0))
    results["away_win_prob"] = float(np.mean(margins < 0))

    results["median_home"] = int(np.median(home_scores))
    results["median_away"] = int(np.median(away_scores))

    results["margin_p10"] = float(np.percentile(margins, 10))
    results["margin_p50"] = float(np.percentile(margins, 50))
    results["margin_p90"] = float(np.percentile(margins, 90))

    results["total_p10"] = float(np.percentile(totals, 10))
    results["total_p50"] = float(np.percentile(totals, 50))
    results["total_p90"] = float(np.percentile(totals, 90))

    if spread is not None:
        try:
            spread = float(spread)
            spread_result = margins + spread
            results["home_cover_prob"] = float(np.mean(spread_result > 0))
            results["away_cover_prob"] = float(np.mean(spread_result < 0))
        except:
            pass

    if total is not None:
        try:
            total = float(total)
            results["over_prob"] = float(np.mean(totals > total))
            results["under_prob"] = float(np.mean(totals < total))
        except:
            pass

    return results