from __future__ import annotations

import pandas as pd

from monte_carlo_sim import simulate_game


def confidence_band(conf: float) -> str:
    if conf >= 90:
        return "Gold"
    if conf >= 75:
        return "Dark Green"
    if conf >= 60:
        return "Green"
    if conf >= 45:
        return "Yellow"
    if conf >= 25:
        return "Amber"
    return "Red"


def band_emoji(conf: float) -> str:
    if conf >= 90:
        return "🟨🟨"
    if conf >= 75:
        return "🟩🟩"
    if conf >= 60:
        return "🟩"
    if conf >= 45:
        return "🟨"
    if conf >= 25:
        return "🟧"
    return "🟥"


def _get(row: pd.Series, key: str, default=""):
    return row[key] if key in row and pd.notna(row[key]) else default


def _to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _pct_text(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def _pretty_line(value) -> str:
    if value is None or value == "" or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.1f}"
    except Exception:
        return str(value)


def _clean_str(value, default=""):
    if value is None or pd.isna(value):
        return default
    s = str(value).strip()
    if s.lower() in {"nan", "none", ""}:
        return default
    return s


def _infer_matchup(row: pd.Series) -> str:
    matchup = _clean_str(_get(row, "matchup", ""))
    if matchup:
        return matchup

    away = _clean_str(_get(row, "away_team", ""))
    home = _clean_str(_get(row, "home_team", ""))
    if away and home:
        return f"{away} @ {home}"

    game_id = _clean_str(_get(row, "game_id", ""))
    if game_id:
        parts = game_id.split("_")
        if len(parts) >= 4:
            return f"{parts[-2]} @ {parts[-1]}"

    return "Unknown matchup"


def _infer_play_label(row: pd.Series) -> str:
    matchup = _infer_matchup(row)
    label = _clean_str(_get(row, "label", ""))
    if label and "parlay leg" not in label.lower():
        if "|" in label:
            return label
        return f"{matchup} | {label}"

    market = _clean_str(_get(row, "market", "moneyline")).lower()
    side = _clean_str(_get(row, "side", ""))
    line = _get(row, "line", _get(row, "market_spread", _get(row, "market_total", "")))

    if market == "moneyline":
        text = f"{side} ML" if side else "Moneyline"
    elif market == "spread":
        try:
            text = f"{side} {float(line):+.1f}" if side else f"Spread {float(line):+.1f}"
        except Exception:
            text = f"{side} spread" if side else "Spread"
    elif market == "total":
        try:
            if side:
                text = f"{side.title()} {float(line):.1f}"
            else:
                text = f"Total {float(line):.1f}"
        except Exception:
            text = f"{side} total" if side else "Total"
    elif "team_total" in market:
        try:
            text = f"{side} Team Total {float(line):.1f}" if side else f"Team Total {float(line):.1f}"
        except Exception:
            text = f"{side} team total" if side else "Team Total"
    else:
        text = f"{side} {market}".strip() if side else (market or "Edge")

    return f"{matchup} | {text}" if matchup else text


def _market_context_sentence(row: pd.Series) -> str:
    market = _clean_str(_get(row, "market", "moneyline")).lower()
    side = _clean_str(_get(row, "side", ""))
    spread = _to_float(_get(row, "market_spread", _get(row, "spread", None)), None)
    total = _to_float(_get(row, "market_total", _get(row, "total", None)), None)

    if market == "moneyline":
        return "This is a straight-up win position."
    if market == "spread":
        if spread is not None:
            return f"Current spread context is {_pretty_line(spread)} on the home side."
        return "This is a spread-based position."
    if market == "total":
        if total is not None:
            return f"Current total context is {_pretty_line(total)}."
        return "This is a total-based position."
    if "team_total" in market:
        return f"This is a team-total position tied to {side or 'one side'}."
    return "This is a market-derived position."


def edge_blurb(row: pd.Series) -> str:
    play_label = _infer_play_label(row)
    confidence = _to_float(_get(row, "confidence", 50), 50.0)
    edge_score = _to_float(_get(row, "edge_score", _get(row, "edge", 0)), 0.0)

    market_type = _clean_str(_get(row, "market_type", _get(row, "market", "moneyline"))).lower()
    side = _clean_str(_get(row, "side", ""))
    spread = _to_float(_get(row, "market_spread", _get(row, "spread", None)), None)
    total = _to_float(_get(row, "market_total", _get(row, "total", None)), None)
    away_score = _to_float(_get(row, "model_away_score", None), None)
    home_score = _to_float(_get(row, "model_home_score", None), None)

    opener = (
        f"{play_label}. "
        f"Confidence {confidence:.0f} — {band_emoji(confidence)} {confidence_band(confidence)}. "
    )

    if away_score is not None and home_score is not None:
        sims = simulate_game(
            home_mean=home_score,
            away_mean=away_score,
            spread=spread,
            total=total,
            sims=20000,
        )

        if market_type == "moneyline":
            if side.lower() == _clean_str(_get(row, "home_team", "")).lower() or side.lower() == "home":
                sim_line = f"Home win rate is {_pct_text(sims['home_win_prob'])}."
            elif side.lower() == _clean_str(_get(row, "away_team", "")).lower() or side.lower() == "away":
                sim_line = f"Away win rate is {_pct_text(sims['away_win_prob'])}."
            else:
                sim_line = (
                    f"Home win rate {_pct_text(sims['home_win_prob'])}; "
                    f"away win rate {_pct_text(sims['away_win_prob'])}."
                )

            return (
                opener
                + f"Median score: {sims['median_home']} to {sims['median_away']}. "
                + sim_line
                + f"Margin band runs {sims['margin_p10']:.1f} to {sims['margin_p90']:.1f}. "
                + _market_context_sentence(row)
            )

        if market_type == "spread":
            if side.lower() == _clean_str(_get(row, "home_team", "")).lower() or side.lower() == "home":
                cover_line = f"Home cover probability is {_pct_text(sims.get('home_cover_prob'))}."
            elif side.lower() == _clean_str(_get(row, "away_team", "")).lower() or side.lower() == "away":
                cover_line = f"Away cover probability is {_pct_text(sims.get('away_cover_prob'))}."
            else:
                cover_line = (
                    f"Home cover {_pct_text(sims.get('home_cover_prob'))}; "
                    f"away cover {_pct_text(sims.get('away_cover_prob'))}."
                )

            return (
                opener
                + f"Median score: {sims['median_home']} to {sims['median_away']}. "
                + cover_line
                + f"Projected margin band runs {sims['margin_p10']:.1f} to {sims['margin_p90']:.1f}. "
                + _market_context_sentence(row)
            )

        if market_type == "total":
            if side.lower() == "over":
                total_line = f"Over hits in {_pct_text(sims.get('over_prob'))} of sims."
            elif side.lower() == "under":
                total_line = f"Under hits in {_pct_text(sims.get('under_prob'))} of sims."
            else:
                total_line = (
                    f"Over {_pct_text(sims.get('over_prob'))}; "
                    f"under {_pct_text(sims.get('under_prob'))}."
                )

            return (
                opener
                + f"Median total is {sims['total_p50']:.1f}. "
                + total_line
                + f"Scoring band runs {sims['total_p10']:.1f} to {sims['total_p90']:.1f}. "
                + _market_context_sentence(row)
            )

        return (
            opener
            + f"Median score: {sims['median_home']} to {sims['median_away']}. "
            + f"Home win probability is {_pct_text(sims['home_win_prob'])}. "
            + _market_context_sentence(row)
        )

    if confidence >= 75:
        base = "Premium-grade edge with stronger support than the average board play."
    elif confidence >= 60:
        base = "Playable edge with enough support to stay on the card."
    elif edge_score >= 0.02:
        base = "Smaller but still measurable edge that needs disciplined pricing."
    else:
        base = "This is more informational than aggressive."

    return opener + base + " " + _market_context_sentence(row)


def game_script(row: pd.Series) -> str:
    matchup = _infer_matchup(row)
    away = _clean_str(_get(row, "away_team", "Away"))
    home = _clean_str(_get(row, "home_team", "Home"))

    away_score = _to_float(_get(row, "model_away_score", None), None)
    home_score = _to_float(_get(row, "model_home_score", None), None)
    total = _to_float(_get(row, "market_total", _get(row, "total", None)), None)
    spread = _to_float(_get(row, "market_spread", _get(row, "spread", None)), None)

    if away_score is None or home_score is None:
        return (
            f"{matchup} has limited projection detail in the current report inputs. "
            f"Canonical games still need usable model_home_score and model_away_score."
        )

    sims = simulate_game(
        home_mean=home_score,
        away_mean=away_score,
        spread=spread,
        total=total,
        sims=20000,
    )

    angles = []

    if pd.notna(sims.get("home_win_prob")) and sims["home_win_prob"] >= 0.60:
        angles.append(f"{home} ML")
    elif pd.notna(sims.get("away_win_prob")) and sims["away_win_prob"] >= 0.60:
        angles.append(f"{away} ML")

    if pd.notna(sims.get("home_cover_prob")) and sims["home_cover_prob"] >= 0.57:
        angles.append(f"{home} spread")
    elif pd.notna(sims.get("away_cover_prob")) and sims["away_cover_prob"] >= 0.57:
        angles.append(f"{away} spread")

    if pd.notna(sims.get("over_prob")) and sims["over_prob"] >= 0.57:
        angles.append("Game Over")
    elif pd.notna(sims.get("under_prob")) and sims["under_prob"] >= 0.57:
        angles.append("Game Under")

    if not angles:
        angles_text = "No premium angle clears threshold, but the distribution still gives useful context."
    else:
        angles_text = " | ".join(angles[:3])

    why_bits = []
    if pd.notna(sims.get("home_win_prob")):
        why_bits.append(f"{home} wins {_pct_text(sims['home_win_prob'])}")
    if pd.notna(sims.get("home_cover_prob")):
        why_bits.append(f"home cover {_pct_text(sims['home_cover_prob'])}")
    if pd.notna(sims.get("over_prob")):
        why_bits.append(f"over {_pct_text(sims['over_prob'])}")

    why_text = "; ".join(why_bits) if why_bits else "distribution still developing"

    return (
        f"{matchup}. "
        f"Line: {home} {_pretty_line(spread)} | Total {_pretty_line(total)}. "
        f"Model medians: {home} {sims['median_home']}, {away} {sims['median_away']}. "
        f"Angles: {angles_text}. "
        f"Why (20k-sim insights): {why_text}. "
        f"Margin band {sims['margin_p10']:.1f} to {sims['margin_p90']:.1f}; "
        f"total band {sims['total_p10']:.1f} to {sims['total_p90']:.1f}."
    )


def parlay_blurb(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "No parlay portfolio available."

    avg_score = None
    if "parlay_score" in rows.columns:
        try:
            avg_score = pd.to_numeric(rows["parlay_score"], errors="coerce").mean()
        except Exception:
            avg_score = None

    avg_prob = None
    if "p_win" in rows.columns:
        try:
            avg_prob = pd.to_numeric(rows["p_win"], errors="coerce").mean()
        except Exception:
            avg_prob = None

    sample_labels = []
    if "label" in rows.columns:
        sample_labels = [
            str(x) for x in rows["label"].dropna().astype(str).head(2).tolist()
            if str(x).strip() and str(x).lower() != "nan"
        ]

    leg_count = len(rows)
    parts = [f"{leg_count}-leg structure built around script-aligned positions"]

    if avg_prob is not None and pd.notna(avg_prob):
        parts.append(f"average leg win probability {_pct_text(avg_prob)}")

    if avg_score is not None and pd.notna(avg_score):
        parts.append(f"average parlay score {avg_score:.2f}")

    if sample_labels:
        parts.append(f"example legs: {'; '.join(sample_labels)}")

    return ". ".join(parts) + "."