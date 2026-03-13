from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "COMMANDERS": ["Washington Commanders", "Washington", "Commanders", "Washington Football Team", "Football Team", "WFT", "Redskins", "Skins", "WAS", "WSH", "WDC", "DC"],
    "COWBOYS": ["Dallas Cowboys", "Dallas", "Cowboys", "America's Team", "Americas Team", "DAL", "DLS"],
    "EAGLES": ["Philadelphia Eagles", "Philadelphia", "Eagles", "PHI", "PHL"],
    "GIANTS": ["New York Giants", "NY Giants", "Giants", "NYG", "New York NYG"],
    "PACKERS": ["Green Bay Packers", "Green Bay", "Packers", "GB", "GNB", "G Bay"],
    "BEARS": ["Chicago Bears", "Chicago", "Bears", "CHI"],
    "VIKINGS": ["Minnesota Vikings", "Minnesota", "Vikings", "MIN", "MINN"],
    "LIONS": ["Detroit Lions", "Detroit", "Lions", "DET"],
    "SAINTS": ["New Orleans Saints", "New Orleans", "Saints", "NO", "NOS", "NOLA", "N.O.", "N O"],
    "BUCCANEERS": ["Tampa Bay Buccaneers", "Tampa Bay", "Buccaneers", "Bucs", "TB", "TBB", "TAMPA"],
    "FALCONS": ["Atlanta Falcons", "Atlanta", "Falcons", "ATL"],
    "PANTHERS": ["Carolina Panthers", "Carolina", "Panthers", "CAR"],
    "RAMS": ["Los Angeles Rams", "LA Rams", "Rams", "LAR", "St. Louis Rams", "St Louis Rams", "St Louis", "Los Angelos Rams", "LA", "L A"],
    "49ERS": ["San Francisco 49ers", "SF 49ers", "Forty Niners", "Forty-Niners", "49ers", "Niners", "San Francisco", "SF", "SFO"],
    "SEAHAWKS": ["Seattle Seahawks", "Seattle", "Seahawks", "SEA"],
    "CARDINALS": ["Arizona Cardinals", "Cardinals", "AZ Cardinals", "ARI", "Phoenix Cardinals", "Phoenix", "St. Louis Cardinals", "St Louis Cardinals", "St Louis"],
    "PATRIOTS": ["New England Patriots", "New England", "Patriots", "NE", "NWE", "N.E."],
    "BILLS": ["Buffalo Bills", "Buffalo", "Bills", "BUF"],
    "JETS": ["New York Jets", "NY Jets", "Jets", "NYJ"],
    "DOLPHINS": ["Miami Dolphins", "Miami", "Dolphins", "MIA"],
    "RAVENS": ["Baltimore Ravens", "Baltimore", "Ravens", "BAL"],
    "BENGALS": ["Cincinnati Bengals", "Cincinnati", "Bengals", "CIN"],
    "BROWNS": ["Cleveland Browns", "Cleveland", "Browns", "CLE"],
    "STEELERS": ["Pittsburgh Steelers", "Pittsburgh", "Steelers", "PIT"],
    "COLTS": ["Indianapolis Colts", "Indianapolis", "Colts", "IND", "Baltimore Colts"],
    "TITANS": ["Tennessee Titans", "Tennessee", "Titans", "TEN", "TN", "Tennessee Oilers", "Houston Oilers", "Oilers", "HOU Oilers"],
    "JAGUARS": ["Jacksonville Jaguars", "Jacksonville", "Jaguars", "Jags", "JAX", "JAC"],
    "TEXANS": ["Houston Texans", "Houston", "Texans", "HOU"],
    "CHIEFS": ["Kansas City Chiefs", "Kansas City", "KC Chiefs", "Chiefs", "KC", "KCC", "KANSAS_CITY"],
    "RAIDERS": ["Las Vegas Raiders", "Las Vegas", "LV", "Raiders", "Oakland Raiders", "Oakland", "Los Angeles Raiders", "LA Raiders", "Los Angeles", "LVR", "OAK", "L.A."],
    "CHARGERS": ["Los Angeles Chargers", "LA Chargers", "Chargers", "LAC", "San Diego Chargers", "San Diego", "SD", "LA", "L.A."],
    "BRONCOS": ["Denver Broncos", "Denver", "Broncos", "DEN"],
}

ALIAS_MAP: dict[str, str] = {}
for canon, aliases in TEAM_ALIASES.items():
    ALIAS_MAP[canon] = canon
    for alias in aliases:
        key = re.sub(r"\s+", "_", re.sub(r"[^A-Z0-9 ]+", "", str(alias).upper()).strip())
        ALIAS_MAP[key] = canon


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def nickify(series: pd.Series) -> pd.Series:
    s = series.astype("string").fillna("")
    cleaned = (
        s.str.upper()
        .str.replace(r"[^A-Z0-9 ]+", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )
    mapped = cleaned.replace(ALIAS_MAP)
    return mapped.str.replace(r"_+", "_", regex=True).str.strip("_")


def ensure_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home" not in out.columns:
        if "home_team" in out.columns:
            out["home"] = out["home_team"]
        else:
            raise KeyError("Missing home/home_team column")
    if "away" not in out.columns:
        if "away_team" in out.columns:
            out["away"] = out["away_team"]
        else:
            raise KeyError("Missing away/away_team column")
    return out


def ensure_season_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "season" not in out.columns:
        if "date" in out.columns:
            dt = pd.to_datetime(out["date"], errors="coerce")
            out["season"] = (dt.dt.year - (dt.dt.month <= 2).astype("Int64")).astype("Int64")
        else:
            raise KeyError("Missing season column and no date column available to derive it")
    if "week" not in out.columns:
        raise KeyError("Missing week column in target scores file")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    return out


def choose_sheet(xlsx_path: Path) -> str | int:
    xl = pd.ExcelFile(xlsx_path)
    # Prefer a sheet name that sounds like the master game sheet.
    for name in xl.sheet_names:
        low = name.lower()
        if "game" in low and ("master" in low or "ev" in low or "tier" in low):
            return name
    return xl.sheet_names[0]


def build_keys(df: pd.DataFrame) -> pd.Series:
    return (
        df["season"].astype("string")
        + "|"
        + df["week"].astype("string")
        + "|"
        + df["_home_nick"]
        + "|"
        + df["_away_nick"]
    )


def merge_scores(source_xlsx: Path, target_csv: Path, output_csv: Path, overwrite: bool) -> None:
    sheet = choose_sheet(source_xlsx)
    src = pd.read_excel(source_xlsx, sheet_name=sheet)
    tgt = pd.read_csv(target_csv, low_memory=False)

    src = clean_columns(src)
    tgt = clean_columns(tgt)

    src = ensure_team_cols(src)
    tgt = ensure_team_cols(tgt)
    tgt = ensure_season_week(tgt)

    required_src = ["season", "week", "spread_close", "total_close"]
    missing_src = [c for c in required_src if c not in src.columns]
    if missing_src:
        raise KeyError(f"Source workbook is missing required columns: {missing_src}")

    src["season"] = pd.to_numeric(src["season"], errors="coerce").astype("Int64")
    src["week"] = pd.to_numeric(src["week"], errors="coerce").astype("Int64")
    src["spread_close"] = pd.to_numeric(src["spread_close"], errors="coerce")
    src["total_close"] = pd.to_numeric(src["total_close"], errors="coerce")

    src["_home_nick"] = nickify(src["home"])
    src["_away_nick"] = nickify(src["away"])
    tgt["_home_nick"] = nickify(tgt["home"])
    tgt["_away_nick"] = nickify(tgt["away"])

    src = src.loc[src["season"].notna() & src["week"].notna()].copy()
    src["_join_key"] = build_keys(src)
    tgt["_join_key"] = build_keys(tgt)

    dupes = src[src.duplicated("_join_key", keep=False)].sort_values("_join_key")
    if not dupes.empty:
        dupes_path = output_csv.with_name(output_csv.stem + "_source_duplicates.csv")
        dupes.to_csv(dupes_path, index=False)
        raise ValueError(
            "Duplicate source rows found for the same season/week/home/away key. "
            f"Review: {dupes_path}"
        )

    src_small = src[["_join_key", "spread_close", "total_close"]].rename(columns={"spread_close": "_spread_home_src", "total_close": "_total_close_src"})
    out = tgt.merge(src_small, on="_join_key", how="left")

    if "spread_home" not in out.columns:
        out["spread_home"] = pd.NA
    if "total_close" not in out.columns:
        out["total_close"] = pd.NA

    out["spread_home"] = pd.to_numeric(out["spread_home"], errors="coerce")
    out["total_close"] = pd.to_numeric(out["total_close"], errors="coerce")

    if overwrite:
        out["spread_home"] = out["_spread_home_src"].combine_first(out["spread_home"])
        out["total_close"] = out["_total_close_src"].combine_first(out["total_close"])
    else:
        out["spread_home"] = out["spread_home"].combine_first(out["_spread_home_src"])
        out["total_close"] = out["total_close"].combine_first(out["_total_close_src"])

    matched = out["_spread_home_src"].notna() | out["_total_close_src"].notna()
    filled_spread = out["spread_home"].notna().sum()
    filled_total = out["total_close"].notna().sum()

    unmatched = out.loc[~matched, [c for c in ["season", "week", "date", "home", "away", "_join_key"] if c in out.columns]].copy()
    unmatched_path = output_csv.with_name(output_csv.stem + "_unmatched_rows.csv")
    unmatched.to_csv(unmatched_path, index=False)

    out = out.drop(columns=["_spread_home_src", "_total_close_src", "_home_nick", "_away_nick", "_join_key"], errors="ignore")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Source sheet: {sheet}")
    print(f"Target rows: {len(tgt):,}")
    print(f"Matched rows: {int(matched.sum()):,}")
    print(f"spread_home populated rows: {int(filled_spread):,}")
    print(f"total_close populated rows: {int(filled_total):,}")
    print(f"Wrote: {output_csv}")
    print(f"Unmatched rows report: {unmatched_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge historical closing spread/total into scores_1966-2025.csv")
    parser.add_argument("--source-xlsx", required=True, help="Path to games_master_with_ev_and_tiers.xlsx")
    parser.add_argument("--target-csv", required=True, help="Path to scores_1966-2025.csv")
    parser.add_argument("--output-csv", required=True, help="Path to write merged scores CSV")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing spread_home/total_close values")
    args = parser.parse_args()

    merge_scores(
        source_xlsx=Path(args.source_xlsx),
        target_csv=Path(args.target_csv),
        output_csv=Path(args.output_csv),
        overwrite=args.overwrite,
    )
