from pathlib import Path
import pandas as pd
import numpy as np
import re
import sys
import os

# ------
# Config
# ------

# repository/
BASE = Path(__file__).resolve().parents[1]  
#  import src.some_module
sys.path.insert(0, str(BASE))              

# ENTSO-E API key
API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_KEY_HERE")

# data/
DATA_DIR = BASE / "data"
OUTDIR = DATA_DIR / "processed_final_variables"
OUTDIR.mkdir(parents=True, exist_ok=True)
print("OUTDIR =", OUTDIR)

# CSV paths 
EMBER_CSV = DATA_DIR / "raw_data" / "ember_hourly_price_all_countries.csv"
AREAS_CSV = DATA_DIR / "processed_data" / "areas_price_metrics.csv"

# import helpers
from src.price_metric_helpers import (
    TZ, METRIC_COLS,
    make_client,
    to_brussels_from_utc,
    to_hourly_mean,
    fill_small_gaps_hourly,
    compute_metrics,
    fetch_entsoe_prices,
    fetch_entsoe_load,
    ember_hourly_prices,
    combine_zones_hourly,
)

# settings
YEARS_DEFAULT_BASELINE = [2017, 2018, 2019]
TARGET_YEAR = 2022
ALL_YEARS = YEARS_DEFAULT_BASELINE + [TARGET_YEAR]


# BZ order
BZ_ORDER = [
    "AT","BE","BG","HR","CZ","DK1","DK2","EE","FI","FR","DE-LU","GR","HU","IE",
    "IT-CN","IT-CS","IT-N","IT-Sar","IT-Sic","IT-S",
    "LV","LT","NL",
    "NO1","NO2","NO3","NO4","NO5",
    "PL","RO",
    "SE1","SE2","SE3","SE4",
    "RS","SK","SI","CH"
]

def parse_baseline_years(x):
    # default
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return YEARS_DEFAULT_BASELINE

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return YEARS_DEFAULT_BASELINE

    # split on ; or ,
    parts = re.split(r"[;,]", s)

    years = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        years.append(int(float(p)))  
    return years

def compute_one_area_year(client, ember_df, area_row, year: int):
    label = area_row["label"]
    area_type = area_row["area_type"]
    price_source = area_row["price_source"]

    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end   = pd.Timestamp(f"{year+1}-01-01", tz=TZ)

    # --- price + load fetching depending on type ---
    if price_source == "ember":
        # Ember hourly price + ENTSO-E load
        ember_country = area_row["ember_country"]
        entsoe_domain = area_row["entsoe_domain"]

        price_raw = ember_hourly_prices(ember_df, ember_country, start, end)
        load_raw  = fetch_entsoe_load(client, entsoe_domain, start, end)

        price_h = to_hourly_mean(price_raw)
        load_h  = to_hourly_mean(load_raw) if load_raw is not None else None

        price_h_filled, imputed = fill_small_gaps_hourly(price_h, start, end)

    elif price_source == "entsoe":
        entsoe_domain = area_row["entsoe_domain"]

        price_raw = fetch_entsoe_prices(client, entsoe_domain, start, end)
        load_raw  = fetch_entsoe_load(client, entsoe_domain, start, end)

        price_h = to_hourly_mean(price_raw) if price_raw is not None else pd.Series(dtype=float)
        load_h  = to_hourly_mean(load_raw) if load_raw is not None else None

        price_h_filled, imputed = fill_small_gaps_hourly(price_h, start, end)

    elif price_source == "synthetic":
        # components e.g. "IT-S|IT-CALA" refers to other labels in registry
        comps = str(area_row["components"]).split("|")

        prices, loads = [], []
        total_imputed = 0

        for comp_label in comps:
            comp_row = AREAS.loc[AREAS["label"] == comp_label].iloc[0]
            entsoe_domain = comp_row["entsoe_domain"]

            p_raw = fetch_entsoe_prices(client, entsoe_domain, start, end)
            L_raw = fetch_entsoe_load(client, entsoe_domain, start, end)

            p_h = to_hourly_mean(p_raw) if p_raw is not None else pd.Series(dtype=float)
            L_h = to_hourly_mean(L_raw) if L_raw is not None else pd.Series(dtype=float)

            p_h_filled, imp = fill_small_gaps_hourly(p_h, start, end)

            prices.append(p_h_filled)
            loads.append(L_h)
            total_imputed += int(imp)

        price_h_filled, load_h_combined = combine_zones_hourly(prices, loads)
        load_h = load_h_combined
        imputed = total_imputed

    else:
        raise ValueError(f"Unknown price_source: {price_source}")

    # --- compute metrics ---
    metrics = compute_metrics(price_h_filled, load_h)

    return {
        "label": label,
        "year": year,
        "area_type": area_type,
        **metrics,
        "imputed_price_hours": int(imputed),
    }
def build_deltas(df_all_years: pd.DataFrame) -> pd.DataFrame:

    rows = []

    areas_index = AREAS.set_index("label", drop=False)

    for label, g in df_all_years.groupby("label"):
        # baseline rule for legacy South 
        if label == "IT-S-legacy":
            baseline_years = YEARS_DEFAULT_BASELINE  
            g_base = df_all_years[df_all_years["label"] == "IT-S"]
            if g_base.empty:
                raise ValueError("Missing baseline series for IT-S (needed to compute IT-S-legacy deltas).")
        else:
            # baseline override from areas  
            if label in areas_index.index:
                baseline_years = parse_baseline_years(areas_index.loc[label, "baseline_years"])
            else:
                baseline_years = YEARS_DEFAULT_BASELINE
            g_base = g

        base = g_base[g_base["year"].isin(baseline_years)][METRIC_COLS].mean()
        y22  = g[g["year"] == TARGET_YEAR][METRIC_COLS].mean()

        out = {
            "label": label,
            "delta_price": y22.get("annual_price_load_weighted_EUR_per_MWh", np.nan) - base.get("annual_price_load_weighted_EUR_per_MWh", np.nan),
            "delta_p95":   y22.get("daily_price_p95_EUR_per_MWh", np.nan) - base.get("daily_price_p95_EUR_per_MWh", np.nan),
            "delta_cv":    y22.get("daily_price_cv", np.nan) - base.get("daily_price_cv", np.nan),
        }
        rows.append(out)

    deltas = pd.DataFrame(rows)

    # rounding
    deltas["delta_price"] = deltas["delta_price"].round(2)
    deltas["delta_p95"]   = deltas["delta_p95"].round(2)
    deltas["delta_cv"]    = deltas["delta_cv"].round(5)

    # enforce your order
    order_map = {lab: i for i, lab in enumerate(BZ_ORDER)}
    deltas["sort_key"] = deltas["label"].map(order_map).fillna(9999).astype(int)
    deltas = deltas.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

    return deltas


if __name__ == "__main__":
    # load registry
    AREAS = pd.read_csv(AREAS_CSV)

    # load Ember
    ember_df = None
    if (AREAS["price_source"] == "ember").any():
        if not EMBER_CSV.exists():
            raise FileNotFoundError(f"Missing Ember file: {EMBER_CSV}")
        ember_df = pd.read_csv(EMBER_CSV)
        ember_df["time_brussels"] = to_brussels_from_utc(ember_df["Datetime (UTC)"])
        ember_df = ember_df.dropna(subset=["time_brussels"])

    client = make_client() 

    # compute all years
    rows = []
    for _, area_row in AREAS.iterrows():
        baseline_years = parse_baseline_years(area_row.get("baseline_years", ""))
        years_needed = sorted(set(baseline_years + [TARGET_YEAR]))

        print(f"\n=== {area_row['label']} years: {years_needed} ===")

        for year in years_needed:
            rows.append(compute_one_area_year(client, ember_df, area_row, year))

    df_all = pd.DataFrame(rows).sort_values(["label", "year"]).reset_index(drop=True)

    out_all_years = OUTDIR / "price_metrics_all_years.csv"
    df_all.to_csv(out_all_years, index=False)
    print("Saved:", out_all_years)

    df_deltas = build_deltas(df_all)

    out_deltas = OUTDIR / "price_metrics_2022_deltas.csv"
    df_deltas.to_csv(out_deltas, index=False)
    print("Saved:", out_deltas)
