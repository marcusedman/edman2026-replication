# Compute BZ load metrics for years: baseline + target

from entsoe import EntsoePandasClient
import pandas as pd
import numpy as np
from pathlib import Path
import os
import time

# Paths

# API key
API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_KEY_HERE")
if not API_KEY or API_KEY == "YOUR_KEY_HERE":
    raise RuntimeError("Missing ENTSOE_API_KEY environment variable")



# repository/
BASE = Path(__file__).resolve().parents[1]  


OUTDIR = BASE / "data" / "processed_final_variables"
OUTDIR.mkdir(parents=True, exist_ok=True)


TZ = "Europe/Brussels"

YEARS_BASELINE = [2019]
TARGET_YEAR = 2022
ALL_YEARS = YEARS_BASELINE + [TARGET_YEAR]

# ---
BZ = {
    "IE": "10Y1001A1001A59C",
    "DE-LU": "DE_LU",
    "SE1": "SE_1",
    "SE2": "SE_2", 
    "SE3": "SE_3", 
    "SE4": "SE_4",
    "NO1": "NO_1", "NO2": "NO_2", "NO3": "NO_3", "NO4": "NO_4", "NO5": "NO_5",
    "DK1": "DK_1", "DK2": "DK_2",
    "IT-CN": "IT_CNOR", "IT-CS": "IT_CSUD", "IT-N": "IT_NORD",
    "IT-Sar": "IT_SARD", "IT-Sic": "IT_SICI",
    "AT": "AT",
    "BE": "BE",
    "BG": "BG",
    "HR": "HR",
    "CZ": "CZ",
    "EE": "EE",
    "FI": "FI",
    "FR": "FR",
    "GR": "GR",
    "HU": "HU",
    "LV": "LV",
    "LT": "LT",
    "NL": "NL",
    "PL": "PL",
    "RO": "RO",
    "RS": "RS",
    "SK": "SK",
    "SI": "SI",
    "CH": "CH",
    
}

# Italy legacy south reconstruction
IT_COMPONENTS = {
    "IT-S": "IT_SUD",
    "IT-CALA": "IT_CALA",
}
LEGACY_IT_LABEL = "IT-S-legacy"
LEGACY_IT_BZ_KEY = "IT_S_LEGACY" # key for it legacy zone

BZ_ORDER = [
    "AT","BE","BG","HR","CZ","DK1","DK2","EE","FI","FR","DE-LU","GR","HU","IE",
    "IT-CN","IT-CS","IT-N","IT-Sar","IT-Sic","IT-S-legacy",
    "LV","LT","NL",
    "NO1","NO2","NO3","NO4","NO5",
    "PL","RO",
    "SE1","SE2","SE3","SE4",
    "RS","SK","SI","CH"
]

LOAD_METRIC_COLS = [
    "annual_total_load_MWh",
    "daily_load_cv",
    "daily_load_p95_MW",
]


# Helpers

def ensure_tz(s: pd.Series, tz: str = TZ) -> pd.Series:
    """Ensure index converted to the correct timezone."""
    if s is None or s.empty:
        return s
    idx = s.index
    if getattr(idx, "tz", None) is None:
        s.index = s.index.tz_localize(tz)
    else:
        s.index = s.index.tz_convert(tz)
    return s


def to_hourly_mean(s: pd.Series) -> pd.Series:
    """Resample to hourly mean."""
    if s is None or s.empty:
        return s
    return s.sort_index().resample("h").mean()


def fill_small_gaps_hourly(
    h: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ffill_limit: int = 3,
    bfill_seed: int = 1,
) -> tuple[pd.Series, int]:
    """
    Reindex to full year hourly index, then fill tiny gaps.
    Returns (filled_series, imputed_hours_count).
    """
    full_idx = pd.date_range(start, end, freq="h", tz=start.tz, inclusive="left")

    if h is None or h.empty:
        return pd.Series(index=full_idx, dtype=float), 0

    h = ensure_tz(h, TZ)
    h_full = h.reindex(full_idx)

    original_non_na = int(h_full.notna().sum())

    # seed first (small backfill), then forward fill small gaps
    h_full = h_full.bfill(limit=bfill_seed).ffill(limit=ffill_limit)

    imputed = int(h_full.notna().sum() - original_non_na)
    return h_full, imputed


def daily_avg(s_hourly: pd.Series, min_hours: int = 20) -> pd.Series:
    """Daily averages, requiring at least min_hours per day."""
    if s_hourly is None or s_hourly.empty:
        return pd.Series(dtype=float)
    g = s_hourly.resample("D")
    mean = g.mean()
    cnt = g.count()
    return mean.where(cnt >= min_hours)


def _coerce_numeric_series(x):
    """entsoe-py may return DataFrame or Series"""
    if isinstance(x, pd.DataFrame):
        x = x.apply(pd.to_numeric, errors="coerce").select_dtypes("number")
        return None if x.empty else x.iloc[:, 0]
    return pd.to_numeric(x, errors="coerce")


# ENTSO-E fetch (chunked due to API limits)

def fetch_load(
    client: EntsoePandasClient,
    bz_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_chunk_days: int = 10,
    use_day_ahead_forecast: bool = True,
    process_type: str = "A01",  # day-ahead
) -> pd.Series | None:
    try:
        parts = []
        chunk_start = start

        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=max_chunk_days), end)

            if use_day_ahead_forecast:
                # Day-ahead Total Load forecast
                try:
                    x = client.query_load_forecast(
                        bz_key, start=chunk_start, end=chunk_end, process_type=process_type
                    )
                except TypeError:
                    # fallback for older entsoe-py versions
                    x = client.query_load_forecast(bz_key, start=chunk_start, end=chunk_end)
            else:
                # Actual load
                x = client.query_load(bz_key, start=chunk_start, end=chunk_end)

            s = _coerce_numeric_series(x)

            if s is not None and not s.dropna().empty:
                parts.append(s)

            chunk_start = chunk_end
            time.sleep(0.2)

        if not parts:
            raise ValueError("no non-empty load chunks returned")

        s_all = pd.concat(parts).sort_index()
        s_all = s_all[~s_all.index.duplicated(keep="first")]
        s_all = ensure_tz(s_all, TZ)

        if s_all.dropna().empty:
            raise ValueError("empty after concat")

        tag = "DA_FORECAST" if use_day_ahead_forecast else "ACTUAL"
        print(f"[OK] load {tag} {bz_key} (n={int(s_all.notna().sum())})")
        return s_all

    except Exception as e:
        print(f"[MISS] load {bz_key} ({e})")
        return None


# Metrics for one BZ + year
def compute_load_metrics_for_bz_year(
    client: EntsoePandasClient,
    label: str,
    bz_key: str,
    year: int,
) -> dict:
    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = pd.Timestamp(f"{year+1}-01-01", tz=TZ)

    # IE: use actual load instead of forecast (both 2019 + 2022)
    use_forecast = (label != "IE")

    load_raw = fetch_load(
        client,
        bz_key,
        start,
        end,
        use_day_ahead_forecast=use_forecast
    )

    if load_raw is None or load_raw.dropna().empty:
        print(f"[MISS] load {label} {year} (no series)")
        return {
            "label": label,
            "bz_key": bz_key,
            "year": year,
            "annual_total_load_MWh": np.nan,
            "daily_load_cv": np.nan,
            "daily_load_p95_MW": np.nan,
            "imputed_load_hours": 0,
            "valid_load_hours": 0,
        }

    # Hourly + fill small gaps
    load_h = to_hourly_mean(load_raw)
    load_h_filled, imputed_load = fill_small_gaps_hourly(
        load_h, start=start, end=end, ffill_limit=3, bfill_seed=1
    )

    expected_hours = len(load_h_filled)
    valid_hours = int(load_h_filled.notna().sum())

    # Total annual load in MWh (hourly MW summed across the year)
    if expected_hours > 0 and valid_hours < 0.95 * expected_hours:
        total_load_mwh = np.nan
    else:
        total_load_mwh = float(load_h_filled.sum())

    # Daily load metrics 
    daily_load = daily_avg(load_h_filled, min_hours=20)

    load_cv = (
        np.nan
        if daily_load.empty or daily_load.mean() == 0 or np.isnan(daily_load.mean())
        else float(daily_load.std(ddof=0) / abs(daily_load.mean()))
    )
    load_p95 = np.nan if daily_load.empty else float(daily_load.quantile(0.95))

    return {
        "label": label,
        "bz_key": bz_key,
        "year": year,
        "annual_total_load_MWh": total_load_mwh,
        "daily_load_cv": load_cv,
        "daily_load_p95_MW": load_p95,
        "imputed_load_hours": int(imputed_load),
        "valid_load_hours": int(valid_hours),
    }


# IT-S-legacy composition

def legacy_it_components_for_year(year: int) -> list[str]:
    """
    Legacy IT south definition:
    - 2019: Calabria included inside IT_SUD -> IT-S only
    - 2022: Calabria split out -> IT-S + IT-CALA
    """
    if year == 2019:
        return ["IT-S"]
    if year == 2022:
        return ["IT-S", "IT-CALA"]
    return ["IT-S"]


def compute_it_south_legacy_metrics_for_year(
    client: EntsoePandasClient,
    year: int,
) -> dict:

    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = pd.Timestamp(f"{year+1}-01-01", tz=TZ)

    components = legacy_it_components_for_year(year)

    hourly_parts = []
    total_imputed = 0

    for comp_label in components:
        comp_bz_key = IT_COMPONENTS[comp_label]

        raw = fetch_load(
            client,
            bz_key=comp_bz_key,
            start=start,
            end=end,
            use_day_ahead_forecast=True,  # legacy IT use forecast
        )

        if raw is None or raw.dropna().empty:
            print(f"[MISS] legacy IT component {comp_label} ({comp_bz_key}) {year} (no series)")
            return {
                "label": LEGACY_IT_LABEL,
                "bz_key": LEGACY_IT_BZ_KEY,
                "year": year,
                "annual_total_load_MWh": np.nan,
                "daily_load_cv": np.nan,
                "daily_load_p95_MW": np.nan,
                "imputed_load_hours": 0,
                "valid_load_hours": 0,
            }

        load_h = to_hourly_mean(raw)
        load_h_filled, imputed = fill_small_gaps_hourly(
            load_h, start=start, end=end, ffill_limit=3, bfill_seed=1
        )

        hourly_parts.append(load_h_filled)
        total_imputed += int(imputed)

    # Sum components hour-by-hour
    legacy_hourly = pd.concat(hourly_parts, axis=1).sum(axis=1, min_count=1)

    expected_hours = len(legacy_hourly)
    valid_hours = int(legacy_hourly.notna().sum())

    if expected_hours > 0 and valid_hours < 0.95 * expected_hours:
        total_load_mwh = np.nan
    else:
        total_load_mwh = float(legacy_hourly.sum())

    daily_load = daily_avg(legacy_hourly, min_hours=20)

    load_cv = (
        np.nan
        if daily_load.empty or daily_load.mean() == 0 or np.isnan(daily_load.mean())
        else float(daily_load.std(ddof=0) / abs(daily_load.mean()))
    )
    load_p95 = np.nan if daily_load.empty else float(daily_load.quantile(0.95))

    return {
        "label": LEGACY_IT_LABEL,
        "bz_key": LEGACY_IT_BZ_KEY,
        "year": year,
        "annual_total_load_MWh": total_load_mwh,
        "daily_load_cv": load_cv,
        "daily_load_p95_MW": load_p95,
        "imputed_load_hours": int(total_imputed),
        "valid_load_hours": int(valid_hours),
    }


def main():
    client = EntsoePandasClient(api_key=api_key)

    rows = []

    # ---- normal zones ----
    for label, bz_key in BZ.items():
        print(f"\n=== {label} ({bz_key}) ===")
        for year in ALL_YEARS:
            row = compute_load_metrics_for_bz_year(client, label, bz_key, year)
            rows.append(row)
            print(f"[DONE] {label} {year}")

    # ---- add IT-S-legacy ----
    for year in ALL_YEARS:
        print(f"\n=== {LEGACY_IT_LABEL} ({year}) ===")
        row = compute_it_south_legacy_metrics_for_year(client, year)
        rows.append(row)
        print(f"[DONE] {LEGACY_IT_LABEL} {year}")

    # sort
    df = pd.DataFrame(rows)
    df["bz_sort"] = pd.Categorical(df["label"], categories=BZ_ORDER, ordered=True)

    df = (
        df.sort_values(["bz_sort", "year"])
          .drop(columns=["bz_sort"])
          .reset_index(drop=True)
    )

    # Output dir
    outdir = OUTDIR


    # --------------------------
    # Baseline avg (2019 baseline)
    # --------------------------
    base = df[df["year"].isin(YEARS_BASELINE)].copy()

    base_avg = (
        base.groupby(["label", "bz_key"], dropna=False)[LOAD_METRIC_COLS]
        .mean()
        .rename(columns={c: f"{c}_baseline" for c in LOAD_METRIC_COLS})
    )

    # --------------------------
    # 2022 metrics
    # --------------------------
    y22 = df[df["year"] == TARGET_YEAR].copy().set_index(["label", "bz_key"])
    combined = base_avg.join(y22[LOAD_METRIC_COLS], how="inner")

    # --------------------------
    # Load deltas = 2022 - baseline
    # --------------------------
    combined["delta_load_pct"] = (
        (combined["annual_total_load_MWh"] - combined["annual_total_load_MWh_baseline"])
        / combined["annual_total_load_MWh_baseline"]
        * 100.0
    )

    combined["delta_load_p95_pct"] = np.where(
        combined["daily_load_p95_MW_baseline"].notna() & (combined["daily_load_p95_MW_baseline"] != 0),
        (combined["daily_load_p95_MW"] - combined["daily_load_p95_MW_baseline"])
        / combined["daily_load_p95_MW_baseline"]
        * 100.0,
        np.nan,
    )

    combined["delta_load_cv"] = (
        combined["daily_load_cv"] - combined["daily_load_cv_baseline"]
    )

    out_deltas = combined.reset_index()[[
        "label",
        "bz_key",
        "delta_load_pct",
        "delta_load_p95_pct",
        "delta_load_cv",
    ]].copy()

    # Canon bidding zone index
    out_deltas = out_deltas.rename(columns={"label": "bidding_zone"})

    # rounding
    out_deltas["delta_load_pct"] = out_deltas["delta_load_pct"].round(2)
    out_deltas["delta_load_p95_pct"] = out_deltas["delta_load_p95_pct"].round(2)
    out_deltas["delta_load_cv"] = out_deltas["delta_load_cv"].round(5)

    out_deltas["bz_sort"] = pd.Categorical(out_deltas["label"], categories=BZ_ORDER, ordered=True)
    
    out_deltas = (
        out_deltas.sort_values(["bz_sort"])
                  .drop(columns=["bz_sort"])
                  .reset_index(drop=True)
    )

    deltas_file = outdir / "load_metrics_bz_2022_deltas.csv"
    out_deltas.to_csv(deltas_file, index=False)
    print("Saved:", deltas_file)


if __name__ == "__main__":
    main()
