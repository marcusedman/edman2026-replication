from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path

YEARS = [2017, 2018, 2019]

api_key = os.getenv("ENTSOE_API_KEY", "YOUR_KEY_HERE") 

ENTSOE_ZONES = [
    # Denmark
    ("DK1", "DK_1"),
    ("DK2", "DK_2"),

    # Norway
    ("NO1", "NO_1"),
    ("NO2", "NO_2"),
    ("NO3", "NO_3"),
    ("NO4", "NO_4"),
    ("NO5", "NO_5"),

    # Sweden
    ("SE1", "SE_1"),
    ("SE2", "SE_2"),
    ("SE3", "SE_3"),
    ("SE4", "SE_4"),

    # Italy
    ("IT-CN",  "IT_CNOR"),
    ("IT-CS",  "IT_CSUD"),
    ("IT-N",   "IT_NORD"),
    ("IT-Sar", "IT_SARD"),
    ("IT-Sic", "IT_SICI"),
    ("IT-S",   "IT_SUD"),

    # Switzerland
    ("CH", "CH"),
]

# -------------------------
# Generation corrections
# -------------------------
# Replace generation totals (GWh -> TWh)
SE_GEN_OVERRIDES_GWH = {
    ("SE1", 2017): 22143, ("SE1", 2018): 21835, ("SE1", 2019): 22134,
    ("SE2", 2017): 44862, ("SE2", 2018): 42649, ("SE2", 2019): 45191,
    ("SE3", 2017): 88723, ("SE3", 2018): 91228, ("SE3", 2019): 92610,
    ("SE4", 2017):  8520, ("SE4", 2018):  7672, ("SE4", 2019):  8490,
}

CH_GEN_OVERRIDES_TWH = {
    ("CH", 2017): 54.7, ("CH", 2018): 60.7, ("CH", 2019): 64.9,
}

# Add extra generation totals (TWh)
ITALY_EXTRA_GEN_TWH = {
    ("IT-Sic", 2017): 1.32, ("IT-Sic", 2018): 1.36,
    ("IT-S",   2017): 20.87, ("IT-S",   2018): 19.07, ("IT-S",   2019): 13.64,
}

# Helpers

def pei_realized(gen_twh: float, load_twh: float) -> float:
    """Signed PEI: +export orientation, -import dependence."""
    if pd.isna(gen_twh) or pd.isna(load_twh) or (gen_twh == 0 and load_twh == 0):
        return np.nan
    if gen_twh >= load_twh and gen_twh != 0:
        return (gen_twh - load_twh) / gen_twh
    if load_twh > gen_twh and load_twh != 0:
        return - (load_twh - gen_twh) / load_twh
    return np.nan


def to_series(ts):
    """Convert Series/DataFrame to a single MW Series."""
    if ts is None:
        return None

    if isinstance(ts, pd.Series):
        return ts

    if isinstance(ts, pd.DataFrame):
        if ts.shape[1] == 1:
            return ts.iloc[:, 0]
        return ts.sum(axis=1, numeric_only=True)

    return None


def integrate_mw_series_to_mwh(ts) -> tuple[float, int]:
    """
    Convert MW time series to MWh using timestep length (hours).
    Returns: (total_mwh, n_points_used_after_dropna)
    """
    s = to_series(ts)
    if s is None:
        return (np.nan, 0)

    s = s.dropna()
    n = int(len(s))
    if n < 2:
        return (np.nan, n)

    dt_hours = pd.Series(s.index[1:] - s.index[:-1]).median() / pd.Timedelta(hours=1)
    if pd.isna(dt_hours) or dt_hours <= 0:
        return (np.nan, n)

    total_mwh = float(s.sum()) * float(dt_hours)
    return (total_mwh, n)



def apply_generation_corrections(zone: str, year: int, gen_twh: float) -> tuple[float, str | None]:
    """
    Apply overrides/corrections to generation totals.
    1) Sweden override (SCB)
    2) Switzerland override (Swiss EnergyCharts)
    3) Italy extra (from production hubs)
    """

    if (zone, year) in SE_GEN_OVERRIDES_GWH:
        corrected = SE_GEN_OVERRIDES_GWH[(zone, year)] / 1000.0
        return corrected, f"SE override -> {corrected:.6f} TWh"

    if (zone, year) in CH_GEN_OVERRIDES_TWH:
        corrected = CH_GEN_OVERRIDES_TWH[(zone, year)]
        return corrected, f"CH override -> {corrected:.6f} TWh"

    if (zone, year) in ITALY_EXTRA_GEN_TWH and pd.notna(gen_twh):
        corrected = gen_twh + ITALY_EXTRA_GEN_TWH[(zone, year)]
        return corrected, f"+Italy extra -> {corrected:.6f} TWh"

    return gen_twh, None


def fetch_generation_total_gross_mw(client, bz_key: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series | None, int, int]:
    """
    Fetch generation by technology and return total gross generation as MW time series.
    """
    try:
        df = client.query_generation(bz_key, start=start, end=end, psr_type=None)

        if isinstance(df, pd.Series):
            # If query_generation returns already total MW series
            return df.dropna(), 1, 0

        if not isinstance(df, pd.DataFrame):
            return None, 0, 0

        # Keep numeric columns only
        df = df.select_dtypes("number")
        if df.empty:
            return None, 0, 0

        # Case 1: MultiIndex columns (tech, measure)
        if isinstance(df.columns, pd.MultiIndex):
            lvl = -1
            measures = pd.Index([c[lvl] for c in df.columns])
            has_agg = (measures == "Actual Aggregated").any()

            if has_agg:
                gross_cols = [c for c in df.columns if c[lvl] == "Actual Aggregated"]
                cons_cols = [c for c in df.columns if c[lvl] == "Actual Consumption"]
                total = df[gross_cols].sum(axis=1, numeric_only=True).dropna()
                return total, len(gross_cols), len(cons_cols)

            # Fallback: if no measure-level labels, sum all numeric
            total = df.sum(axis=1, numeric_only=True).dropna()
            return total, df.shape[1], 0

        # Case 2: Flat string columns with suffix
        if all(isinstance(c, str) for c in df.columns):
            gross_cols = [c for c in df.columns if c.endswith("Actual Aggregated")]
            cons_cols = [c for c in df.columns if c.endswith("Actual Consumption")]

            if gross_cols:
                total = df[gross_cols].sum(axis=1, numeric_only=True).dropna()
                return total, len(gross_cols), len(cons_cols)

            # Case 3: Plain technology names (already gross)
            total = df.sum(axis=1, numeric_only=True).dropna()
            return total, df.shape[1], 0

        # Final fallback
        total = df.sum(axis=1, numeric_only=True).dropna()
        return total, df.shape[1], 0

    except Exception as e:
        print(f"[MISS] gen {bz_key} ({type(e).__name__}: {e})")
        return None, 0, 0


# Main

def main():
    # outputs
    REPO_ROOT = Path(__file__).resolve().parents[1]
    OUTDIR = REPO_ROOT / "data" / "processed_data"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTDIR / "pei_avg_2017_2019_entsoe.csv"

    # ENTSO-E API
    try:
        from entsoe import EntsoePandasClient
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'entsoe-py'. Install it with:\n\n"
            "  pip install entsoe-py\n"
        ) from e


    client = EntsoePandasClient(api_key=api_key)

    rows = []

    print("\n ENTSO-E PEI computation (gross generation)\n")

    for zone, domain in ENTSOE_ZONES:
        print(f"\n--- {zone} ({domain}) ---")

        for year in YEARS:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

            # ---- LOAD ----
            load_twh = np.nan
            try:
                load_ts = client.query_load(domain, start=start, end=end)
                load_mwh, n_load = integrate_mw_series_to_mwh(load_ts)
                load_twh = load_mwh / 1e6 if pd.notna(load_mwh) else np.nan

                if pd.notna(load_twh):
                    print(f"{zone} Load sum OK    {year} (n={n_load})")
                else:
                    print(f"{zone} Load sum EMPTY {year} (n={n_load})")
            except Exception as e:
                print(f"{zone} Load FAIL      {year} ({type(e).__name__}: {e})")

            # ---- GENERATION (GROSS ONLY) ----
            gen_twh = np.nan
            n_gross_cols, n_cons_cols = 0, 0
            try:
                gen_total_mw, n_gross_cols, n_cons_cols = fetch_generation_total_gross_mw(
                    client, domain, start, end
                )

                gen_mwh, n_gen = integrate_mw_series_to_mwh(gen_total_mw)
                gen_twh = gen_mwh / 1e6 if pd.notna(gen_mwh) else np.nan

                extra = f"[gross cols={n_gross_cols}"
                if n_cons_cols > 0:
                    extra += f", cons cols={n_cons_cols}"
                extra += "]"

                if pd.notna(gen_twh):
                    print(f"{zone} Gen  sum OK    {year} (n={n_gen}) {extra}")
                else:
                    print(f"{zone} Gen  sum EMPTY {year} (n={n_gen}) {extra}")
            except Exception as e:
                print(f"{zone} Gen  FAIL      {year} ({type(e).__name__}: {e})")

            # ---- Apply corrections ----
            gen_twh_corr, note = apply_generation_corrections(zone, year, gen_twh)
            if note is not None:
                print(f"{zone} Gen  corrected {year} ({note})")

            # ---- PEI ----
            pei = pei_realized(gen_twh_corr, load_twh)

            rows.append({
                "zone": zone,
                "year": year,
                "PEI_realized": pei
            })

    print("\n=== ENTSO-E PEI computation finished ===\n")

    panel = pd.DataFrame(rows)

    # Wide output
    wide = (
        panel.pivot_table(index="zone", columns="year", values="PEI_realized", aggfunc="first")
        .rename(columns={2017: "PEI_17", 2018: "PEI_18", 2019: "PEI_19"})
        .reset_index()
    )

    wide["PEI_avg"] = wide[["PEI_17", "PEI_18", "PEI_19"]].mean(axis=1, skipna=True)
    wide["n_years"] = wide[["PEI_17", "PEI_18", "PEI_19"]].notna().sum(axis=1)

    for c in ["PEI_17", "PEI_18", "PEI_19", "PEI_avg"]:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").round(3)

    # Zone order
    order = [z for z, _ in ENTSOE_ZONES]
    wide["__rank"] = wide["zone"].map({z: i for i, z in enumerate(order)})
    wide = wide.sort_values("__rank").drop(columns="__rank").reset_index(drop=True)

    wide.to_csv(out_file, index=False)
    print("Saved:", out_file)
    print("\nPreview:\n")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    if not api_key or api_key == "YOUR_KEY_HERE":
       raise RuntimeError("Missing ENTSOE_API_KEY environment variable")
    main()
