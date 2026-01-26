# Fetch ENTSO-E generation data per type for 2019, process it, and save to CSV.
# Includes debug option for inspecting fetched data.

from entsoe import EntsoePandasClient
import pandas as pd
import os
from pathlib import Path

# ---------------------------
# Config
# ---------------------------

# API key for ENTSO-E
API_KEY = os.environ.get("ENTSOE_API_KEY", "YOUR_API_KEY_HERE")

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTDIR = DATA_DIR / "processed_data"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUTDIR / "entsoe_generation_per_type_2019.csv"

# Zones to fetch
ZONES = [
    "IE_SEM","DE_LU","NO","SE","AT","BE","BG","HR","CZ","DK","EE","FI","FR","GR","HU","IT",
    "LV","LT","NL","PL","RO","RS","SK","SI","CH"
]

# remove unwanted techs
DROP_TECHS = {
    "Solar",
    "Wind Offshore",
    "Wind Onshore",
    "Geothermal",
    "Hydro Pumped Storage",
}

# Rollup mapping
OIL_COMPONENTS = {"Fossil Oil", "Fossil Oil shale"}
GAS_COMPONENTS = {"Fossil Gas", "Fossil Coal-derived gas"}

# Debug options, set to True for debug prints
DEBUG = False  


# ---------------------------
# Helpers
# ---------------------------
def dbg(msg: str):
    if DEBUG:
        print(msg)


def power_timeseries_to_mwh(df: pd.DataFrame) -> pd.Series:
    """
    Integrate MW -> MWh using timestep lengths.
    Returns annual MWh per column.
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    dt_h = df.index.to_series().diff().dt.total_seconds().div(3600.0)

    # fill first timestep with next timestep length
    if len(dt_h) > 1:
        dt_h.iloc[0] = dt_h.iloc[1]
    else:
        dt_h.iloc[0] = 0.0

    return (df.mul(dt_h, axis=0)).sum()


def one_column_per_tech(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force 1 column per technology.

    - If flat columns: return as-is.
    - If MultiIndex: keep only 'Actual Aggregated' per tech (drop consumption).
    """
    if df is None or df.empty:
        return df

    # already one tech per column
    if not isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = out.columns.astype(str).str.strip()
        return out

    # MultiIndex: (tech, metric)
    tech = df.columns.get_level_values(0).astype(str).str.strip()
    metric = df.columns.get_level_values(1).astype(str).str.strip()

    # entsoe-py sometimes returns empty metric -> treat as Actual Aggregated
    metric = metric.where(~metric.isin(["", "None", "nan"]), "Actual Aggregated")

    tmp = df.copy()
    tmp.columns = pd.MultiIndex.from_arrays([tech, metric])

    # prefer only Actual Aggregated
    if "Actual Aggregated" in tmp.columns.get_level_values(1):
        out = tmp.xs("Actual Aggregated", level=1, axis=1)
    else:
        # fallback: pick first metric per tech (avoid double counting)
        out = tmp.T.groupby(level=0).first().T

    out.columns = out.columns.astype(str).str.strip()
    return out


def apply_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """Create Oil/Gas rollups and drop their component columns."""
    if df is None or df.empty:
        return df

    out = df.copy()

    oil_cols = sorted(OIL_COMPONENTS.intersection(out.columns))
    gas_cols = sorted(GAS_COMPONENTS.intersection(out.columns))

    if oil_cols:
        out["Oil"] = out[oil_cols].sum(axis=1, skipna=True)
        out = out.drop(columns=oil_cols)

    if gas_cols:
        out["Gas"] = out[gas_cols].sum(axis=1, skipna=True)
        out = out.drop(columns=gas_cols)

    return out


def fetch_generation(client: EntsoePandasClient, zone_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Fetch generation for one zone"""
    try:
        df = client.query_generation(zone_code, start=start, end=end, psr_type=None)
        return df
    except Exception as e:
        print(f"[ERROR] fetch failed for {zone_code}: {e}")
        return None


# ---------------------------
# Main
# ---------------------------
def main():
    print("ROOT =", ROOT)
    print("OUT_CSV =", OUT_CSV)

    client = EntsoePandasClient(api_key=API_KEY)

    start = pd.Timestamp("2019-01-01 00:00:00", tz="Europe/Brussels")
    end = pd.Timestamp("2020-01-01 00:00:00", tz="Europe/Brussels")

    rows = []

    for zone_code in ZONES:
        print(f"\n--- {zone_code} ---")

        df = fetch_generation(client, zone_code, start, end)

        if DEBUG:
            print("\n" + "=" * 90)
            print(f"[DEBUG] Raw fetch for {zone_code}")
            if df is None:
                print("df is None")
            else:
                print("shape:", df.shape)
                print("columns type:", type(df.columns))
                print("head columns:", df.columns[:10])
                print(df.head(2))
                if isinstance(df.columns, pd.MultiIndex):
                    metrics = df.columns.get_level_values(1).unique().tolist()
                    print("[DEBUG] Metrics present:", metrics)
            print("=" * 90 + "\n")

        if df is None or df.empty:
            print(f"[WARN] No generation data for {zone_code}")
            continue

        # 1) Make it 1 column per tech (avoid double counting)
        df = one_column_per_tech(df)

        # 2) Drop unwanted techs
        df = df.drop(columns=[t for t in DROP_TECHS if t in df.columns], errors="ignore")

        # 3) Apply rollups
        df = apply_rollups(df)

        dbg(f"[DEBUG] Cleaned columns for {zone_code}: {list(df.columns)}")

        # 4) MW -> annual MWh per tech
        annual_mwh = power_timeseries_to_mwh(df)

        dbg(f"[DEBUG] {zone_code} annual total MWh = {annual_mwh.sum():,.0f}")

        rows.append(annual_mwh.to_frame(name=zone_code).T)
        print(f"[OK] {zone_code}")

    if not rows:
        print("No data collected.")
        return

    out = pd.concat(rows, axis=0).fillna(0.0)

    # Put Oil & Gas first if present
    front = [c for c in ["Oil", "Gas"] if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]

    out = out.round(0).astype("int64")
    out.index.name = "bidding_zone"

    out.to_csv(OUT_CSV)
    print(f"\nWrote: {OUT_CSV.resolve()}")
    print(f"[DONE] shape={out.shape}, zones={out.shape[0]}, techs={out.shape[1]}")


if __name__ == "__main__":
    main()