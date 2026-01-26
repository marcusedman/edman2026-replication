import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTDIR = DATA_DIR / "processed_data"
OUTDIR.mkdir(parents=True, exist_ok=True)

# paths
RAW_IRENA = DATA_DIR / "raw_data" / "irena_raw_cap_gen_solar_wind_geo_17_21.csv"
OUT_FILE = OUTDIR / "irena_capacity_factors_2017_2021.csv"


YEARS = [2017, 2018, 2019, 2020, 2021]

COUNTRY_MAP = {
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Croatia": "HR",
    "Czechia": "CZ", "Denmark": "DK", "Estonia": "EE", "Finland": "FI",
    "France": "FR", "Germany": "DE", "Greece": "GR", "Hungary": "HU",
    "Ireland": "IE", "Italy": "IT", "Latvia": "LV", "Lithuania": "LT",
    "Netherlands (Kingdom of the)": "NL", "Poland": "PL", "Romania": "RO",
    "Serbia": "RS", "Slovakia": "SK", "Slovenia": "SI", "Switzerland": "CH",
    "Norway": "NO", "Sweden": "SE",
}

DT_CAP = "Electricity Installed Capacity (MW)"
DT_GEN = "Electricity Generation (GWh)"

def main():
    if not RAW_IRENA.exists():
        raise FileNotFoundError(f"Input CSV not found: {RAW_IRENA.resolve()}")

    df = pd.read_csv(
        RAW_IRENA,
        na_values=["-", "", "n/e"],
        keep_default_na=True,
        encoding="utf-8-sig",
        engine="python",
    )

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df = df[
        df["Year"].isin(YEARS)
        & df["Grid connection"].astype(str).str.contains("On-grid", na=False)
        & df["Data Type"].isin([DT_CAP, DT_GEN])
        & df["Country/area"].isin(COUNTRY_MAP)
    ].copy()

    df["Country"] = df["Country/area"].map(COUNTRY_MAP)
    df["Electricity statistics"] = pd.to_numeric(df["Electricity statistics"], errors="coerce")

    # Pivot to one row per (Country, Technology, Year) with Cap_MW and Gen_GWh
    piv = (
        df.pivot_table(
            index=["Country", "Technology", "Year"],
            columns="Data Type",
            values="Electricity statistics",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={DT_CAP: "Cap_MW", DT_GEN: "Gen_GWh"})
    )

    # Compute year-level numerator/denominator
    piv["Gen_MWh"] = piv["Gen_GWh"] * 1000.0
    piv["Den_MWh"] = piv["Cap_MW"] * 8760.0

    # Use only years with both values present
    valid = piv["Gen_MWh"].notna() & piv["Den_MWh"].notna() & (piv["Cap_MW"] > 0)
    pv = piv[valid].copy()

    # Weighted 5-year CF per Country × Technology
    sums = (
        pv.groupby(["Country", "Technology"], as_index=False)[["Gen_MWh", "Den_MWh"]]
        .sum()
        .rename(columns={"Gen_MWh": "Gen_MWh_sum", "Den_MWh": "Den_MWh_sum"})
    )

    sums["CapacityFactor"] = sums["Gen_MWh_sum"] / sums["Den_MWh_sum"]
    out = sums[["Country", "Technology", "CapacityFactor"]].copy()
    out["CapacityFactor"] = out["CapacityFactor"].round(6)

    out = out.sort_values(["Country", "Technology"]).reset_index(drop=True)

    # write to your configured output path
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)
    print(f"[OK] Wrote {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

