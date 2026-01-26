"""
Merge
pei_csvs/pei_avg_2017_2019_entsoe.csv
pei_avg_2017_2019_eurostat.csv
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

FINAL_ORDER = [
    "AT","BE","BG","HR","CZ","DK1","DK2","EE","FI","FR","DE-LU","GR","HU","IE",
    "IT-CN","IT-CS","IT-N","IT-Sar","IT-Sic","IT-S",
    "LV","LT","NL",
    "NO1","NO2","NO3","NO4","NO5",
    "PL","RO",
    "SE1","SE2","SE3","SE4",
    "RS","SK","SI","CH"
]

KEEP_COLS = ["zone", "PEI_avg"]


def read_pei_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize zone
    df["zone"] = df["zone"].astype(str).str.strip()

    # expected cols only
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    # numeric conversion
    for c in ["PEI_avg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PEI_DIR = REPO_ROOT / "data" / "processed_data" 
    PEI_DIR.mkdir(parents=True, exist_ok=True)

    entsoe_file = PEI_DIR / "pei_avg_2017_2019_entsoe.csv"
    eurostat_file = PEI_DIR / "pei_avg_2017_2019_eurostat.csv"
    out_file = REPO_ROOT / "data" / "processed_final_variables" / "pei_avg_2017_2019_final.csv"

    if not entsoe_file.exists():
        raise FileNotFoundError(f"Missing: {entsoe_file}")
    if not eurostat_file.exists():
        raise FileNotFoundError(f"Missing: {eurostat_file}")

    entsoe = read_pei_csv(entsoe_file)
    euro = read_pei_csv(eurostat_file)

    # Merge (concat)
    merged = pd.concat([euro, entsoe], ignore_index=True)

    # Sort by required order
    rank = {z: i for i, z in enumerate(FINAL_ORDER)}
    merged["__rank"] = merged["zone"].map(rank)

    # Keep only the final order zones (strict)
    merged = merged[merged["zone"].isin(FINAL_ORDER)].copy()

    merged = (
        merged.sort_values("__rank")
        .drop(columns="__rank")
        .reset_index(drop=True)
        .rename(columns={"zone": "bidding_zone"})
    )

    # Round 
    for c in ["PEI_avg"]:
        if c in merged.columns:
            merged[c] = merged[c].round(3)
    # rename PEI_avg to PEI
    merged = merged.rename(columns={"PEI_avg": "PEI"})
    
    if "n_years" in merged.columns:
        merged["n_years"] = merged["n_years"].astype("Int64")

    # check missing zones
    missing = [z for z in FINAL_ORDER if z not in set(merged["bidding_zone"])]
    if missing:
        print("\n[WARNING] Missing zones in final output:")
        for z in missing:
            print(" -", z)

    merged.to_csv(out_file, index=False)
    print("\nSaved:", out_file)
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
