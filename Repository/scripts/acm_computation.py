import re
import pandas as pd
from pathlib import Path


# repository/
ROOT = Path(__file__).resolve().parents[1]  

# data/
DATA_DIR = ROOT / "data"
OUTDIR = DATA_DIR / "processed_data"

# CSV paths
CAPS_IN = DATA_DIR / "processed_data" / "installed_Capacities_2021_harmonized.csv"
CF_IN = DATA_DIR / "processed_data" / "complete_cf_countries_2019.csv"



def parent_country_from_bidding_zone(bz: str) -> str:
    bz = str(bz).strip()

    # already a 2-letter country code
    if re.fullmatch(r"[A-Z]{2}", bz):
        return bz

    # DK1/DK2 -> DK
    if re.fullmatch(r"DK[12]", bz):
        return "DK"

    # NO1..NO5 -> NO
    if re.fullmatch(r"NO[1-5]", bz):
        return "NO"

    # SE1..SE4 -> SE
    if re.fullmatch(r"SE[1-4]", bz):
        return "SE"

    # IT-* zones -> IT
    if bz.startswith("IT-"):
        return "IT"

    # DE-LU -> DE
    if bz == "DE-LU":
        return "DE"

    return bz  # fallback


def main():
    # --- Load inputs ---
    cf = pd.read_csv(CF_IN)         # Country, Tech, CapacityFactor, Source
    caps = pd.read_csv(CAPS_IN)     # bidding_zone + tech columns

    # --- Normalize CAPS input ---
    if "bidding_zone" not in caps.columns:
        raise ValueError("CAPS file must contain a 'bidding_zone' column.")

    # Map bidding_zone ->  Country for joining CF
    caps["Country"] = caps["bidding_zone"].apply(parent_country_from_bidding_zone)

    # --- long format
    cap_long = caps.melt(
        id_vars=["bidding_zone", "Country"],
        var_name="Tech",
        value_name="Cap_MW_2021",
    )

    # Drop rows with zero or NaN capacity
    cap_long["Cap_MW_2021"] = pd.to_numeric(cap_long["Cap_MW_2021"], errors="coerce")
    cap_long = cap_long[cap_long["Cap_MW_2021"].fillna(0) > 0].copy()

    # --- Merge with CFs by parent country + tech ---
    cf_needed_cols = ["Country", "Tech", "CapacityFactor"]
    if not set(cf_needed_cols).issubset(cf.columns):
        raise ValueError("CF file must contain columns: Country, Tech, CapacityFactor")

    merged = cap_long.merge(cf[cf_needed_cols], on=["Country", "Tech"], how="left")
    
    # --- Handle missing CFs (assign default) ---
    missing_mask = merged["CapacityFactor"].isna()
    missing_cf = merged[missing_mask]

    if not missing_cf.empty:
        print("Capacities have installed capacity with no CF:")
        print(
            missing_cf[["bidding_zone", "Country", "Tech", "Cap_MW_2021"]]
            .sort_values(["Country", "bidding_zone", "Tech"])
            .to_string(index=False)
        )
        print("\nAssigning default CapacityFactor = 0.2 to the rows above.\n")
        merged.loc[missing_mask, "CapacityFactor"] = 0.2
    else:
        print("All non-zero installed capacities have an associated capacity factor.")

    # ACM = installed capacity * CF
    merged["ACM_MW_2021"] = merged["Cap_MW_2021"] * merged["CapacityFactor"]
    merged["ACM_MW_2021"] = merged["ACM_MW_2021"].round(0)

    # Final output
    out = merged[
        [
            "bidding_zone",
            "Country",
            "Tech",
            "Cap_MW_2021",
            "CapacityFactor",
            "ACM_MW_2021",
        ]
    ].sort_values(["Country", "bidding_zone", "Tech"]).reset_index(drop=True)

    # output
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "adjusted_capacity_mix_2021.csv"
    out.to_csv(out_path, index=False)

    print(f"[OK] Wrote {out_path.resolve()}  (rows: {len(out)})")

if __name__ == "__main__":
    main()