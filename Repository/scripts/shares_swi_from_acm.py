import numpy as np
import pandas as pd
from pathlib import Path

# repository/
ROOT = Path(__file__).resolve().parents[1]  # repository/

# data/
DATA_DIR = ROOT / "data"
OUTDIR = DATA_DIR / "processed_final_variables"
OUTDIR.mkdir(parents=True, exist_ok=True)

# CSV paths
ACM_CSV = DATA_DIR / "processed_data" / "installed_capacity_adjusted_2021.csv"
OUT_FILE = OUTDIR / "shares_swi_from_acm.csv"

# Desired output order (matches your input file)
BZ_ORDER = [
    "AT", "BE", "BG", "HR", "CZ", "DK1", "DK2", "EE", "FI", "FR", "DE-LU", "GR", "HU", "IE",
    "IT-CN", "IT-CS", "IT-N", "IT-Sar", "IT-Sic", "IT-S",
    "LV", "LT", "NL",
    "NO1", "NO2", "NO3", "NO4", "NO5",
    "PL", "RO",
    "SE1", "SE2", "SE3", "SE4",
    "RS", "SK", "SI", "CH"
]

# VRE techs
VRE_TECHS = [
    "Solar",
    "Wind Onshore",
    "Wind Offshore",
    "Hydro Run-of-river and poundage",
]


def shannon_index(values):
    """Shannon Wiener index = -sum_i p_i ln(p_i)"""
    vals = np.array(values, dtype=float)
    total = vals.sum()
    if total <= 0:
        return np.nan
    p = vals / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def main():
    df = pd.read_csv(ACM_CSV)

    # Clean names
    df["bidding_zone"] = df["bidding_zone"].astype(str).str.strip()
    df["Tech"] = df["Tech"].astype(str).str.strip()

    # Aggregate ACM per bidding_zone & Tech
    cap = (
        df.groupby(["bidding_zone", "Tech"], as_index=False)["ACM_MW_2021"]
          .sum()
    )

    # Total ACM per bidding_zone
    total_cap = (
        cap.groupby("bidding_zone", as_index=False)["ACM_MW_2021"]
           .sum()
           .rename(columns={"ACM_MW_2021": "Total_ACM_MW"})
    )

    cap = cap.merge(total_cap, on="bidding_zone", how="left")

    # Shannon index per bidding_zone
    H = (
        cap.groupby("bidding_zone")["ACM_MW_2021"]
           .apply(shannon_index)
           .rename("SWI")
           .reset_index()
    )

    # Helper functions
    def acm_for_tech(area_caps, tech_name):
        return float(area_caps.loc[area_caps["Tech"] == tech_name, "ACM_MW_2021"].sum())

    def acm_for_techs(area_caps, tech_names):
        return float(area_caps.loc[area_caps["Tech"].isin(tech_names), "ACM_MW_2021"].sum())

    # Shares per bidding_zone
    rows = []
    for bz, group in cap.groupby("bidding_zone"):
        total = float(group["Total_ACM_MW"].iloc[0]) if not group.empty else 0.0

        if total > 0:
            acm_hydro_res = acm_for_tech(group, "Hydro Water Reservoir")
            acm_vre = acm_for_techs(group, VRE_TECHS)
            acm_gas = acm_for_tech(group, "Gas")

            share_hydro_res = acm_hydro_res / total
            share_vre = acm_vre / total
            share_gas = acm_gas / total
        else:
            share_hydro_res = np.nan
            share_vre = np.nan
            share_gas = np.nan

        rows.append({
            "bidding_zone": bz,
            "Total_ACM_MW": total,
            "share_hydro_reservoir": share_hydro_res,
            "share_vre": share_vre,
            "share_gas": share_gas,
        })

    shares = pd.DataFrame(rows)

    # Merge SWI and shares
    out = shares.merge(H, on="bidding_zone", how="left")

    # Force correct ordering
    out["bidding_zone"] = pd.Categorical(out["bidding_zone"], categories=BZ_ORDER, ordered=True)
    out = out.sort_values("bidding_zone")

    # Keep final output columns and order
    out = out[[
        "bidding_zone",
        "SWI",
        "share_hydro_reservoir",
        "share_vre",
        "share_gas",
    ]]

    # Rounding
    out["SWI"] = out["SWI"].round(2)
    out["share_hydro_reservoir"] = out["share_hydro_reservoir"].round(3)
    out["share_vre"] = out["share_vre"].round(3)
    out["share_gas"] = out["share_gas"].round(3)

    # Set index to bidding_zone
    out = out.set_index("bidding_zone")

    out.to_csv(OUT_FILE, index=True)
    print(f"[OK] Wrote {OUT_FILE.resolve()} (rows: {len(out)})")

    # Optional: warn if something is missing from the desired order
    missing = [bz for bz in BZ_ORDER if bz not in out.index]
    extra = [bz for bz in out.index if bz not in BZ_ORDER]

    if missing:
        print("\n[WARN] Missing zones (in BZ_ORDER but not in input):")
        print(missing)

    if extra:
        print("\n[WARN] Extra zones (in input but not in BZ_ORDER):")
        print(extra)


if __name__ == "__main__":
    main()
