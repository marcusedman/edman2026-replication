import numpy as np
import pandas as pd
from pathlib import Path

# repository/
ROOT = Path(__file__).resolve().parents[1]

# CSV paths
IN_FILE = ROOT / "data" / "processed_data" / "installed_capacity_adjusted_2021.csv"
OUT_FILE = ROOT / "data" / "processed_final_variables" / "stirling_diversity_2021.csv"


# output order
BZ_ORDER = [
    "AT","BE","BG","HR","CZ","DK1","DK2","EE","FI","FR","DE-LU","GR","HU","IE",
    "IT-CN","IT-CS","IT-N","IT-Sar","IT-Sic","IT-S",
    "LV","LT","NL",
    "NO1","NO2","NO3","NO4","NO5",
    "PL","RO",
    "SE1","SE2","SE3","SE4",
    "RS","SK","SI","CH"
]

# Tech parameters (D, C)

TECH_PARAMS = {
    "Biomass": (0.50, 0.30),
    "Gas": (0.95, 0.50),
    "Geothermal": (0.20, 0.00),
    "Hydro Run-of-river and poundage": (0.10, 0.00),
    "Hydro Water Reservoir": (1.00, 0.00),
    "Oil": (0.90, 1.00),
    "Other": (0.60, 0.60),
    "Solar": (0.00, 0.00),
    "Waste": (0.40, 0.20),
    "Wind Onshore": (0.00, 0.00),
    "Nuclear": (0.20, 0.10),
    "Wind Offshore": (0.00, 0.00),
    "Fossil Brown coal/Lignite": (0.40, 0.20),
    "Fossil Hard coal": (0.50, 0.40),
    "Other renewable": (0.20, 0.00),
    "Fossil Peat": (0.40, 0.20),
    "Marine": (0.00, 0.00),
}


def calculate_stirling_index(group: pd.DataFrame) -> float:
    """
    Stirling Index = sum_i sum_j p_i * p_j * d_ij
    where d_ij = sqrt((D_i - D_j)^2 + (C_i - C_j)^2)
    """
    df_active = group[group["ACM_MW_2021"] > 0].copy()

    total_cap = df_active["ACM_MW_2021"].sum()
    if total_cap <= 0:
        return np.nan

    df_active["p"] = df_active["ACM_MW_2021"] / total_cap

    # p, D, C vectors
    p = df_active["p"].to_numpy(dtype=float)
    D = df_active["Tech"].map(lambda t: TECH_PARAMS[t][0]).to_numpy(dtype=float)
    C = df_active["Tech"].map(lambda t: TECH_PARAMS[t][1]).to_numpy(dtype=float)

    # Pairwise distance matrix
    diff_D = D[:, None] - D
    diff_C = C[:, None] - C
    d_matrix = np.sqrt(diff_D**2 + diff_C**2)

    # Stirling = p^T * d_matrix * p
    return float(p @ (d_matrix @ p))


def main():
    print(f"Reading {IN_FILE}...")
    df = pd.read_csv(IN_FILE)
    # Clean strings
    df["bidding_zone"] = df["bidding_zone"].astype(str).str.strip()
    df["Tech"] = df["Tech"].astype(str).str.strip()
    
    # Aggregate ACM per bidding_zone & Tech (safe if duplicates exist)
    cap = (
        df.groupby(["bidding_zone", "Tech"], as_index=False)["ACM_MW_2021"]
          .sum()
    )

    # Check missing TECH_PARAMS
    unknown_techs = sorted(set(cap["Tech"].unique()) - set(TECH_PARAMS.keys()))
    if unknown_techs:
        raise ValueError(
            "Missing TECH_PARAMS:\n"
            + "\n".join(unknown_techs)
        )

    print("Calculating Stirling Index...")
    out = (
        cap.groupby("bidding_zone")
           .apply(calculate_stirling_index)
           .rename("stirling_index_2021")
           .reset_index()
    )
    # round to 3 decimals
    out["stirling_index_2021"] = out["stirling_index_2021"].round(3)
    # Order and set index = bidding_zone
    out["bidding_zone"] = pd.Categorical(out["bidding_zone"], categories=BZ_ORDER, ordered=True)
    out = out.sort_values("bidding_zone").set_index("bidding_zone")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=True)

    print(f"[OK] Wrote {OUT_FILE.resolve()} (rows: {len(out)})")


if __name__ == "__main__":
    main()
