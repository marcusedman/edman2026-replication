import pandas as pd
from pathlib import Path


# Paths
ROOT = Path(__file__).resolve().parents[1]   
DATA_DIR = ROOT / "data"                    
PROCESSED_DIR = DATA_DIR / "processed_data" 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Paths
ENTSOE_IN = PROCESSED_DIR / "entsoe_capacity_factors_2019.csv"
IRENA_IN  = PROCESSED_DIR / "IRENA_capacity_factors_2017_2021.csv"
OUT       = PROCESSED_DIR / "complete_cf_countries_2019.csv"


# Load
entsoe = pd.read_csv(ENTSOE_IN)
entsoe = entsoe.rename(columns={"bidding_zone": "Country"})
irena  = pd.read_csv(IRENA_IN)

# Check columns
if "Tech" not in entsoe.columns:
    if "Technology" in entsoe.columns:
        entsoe = entsoe.rename(columns={"Technology": "Tech"})
    else:
        raise ValueError("ENTSO-E missing Tech column")
if "CapacityFactor" not in entsoe.columns:
    raise ValueError("ENTSO-E missing 'CapacityFactor'.")

required_irena_cols = {"Country", "Technology", "CapacityFactor"}
if not required_irena_cols.issubset(irena.columns):
    missing = required_irena_cols - set(irena.columns)
    raise ValueError(f"IRENA file is missing columns: {', '.join(missing)}")

# Map IRENA tech names -> our labels
TECH_MAP = {
    "Solar photovoltaic": "Solar",
    "Onshore wind energy": "Wind Onshore",
    "Offshore wind energy": "Wind Offshore",
    "Geothermal energy": "Geothermal",
}
irena["Tech"] = irena["Technology"].map(TECH_MAP)

# Keep only the requested IRENA techs and rows with CF present
irena_subset = (
    irena[irena["Tech"].isin(TECH_MAP.values())]
    .dropna(subset=["CapacityFactor"])
    .copy()[["Country", "Tech", "CapacityFactor"]]
)
irena_subset["Source"] = "IRENA"

# ENTSO-E: keep all techs
entsoe_cf = entsoe[["Country", "Tech", "CapacityFactor"]].copy()
entsoe_cf["Source"] = "ENTSOE"

# FULL OUTER MERGE so IRENA-only rows are not lost
merged = entsoe_cf.merge(
    irena_subset,
    on=["Country", "Tech"],
    how="outer",
    suffixes=("_ENTSOE", "_IRENA")
)

# Prefer IRENA when available
merged["CapacityFactor"] = merged["CapacityFactor_IRENA"].combine_first(
    merged["CapacityFactor_ENTSOE"]
)
merged["Source"] = merged.apply(
    lambda r: "IRENA" if pd.notna(r.get("CapacityFactor_IRENA")) else "ENTSOE",
    axis=1
)

# Drop categories with no CF
merged = merged.dropna(subset=["CapacityFactor"])

# Build base output
out = merged[["Country", "Tech", "CapacityFactor", "Source"]].sort_values(
    ["Country", "Tech"]
).reset_index(drop=True)

# ----- Manual CF overrides -----
OVERRIDES = [
    # Sweden
    {"Country": "SE", "Tech": "Hydro Water Reservoir",           "CapacityFactor": 0.459, "Source": "SCB"},
    {"Country": "SE", "Tech": "Other",                           "CapacityFactor": 0.22,  "Source": "SCB"},
    {"Country": "SE", "Tech": "Gas",                             "CapacityFactor": 0.22,  "Source": "SCB"},
    {"Country": "SE", "Tech": "Hydro Run-of-river and poundage", "CapacityFactor": 0.459, "Source": "SCB"},

    # Switzerland (energy-charts)
    {"Country": "CH", "Tech": "Hydro Run-of-river and poundage", "CapacityFactor": 0.49,  "Source": "energy-charts (CH)"},
    {"Country": "CH", "Tech": "Hydro Water Reservoir",           "CapacityFactor": 0.27,  "Source": "energy-charts (CH)"},

    # Other
    {"Country": "RS", "Tech": "Biomass",                         "CapacityFactor": 0.24,  "Source": "Irena 24MW and 11.39GWh (2019)"},
    {"Country": "GR", "Tech": "Other",                           "CapacityFactor": 0.41,  "Source": "Ember 1.44GW and 5.23 TWh (2019)"},
]

def apply_overrides(df: pd.DataFrame, overrides: list[dict]) -> pd.DataFrame:
    df = df.copy()
    for o in overrides:
        mask = (df["Country"] == o["Country"]) & (df["Tech"] == o["Tech"])
        if mask.any():
            df.loc[mask, "CapacityFactor"] = float(o["CapacityFactor"])
            df.loc[mask, "Source"] = o["Source"]
        else:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Country": o["Country"],
                                "Tech": o["Tech"],
                                "CapacityFactor": float(o["CapacityFactor"]),
                                "Source": o["Source"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return df

out = apply_overrides(out, OVERRIDES).sort_values(["Country", "Tech"]).reset_index(drop=True)
# -------------------------------

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

n_irena = int((out["Source"] == "IRENA").sum())
n_overrides = len(OVERRIDES)
print(f"[OK] Stats ... (rows={len(out)}, IRENA_rows={n_irena}, manual_overrides={n_overrides})")
