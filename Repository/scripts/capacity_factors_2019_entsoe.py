# Compute entsoe capacity factors per country and technology 

import argparse
from pathlib import Path
import pandas as pd

# ------
# Config
# ------
HOURS_PER_YEAR_2019 = 8760.0

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTDIR = DATA_DIR / "processed_data"
OUTDIR.mkdir(parents=True, exist_ok=True)

GEN_PER_TYPE = DATA_DIR / "processed_data" / "entsoe_generation_per_type_2019.csv"
CAPACITY_RAW = DATA_DIR / "raw_data" / "entsoe_net_generation_capacity_2019_raw.csv"
OUT_FILE = OUTDIR / "entsoe_capacity_factors_2019.csv"

# Zone codes to country codes
BZ_MAP_TO_CAP = {
    "DE_LU": "DE",
    "IE_SEM": "IE",
}

# Rollup mappings
ROLLUP_CAPACITY_GROUPS = {
    "Oil": {"Fossil Oil", "Fossil Oil shale"},
    "Gas": {"Fossil Gas", "Fossil Coal-derived gas"},
}


# ---------------------------
# Helpers
# ---------------------------
def canon_tech(txt: str) -> str:
    if not isinstance(txt, str):
        return txt
    return txt.split(" - ")[0].strip()


def load_generation_wide(gen_csv: Path | str) -> pd.DataFrame:
    gen = pd.read_csv(gen_csv)

    if "bidding_zone" not in gen.columns:
        raise ValueError(f"Generation file missing 'bidding_zone': {gen_csv}")

    gen["bidding_zone"] = gen["bidding_zone"].astype(str).str.strip()
    gen["bidding_zone"] = gen["bidding_zone"].replace(BZ_MAP_TO_CAP)

    value_cols = [c for c in gen.columns if c != "bidding_zone"]

    gen_long = gen.melt(
        id_vars=["bidding_zone"],
        value_vars=value_cols,
        var_name="TechRaw",
        value_name="Gen_MWh",
    )

    gen_long["Tech"] = gen_long["TechRaw"].apply(canon_tech)
    gen_long["Gen_MWh"] = pd.to_numeric(gen_long["Gen_MWh"], errors="coerce").astype(float)

    gen_long = (
        gen_long.groupby(["bidding_zone", "Tech"], as_index=False)["Gen_MWh"]
        .sum()
        .query("Gen_MWh.notna()")
    )

    # Fractional splits 
    REALLOC_SPLIT = {
        ("NL", "Other"): {"Gas": 0.75, "Fossil Hard coal": 0.25}, # Source: Ember
        ("IT", "Other"): {"Gas": 0.70, "Fossil Hard coal": 0.15}, # Source: Ember
        ("BE", "Other"): {"Gas": 1.0},                            # Source: Ember
        ("RS", "Other"): {"Fossil Brown coal/Lignite": 0.60},     # Source: Ember
        ("LV", "Other"): {"Gas": 1},                              # Source: Ember
    }

    for (bz, src), mapping in REALLOC_SPLIT.items():
        mask_src = (gen_long["bidding_zone"] == bz) & (gen_long["Tech"] == src)
        mwh = float(gen_long.loc[mask_src, "Gen_MWh"].sum())

        if mwh <= 0:
            continue

        frac_total = float(sum(mapping.values()))
        if frac_total < 0 or frac_total > 1 + 1e-9:
            raise ValueError(f"Fractions for {(bz, src)} must sum to <= 1. Got {frac_total}")

        for dst, frac in mapping.items():
            if dst == src:
                raise ValueError(f"Do not include source tech itself for {(bz, src)}")

            add = mwh * float(frac)
            mask_dst = (gen_long["bidding_zone"] == bz) & (gen_long["Tech"] == dst)

            if mask_dst.any():
                gen_long.loc[mask_dst, "Gen_MWh"] = gen_long.loc[mask_dst, "Gen_MWh"] + add
            else:
                gen_long = pd.concat(
                    [gen_long, pd.DataFrame([{"bidding_zone": bz, "Tech": dst, "Gen_MWh": add}])],
                    ignore_index=True,
                )

        
        gen_long.loc[mask_src, "Gen_MWh"] = mwh * (1.0 - frac_total)

    return gen_long.groupby(["bidding_zone", "Tech"], as_index=False)["Gen_MWh"].sum()


def load_capacity_long(cap_csv: Path | str, capacity_year: int) -> pd.DataFrame:
    cap = pd.read_csv(cap_csv, sep=None, engine="python")

    required = {"Year", "MeasureItem", "Category", "Country", "ProvidedValue"}
    missing = required - set(cap.columns)
    if missing:
        raise ValueError(f"Capacity file missing columns: {missing}")

    cap = cap[
        (cap["Year"] == capacity_year)
        & (cap["MeasureItem"].astype(str).str.contains("Net Generating Capacity", na=False))
    ].copy()

    if cap.empty:
        raise ValueError(
            f"Capacity file has no rows for Year={capacity_year}. "
            f"Check which years exist in the file."
        )

    cap["bidding_zone"] = cap["Country"].astype(str).str.strip()
    cap["Tech"] = cap["Category"].apply(canon_tech)
    cap["Cap_MW"] = pd.to_numeric(cap["ProvidedValue"], errors="coerce").astype(float)

    cap = cap.groupby(["bidding_zone", "Tech"], as_index=False)["Cap_MW"].sum()
    return cap


def build_rollup_capacity(cap_df: pd.DataFrame) -> pd.DataFrame:
    """Add Oil/Gas rollup tech rows to capacity table."""
    rows = []
    for rollup, parts in ROLLUP_CAPACITY_GROUPS.items():
        part_cap = (
            cap_df[cap_df["Tech"].isin(parts)]
            .groupby("bidding_zone", as_index=False)["Cap_MW"]
            .sum()
        )
        if not part_cap.empty:
            part_cap["Tech"] = rollup
            rows.append(part_cap)

    if rows:
        rollups = pd.concat(rows, ignore_index=True)
        return pd.concat([cap_df, rollups], ignore_index=True)

    return cap_df


def compute_capacity_factors(gen_long: pd.DataFrame, cap_df: pd.DataFrame) -> pd.DataFrame:
    cap_all = build_rollup_capacity(cap_df)

    # quick coverage check
    missing_zones = sorted(set(gen_long["bidding_zone"]) - set(cap_all["bidding_zone"]))
    if missing_zones:
        print("[WARN] Zones in generation but missing in capacity:", missing_zones)

    df = gen_long.merge(cap_all, on=["bidding_zone", "Tech"], how="left")

    df["GhostGen"] = (df["Gen_MWh"].fillna(0) > 0) & (df["Cap_MW"].isna() | (df["Cap_MW"] <= 0))

    denom = df["Cap_MW"] * HOURS_PER_YEAR_2019
    df["CapacityFactor"] = df["Gen_MWh"] / denom
    df.loc[(df["Cap_MW"].isna()) | (df["Cap_MW"] <= 0), "CapacityFactor"] = pd.NA

    #  ghost generation rows, assume CF = 0.2
    df.loc[df["GhostGen"], "CapacityFactor"] = 0.2

    return df[["bidding_zone", "Tech", "Gen_MWh", "Cap_MW", "CapacityFactor", "GhostGen"]]


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute capacity factors from gen (2019) + capacity file (country ISO2).")
    ap.add_argument("--gen", type=Path, default=GEN_PER_TYPE, help="Generation wide CSV (2019).")
    ap.add_argument("--cap", type=Path, default=CAPACITY_RAW, help="Capacity raw CSV (; separated).")
    ap.add_argument("--cap_year", type=int, default=2019, help="Capacity year to use.")
    ap.add_argument("--out", type=Path, default=OUT_FILE, help="Output CSV (long).")
    args = ap.parse_args()

    print("ROOT =", ROOT)
    print("GEN_PER_TYPE =", args.gen)
    print("CAPACITY_RAW =", args.cap)
    print("CAPACITY YEAR =", args.cap_year)
    print("OUT_FILE =", args.out)

    gen_long = load_generation_wide(args.gen)
    cap_df = load_capacity_long(args.cap, capacity_year=args.cap_year)

    result_long = compute_capacity_factors(gen_long, cap_df)

    ghost_df = result_long.loc[result_long["GhostGen"]].sort_values(["bidding_zone", "Tech"]).copy()

   
    result_long.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
