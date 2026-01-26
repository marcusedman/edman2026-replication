import pandas as pd
from pathlib import Path


# Config

ROOT = Path(__file__).resolve().parents[1]  # repository/
DATA_DIR = ROOT / "data" / "processed_final_variables"

FILES = [
    "price_metrics_2022_deltas.csv",
    "pei_avg_2017_2019_final.csv",
    "shares_swi_from_acm.csv",
    "stirling_diversity_2021.csv",
    "load_metrics_bz_2022_deltas.csv",
    "gas_price_deltas.csv",
]

INDEX_COL = "bidding_zone"
OUTFILE = DATA_DIR / "final_variables_merged.csv"


def read_with_index(path: Path, index_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if index_col not in df.columns:
        raise ValueError(f"[{path.name}] missing required index column '{index_col}'")

    # ensure index is consistent (string)
    df[index_col] = df[index_col].astype(str).str.strip()
    df = df.set_index(index_col)

    # safety: no duplicate index entries
    if df.index.duplicated().any():
        dups = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"[{path.name}] has duplicated '{index_col}' values: {dups}")

    return df


def check_same_index(dfs: list[pd.DataFrame], names: list[str]) -> None:
    ref = dfs[0].index
    ref_name = names[0]

    for df, name in zip(dfs[1:], names[1:]):
        if not ref.equals(df.index):
            missing_in_df = ref.difference(df.index).tolist()
            extra_in_df = df.index.difference(ref).tolist()

            msg = [
                f"Index mismatch between '{ref_name}' and '{name}'.",
                f"- Same length? {len(ref)} vs {len(df.index)}",
                f"- Missing in {name}: {missing_in_df[:20]}{' ...' if len(missing_in_df) > 20 else ''}",
                f"- Extra in {name}: {extra_in_df[:20]}{' ...' if len(extra_in_df) > 20 else ''}",
                "",
                "Fix: ensure all files contain the same bidding_zone values in the same order.",
            ]
            raise ValueError("\n".join(msg))


def main():
    paths = [DATA_DIR / f for f in FILES]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # read all
    dfs = [read_with_index(p, INDEX_COL) for p in paths]

    # check same index + same order
    check_same_index(dfs, FILES)

    # merge columns 
    merged = pd.concat(dfs, axis=1)

    # check no duplicate column names
    if merged.columns.duplicated().any():
        dup_cols = merged.columns[merged.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names found after merge: {dup_cols}\n"
            "Fix: rename columns in inputs or add suffixes before merging."
        )

    # save
    merged.to_csv(OUTFILE, index=True)
    print(f"Saved merged file: {OUTFILE}")
    print(f"Rows: {merged.shape[0]}, Columns: {merged.shape[1]}")


if __name__ == "__main__":
    main()
