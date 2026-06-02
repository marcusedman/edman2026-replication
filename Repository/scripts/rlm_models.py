"""
rlm_models.py
-------------
Robust linear models (RLM, Tukey biweight, H3 covariance), including VIF checks and correlation matrices.
The outcome variables (delta_price, delta_p95, delta_cv) are loaded from the annual delta file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ======================
# CONFIG 
# ======================

OUTCOME_YEAR   = "2022"   # "2022" or "2023"

EXTRA_CONTROLS = ["PEI"]  # list of additional regressors to add to all models (add "IC_95" for Table 6 models)

EXCLUDE_ZONES  = []

# ===========================================================================

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_final_variables"

MERGED_CSV = DATA_DIR / "final_variables_merged.csv"
ANNUAL_CSV = DATA_DIR / "annual_delta_price_metrics.csv"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

BASE_MODELS = {
    "Model 1  (Price ~ SWI)":                          "delta_price ~ SWI",
    "Model 2  (P95 ~ SWI)":                            "delta_p95   ~ SWI",
    "Model 3  (CV ~ SWI)":                             "delta_cv    ~ SWI",

    "Model 4  (Price ~ Stirling)":                     "delta_price ~ stirling_index_2021",
    "Model 5  (P95 ~ Stirling)":                       "delta_p95   ~ stirling_index_2021",
    "Model 6  (CV ~ Stirling)":                        "delta_cv    ~ stirling_index_2021",

    "Model 7  (Price ~ Tech shares)":                  "delta_price ~ share_vre + share_hydro_reservoir + share_gas",
    "Model 8  (P95 ~ Tech shares)":                    "delta_p95   ~ share_vre + share_hydro_reservoir + share_gas",
    "Model 9  (CV ~ Tech shares)":                     "delta_cv    ~ share_vre + share_hydro_reservoir + share_gas",

    "Model 10 (Price ~ SWI + hydro)":                  "delta_price ~ SWI + share_hydro_reservoir",
}


def build_models(base: dict[str, str], extras: list[str]) -> dict[str, str]:
    """Append EXTRA_CONTROLS to each base formula's RHS."""
    if not extras:
        return dict(base)
    suffix = " + " + " + ".join(extras)
    return {name: formula + suffix for name, formula in base.items()}


MODELS = build_models(BASE_MODELS, EXTRA_CONTROLS)


# ---------------------------------------------------------------------------
# Load and merge data
# ---------------------------------------------------------------------------

def load_outcomes(year: str) -> pd.DataFrame:
    """
    Load annual delta columns
    """
    df = pd.read_csv(ANNUAL_CSV)

    rename = {
        f"delta_price_{year}": "delta_price",
        f"delta_p95_{year}":   "delta_p95",
        f"delta_cv_{year}":    "delta_cv",
    }

    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected columns not found in {ANNUAL_CSV.name}: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.rename(columns=rename)
    df = df[["label", "delta_price", "delta_p95", "delta_cv"]]
    df["label"] = df["label"].replace("IT-S-legacy", "IT-S")
    return df


def build_dataset() -> tuple[pd.DataFrame, str]:
    """Merge outcomes with independent variables; return (df, description)."""
    outcomes = load_outcomes(OUTCOME_YEAR)

    merged = pd.read_csv(MERGED_CSV)
    if "bidding_zone" in merged.columns and "label" not in merged.columns:
        merged = merged.rename(columns={"bidding_zone": "label"})
    merged["label"] = merged["label"].astype(str).str.strip()

    df = merged.merge(outcomes, on="label", how="inner", suffixes=("_ref", ""))

    # Drop possible duplicate columns from merge
    for base in ("delta_price", "delta_p95", "delta_cv"):
        ref_col = f"{base}_ref"
        if ref_col in df.columns:
            df = df.drop(columns=ref_col)

    if EXCLUDE_ZONES:
        before = len(df)
        df = df[~df["label"].isin(EXCLUDE_ZONES)]
        print(f"Excluded zones    : {EXCLUDE_ZONES}  ({before - len(df)} removed)")

    desc = f"Annual {OUTCOME_YEAR}"
    print(f"Outcome source  : {desc}")
    print(f"Extra controls  : {EXTRA_CONTROLS if EXTRA_CONTROLS else '(none)'}")
    print(f"Zones in dataset: {len(df)}")
    print(f"Zones used      : {sorted(df['label'].tolist())}\n")

    return df, desc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_pseudo_r2(res) -> float:
    """Weighted pseudo-R² (Rw²) using RLM final weights."""
    y    = np.asarray(res.model.endog, dtype=float)
    yhat = np.asarray(res.fittedvalues, dtype=float)
    w    = np.asarray(res.weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan
    ybar_w = (w * y).sum() / w_sum
    num    = (w * (y - yhat) ** 2).sum()
    den    = (w * (y - ybar_w) ** 2).sum()
    return float(1 - num / den) if den > 0 else np.nan


def compute_vif(exog: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in exog.columns if c.lower() != "intercept"]
    X    = np.asarray(exog[cols], dtype=float)
    rows = []
    for i, c in enumerate(cols):
        X_with_const = sm.add_constant(X, has_constant="add")
        try:
            vif = variance_inflation_factor(X_with_const, i + 1)
        except Exception:
            vif = np.nan
        rows.append({"variable": c, "VIF": vif})
    return pd.DataFrame(rows)


def print_correlations(df: pd.DataFrame, outcome_vars: list[str],
                        predictor_vars: list[str]) -> None:
    all_vars      = outcome_vars + predictor_vars
    available     = [v for v in all_vars if v in df.columns]
    missing       = [v for v in all_vars if v not in df.columns]

    if missing:
        print(f"  Warning: variables not found in dataset: {missing}")
    if len(available) < 2:
        print("  Not enough variables for correlation matrix")
        return

    corr = df[available].corr(method="pearson")
    print("\n" + "=" * 80)
    print("  PEARSON CORRELATIONS (r values)")
    print("=" * 80)
    print(corr.round(3).to_string())
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------------

def run_models(df: pd.DataFrame, desc: str) -> None:
    sep_thick = "=" * 80
    sep_thin  = "-" * 80

    print(f"\n{sep_thick}")
    print(f"  RLM RESULTS  |  Tukey Biweight  |  cov=H3")
    print(f"  Outcome: {desc}")
    print(sep_thick)

    # Correlations
    outcome_vars   = ["delta_price", "delta_p95", "delta_cv"]
    all_predictors = set()
    for formula in MODELS.values():
        if "~" in formula:
            for term in formula.split("~")[1].split("+"):
                t = term.strip()
                if t and t not in outcome_vars:
                    all_predictors.add(t)
    print(f"\n  Computing correlations for: {outcome_vars + sorted(all_predictors)}")
    print_correlations(df, outcome_vars, sorted(all_predictors))

    for name, formula in MODELS.items():
        print(f"\n{name}")
        print(f"  Formula : {formula}")
        print(sep_thin)

        try:
            model = smf.rlm(
                formula=formula,
                data=df,
                M=sm.robust.norms.TukeyBiweight(),
            )
            res = model.fit(cov="H3")

            summary_df = pd.DataFrame({
                "coef":    res.params,
                "std_err": res.bse,
                "P>|z|":   res.pvalues,
                "[0.025":  res.conf_int()[0],
                "0.975]":  res.conf_int()[1],
            })
            print(summary_df.round(4).to_string())

            print(sep_thin)
            print(f"  Weighted pseudo-R² (Rw²) : {weighted_pseudo_r2(res):.4f}")

            exog_df = pd.DataFrame(model.exog, columns=model.exog_names)
            vif_df  = compute_vif(exog_df)
            print(sep_thin)
            print("  VIF (variance inflation factors):")
            print(vif_df.round(3).to_string(index=False))
            print(sep_thick)

        except Exception as exc:
            print(f"  ERROR: {exc}")
            print(sep_thick)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, desc = build_dataset()
    run_models(df, desc)