import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # repository/
DATA_FILE = ROOT / "data" / "processed_final_variables" / "final_variables_merged.csv"

df = pd.read_csv(DATA_FILE)

# If bidding_zone exists, keep it as a label column (not required for regression)
if "bidding_zone" in df.columns:
    df["bidding_zone"] = df["bidding_zone"].astype(str).str.strip()

# Models (base)
models_base = {

    "Model 1 (Price ~ SWI)": "delta_price ~ SWI + PEI",
    "Model 2 (P95 ~ SWI)":   "delta_p95 ~ SWI + PEI",
    "Model 3 (CV ~ SWI)":    "delta_cv ~ SWI + PEI",


    "Model 4 (Price ~ Stirling)": "delta_price ~ stirling_index_2021 + PEI",
    "Model 5 (P95 ~ Stirling)":   "delta_p95 ~ stirling_index_2021 + PEI",
    "Model 6 (CV ~ Stirling)":    "delta_cv ~ stirling_index_2021 + PEI",


    "Model 7 (Price ~ Tech shares)": "delta_price ~ share_hydro_reservoir + share_vre + share_gas + PEI",
    "Model 8 (P95 ~ Tech shares)":   "delta_p95 ~ share_hydro_reservoir + share_vre + share_gas + PEI",
    "Model 9 (CV ~ Tech shares)":    "delta_cv ~ share_hydro_reservoir + share_vre + share_gas + PEI",
    # --- Extended model 10 ---
    "Model 10 (Price ~ SWI + share_hydro_reservoir)": "delta_price ~ SWI + share_hydro_reservoir + PEI",
}

# Models (with controls)

models_controls = {
    "Model 11 (Price ~ SWI + controls)": "delta_price ~ SWI + PEI + delta_load_pct + delta_price_gas",
    "Model 12 (P95 ~ SWI + controls)":   "delta_p95 ~ SWI + PEI + delta_load_p95_pct + delta_p95_gas",
    "Model 13 (CV ~ SWI + controls)":    "delta_cv ~ SWI + PEI + delta_load_cv + delta_cv_gas",

    "Model 14 (Price ~ Stirling + controls)": "delta_price ~ stirling_index_2021 + PEI + delta_load_pct + delta_price_gas",
    "Model 15 (P95 ~ Stirling + controls)":   "delta_p95 ~ stirling_index_2021 + PEI + delta_load_p95_pct + delta_p95_gas",
    "Model 16 (CV ~ Stirling + controls)":    "delta_cv ~ stirling_index_2021 + PEI + delta_load_cv + delta_cv_gas",

    "Model 17 (Price ~ shares + controls)": "delta_price ~ share_hydro_reservoir + share_vre + share_gas + PEI + delta_load_pct + delta_price_gas",
    "Model 18 (P95 ~ shares + controls)":   "delta_p95 ~ share_hydro_reservoir + share_vre + share_gas + PEI + delta_load_p95_pct + delta_p95_gas",
    "Model 19 (CV ~ shares + controls)":    "delta_cv ~ share_hydro_reservoir + share_vre + share_gas + PEI + delta_load_cv + delta_cv_gas",
}

# Model 
def run_models(models: dict, df: pd.DataFrame, title: str):
    print(f"\n\n=== {title} ===")
    print("--- MODEL RESULTS (RLM: TukeyBiweight, cov=H3) ---")

    for name, formula in models.items():
        print(f"\n{name}")
        print(f"Formula: {formula}")
        print("-" * 80)

        try:
            res = smf.rlm(
                formula=formula,
                data=df,
                M=sm.robust.norms.TukeyBiweight(),
            ).fit(cov="H3")

            summary_df = pd.DataFrame({
                "coef": res.params,
                "std err": res.bse,
                "P>|z|": res.pvalues,
                "[0.025": res.conf_int()[0],
                "0.975]": res.conf_int()[1],
            })

            print(summary_df.round(4).to_string())

            # Weighted pseudo-R^2 (Rw^2) 
            y = res.model.endog
            yhat = res.fittedvalues
            w = np.asarray(res.weights, dtype=float)

            w_sum = np.sum(w)
            if w_sum <= 0:
                rw2 = np.nan
            else:
                ybar_w = np.sum(w * y) / w_sum
                num = np.sum(w * (y - yhat) ** 2)
                den = np.sum(w * (y - ybar_w) ** 2)
                rw2 = 1 - num / den if den > 0 else np.nan

            print("-" * 60)
            print(f"Weighted pseudo-R² (Rw²): {rw2:.4f}")
            print("=" * 80)

        except Exception as e:
            print(f"Error: {e}")


# Run models
if __name__ == "__main__":
    run_models(models_base, df, title="BASE MODELS")
    run_models(models_controls, df, title="MODELS WITH CONTROLS")
