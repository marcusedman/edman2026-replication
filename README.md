# Repository — "Do Generation Diversity and Renewables Mitigate Price Shocks?"

This repository reproduces the data and regression results from the accompanying journal article:
"Do Generation Diversity and Renewables Mitigate Price Shocks? Empirical Evidence from the 2022 European Energy Crisis"

It computes:
- Capacity factors (ENTSO-E + IRENA, with manual overrides)
- Adjusted Capacity Mix (ACM) for 2021
- Diversity metrics (Shannon–Wiener Index, Stirling Index)
- Generation ACM shares (Reservoir hydro, Variable Renewable Energy, Natural Gas)
- Load controls (2019 vs 2022 deltas)
- Trade metric (PEI) (ENTSO-E + Eurostat)
- Final merged dataset used in regressions
- Robust linear models (RLM) with HC3 covariance and model fit metric
---

## Repository structure
repository/
├─ scripts/ # main pipeline scripts
├─ src/ # helper module
├─ data/
│ ├─ raw_data/ # raw input datasets 
│ ├─ processed_data/ # intermediate processed outputs
│ └─ processed_final_variables/ # final merged variables (model-ready)

---

## Requirements

Python 3.10+ recommended.

Install dependencies:

- bash
pip install -r requirements.txt

## Short pipeline — reproduce regression results (no API key required)
**1. Merge all independent variables**
Run: 'python scripts/merge_all_variables.py'
Output: `data/processed_final_variables/final_variables_merged.csv`
2) Run robust models
input: final_variables_merged.csv
input: annual_delta_price_metrics

**2. run /scripts/rlm_models.py**
output: Robust linear models for tables (1-10) and diagnostics. Possible to include IC95 in config for tables 1c-9c)


## Replication pipeline full (API key required))

ENTSO-E API key
Some scripts require access to the ENTSO-E Transparency Platform API.
Set your API key as an environment variable:
export ENTSOE_API_KEY="YOUR_KEY_HERE"

Run the scripts in the order below to regenerate the final dataset and regression models.
1) ENTSO-E generation totals (2019)
Fetch annual generation per bidding zone and technology:
python scripts/entsoe_fetch_gen_per_type_2019.py
Output
data/processed_data/entsoe_generation_per_type_2019.csv

2) ENTSO-E capacity factors (2019)
Combine ENTSO-E generation totals with ENTSO-E net generation capacity data:
python scripts/capacity_factors_2019_entsoe.py
Output
data/processed_data/entsoe_capacity_factors_2019.csv

3) IRENA capacity factors (2017–2021)
Compute 5-year weighted capacity factors for Solar/Wind/Geothermal:
python scripts/irena_capacity_factors_2017_2021.py
Output
data/processed_data/IRENA_capacity_factors_2017_2021.csv

4) Combine ENTSO-E + IRENA capacity factors
Merge ENTSO-E and IRENA factors
python scripts/combine_capacity_factors_irena_entsoe.py
Output
data/processed_data/complete_cf_countries_2019.csv

5) Compute Adjusted Capacity Mix (ACM) for 2021
ACM is computed as:
ACM = Installed Capacity (MW, 2021) × Historical Capacity Factors
python scripts/acm_computation.py
Output
data/processed_data/installed_capacity_adjusted_2021.csv

6) Compute SWI + technology shares from ACM
Computes Shannon–Wiener Index (SWI) and selected ACM shares:
python scripts/shares_swi_from_acm.py
Output
data/processed_final_variables/shares_swi_from_acm.csv'

7) Compute Stirling diversity index (2021)
Computes Stirling diversity using disparity dimensions for dispatchability and operating cost:
python scripts/stirling_index_computation.py
Output
data/processed_final_variables/stirling_diversity_2021.csv

8) Compute load metrics (2019 baseline vs 2022 deltas)
Computes demand change controls:
annual load delta (%)
daily p95 load delta (%)
daily CV delta
python scripts/load_change_forecast_compute.py
Output
data/processed_final_variables/load_metrics_bz_2022_deltas.csv

10) Compute trade exposure (PEI)
10a) PEI from ENTSO-E (2017–2019 average)
python scripts/PEI_entsoe_computation.py
Output
data/processed_data/pei_avg_2017_2019_entsoe.csv
10b) PEI from Eurostat (2017–2019 average)
python scripts/PEI_eurostat_computation.py
Output
data/processed_data/pei_avg_2017_2019_eurostat.csv
10c) Merge PEI metrics (final)
python scripts/merge_pei.py
Output
data/processed_final_variables/pei_avg_2017_2019_final.csv

11) Merge all final variables
Combines all independent metrics into a single model-ready table
python scripts/merge_all_variables.py
Output
processed_final_variables/final_variables_merged.csv

12) Run robust regression models (RLM)
Runs all 19 Robust Linear Models (Tukey Biweight, HC3 covariance) and prints:
coefficients
robust standard errors
p-values
confidence intervals
weighted pseudo-R2
run scripts/rlm_models.py
Output
printed regression tables to console


