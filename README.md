# Repository — "Do Generation Diversity and Renewables Mitigate Price Shocks?"

This repository reproduces the data and regression results from the accompanying journal article:
"Do Generation Diversity and Renewables Mitigate Price Shocks? Empirical Evidence from the 2022 European Energy Crisis"

It computes:
- Capacity factors (ENTSO-E + IRENA, with manual overrides)
- Adjusted Capacity Mix (ACM) for 2021
- Diversity metrics (Shannon–Wiener Index, Stirling Index)
- Load controls (2019 vs 2022 deltas)
- Price metrics (2017–2019 baseline vs 2022)
- Trade exposure metric (PEI) (ENTSO-E + Eurostat)
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

ENTSO-E API key
Some scripts require access to the ENTSO-E Transparency Platform API.
Set your API key as an environment variable:
export ENTSOE_API_KEY="YOUR_KEY_HERE"


Replication pipeline (run order)

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

9) Compute price metrics (baseline vs 2022 deltas)
Computes price change controls from either:
ENTSO-E day-ahead prices, or
Ember price dataset (for specific areas)
plus load-weighting from ENTSO-E load series
python scripts/price_metrics_computation.py
Outputs
data/processed_final_variables/price_metrics_all_years.csv
data/processed_final_variables/price_metrics_2022_deltas.csv

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
Combines all computed metrics into a single model-ready table
python scripts/merge_all_variables.py
Output
data/processed_final_variables/final_variables_merged.csv

12) Run robust regression models (RLM)
Runs all 19 Robust Linear Models (Tukey Biweight, HC3 covariance) and prints:
coefficients
robust standard errors
p-values
confidence intervals
weighted pseudo-R2
python scripts/RLM_HC3.py
Output
printed regression tables to console

Data
Raw datasets are stored in:
data/raw_data/

Reproducibility and outputs
All scripts are written to be reproducible with fixed input data, and output into:

data/processed_data/ (intermediate results)
data/processed_final_variables/ (final model-ready tables)

