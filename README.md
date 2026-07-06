# EHR-Based Respiratory Disease Facility-Level Forecasting

This repository contains code for short-term forecasting of respiratory disease burden using electronic health record (EHR) data from South Carolina health systems. The workflow is designed to generate probabilistic weekly forecasts of respiratory encounters for COVID-19, influenza, and RSV to support operational planning, situational awareness, and health-system decision-making.

The current implementation focuses on **MUSC Health aggregate forecasting** using a **Quantile Random Forest (QRF)** model with a growth-based target transformation. The workflow can be extended to Prisma Health and individual facility-level forecasts.

---

## Project Overview

Respiratory disease burden can change quickly across health systems and facilities. Timely forecasts of weekly encounters can help health-system partners anticipate near-term demand and support resource planning.

This project uses weekly EHR-derived indicators, including respiratory encounters and positive tests, to forecast future respiratory encounter burden. The current weekly forecasting script:

* Combines COVID-19, influenza, and RSV weekly EHR data.
* Aggregates encounters across selected MUSC facilities.
* Builds lagged, seasonal, growth, momentum, positivity, and cross-facility predictors.
* Fits a Quantile Random Forest model using a growth target transformation.
* Produces 1–4 week ahead probabilistic forecasts.
* Saves a standardized forecast submission CSV.
* Saves model metadata and forecast plots.

---

## Current Model

The main model is:

**Quantile Random Forest with growth target transformation**

The model forecasts future weekly respiratory encounters using the transformed target:

```text
log((y_future + 1) / (y_anchor + 1))
```

where:

* `y_future` is the future weekly encounter count at a given forecast horizon.
* `y_anchor` is the current encounter count at the forecast origin.

Forecasts are transformed back to the original encounter scale using:

```text
max(0, (y_anchor + 1) * exp(prediction) - 1)
```

This transformation helps the model learn relative growth in encounter burden rather than directly modeling raw encounter counts.

---

## Forecast Targets

The current script forecasts:

* **Outcome:** Weekly respiratory encounters
* **Disease grouping:** Combined COVID-19, influenza, and RSV
* **Health system:** MUSC Health
* **Location level:** Health-system aggregate
* **Forecast horizons:** 1, 2, 3, and 4 weeks ahead
* **Forecast type:** Quantile forecast

---

## Forecast Quantiles

The model produces 23 forecast quantiles:

```text
0.01, 0.025, 0.05,
0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
0.95, 0.975, 0.99
```

The median forecast corresponds to the `0.50` quantile.

---

## Features

The QRF-growth model uses several groups of predictors.

### Lagged aggregate features

* `enc_lag1` to `enc_lag13`
* `pos_lag1` to `pos_lag13`

### Seasonal features

* `sin_woy`
* `cos_woy`

### Current burden feature

* `enc_current`

### Encounter growth features

* `enc_growth_1w`
* `enc_growth_2w`
* `enc_growth_4w`
* `enc_growth_8w`
* `enc_growth_accel`

### Encounter momentum features

* `enc_rmean_4w`
* `enc_rmean_13w`
* `enc_momentum_4v13`
* `enc_rank_13w`

### Positive test growth features

* `pos_growth_1w`
* `pos_growth_2w`
* `pos_growth_4w`
* `pos_growth_accel`

### Positivity features

* `positivity_lag1`
* `positivity_lag2`
* `positivity_growth_2w`

### Cross-facility lag features

For each selected facility, the model includes facility-level encounter and positive-test lags:

```text
facenc_<facility>_lag1
facenc_<facility>_lag2
facpos_<facility>_lag1
facpos_<facility>_lag2
```

---

## Repository Structure

A recommended repository structure is:

```text
ehr-respiratory-facility-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config_template.yaml
│
├── scripts/
│   └── run_musc_aggregate_qrf_growth.py
│
├── src/
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── target_transforms.py
│   ├── qrf_model.py
│   ├── forecast_output.py
│   └── plotting.py
│
├── docs/
│   ├── workflow.md
│   └── data_dictionary.md
│
├── model_eval/
│   └── README.md
│
└── final_forecast/
    └── README.md
```

---

## Data

The model uses weekly EHR-derived facility-level data for:

* COVID-19
* Influenza
* RSV

Each disease-specific file should include weekly facility-level values for:

* Facility/location
* Week
* Weekly encounters
* Weekly positive tests

Expected columns include:

```text
Location
Week
Weekly_Encounters
Weekly_Positive_Tests
```

The script can also detect several possible facility/location column names, including:

```text
Location
Facility
DEPARTMENT_CENTER
DEPARTMENT_LOCATION
DEPARTMENT_NAME
FACILITY
```

---

## Data Privacy and Security

Raw EHR data, protected health information, local Box folders, and health-system operational data should **not** be uploaded to GitHub.

This repository should only contain:

* Code
* Documentation
* Configuration templates
* Example or dummy data, if needed
* Non-sensitive metadata

Do **not** commit:

* Raw EHR data
* Patient-level data
* Health-system operational data
* Local Box folder contents
* Files containing PHI
* Private paths
* Credentials
* Access tokens
* Private parameter files if they reveal sensitive information

Use `.gitignore` to exclude data, outputs, local configuration files, and model artifacts.

---

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
quantile-forest
pyyaml
```

If `quantile-forest` is not installed, the current script attempts to install it automatically.

---

## Configuration

The weekly forecast script uses local paths for:

* Input weekly EHR data
* Project base directory
* Saved optimal parameter JSON
* Forecast output directory

For GitHub, these local paths should be moved into a private configuration file.

Example `config/config_template.yaml`:

```yaml
reference_date: "2026-07-04"

input_dir: "PATH_TO_WEEKLY_EHR_DATA"
base_dir: "PATH_TO_PROJECT_FOLDER"

params_json: "PATH_TO_OPTIMAL_PARAMS_JSON"
output_root: "PATH_TO_OUTPUT_FOLDER"

system: "MUSC"
model: "QRF_growth"

train_start: "2020-01-01"
horizons: [1, 2, 3, 4]
max_lag: 13
facility_lags: 2

location_general: "facility"
disease: "respiratory"
population: "health_system"
data_source: "HS"
outcome_measure: "Weekly_Encounters"
```

A local `config.yaml` file can be created from this template, but it should be excluded from GitHub.

---

## Weekly Forecast Workflow

Each week:

1. Update the reference date.
2. Load the most recent weekly EHR data.
3. Determine the latest training data week.
4. Build aggregate and facility-level features.
5. Load previously saved optimal QRF parameters.
6. Fit the final model using all available training rows.
7. Generate 1–4 week ahead quantile forecasts.
8. Save the standardized forecast CSV.
9. Save model metadata.
10. Generate forecast plots.

The script defines:

```text
REFERENCE_DATE = submission week end date
TRAIN_DATA_END = latest data week <= REFERENCE_DATE - 7 days
```

The training data end date is snapped to the nearest available data week.

---

## Running the Forecast

Example command:

```bash
python scripts/run_musc_aggregate_qrf_growth.py
```

Before running, update the weekly reference date:

```python
REFERENCE_DATE_STR = "YYYY-MM-DD"
```

Also confirm that the input paths and optimal parameter JSON path are correct.

---

## Output Files

The weekly forecast script generates the following outputs.

### 1. Submission CSV

Example file name:

```text
YYYY-MM-DD_MUSC_aggregate_QRF_growth_submission.csv
```

This file includes:

* Observed historical rows
* Forecast quantile rows
* Forecast horizons 1–4
* Standardized output columns

### 2. Metadata JSON

Example file name:

```text
YYYY-MM-DD_MUSC_aggregate_QRF_growth_metadata.json
```

This file records:

* Reference date
* Training data end date
* Model name
* Location
* Facilities used
* Forecast horizons
* Model parameters used

### 3. Forecast plots

Example files:

```text
YYYY-MM-DD_MUSC_aggregate_QRF_growth_forecast.png
YYYY-MM-DD_MUSC_aggregate_QRF_growth_forecast.pdf
```

The plot shows:

* Observed historical weekly encounters
* Median forecast
* 50% prediction interval
* 90% prediction interval
* Forecast horizons 1–4

---

## Standard Output Format

The submission CSV follows this column structure:

```text
reference_date
target
location_general
location
target_end_date
value
disease
population
training_validation
estimate_projected_report
imputed
data_source
outcome_measure
output_type
output_type_id
```

For observed historical rows:

```text
training_validation = 1
estimate_projected_report = 2
output_type_id = NA
```

For forecast rows:

```text
training_validation = 0
estimate_projected_report = 1
output_type = quantile
output_type_id = quantile level
```

---

## MUSC Facilities Used in the Aggregate Model

The current MUSC aggregate model uses the following facilities:

```text
MUSC HEALTH BLACK RIVER MEDICAL CENTER
MUSC-DCM HEALTH CHESTER MEDICAL CENTER
MUSC HEALTH COLUMBIA MEDICAL CENTER DOWNTOWN
MUSC HEALTH COLUMBIA MEDICAL CTR NE
MUSC-FLO HEALTH FLORENCE MEDICAL CENTER
MUSC HEALTH KERSHAW MEDICAL CENTER
MUSC-LCC HEALTH LANCASTER MEDICAL CENTER
MUSC-MAO HEALTH MARION MEDICAL CENTER
MUSC ORBG HOSPITAL
MUSC MAIN HOSPITAL
MUSC ASHLEY RIVER TOWER
MUSC SHAWN JENKINS CHILDRENS HOSPITAL
```

Facility names are matched to available EHR location names using normalized string matching.

---

## Model Parameters

The final forecast script uses saved optimal parameters from a JSON file.

Expected key format:

```text
MUSC__h1
MUSC__h2
MUSC__h3
MUSC__h4
```

Each key should contain the optimal lag length and Quantile Random Forest parameters for the corresponding forecast horizon.

---

## Extending to Prisma or Facility-Level Forecasts

The workflow can be extended by:

1. Updating the facility list.
2. Updating input data directories.
3. Updating the health-system label.
4. Training or loading separate optimal parameters.
5. Modifying the output location field.
6. Running the same forecast workflow for each facility or aggregate system.

Possible extensions include:

* Prisma aggregate forecasts
* Individual MUSC facility forecasts
* Individual Prisma facility forecasts
* QRF, ARIMA, XGBoost, and ensemble model comparison
* Direct and recursive forecasting strategies
* Model evaluation across forecast horizons

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.venv/
venv/

# Data and sensitive files
data/
raw_data/
processed_data/
*.csv
*.xlsx
*.xls
*.parquet
*.rds

# Outputs
final_forecast/
model_eval/
outputs/
plots/
*.png
*.pdf
*.json

# Local configuration
config/config.yaml
*.log

# OS files
.DS_Store
Thumbs.db
```

---

## Suggested Repository Name

Recommended names:

```text
ehr-respiratory-facility-forecasting
```

or

```text
sc-ehr-respiratory-forecasting
```

---

## Authors and Contributors

Md Sakhawat Hossain
Center for Public Health Modeling and Response
Clemson University

Additional contributors may be added based on project roles and authorship decisions.

---

## License

A license should be selected before making the repository public. If the repository contains only code and documentation, possible licenses include:

* MIT License
* Apache License 2.0
* BSD 3-Clause License

If the work is part of a funded or institutional project, confirm the appropriate license with the project team before public release.

---

## Disclaimer

This repository is intended for research and operational forecasting development. Forecast outputs should be interpreted in the context of data availability, reporting delays, model uncertainty, and health-system operational knowledge. Raw EHR data and sensitive health-system information are not included in this repository.
