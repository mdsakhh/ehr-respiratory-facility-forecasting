# Facility-Level Respiratory Disease Forecasting

Weekly operational forecasts of respiratory disease encounters (COVID-19 + Influenza + RSV)
for MUSC and Prisma Health, at both the individual-facility and health-system-aggregate level.

Each script loads the latest weekly EHR extract, refits a Quantile Random Forest using
previously selected optimal hyperparameters, produces 1–4 week-ahead quantile forecasts,
and writes **one** submission CSV directly to the shared Forecast-Drop-Off folder.

---

## Repository layout

```
facility-respiratory-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
└── scripts/
    ├── MUSC_individual_weekly_forecast_submission.py    → Hossain_Sakhawat_facility_respiratory_MUSC.csv
    ├── PRISMA_individual_weekly_forecast_submission.py  → Hossain_Sakhawat_facility_respiratory_PRISMA.csv
    ├── MUSC_aggregated_weekly_forecast_submission.py    → Hossain_Sakhawat_facility_aggregated_respiratory_MUSC.csv
    └── PRISMA_aggregated_weekly_forecast_submission.py  → Hossain_Sakhawat_facility_aggregated_respiratory_PRISMA.csv
```

No data, forecasts, or PHI are tracked in this repository — see `.gitignore`.

---

## Models

| Script | Scope | Target transform |
|---|---|---|
| `*_individual_weekly_forecast_submission.py` | One QRF per facility | level-sqrt: `sqrt(y + 0.01)` |
| `*_aggregated_weekly_forecast_submission.py` | One QRF per health system | growth: `log((y_future + 1) / (y_anchor + 1))` |

Both use `quantile_forest.RandomForestQuantileRegressor` with hyperparameters read from
files produced by the separate model-evaluation scripts:

- Individual: `*_QRF_sqrt_selected_params.csv` (per facility × horizon)
- Aggregated: `optimal_params_*_QRF_growth.json` (keys `MUSC__h1` … `PRISMA__h4`)

Aggregated models additionally use cross-facility lag features (`facenc_*_lag1/2`,
`facpos_*_lag1/2`).

---

## Output — submission file only

Each script writes exactly one file and nothing else. **No directories are created**, and
no plots, metadata JSON, or archive copies are produced.

```
C:/Users/AMBLEIC/Box/BoxPHI-PHMR Projects/Forecasting Resources/Forecast-Drop-Off/Weekly Updates/Implementation
```

Set per script via `SUBMISSION_DIR`. This folder must already exist — if it doesn't (e.g.
Box isn't synced), the script raises a clear `FileNotFoundError` rather than silently
creating a stray folder.

Files are written with a fixed name and overwritten each week. Set
`STAMP_SUBMISSION_NAME = True` if the reference date should be appended to the file name
instead.

### CSV schema

| Column | Notes |
|---|---|
| `reference_date` | `YYYY-MM-DD` — first forecast date |
| `target` | empty |
| `location_general` | `facility` |
| `location` | facility name, or `MUSC` / `PRISMA` for aggregated |
| `target_end_date` | `YYYY-MM-DD` |
| `value` | integer encounters |
| `disease` | `respiratory_diseases` |
| `population` | `health_system` |
| `training_validation` | `1` = observed history, `0` = forecast |
| `estimate_projected_report` | `2` = observed, `1` = forecast |
| `imputed` | `0` |
| `data_source` | `HS` |
| `outcome_measure` | `Weekly_Encounters` |
| `output_type` | `quantile` |
| `output_type_id` | `NA` for observed; `0.01` … `0.99` for forecasts |

Quantile levels (23): 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99.

### Date formatting

Every date is rendered through `fmt_date()` (`strftime("%Y-%m-%d")`). Immediately before
writing, `enforce_date_format()` re-parses `reference_date` and `target_end_date` and
re-renders them, raising if anything fails to parse. This holds regardless of how the
`Week` column is formatted in the incoming extracts.

---

## Weekly run procedure

1. Confirm the new weekly extracts have landed in `INPUT_DIR` for each system.
2. In each script, set **one** value:

   ```python
   REFERENCE_DATE_STR = "2026-08-08"   # first forecast date for this submission week
   ```

   `TRAIN_DATA_END` is derived as `REFERENCE_DATE − 7 days`, then snapped down to the
   latest week actually present in the data. Nothing at or after `TRAIN_DATA_END + 1 week`
   is used for fitting or feature construction.

3. Run the four scripts (order does not matter):

   ```bash
   python scripts/MUSC_individual_weekly_forecast_submission.py
   python scripts/PRISMA_individual_weekly_forecast_submission.py
   python scripts/MUSC_aggregated_weekly_forecast_submission.py
   python scripts/PRISMA_aggregated_weekly_forecast_submission.py
   ```

4. Check the console summary for each run:
   - `Matched N/N facilities` — investigate any unmatched names
   - `TV=0 forecast` row count matches `facilities × 4 horizons × 23 quantiles`
   - `Quantile levels : 23 unique`
   - `reference_date` / `target_end_date` printed in `YYYY-MM-DD`

5. Confirm the four CSVs are in the drop-off folder.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Python 3.9+. `quantile-forest` will auto-install on first run if missing, but installing
from `requirements.txt` up front is preferred.

---

## Configuration reference

Per-script settings near the top of each file:

| Setting | Purpose |
|---|---|
| `REFERENCE_DATE_STR` | **The only value that changes weekly.** |
| `INPUT_DIR` | Folder holding the weekly extracts for that system |
| `BASE_DIR` | Root used to locate the saved hyperparameter files |
| `PARAMS_CSV` / `PARAMS_JSON` | Saved optimal hyperparameters |
| `SUBMISSION_DIR` | Shared drop-off folder (must already exist) |
| `SUBMISSION_NAME` | Output file name (no extension) |
| `STAMP_SUBMISSION_NAME` | Append reference date to file name |
| `DATE_FMT` | `%Y-%m-%d` |
| `MIN_TRAIN` | 40 (individual) / 60 (aggregated) minimum training rows |
| `RANDOM_STATE` | 42 |

---

## Troubleshooting

**`Submission folder does not exist`** — `SUBMISSION_DIR` is wrong or Box isn't synced.
The script deliberately does not create it.

**`Unmatched: [...]`** — a facility name in the script list doesn't appear in the extract.
Names are matched exact → normalized → substring. Check for a renamed facility.

**`Skip H=n: only N train rows`** — facility has too little history after feature lags and
NaN dropping. Expected for newly added facilities; that facility gets no forecast rows.

**`Key 'MUSC__h1' not found`** — the aggregated params JSON doesn't have the expected keys.
Re-run the corresponding eval script or fix `AGG_LOCATION`.

**`Warning: using fallback params`** — no saved params for that facility/horizon, so the
first available entry for that horizon is used. Re-run the individual eval script to fix.

**`No data weeks found on or before TRAIN_DATA_END`** — the extract is stale relative to
`REFERENCE_DATE_STR`, or the file didn't refresh.

**Unparseable dates** — `enforce_date_format()` raised, meaning the `Week` column contains
values pandas can't parse. Inspect the extract.
