# =============================================================================
# MUSC INDIVIDUAL FACILITIES — WEEKLY FORECAST SUBMISSION  (QRF_level_sqrt)
#
# WEEKLY OPERATIONAL LOGIC
#   REFERENCE_DATE = first forecast date / submission week
#   TRAIN_DATA_END = REFERENCE_DATE − 7 days  (snapped to nearest data week)
#
# MODEL
#   QRF with level-sqrt target transform:
#     forward : sqrt(y + 0.01)
#     inverse : max(0, p^2 − 0.01)
#   Uses saved optimal parameters from MUSC_QRF_sqrt_selected_params.csv
#   Final fit: all rows with target_end_date <= TRAIN_DATA_END
#   No CV at submission time
#
# OUTPUT
#   ONE file only — the forecast submission CSV, written directly to
#   SUBMISSION_DIR.  No output folders are created, and no plots,
#   metadata, or archive copies are written.
#
#     <SUBMISSION_DIR>/Hossain_Sakhawat_facility_respiratory_MUSC.csv
#
#   Columns:
#     reference_date, target, location_general, location, target_end_date,
#     value, disease, population, training_validation,
#     estimate_projected_report, imputed, data_source,
#     outcome_measure, output_type, output_type_id
#
#   All date columns are written as "YYYY-MM-DD".
#
#   training_validation:
#     1 = training history (observed, not forecast)
#     0 = forecast (test / future)
# =============================================================================

import os, re, json, time, shutil, warnings

import numpy  as np
import pandas as pd

try:
    from quantile_forest import RandomForestQuantileRegressor
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "quantile-forest"])
    from quantile_forest import RandomForestQuantileRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# ── USER SETTINGS — EDIT THIS EVERY WEEK ─────────────────────────────────────
# =============================================================================
REFERENCE_DATE_STR = "2026-08-08"   # ← first forecast date; change each week

# =============================================================================
# ── PATHS — EDIT THESE ───────────────────────────────────────────────────────
# =============================================================================
INPUT_DIR = (
    r"C:\Users\mdsak\Box\BoxPHI-PHMR Projects\Data\MUSC"
    r"\Infectious Disease EHR\Weekly Data\Latest Weekly Data"
)
BASE_DIR = (
    r"C:\Users\mdsak\Box\BoxPHI-PHMR Projects\Sakhawat"
    r"\Rt_Forecast_EHR_Facility"
)
# Saved optimal params from the model-evaluation script
PARAMS_CSV = os.path.join(
    BASE_DIR, "submission", "musc_qrf_sqrt",
    "MUSC_QRF_sqrt_selected_params.csv"
)

# =============================================================================
# ── SUBMISSION DROP-OFF ──────────────────────────────────────────────────────
#   The submission CSV is written directly here. This folder must already
#   exist — the script does not create any directories.
# =============================================================================
SUBMISSION_DIR = (
    r"C:/Users/AMBLEIC/Box/BoxPHI-PHMR Projects/Forecasting Resources"
    r"/Forecast-Drop-Off/Weekly Updates/Implementation"
)
SUBMISSION_NAME = "Hossain_Sakhawat_facility_respiratory_MUSC"   # no extension

# Set True to append the reference date to the drop-off file name
STAMP_SUBMISSION_NAME = False

# Date format used for every date column written to the submission CSV
DATE_FMT = "%Y-%m-%d"

# =============================================================================
# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# =============================================================================
RANDOM_STATE  = 42
np.random.seed(RANDOM_STATE)

TRAIN_START    = pd.to_datetime("2020-01-01")
REFERENCE_DATE = pd.to_datetime(REFERENCE_DATE_STR)
TRAIN_DATA_END = REFERENCE_DATE - pd.Timedelta(days=7)

HORIZONS    = [1, 2, 3, 4]
H_TO_EPR    = {1:0, 2:1, 3:2, 4:3}   # estimate_projected_report
SQRT_SHIFT  = 0.01
MIN_TRAIN   = 40
MODEL_JOBS  = -1

SUBMISSION_QUANTILES = [
    0.01, 0.025, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.95, 0.975, 0.99,
]

MUSC_FACILITIES = [
    "MUSC HEALTH BLACK RIVER MEDICAL CENTER",
    "MUSC-DCM HEALTH CHESTER MEDICAL CENTER",
    "MUSC HEALTH COLUMBIA MEDICAL CENTER DOWNTOWN",
    "MUSC HEALTH COLUMBIA MEDICAL CTR NE",
    "MUSC-FLO HEALTH FLORENCE MEDICAL CENTER",
    "MUSC HEALTH KERSHAW MEDICAL CENTER",
    "MUSC-LCC HEALTH LANCASTER MEDICAL CENTER",
    "MUSC-MAO HEALTH MARION MEDICAL CENTER",
    "MUSC ORBG HOSPITAL",
    "MUSC MAIN HOSPITAL",
    "MUSC ASHLEY RIVER TOWER",
    "MUSC SHAWN JENKINS CHILDRENS HOSPITAL",
]

# Output CSV field constants
LOCATION_GENERAL = "facility"
DISEASE          = "respiratory_diseases"
POPULATION       = "health_system"
DATA_SOURCE      = "HS"
OUTCOME_MEASURE  = "Weekly_Encounters"

LAG_OPTIONS = [2, 3, 4, 5, 6, 7, 8, 13]
MAX_LAG     = max(LAG_OPTIONS)

DATE_COLS = ["reference_date", "target_end_date"]

COL_ORDER = [
    "reference_date","target","location_general","location",
    "target_end_date","value","disease","population",
    "training_validation","estimate_projected_report",
    "imputed","data_source","outcome_measure",
    "output_type","output_type_id",
]

# =============================================================================
# HELPERS
# =============================================================================
def fmt_date(d):
    """Every date written to the submission CSV goes through here → YYYY-MM-DD."""
    return pd.Timestamp(d).strftime(DATE_FMT)

def enforce_date_format(df, cols=DATE_COLS, fmt=DATE_FMT):
    """Final safety net: re-parse and re-render every date column as YYYY-MM-DD."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.strftime(fmt)
            if out[c].isna().any():
                bad = int(out[c].isna().sum())
                raise ValueError(f"{bad} unparseable date(s) in column '{c}'")
    return out

def submission_filename(ref_str):
    if STAMP_SUBMISSION_NAME:
        return f"{SUBMISSION_NAME}_{ref_str}.csv"
    return f"{SUBMISSION_NAME}.csv"

def save_submission(df, path):
    """Write the submission CSV. No directories are created."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Submission folder does not exist: {folder}\n"
            f"Check SUBMISSION_DIR / that Box is synced.")
    tmp = path + f".tmp_{int(time.time())}"
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    if os.path.exists(path): os.remove(path)
    shutil.move(tmp, path)
    print(f"  ✓  Saved: {os.path.basename(path)}")

def normalize_name(x):
    x = str(x).upper().replace("&","AND")
    return re.sub(r"\s+"," ", re.sub(r"[^A-Z0-9]+"," ",x)).strip()

def q_str(q):
    return f"{q:.3f}".rstrip("0").rstrip(".")

def fwd(y):
    return np.sqrt(np.maximum(np.asarray(y,float), 0.) + SQRT_SHIFT)

def inv(p):
    return np.maximum(0., np.power(np.maximum(np.asarray(p,float), 0.), 2.) - SQRT_SHIFT)

def safe_log_ratio(a, b):
    return np.log((np.asarray(a,float)+1.)/(np.asarray(b,float)+1.))

def trailing_rank(s, w):
    def rl(x): return pd.Series(x).rank(pct=True).iloc[-1]
    return s.rolling(w, min_periods=w).apply(rl, raw=False)

def mk_num(df, cols):
    out = df[cols].copy()
    for c in cols: out[c] = pd.to_numeric(out[c],errors="coerce")
    return out.astype(float)

# =============================================================================
# DATA LOADING
# =============================================================================
def find_location_col(df):
    for c in ["Location","Facility","DEPARTMENT_CENTER",
               "DEPARTMENT_LOCATION","DEPARTMENT_NAME","FACILITY"]:
        if c in df.columns: return c
    raise ValueError(f"No location col: {list(df.columns)}")

def find_disease_file(input_dir, disease):
    patterns = {"COVID":["covid","covid-19"],"Flu":["influenza","flu"],"RSV":["rsv"]}[disease]
    all_csv  = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    cands = []
    for f in all_csv:
        fl = f.lower()
        if not any(p in fl for p in patterns): continue
        score = sum(t in fl for t in ["weekly","facility","burden","dx","musc","health"])
        if "weekly"   in fl: score += 2
        if "facility" in fl: score += 2
        cands.append((score, f))
    if not cands: raise FileNotFoundError(f"No file for {disease} in {input_dir}")
    return sorted(cands, key=lambda x:(-x[0],x[1]))[0][1]

def load_disease_file(input_dir, fname):
    df = pd.read_csv(os.path.join(input_dir, fname))
    df = df.rename(columns={find_location_col(df):"Location"})
    df["Week"] = pd.to_datetime(df["Week"],errors="coerce")
    for c in ["Weekly_Encounters","Weekly_Positive_Tests"]:
        df[c] = pd.to_numeric(df[c],errors="coerce").fillna(0.)
    return df[["Location","Week","Weekly_Encounters",
               "Weekly_Positive_Tests"]].dropna(subset=["Week","Location"])

def match_facilities(raw, requested):
    actual = sorted(raw["Location"].dropna().astype(str).unique())
    anmap  = {normalize_name(a):a for a in actual}
    anlist = [(normalize_name(a),a) for a in actual]
    matched, unmatched = [], []
    for req in requested:
        s = str(req).strip()
        if s in actual: matched.append(s); continue
        n = normalize_name(s)
        if n in anmap: matched.append(anmap[n]); continue
        cont = [a for an,a in anlist if n in an or an in n]
        if   len(cont)==1: matched.append(cont[0])
        elif len(cont)>1:  matched.append(sorted(cont,key=len)[0])
        else:              unmatched.append(s)
    matched = sorted(list(dict.fromkeys(matched)))
    print(f"  Matched {len(matched)}/{len(requested)} facilities")
    if unmatched: print(f"  Unmatched: {unmatched}")
    return matched

def load_musc_data(input_dir, facilities):
    frames = []
    for dis in ["COVID","Flu","RSV"]:
        fn = find_disease_file(input_dir, dis)
        print(f"  {dis}: {fn}")
        frames.append(load_disease_file(input_dir, fn))
    raw = pd.concat(frames, ignore_index=True)
    raw = (raw.groupby(["Location","Week"],as_index=False)
              [["Weekly_Encounters","Weekly_Positive_Tests"]].sum())
    raw = raw[raw["Week"]>=TRAIN_START].copy()
    matched = match_facilities(raw, facilities)
    return raw[raw["Location"].isin(matched)].copy(), matched

# =============================================================================
# FEATURE ENGINEERING  (full feature set, same as model-evaluation script)
# =============================================================================
EXTRA_FEATURES = [
    "sin_woy","cos_woy","enc_current",
    "enc_growth_1w","enc_growth_2w","enc_growth_4w","enc_growth_8w","enc_growth_accel",
    "pos_growth_1w","pos_growth_2w","pos_growth_4w","pos_growth_accel",
    "enc_rmean_4w","enc_rmean_13w","enc_momentum_4v13","enc_rank_13w",
    "positivity_lag1","positivity_lag2","positivity_growth_2w",
]

def compute_features(df):
    df  = df.sort_values("Week").copy().reset_index(drop=True)
    enc = pd.to_numeric(df["Weekly_Encounters"],    errors="coerce").astype(float)
    pos = pd.to_numeric(df["Weekly_Positive_Tests"],errors="coerce").astype(float)
    for k in range(1, MAX_LAG+1):
        df[f"enc_lag{k}"] = enc.shift(k)
        df[f"pos_lag{k}"] = pos.shift(k)
    woy = pd.to_datetime(df["Week"]).dt.isocalendar().week.astype(float)
    df["sin_woy"]          = np.sin(2.*np.pi*woy/52.)
    df["cos_woy"]          = np.cos(2.*np.pi*woy/52.)
    df["enc_current"]      = enc
    df["enc_growth_1w"]    = safe_log_ratio(enc.shift(1), enc.shift(2))
    df["enc_growth_2w"]    = safe_log_ratio(enc.shift(1), enc.shift(3))
    df["enc_growth_4w"]    = safe_log_ratio(enc.shift(1), enc.shift(5))
    df["enc_growth_8w"]    = safe_log_ratio(enc.shift(1), enc.shift(9))
    df["enc_growth_accel"] = df["enc_growth_1w"] - df["enc_growth_1w"].shift(1)
    df["pos_growth_1w"]    = safe_log_ratio(pos.shift(1), pos.shift(2))
    df["pos_growth_2w"]    = safe_log_ratio(pos.shift(1), pos.shift(3))
    df["pos_growth_4w"]    = safe_log_ratio(pos.shift(1), pos.shift(5))
    df["pos_growth_accel"] = df["pos_growth_1w"] - df["pos_growth_1w"].shift(1)
    df["enc_rmean_4w"]     = enc.shift(1).rolling(4,  min_periods=4 ).mean()
    df["enc_rmean_13w"]    = enc.shift(1).rolling(13, min_periods=13).mean()
    df["enc_momentum_4v13"]= df["enc_rmean_4w"] - df["enc_rmean_13w"]
    df["enc_rank_13w"]     = trailing_rank(enc.shift(1), 13)
    pos_rate = pos/(enc+1.)
    df["positivity_lag1"]      = pos_rate.shift(1)
    df["positivity_lag2"]      = pos_rate.shift(2)
    df["positivity_growth_2w"] = pos_rate.shift(1) - pos_rate.shift(3)
    for c in df.columns:
        if c not in ["Location","Week"]:
            df[c] = pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    return df

def feat_cols_for_lag(df, lag_n):
    cols  = [f"enc_lag{k}" for k in range(1,lag_n+1)]
    cols += [f"pos_lag{k}" for k in range(1,lag_n+1)]
    cols += EXTRA_FEATURES
    return [c for c in dict.fromkeys(cols) if c in df.columns]

# =============================================================================
# LOAD SAVED OPTIMAL PARAMETERS
# =============================================================================
def load_saved_params(params_csv):
    """
    Load the CSV produced by the model-evaluation script.
    Returns dict keyed by (location_normalized, horizon) → {lag_n, params}
    """
    df = pd.read_csv(params_csv)
    store = {}
    for _, row in df.iterrows():
        loc = str(row["location"])
        h   = int(row["forecast_horizon"])
        store[(normalize_name(loc), h)] = {
            "lag_n":  int(row["best_lag"]),
            "params": json.loads(row["best_params_json"]),
        }
    print(f"  Loaded {len(store)} param entries from {os.path.basename(params_csv)}")
    return store

def get_params(store, loc_name, horizon):
    key = (normalize_name(loc_name), horizon)
    if key in store:
        return store[key]["lag_n"], store[key]["params"]
    # Fallback: horizon-only match (first location with that horizon)
    for (loc_n, h), v in store.items():
        if h == horizon:
            print(f"  Warning: using fallback params for {loc_name} H={horizon}")
            return v["lag_n"], v["params"]
    raise KeyError(f"No saved params for {loc_name}, H={horizon}")

# =============================================================================
# BUILD DATASET FOR FINAL FIT
# =============================================================================
def build_train_dataset(loc_df, horizon, lag_n):
    df = (loc_df[["Week","Weekly_Encounters","Weekly_Positive_Tests"]]
          .copy().sort_values("Week").reset_index(drop=True))
    df = compute_features(df)
    fc = feat_cols_for_lag(df, lag_n)
    df["y_future"]        = df["Weekly_Encounters"].shift(-horizon).astype(float)
    df["target"]          = fwd(df["y_future"])
    df["target_end_date"] = df["Week"] + pd.Timedelta(weeks=horizon)
    for c in fc+["target","y_future"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    df = df.dropna(subset=fc+["target","y_future"]).copy()
    # Final fit: only rows whose target_end_date <= TRAIN_DATA_END (no leakage)
    df = df[df["target_end_date"] <= TRAIN_DATA_END].copy()
    return df, fc

def build_forecast_row(loc_df, lag_n):
    """
    Build the single feature row for forecasting from TRAIN_DATA_END.
    Leakage-safe: uses only data up to TRAIN_DATA_END.
    """
    ser = (loc_df[loc_df["Week"] <= TRAIN_DATA_END]
           [["Week","Weekly_Encounters","Weekly_Positive_Tests"]]
           .copy().sort_values("Week").reset_index(drop=True))
    if len(ser)==0: return None, None
    ser = compute_features(ser)
    fc  = feat_cols_for_lag(ser, lag_n)
    row = ser[ser["Week"]==TRAIN_DATA_END].copy()
    if len(row)==0: return None, None
    for c in fc:
        if c not in row.columns: row[c]=np.nan
        row[c] = pd.to_numeric(row[c],errors="coerce")
    X = row[fc].astype(float)
    if X.isna().any().any(): return None, None
    return X, fc

# =============================================================================
# OUTPUT ROW BUILDERS
# =============================================================================
def obs_history_rows(loc_name, loc_df):
    """
    One row per historical week.
    training_validation       = 1    (observed/training)
    estimate_projected_report = 2    (observed)
    output_type_id            = NA   (not a quantile forecast)
    """
    rows = []
    hist = loc_df[loc_df["Week"] <= TRAIN_DATA_END].sort_values("Week")
    for _, r in hist.iterrows():
        for h in HORIZONS:
            rows.append({
                "reference_date":            fmt_date(REFERENCE_DATE),
                "target":                    None,
                "location_general":          LOCATION_GENERAL,
                "location":                  loc_name,
                "target_end_date":           fmt_date(r["Week"]),
                "value":                     int(round(max(0., float(r["Weekly_Encounters"])))),
                "disease":                   DISEASE,
                "population":                POPULATION,
                "training_validation":       1,
                "estimate_projected_report": 2,
                "imputed":                   0,
                "data_source":               DATA_SOURCE,
                "outcome_measure":           OUTCOME_MEASURE,
                "output_type":               "quantile",
                "output_type_id":            "NA",
            })
    return pd.DataFrame(rows).drop_duplicates(
        subset=["location","target_end_date","estimate_projected_report"]
    ).to_dict("records")


def forecast_quantile_rows(loc_name, h, q_map):
    """
    23 quantile rows per forecast horizon.
    training_validation       = 0              (forecast)
    estimate_projected_report = 1              (forecast)
    output_type_id            = q_str(q)       (0.01 … 0.99)
    """
    tend = REFERENCE_DATE + pd.Timedelta(weeks=h-1)
    rows = []
    for q, val in sorted(q_map.items()):
        rows.append({
            "reference_date":            fmt_date(REFERENCE_DATE),
            "target":                    None,
            "location_general":          LOCATION_GENERAL,
            "location":                  loc_name,
            "target_end_date":           fmt_date(tend),
            "value":                     int(round(max(0., float(val)))),
            "disease":                   DISEASE,
            "population":                POPULATION,
            "training_validation":       0,
            "estimate_projected_report": 1,
            "imputed":                   0,
            "data_source":               DATA_SOURCE,
            "outcome_measure":           OUTCOME_MEASURE,
            "output_type":               "quantile",
            "output_type_id":            q_str(float(q)),
        })
    return rows

# =============================================================================
# PROCESS ONE FACILITY
# =============================================================================
def process_facility(loc_name, loc_df, param_store):
    obs_rows  = []         # observed history rows
    fc_q_rows = []         # quantile rows (all 23 quantiles)

    # Observed history rows
    obs_rows.extend(obs_history_rows(loc_name, loc_df))

    # Forecast for each horizon
    for h in HORIZONS:
        try:
            lag_n, best_params = get_params(param_store, loc_name, h)
        except KeyError as e:
            print(f"    {e}"); continue

        # Build training dataset
        train_df, fc = build_train_dataset(loc_df, h, lag_n)
        if len(train_df) < MIN_TRAIN:
            print(f"    Skip H={h}: only {len(train_df)} train rows"); continue

        X_tr = mk_num(train_df, fc)
        y_tr = train_df["target"].values
        vm   = np.isfinite(X_tr).all(1) & np.isfinite(y_tr)
        X_tr = X_tr.loc[vm].values; y_tr = y_tr[vm]
        if len(X_tr) < MIN_TRAIN:
            print(f"    Skip H={h}: only {len(X_tr)} valid rows"); continue

        # Build forecast feature row
        X_fc, _ = build_forecast_row(loc_df, lag_n)
        if X_fc is None:
            print(f"    Skip H={h}: could not build forecast row"); continue
        # Re-align columns
        for c in fc:
            if c not in X_fc.columns: X_fc[c] = np.nan
        X_fc_vals = X_fc[fc].astype(float).values

        # Fit final model
        params = dict(best_params)
        params.update({"random_state":RANDOM_STATE,"n_jobs":MODEL_JOBS})
        model = RandomForestQuantileRegressor(**params)
        model.fit(X_tr, y_tr)

        # Predict all submission quantiles
        pq = np.asarray(model.predict(X_fc_vals, quantiles=SUBMISSION_QUANTILES), float)
        if pq.ndim == 2: pq = pq[0]
        pq = np.maximum(0., pq)

        # Back-transform (level_sqrt inverse)
        pq_level = inv(pq)

        q_map = {q: float(pq_level[i]) for i,q in enumerate(SUBMISSION_QUANTILES)}
        pred_median = q_map[0.50]

        # All 23 quantile rows
        fc_q_rows.extend(forecast_quantile_rows(loc_name, h, q_map))

        print(f"    H={h}  lag={lag_n}  median={pred_median:.1f}  "
              f"90%CI=[{q_map[0.05]:.1f}, {q_map[0.95]:.1f}]")

    return obs_rows, fc_q_rows

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0      = time.time()
    ref_str = REFERENCE_DATE.strftime(DATE_FMT)

    submission_path = os.path.join(SUBMISSION_DIR, submission_filename(ref_str))

    print("="*70)
    print("MUSC INDIVIDUAL FACILITIES — WEEKLY FORECAST SUBMISSION")
    print(f"  Reference date : {ref_str}")
    print(f"  Train data end : {fmt_date(TRAIN_DATA_END)}")
    print(f"  Params CSV     : {PARAMS_CSV}")
    print(f"  Submission     : {submission_path}")
    print("="*70)

    # Load data
    print("\nLoading MUSC data …")
    raw_fac, matched = load_musc_data(INPUT_DIR, MUSC_FACILITIES)
    print(f"  {len(matched)} facilities | "
          f"{fmt_date(raw_fac['Week'].min())} → {fmt_date(raw_fac['Week'].max())}")

    # Snap TRAIN_DATA_END to the latest available data week <= computed TRAIN_DATA_END
    avail_weeks = pd.to_datetime(sorted(raw_fac["Week"].unique()))
    valid_weeks = avail_weeks[avail_weeks <= TRAIN_DATA_END]
    if len(valid_weeks) == 0:
        raise ValueError(
            f"No data weeks found on or before TRAIN_DATA_END={fmt_date(TRAIN_DATA_END)}. "
            f"Earliest available: {fmt_date(avail_weeks.min())}")
    TRAIN_DATA_END = valid_weeks.max()
    print(f"  TRAIN_DATA_END snapped to nearest data week: {fmt_date(TRAIN_DATA_END)}")

    # Load saved params
    print("\nLoading saved optimal parameters …")
    param_store = load_saved_params(PARAMS_CSV)

    # Process each facility
    all_obs = []; all_fc_q = []

    for loc in sorted(matched):
        print(f"\n  [{loc}]")
        loc_df = (raw_fac[raw_fac["Location"]==loc]
                  .copy().sort_values("Week").reset_index(drop=True))
        obs_rows, fc_q_rows = process_facility(loc, loc_df, param_store)
        all_obs.extend(obs_rows)
        all_fc_q.extend(fc_q_rows)

    # ── Build final submission CSV ────────────────────────────────────────────
    # Observed rows: TV=1, EPR=2, output_type_id="NA"
    # Forecast rows: TV=0, EPR=1, output_type_id=q_str(q)  (23 quantiles × 4 horizons)
    obs_dedup = pd.DataFrame(all_obs).drop_duplicates(
        subset=["location","target_end_date","estimate_projected_report"])
    fc_q_df   = pd.DataFrame(all_fc_q)

    if fc_q_df.empty:
        raise RuntimeError("No quantile forecast rows were generated — check that "
                           "process_facility returned fc_q_rows correctly.")

    final_df = pd.concat([obs_dedup, fc_q_df], ignore_index=True)[COL_ORDER]

    # Enforce YYYY-MM-DD on every date column before writing
    final_df = enforce_date_format(final_df)

    # Write the submission file directly to the drop-off folder
    print()
    save_submission(final_df, submission_path)

    n_obs = (final_df["training_validation"] == 1).sum()
    n_fc  = (final_df["training_validation"] == 0).sum()
    n_fac = final_df[final_df["training_validation"]==0]["location"].nunique()
    print(f"\n  Total rows       : {len(final_df)}")
    print(f"  TV=1 observed    : {n_obs}")
    print(f"  TV=0 forecast    : {n_fc}  "
          f"({n_fac} facilities × {len(HORIZONS)} horizons × 23 quantiles = "
          f"{n_fac * len(HORIZONS) * 23} expected)")
    fc_qtypes = sorted(final_df[final_df["training_validation"]==0]["output_type_id"].unique())
    print(f"  Quantile levels  : {len(fc_qtypes)} unique  {fc_qtypes}")
    print(f"  reference_date   : {sorted(final_df['reference_date'].unique())}")
    print(f"  target_end_date  : {min(final_df['target_end_date'])} → "
          f"{max(final_df['target_end_date'])}")

    print(f"\n{'='*70}")
    print(f"Done.  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"Submission: {submission_path}")
    print("="*70)
