# =============================================================================
# MUSC AGGREGATED — WEEKLY FORECAST SUBMISSION  (QRF_growth)
#
# WEEKLY OPERATIONAL LOGIC
#   REFERENCE_DATE = first forecast date / submission week  ← change each week
#   TRAIN_DATA_END = latest data week ≤ REFERENCE_DATE − 7 days
#                   (snapped to nearest available data week)
#
# MODEL
#   QRF with growth target transform:
#     forward : log((y_future + 1) / (y_anchor + 1))
#     inverse : max(0, (y_anchor + 1) * exp(p) − 1)
#
#   Uses saved optimal parameters from:
#     optimal_params_MUSC_QRF_growth.json
#     key format: "MUSC__h1" … "MUSC__h4"
#
#   Final fit on ALL rows with target_end_date ≤ TRAIN_DATA_END
#   No CV at submission time.
#
# FEATURES
#   enc_lag1..lag_n, pos_lag1..lag_n
#   sin_woy, cos_woy, enc_current
#   enc_growth_1/2/4/8w, enc_growth_accel
#   enc_rmean_4w/13w, enc_momentum_4v13, enc_rank_13w
#   pos_growth_1/2/4w, pos_growth_accel
#   positivity_lag1/2, positivity_growth_2w
#   facenc_<fac>_lag1/2, facpos_<fac>_lag1/2  (cross-facility lags)
#
# OUTPUT
#   ONE file only — the forecast submission CSV, written directly to
#   SUBMISSION_DIR.  No output folders are created, and no plots,
#   metadata, or archive copies are written.
#
#     <SUBMISSION_DIR>/Hossain_Sakhawat_facility_aggregated_respiratory_MUSC.csv
#
#   Columns:
#     reference_date, target, location_general, location,
#     target_end_date, value, disease, population,
#     training_validation, estimate_projected_report,
#     imputed, data_source, outcome_measure,
#     output_type, output_type_id
#
#   All date columns are written as "YYYY-MM-DD".
#
#   training_validation = 1  →  observed history rows
#   training_validation = 0  →  forecast quantile rows  (H=1..4)
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
# ── USER SETTINGS — CHANGE EACH WEEK ─────────────────────────────────────────
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
# JSON saved by the aggregate model-evaluation script
PARAMS_JSON = os.path.join(
    BASE_DIR, "model_eval", "musc_aggregate_qrf_growth_eval_v1",
    "optimal_parameters", "optimal_params_MUSC_QRF_growth.json"
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
SUBMISSION_NAME = "Hossain_Sakhawat_facility_aggregated_respiratory_MUSC"  # no ext

# Set True to append the reference date to the drop-off file name
STAMP_SUBMISSION_NAME = False

# Date format used for every date column written to the submission CSV
DATE_FMT = "%Y-%m-%d"

# =============================================================================
# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

TRAIN_START    = pd.to_datetime("2020-01-01")
REFERENCE_DATE = pd.to_datetime(REFERENCE_DATE_STR)
TRAIN_DATA_END = REFERENCE_DATE - pd.Timedelta(days=7)   # snapped after load

HORIZONS      = [1, 2, 3, 4]
H_TO_EPR      = {1:0, 2:1, 3:2, 4:3}
N_FAC_LAGS    = 2
MIN_TRAIN     = 60
MODEL_JOBS    = -1
AGG_LOCATION  = "MUSC"

SUBMISSION_QUANTILES = np.array([
    0.01, 0.025, 0.05,
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.95, 0.975, 0.99,
], dtype=float)
N_Q        = len(SUBMISSION_QUANTILES)
MEDIAN_IDX = int(np.where(np.isclose(SUBMISSION_QUANTILES, 0.50))[0][0])

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

LOCATION_GENERAL = "facility"
DISEASE          = "respiratory_diseases"
POPULATION       = "health_system"
DATA_SOURCE      = "HS"
OUTCOME_MEASURE  = "Weekly_Encounters"
MAX_LAG          = 13

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
def q_str(q): return f"{q:.3f}".rstrip("0").rstrip(".")

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
    print(f"  ✓  Saved: {os.path.basename(path)}", flush=True)

def normalize_name(x):
    x = str(x).upper().replace("&","AND")
    return re.sub(r"\s+"," ",re.sub(r"[^A-Z0-9]+"," ",x)).strip()

def safe_fac_id(name):
    return re.sub(r"[^A-Za-z0-9]+","_",str(name)).strip("_")[:60]

def monotone(arr):
    return np.maximum.accumulate(np.asarray(arr, dtype=float))

def mk_num(df, cols):
    out = df[cols].copy()
    for c in cols: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype(float)

# =============================================================================
# TARGET TRANSFORMS  (growth)
# =============================================================================
def fwd(yf, ya):
    yf,ya = np.asarray(yf,float), np.asarray(ya,float)
    return np.log((yf+1.)/(ya+1.))

def inv(p, ya):
    p,ya = np.asarray(p,float), np.asarray(ya,float)
    return np.maximum(0., (ya+1.)*np.exp(p)-1.)

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
        if "weekly" in fl:   score += 2
        if "facility" in fl: score += 2
        cands.append((score, f))
    if not cands: raise FileNotFoundError(f"No file for {disease}")
    return sorted(cands, key=lambda x:(-x[0],x[1]))[0][1]

def load_disease_file(input_dir, fname):
    df = pd.read_csv(os.path.join(input_dir, fname))
    df = df.rename(columns={find_location_col(df):"Location"})
    df["Week"] = pd.to_datetime(df["Week"], errors="coerce")
    for c in ["Weekly_Encounters","Weekly_Positive_Tests"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.)
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
    print(f"  Matched {len(matched)}/{len(requested)} facilities", flush=True)
    if unmatched: print(f"  Unmatched: {unmatched}", flush=True)
    return matched

def load_musc_data(input_dir, facilities):
    frames = []
    for dis in ["COVID","Flu","RSV"]:
        fn = find_disease_file(input_dir, dis)
        print(f"  {dis}: {fn}", flush=True)
        frames.append(load_disease_file(input_dir, fn))
    raw = pd.concat(frames, ignore_index=True)
    raw = (raw.groupby(["Location","Week"],as_index=False)
              [["Weekly_Encounters","Weekly_Positive_Tests"]].sum())
    raw = raw[raw["Week"]>=TRAIN_START].copy()
    matched = match_facilities(raw, facilities)
    return raw[raw["Location"].isin(matched)].copy(), matched

def build_aggregate(raw):
    agg = (raw.groupby("Week",as_index=False)
              [["Weekly_Encounters","Weekly_Positive_Tests"]].sum())
    agg["Location"] = AGG_LOCATION
    return agg.sort_values("Week").reset_index(drop=True)

def build_facility_wide(fac_df, facilities, value_col, prefix):
    sub  = fac_df[fac_df["Location"].isin(facilities)].copy()
    wide = sub.pivot_table(index="Week", columns="Location",
                            values=value_col, aggfunc="sum", fill_value=0.)
    for f in facilities:
        if f not in wide.columns: wide[f]=0.
    wide = wide[facilities].fillna(0.).sort_index().reset_index()
    rmap = {f:f"{prefix}_{safe_fac_id(f)}" for f in facilities}
    return wide.rename(columns=rmap), [rmap[f] for f in facilities]

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
EXTRA_FEATURES = [
    "sin_woy","cos_woy","enc_current",
    "enc_growth_1w","enc_growth_2w","enc_growth_4w","enc_growth_8w","enc_growth_accel",
    "pos_growth_1w","pos_growth_2w","pos_growth_4w","pos_growth_accel",
    "enc_rmean_4w","enc_rmean_13w","enc_momentum_4v13","enc_rank_13w",
    "positivity_lag1","positivity_lag2","positivity_growth_2w",
]

def safe_log_ratio(a,b):
    return np.log((np.asarray(a,float)+1.)/(np.asarray(b,float)+1.))

def trailing_rank(s,w):
    def rl(x): return pd.Series(x).rank(pct=True).iloc[-1]
    return s.rolling(w,min_periods=w).apply(rl,raw=False)

def compute_features(df):
    df  = df.sort_values("Week").copy().reset_index(drop=True)
    enc = pd.to_numeric(df["Weekly_Encounters"],    errors="coerce").astype(float)
    pos = pd.to_numeric(df["Weekly_Positive_Tests"],errors="coerce").astype(float)
    for k in range(1,MAX_LAG+1):
        df[f"enc_lag{k}"] = enc.shift(k)
        df[f"pos_lag{k}"] = pos.shift(k)
    woy = pd.to_datetime(df["Week"]).dt.isocalendar().week.astype(float)
    df["sin_woy"]          = np.sin(2.*np.pi*woy/52.)
    df["cos_woy"]          = np.cos(2.*np.pi*woy/52.)
    df["enc_current"]      = enc
    df["enc_growth_1w"]    = safe_log_ratio(enc.shift(1),enc.shift(2))
    df["enc_growth_2w"]    = safe_log_ratio(enc.shift(1),enc.shift(3))
    df["enc_growth_4w"]    = safe_log_ratio(enc.shift(1),enc.shift(5))
    df["enc_growth_8w"]    = safe_log_ratio(enc.shift(1),enc.shift(9))
    df["enc_growth_accel"] = df["enc_growth_1w"]-df["enc_growth_1w"].shift(1)
    df["pos_growth_1w"]    = safe_log_ratio(pos.shift(1),pos.shift(2))
    df["pos_growth_2w"]    = safe_log_ratio(pos.shift(1),pos.shift(3))
    df["pos_growth_4w"]    = safe_log_ratio(pos.shift(1),pos.shift(5))
    df["pos_growth_accel"] = df["pos_growth_1w"]-df["pos_growth_1w"].shift(1)
    df["enc_rmean_4w"]     = enc.shift(1).rolling(4, min_periods=4 ).mean()
    df["enc_rmean_13w"]    = enc.shift(1).rolling(13,min_periods=13).mean()
    df["enc_momentum_4v13"]= df["enc_rmean_4w"]-df["enc_rmean_13w"]
    df["enc_rank_13w"]     = trailing_rank(enc.shift(1),13)
    pos_rate = pos/(enc+1.)
    df["positivity_lag1"]      = pos_rate.shift(1)
    df["positivity_lag2"]      = pos_rate.shift(2)
    df["positivity_growth_2w"] = pos_rate.shift(1)-pos_rate.shift(3)
    for c in df.columns:
        if c not in ["Location","Week"]:
            df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    return df

def feat_cols_for_lag(df, lag_n):
    cols  = [f"enc_lag{k}" for k in range(1,lag_n+1)]
    cols += [f"pos_lag{k}" for k in range(1,lag_n+1)]
    cols += EXTRA_FEATURES
    return [c for c in dict.fromkeys(cols) if c in df.columns]

def add_facility_lags(df, fac_enc_wide, fac_enc_ids, fac_pos_wide, fac_pos_ids):
    out, fac_cols = df.copy(), []
    for wide, ids in [(fac_enc_wide,fac_enc_ids),(fac_pos_wide,fac_pos_ids)]:
        lagged = wide.sort_values("Week").copy(); new_cols=[]
        for col in ids:
            for k in range(1,N_FAC_LAGS+1):
                nm = f"{col}_lag{k}"
                lagged[nm] = pd.to_numeric(lagged[col],errors="coerce").shift(k)
                new_cols.append(nm)
        out = out.merge(lagged[["Week"]+new_cols],on="Week",how="left")
        fac_cols.extend(new_cols)
    return out, fac_cols

# =============================================================================
# DATASET BUILDER
# =============================================================================
def build_train_dataset(agg_df, horizon, lag_n,
                         fac_enc_wide, fac_enc_ids, fac_pos_wide, fac_pos_ids):
    """Build supervised dataset; filter to target_end_date ≤ TRAIN_DATA_END."""
    df = agg_df[["Week","Weekly_Encounters","Weekly_Positive_Tests"]].copy()
    df = df.sort_values("Week").reset_index(drop=True)
    df = compute_features(df)
    df, fac_cols = add_facility_lags(df,fac_enc_wide,fac_enc_ids,
                                      fac_pos_wide,fac_pos_ids)
    all_feat = feat_cols_for_lag(df, lag_n) + fac_cols
    df["y_anchor"]        = pd.to_numeric(df["Weekly_Encounters"],errors="coerce").astype(float)
    df["y_future"]        = df["Weekly_Encounters"].shift(-horizon).astype(float)
    df["target"]          = fwd(df["y_future"], df["y_anchor"])
    df["target_end_date"] = df["Week"] + pd.Timedelta(weeks=horizon)
    for c in all_feat+["target","y_anchor","y_future"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
    df = df.dropna(subset=all_feat+["target","y_future"]).copy()
    # Final fit uses all data whose target is known at TRAIN_DATA_END
    df = df[df["target_end_date"] <= TRAIN_DATA_END].copy()
    return df, all_feat

def build_forecast_row(agg_df, lag_n,
                        fac_enc_wide, fac_enc_ids, fac_pos_wide, fac_pos_ids):
    """
    Build feature row for forecasting from TRAIN_DATA_END.
    All tables truncated to Week ≤ TRAIN_DATA_END — no leakage.
    """
    series = (agg_df[agg_df["Week"]<=TRAIN_DATA_END]
              [["Week","Weekly_Encounters","Weekly_Positive_Tests"]]
              .copy().sort_values("Week").reset_index(drop=True))
    if len(series)==0: return None
    series = compute_features(series)
    fe_trunc = fac_enc_wide[fac_enc_wide["Week"]<=TRAIN_DATA_END].copy()
    fp_trunc = fac_pos_wide[fac_pos_wide["Week"]<=TRAIN_DATA_END].copy()
    series, fac_cols = add_facility_lags(series,fe_trunc,fac_enc_ids,
                                          fp_trunc,fac_pos_ids)
    fc  = feat_cols_for_lag(series, lag_n) + fac_cols
    row = series[series["Week"]==TRAIN_DATA_END].copy()
    if len(row)==0: return None
    for c in fc:
        if c not in row.columns: row[c]=np.nan
        row[c]=pd.to_numeric(row[c],errors="coerce")
    X = row[fc].astype(float)
    if X.isna().any().any(): return None
    return X, fc

# =============================================================================
# QRF
# =============================================================================
def build_qrf(params):
    p = dict(params); p.update({"random_state":RANDOM_STATE,"n_jobs":MODEL_JOBS})
    return RandomForestQuantileRegressor(**p)

def qrf_predict_all_q(model, X):
    pred = np.asarray(model.predict(X,quantiles=list(SUBMISSION_QUANTILES)),float)
    if pred.ndim==1: pred=pred.reshape(1,-1)
    if pred.ndim==3: pred=pred.squeeze()
    if pred.shape[0]==N_Q and pred.shape[1]!=N_Q: pred=pred.T
    return np.apply_along_axis(monotone,1,pred)

# =============================================================================
# LOAD SAVED OPTIMAL PARAMETERS
# =============================================================================
def load_saved_params(json_path):
    """
    Load optimal params JSON saved by the eval script.
    Expects keys "MUSC__h1" … "MUSC__h4".
    Returns dict {horizon: {lag_n, n_estimators, max_depth, ...}}.
    """
    with open(json_path) as f:
        raw = json.load(f)
    store = {}
    for h in HORIZONS:
        key = f"{AGG_LOCATION}__h{h}"
        if key not in raw:
            raise KeyError(f"Key '{key}' not found in {json_path}. "
                           f"Available keys: {list(raw.keys())}")
        store[h] = raw[key]
    print(f"  Loaded params for H=1..4 from {os.path.basename(json_path)}")
    return store

# =============================================================================
# OUTPUT ROW BUILDERS
# =============================================================================
def obs_history_rows(agg_df, ref_str):
    """
    One row per historical week.
    training_validation       = 1   (observed/training)
    estimate_projected_report = 2   (observed)
    output_type_id            = NA  (not a quantile forecast)
    """
    rows = []
    hist = agg_df[agg_df["Week"] <= TRAIN_DATA_END].sort_values("Week")
    for h in HORIZONS:
        for _, r in hist.iterrows():
            rows.append({
                "reference_date":            ref_str,
                "target":                    None,
                "location_general":          LOCATION_GENERAL,
                "location":                  AGG_LOCATION,
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
        subset=["location","target_end_date","training_validation",
                "estimate_projected_report"]).to_dict("records")


def forecast_quantile_rows_agg(ref_str, tend, q_map, epr):
    """
    23 quantile rows for one horizon (aggregate forecast).
    training_validation       = 0         (forecast)
    estimate_projected_report = 1         (forecast)
    output_type_id            = q_str(q)  (0.01 … 0.99)
    """
    rows = []
    for q, val in sorted(q_map.items()):
        rows.append({
            "reference_date":            ref_str,
            "target":                    None,
            "location_general":          LOCATION_GENERAL,
            "location":                  AGG_LOCATION,
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
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0      = time.time()
    ref_str = REFERENCE_DATE.strftime(DATE_FMT)

    submission_path = os.path.join(SUBMISSION_DIR, submission_filename(ref_str))

    print("="*70, flush=True)
    print("MUSC AGGREGATED — WEEKLY FORECAST SUBMISSION", flush=True)
    print(f"  Reference date  : {ref_str}", flush=True)
    print(f"  Params JSON     : {PARAMS_JSON}", flush=True)
    print(f"  Submission      : {submission_path}", flush=True)
    print("="*70, flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading MUSC data …", flush=True)
    raw, matched = load_musc_data(INPUT_DIR, MUSC_FACILITIES)
    agg_df = build_aggregate(raw)
    fac_df = raw.copy()

    # Snap TRAIN_DATA_END to nearest available data week ≤ computed date
    avail = pd.to_datetime(sorted(agg_df["Week"].unique()))
    valid = avail[avail <= TRAIN_DATA_END]
    if len(valid)==0:
        raise ValueError(f"No data on or before {fmt_date(TRAIN_DATA_END)}")
    TRAIN_DATA_END = valid.max()
    print(f"  TRAIN_DATA_END snapped : {fmt_date(TRAIN_DATA_END)}", flush=True)
    print(f"  Aggregate weeks        : {len(agg_df)}", flush=True)
    print(f"  Date range             : "
          f"{fmt_date(agg_df['Week'].min())} → {fmt_date(agg_df['Week'].max())}", flush=True)

    # Cross-facility wide tables
    fac_enc_wide, fac_enc_ids = build_facility_wide(
        fac_df, matched, "Weekly_Encounters",     "facenc")
    fac_pos_wide, fac_pos_ids = build_facility_wide(
        fac_df, matched, "Weekly_Positive_Tests", "facpos")

    # ── Load optimal parameters ────────────────────────────────────────────────
    print("\nLoading saved optimal parameters …", flush=True)
    param_store = load_saved_params(PARAMS_JSON)

    # ── Fit final model and forecast for each horizon ──────────────────────────
    all_fc_q_rows = []   # all 23 quantiles per horizon

    for h in HORIZONS:
        p      = param_store[h]
        lag_n  = int(p["lag_n"])
        params = {k: int(v) if isinstance(v,(int,float)) and k != "max_features"
                  else v
                  for k, v in p.items()
                  if k not in ("lag_n","cv_rmse","train_end")}

        print(f"\n  H={h}  lag={lag_n}  params={params}", flush=True)

        # Build training dataset (all data up to TRAIN_DATA_END)
        train_df, feat_cols = build_train_dataset(
            agg_df, h, lag_n,
            fac_enc_wide, fac_enc_ids, fac_pos_wide, fac_pos_ids)

        if len(train_df) < MIN_TRAIN:
            print(f"    Skip H={h}: only {len(train_df)} train rows"); continue

        X_tr = mk_num(train_df, feat_cols)
        y_tr = train_df["target"].values
        vm   = np.isfinite(X_tr).all(1) & np.isfinite(y_tr)
        X_tr = X_tr.loc[vm].values; y_tr = y_tr[vm]

        if len(X_tr) < MIN_TRAIN:
            print(f"    Skip H={h}: only {len(X_tr)} valid rows"); continue

        # Final fit on all training data
        model = build_qrf(params)
        model.fit(X_tr, y_tr)
        print(f"    Fitted on {len(X_tr)} training rows", flush=True)

        # Build forecast feature row from TRAIN_DATA_END
        result = build_forecast_row(
            agg_df, lag_n,
            fac_enc_wide, fac_enc_ids, fac_pos_wide, fac_pos_ids)
        if result is None:
            print(f"    Skip H={h}: could not build forecast row"); continue
        X_fc, fc = result

        # Align columns
        for c in feat_cols:
            if c not in X_fc.columns: X_fc[c]=np.nan
        X_fc_vals = X_fc[feat_cols].astype(float).values

        # Predict all quantiles, back-transform
        pq    = qrf_predict_all_q(model, X_fc_vals)[0]
        ya    = float(agg_df[agg_df["Week"]==TRAIN_DATA_END]["Weekly_Encounters"].values[0])
        q_lev = monotone(np.maximum(0., inv(pq, np.repeat(ya, N_Q))))

        pred_median = float(q_lev[MEDIAN_IDX])
        q_map = {q: float(q_lev[i]) for i,q in enumerate(SUBMISSION_QUANTILES)}
        tend  = REFERENCE_DATE + pd.Timedelta(weeks=h-1)

        print(f"    Forecast target_end_date={fmt_date(tend)}  "
              f"median={pred_median:.1f}  "
              f"90%CI=[{q_map[0.05]:.1f}, {q_map[0.95]:.1f}]", flush=True)

        all_fc_q_rows.extend(forecast_quantile_rows_agg(ref_str, tend, q_map, H_TO_EPR[h]))

    # ── Observed history rows ──────────────────────────────────────────────────
    obs_rows = obs_history_rows(agg_df, ref_str)

    # ── Build submission: observed (TV=1,EPR=2,NA) + forecast quantiles ───────
    obs_dedup = pd.DataFrame(obs_rows).drop_duplicates(
        subset=["location","target_end_date","estimate_projected_report"])
    fc_q_df   = pd.DataFrame(all_fc_q_rows)

    if fc_q_df.empty:
        raise RuntimeError("No quantile forecast rows were generated — "
                           "check the horizon loop above.")

    final_sub = pd.concat([obs_dedup, fc_q_df], ignore_index=True)[COL_ORDER]

    # Enforce YYYY-MM-DD on every date column before writing
    final_sub = enforce_date_format(final_sub)

    # Write the submission file directly to the drop-off folder
    print()
    save_submission(final_sub, submission_path)

    tv_counts = final_sub["training_validation"].value_counts().to_dict()
    print(f"\n  Total rows      : {len(final_sub)}")
    print(f"  TV=1 (observed) : {tv_counts.get(1,0)}")
    print(f"  TV=0 (forecast) : {tv_counts.get(0,0)}  "
          f"(23 quantiles × {len(HORIZONS)} horizons)")
    print(f"  reference_date  : {sorted(final_sub['reference_date'].unique())}")
    print(f"  target_end_date : {min(final_sub['target_end_date'])} → "
          f"{max(final_sub['target_end_date'])}")

    print(f"\n{'='*70}", flush=True)
    print(f"Done.  Elapsed: {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Submission: {submission_path}", flush=True)
    print("="*70, flush=True)
