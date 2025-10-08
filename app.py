# app.py
import os
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from pandas.api.types import is_datetime64_any_dtype

# =========================
# Streamlit page setup
# =========================
st.set_page_config(page_title="Health Forecast Dashboard", layout="wide")
st.title("🧪 Health Forecast Dashboard — Cases & Deaths")
st.caption("Forecast weekly cases (with Alert/Action thresholds) & deaths (no thresholds). View accuracy, maps, charts, and exportable tables.")

# =========================
# Constants / Defaults
# =========================
OUTDIR = "outputs_nb"
RANDOM_STATE = 42
LAGS = [1, 2, 4]
ROLL_WINDOWS = [4]
BASE_NUMERIC = ["Week", "Year", "Precipitation", "MIN", "MAX"]
BASE_CATEG = ["Health Facility", "Station"]
POTENTIAL_LEAKAGE = ["mean_cases", "std_cases", "alert_threshold", "action_threshold"]

DEFAULT_FAC_TABLE = pd.DataFrame({
    "Health Facility": [
        "ALWAYS FTC","Chadereka RHC","Chawarura RHC","CHIDIKAMWEDZI FTC","CHINYANI FTC",
        "Chiwenga RHC","DAMBAKURIMA RHC","DAVID NELSON CLINIC","Hoya RHC","Hwata RHC",
        "machaya clinic","Muzarabani RHC","Nyamaridza Clinic","Range clinic","St. Albert's Hospital"
    ],
    "Ward": [13,1,21,28,11,24,4,15,17,6,3,7,22,16,10],
    "Longitude": [31.002302,31.20184,31.112028,30.961926,31.205876,31.332398,31.088416,31.118029,31.302773,30.96,31.162,30.8925,31.26132805,31.27273905,31.28189564],
    "Latitude":  [-16.673135,-16.16698667,-16.518236,-16.80073,-16.582612,-16.104893,-16.146154,-16.729417,-16.347057,-16.3237,-16.3507,-16.3925,-16.76743297,-16.47705964,-16.47527736]
})

# =========================
# ML feature helpers
# =========================
def add_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    angle = 2 * np.pi * (df["Week"].astype(float) / 52.0)
    df["week_sin"] = np.sin(angle)
    df["week_cos"] = np.cos(angle)
    return df

def add_lags(df: pd.DataFrame, targets=("Cases", "Deaths")) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["Health Facility", "Year", "Week"], inplace=True)
    for t in targets:
        for L in LAGS:
            df[f"{t}_lag{L}"] = df.groupby("Health Facility")[t].shift(L)
        for W in ROLL_WINDOWS:
            df[f"{t}_roll{W}"] = (
                df.groupby("Health Facility")[t]
                  .shift(1)
                  .rolling(W, min_periods=1).mean()
                  .reset_index(level=0, drop=True)
            )
    return df

def get_feature_columns(df: pd.DataFrame, target: str):
    cols = []
    cols += [c for c in BASE_NUMERIC + BASE_CATEG if c in df.columns]
    cols += [c for c in ["week_sin", "week_cos"] if c in df.columns]
    cols += [c for c in df.columns if any(k in c for k in ["Cases_lag","Cases_roll","Deaths_lag","Deaths_roll"])]
    cols = [c for c in cols if c not in POTENTIAL_LEAKAGE and c != target]
    return list(dict.fromkeys(cols))

def _list_weeks(spec):
    return list(range(1, 53)) if (spec is None or spec == "all") else list(spec)

def _ensure_all_weeks(df_like, y_cols):
    weeks_full = pd.DataFrame({"Week": list(range(1, 53))})
    out = weeks_full.merge(df_like, on="Week", how="left").sort_values("Week")
    keep = ["Week"] + [c for c in out.columns if c != "Week"]
    return out[keep]

# =========================
# Model loading
# =========================
_HAS_SKOPS = False
try:
    import skops.io as skio
    from skops.io import UntrustedTypesFoundException
    _HAS_SKOPS = True
except Exception:
    class UntrustedTypesFoundException(Exception): ...
import joblib

@st.cache_resource(show_spinner=False)
def load_model(path_skops, path_joblib):
    if _HAS_SKOPS and os.path.exists(path_skops):
        try:
            return skio.load(path_skops, trusted=True)
        except UntrustedTypesFoundException:
            pass
        except Exception:
            pass
    if os.path.exists(path_joblib):
        return joblib.load(path_joblib)
    return None

cases_model   = load_model(os.path.join(OUTDIR,"best_cases_pipeline.skops"),
                           os.path.join(OUTDIR,"best_cases_pipeline.joblib"))
deaths_stage1 = load_model(os.path.join(OUTDIR,"best_deaths_stage1_classifier.skops"),
                           os.path.join(OUTDIR,"best_deaths_stage1_classifier.joblib"))
deaths_stage2 = load_model(os.path.join(OUTDIR,"best_deaths_stage2_regressor.skops"),
                           os.path.join(OUTDIR,"best_deaths_stage2_regressor.joblib"))

if cases_model is None:
    st.error(f"❌ Cases model not found under `{OUTDIR}`. Please place your `.skops` or `.joblib` files there.")
    st.stop()

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Forecast Settings")
mode_choice = st.sidebar.selectbox(
    "Mode",
    ["upload_exclude_weather", "no_new_data", "upload_include_weather"],
    index=0
)
years_to_forecast = st.sidebar.text_input("Forecast Years (comma-separated)", "2025")
weeks_input = st.sidebar.text_input("Weeks (1-52, comma-separated or 'all')", "all")
fac_filter_txt = st.sidebar.text_input("Filter Facilities (comma-separated exact names)", "")

def _parse_int_list_or_all(s):
    s = (s or "").strip().lower()
    if s in ("", "all"): return "all"
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except:
        return "all"

FORECAST_YEARS = [int(x.strip()) for x in years_to_forecast.split(",") if x.strip()]
FORECAST_WEEKS = _parse_int_list_or_all(weeks_input)
FACILITY_FILTER = [x.strip() for x in fac_filter_txt.split(",") if x.strip()] or None

st.sidebar.markdown("---")
st.sidebar.header("📄 Upload Data")
hist_csv = st.sidebar.file_uploader("History CSV (required for upload_* modes)", type=["csv"])
weather_csv = st.sidebar.file_uploader("Weather CSV (only for include_weather)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("🗺 Shapefile (Cent_Wards)")
shp_zip  = st.sidebar.file_uploader("ZIP with .shp/.dbf/.shx/.prj", type=["zip"])
shp_parts = st.sidebar.file_uploader("OR shapefile parts", type=["shp","dbf","shx","prj"], accept_multiple_files=True)
poly_name_col = st.sidebar.text_input("Polygon name column", value="admin3Name")
projection = st.sidebar.selectbox("Projection", ["mercator","equirectangular","natural earth"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("⏱ Time Filters (Maps/Tables)")
years_choice  = st.sidebar.text_input("Years (comma or 'all')", "2025")
months_choice = st.sidebar.text_input("Months 1-12 (comma or 'all')", "all")
weeks_choice  = st.sidebar.text_input("Weeks 1-52 (comma or 'all')", "all")

def _parse_list_or_all(s, kind="int"):
    s = (s or "").strip().lower()
    if s in ("", "all", "none"): return "all"
    vals = []
    for p in s.split(","):
        p = p.strip()
        if not p: continue
        vals.append(int(p) if kind=="int" else p)
    return vals if vals else "all"

YEARS_FILTER  = _parse_list_or_all(years_choice, "int")
MONTHS_FILTER = _parse_list_or_all(months_choice, "int")
WEEKS_FILTER  = _parse_list_or_all(weeks_choice, "int")

download_on = st.sidebar.checkbox("Enable CSV downloads", value=True)

# =========================
# Upload readers
# =========================
def _read_upload(u):
    if u is None: return None
    df = pd.read_csv(u)
    df.columns = [c.strip() for c in df.columns]
    return df

df_hist = _read_upload(hist_csv) if mode_choice in ("upload_exclude_weather","upload_include_weather") else None
df_weather = _read_upload(weather_csv) if (mode_choice == "upload_include_weather") else None

# =========================
# Shapefile loaders
# =========================
@st.cache_data(show_spinner=False)
def load_shp_zip(b: bytes):
    tmpdir = Path("./_shp_tmp") / f"z_{np.random.randint(1e9)}"; tmpdir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(b)) as zf: zf.extractall(tmpdir)
    shp_files = list(tmpdir.rglob("*.shp"))
    if not shp_files: raise ValueError("No .shp found in ZIP.")
    return gpd.read_file(shp_files[0]).to_crs(epsg=4326)

@st.cache_data(show_spinner=False)
def load_shp_parts(files: List):
    tmpdir = Path("./_shp_tmp") / f"p_{np.random.randint(1e9)}"; tmpdir.mkdir(parents=True, exist_ok=True)
    for f in files: (tmpdir / f.name).write_bytes(f.read())
    shp_files = list(tmpdir.glob("*.shp"))
    if not shp_files: raise ValueError("Please include the .shp file among the parts.")
    return gpd.read_file(shp_files[0]).to_crs(epsg=4326)

wards_gdf = None
if shp_zip is not None:
    try:
        wards_gdf = load_shp_zip(shp_zip.read())
    except Exception as e:
        st.error(f"Shapefile ZIP error: {e}")
elif shp_parts:
    try:
        wards_gdf = load_shp_parts(shp_parts)
    except Exception as e:
        st.error(f"Shapefile parts error: {e}")

# =========================
# Core utils
# =========================
def ensure_confidence_cols(df_like: pd.DataFrame) -> pd.DataFrame:
    df = df_like.copy()
    if "confidence_rate" in df.columns:
        df["confidence_rate"] = pd.to_numeric(df["confidence_rate"], errors="coerce").clip(0,1)
    elif "confidence_pct" in df.columns:
        df["confidence_rate"] = (pd.to_numeric(df["confidence_pct"], errors="coerce")/100.0).clip(0,1)
    else:
        df["confidence_rate"] = 0.5
    df["confidence_pct"] = (100.0 * df["confidence_rate"]).round(1)
    return df

def to_geojson_serializable(gdf: gpd.GeoDataFrame) -> dict:
    g2 = gdf.copy()
    for col in ["date","validOn","validTo"]:
        if col in g2.columns: g2[col] = g2[col].astype(str)
    geom_name = g2.geometry.name if hasattr(g2,"geometry") else "geometry"
    for c in g2.columns:
        if c == geom_name: continue
        if is_datetime64_any_dtype(g2[c]): g2[c] = g2[c].astype(str)
    return json.loads(g2.to_json())

def to_month(df_like):
    out = df_like.copy()
    if "Month" not in out.columns:
        dt = pd.to_datetime(
            out["Year"].astype(int).astype(str) + out["Week"].astype(int).astype(str) + "1",
            format="%G%V%u", errors="coerce"
        )
        out["Month"] = dt.dt.month
    return out

def apply_time_filters(df_like, years, months, weeks):
    out = to_month(df_like)
    if years != 'all':  out = out[out["Year"].isin(years)]
    if weeks != 'all':  out = out[out["Week"].isin(weeks)]
    if months != 'all': out = out[out["Month"].isin(months)]
    return out

def add_status_column(cases_df: pd.DataFrame) -> pd.DataFrame:
    df = cases_df.copy()
    def _status(row):
        if pd.isna(row.get("alert_threshold")) or pd.isna(row.get("action_threshold")):
            return "Normal"
        if row["Pred_Cases"] > row["action_threshold"]: return "Outbreak"
        if row["Pred_Cases"] > row["alert_threshold"]:  return "Alerting"
        return "Normal"
    df["Status"] = df.apply(_status, axis=1)
    return df

def status_palette(s):
    if s == "Outbreak": return "#ef4444"  # red
    if s == "Alerting": return "#f59e0b"  # amber
    return "#10b981"                     # green

# =========================
# Forecast runner
# =========================
def run_forecast(mode_choice, forecast_years, weeks_spec, df_hist, df_weather, facility_filter):
    # History
    if mode_choice in ("upload_exclude_weather","upload_include_weather"):
        if df_hist is None: raise ValueError("Upload a history CSV for the selected mode.")
        req = ["Health Facility","Station","Year","Week","Cases","Deaths"]
        miss = [c for c in req if c not in df_hist.columns]
        if miss: raise ValueError(f"History CSV missing columns: {miss}")
        db = df_hist.copy()
    elif mode_choice == "no_new_data":
        if df_hist is None: raise RuntimeError("For 'no_new_data' here, please upload a history CSV to act as df.")
        db = df_hist.copy()
    else:
        raise ValueError("unknown mode.")

    for c in ["Year","Week","Cases","Deaths"]:
        db[c] = pd.to_numeric(db[c], errors="coerce")
    db["Cases"]  = db["Cases"].fillna(0).astype(float)
    db["Deaths"] = db["Deaths"].fillna(0).astype(float)

    if facility_filter:
        db = db[db["Health Facility"].isin(facility_filter)].copy()
        if db.empty: raise ValueError("After facility filter, no data remains.")

    # Future grid
    weeks = _list_weeks(weeks_spec)
    fac = (db[["Health Facility","Station"]].drop_duplicates().sort_values(["Health Facility","Station"]))
    years_df = pd.DataFrame({"Year": forecast_years, "key":1})
    weeks_df = pd.DataFrame({"Week": weeks, "key":1})
    future = (fac.assign(key=1).merge(years_df, on="key").merge(weeks_df, on="key").drop(columns="key"))

    # Exogenous (optional)
    include_exog = (mode_choice == "upload_include_weather")
    if include_exog:
        if df_weather is not None:
            w = df_weather.copy()
            if {"Health Facility","Year","Week"}.issubset(w.columns):
                future = future.merge(w[["Health Facility","Year","Week","Precipitation","MIN","MAX"]],
                                      on=["Health Facility","Year","Week"], how="left")
            elif {"Station","Year","Week"}.issubset(w.columns):
                future = future.merge(w[["Station","Year","Week","Precipitation","MIN","MAX"]],
                                      on=["Station","Year","Week"], how="left")
            else:
                st.warning("Weather CSV missing expected keys; using proxy means.")
                df_weather = None
        if df_weather is None:
            recent = db[db["Year"] == db["Year"].max()].sort_values(["Health Facility","Week"])
            recent4 = (recent.groupby("Health Facility").tail(4)
                             .groupby("Health Facility")[["Precipitation","MIN","MAX"]]
                             .mean().reset_index())
            future = future.merge(recent4, on="Health Facility", how="left")

    for c in ["Precipitation","MIN","MAX"]:
        if c not in future.columns: future[c] = np.nan

    work = pd.concat([db.copy(), future.copy()], ignore_index=True, sort=False)
    for t in ["Cases","Deaths"]:
        if t not in work.columns: work[t] = np.nan
    work = add_seasonal(work)
    work = add_lags(work)

    future_ready = work[work["Year"].isin(forecast_years)].copy()

    # Predict CASES
    feat_cases = get_feature_columns(future_ready, target="Cases")
    y_pred_cases = np.asarray(cases_model.predict(future_ready[feat_cases]), dtype=float)
    cases_out = future_ready[["Health Facility","Station","Year","Week"]].copy()
    cases_out["Pred_Cases"] = np.rint(np.maximum(y_pred_cases,0)).astype(int)

    # Confidence (cases) via backtest
    hist_ready = work[work["Year"].isin(sorted(set(db["Year"])))].copy().dropna(subset=["Cases"])
    feat_hist = get_feature_columns(hist_ready, target="Cases")
    y_true_hist = hist_ready["Cases"].astype(float).values
    y_pred_hist = np.asarray(cases_model.predict(hist_ready[feat_hist]), dtype=float)
    ape = np.abs(y_true_hist - y_pred_hist) / np.maximum(1.0, np.abs(y_true_hist))
    hist_ready["_ape"] = ape
    ape_med = (hist_ready.groupby(["Health Facility","Week"], as_index=False)
               .agg(median_ape=("_ape","median"), n_years=("Year","nunique"))).copy()
    ape_med["median_ape"] = ape_med["median_ape"].fillna(1.0).clip(0,2)
    confidence = (1.0 - ape_med["median_ape"]).clip(0,1)
    weight = (ape_med["n_years"]/ape_med["n_years"].max()).fillna(0.5).clip(0.5,1.0)
    ape_med["confidence_rate"] = (0.7*confidence + 0.3*weight).clip(0,1)
    cases_out = cases_out.merge(ape_med[["Health Facility","Week","confidence_rate"]],
                                on=["Health Facility","Week"], how="left")
    cases_out["confidence_rate"] = cases_out["confidence_rate"].fillna(0.5)
    cases_out["confidence_pct"] = (100.0*cases_out["confidence_rate"]).round(1)

    # Predict DEATHS (hurdle if available)
    has_hurdle = (deaths_stage1 is not None) and (deaths_stage2 is not None)
    if has_hurdle:
        feat_deaths = get_feature_columns(future_ready, target="Deaths")
        if hasattr(deaths_stage1.named_steps["model"], "predict_proba"):
            p_pos = deaths_stage1.predict_proba(future_ready[feat_deaths])[:,1]
        else:
            z = deaths_stage1.decision_function(future_ready[feat_deaths]); p_pos = 1/(1+np.exp(-z))
        mu_pos = np.maximum(deaths_stage2.predict(future_ready[feat_deaths]), 0.0)
        deaths_pred = p_pos*mu_pos
    else:
        deaths_pred = np.zeros(len(future_ready))
    deaths_out = future_ready[["Health Facility","Station","Year","Week"]].copy()
    deaths_out["Pred_Deaths"] = np.rint(np.maximum(deaths_pred,0)).astype(int)

    # Confidence (deaths)
    hist_ready_d = work[work["Year"].isin(sorted(set(db["Year"])))].copy().dropna(subset=["Deaths"])
    feat_hist_d = get_feature_columns(hist_ready_d, target="Deaths")
    if has_hurdle:
        if hasattr(deaths_stage1.named_steps["model"], "predict_proba"):
            p_pos_h = deaths_stage1.predict_proba(hist_ready_d[feat_hist_d])[:,1]
        else:
            z = deaths_stage1.decision_function(hist_ready_d[feat_hist_d]); p_pos_h = 1/(1+np.exp(-z))
        mu_pos_h = np.maximum(deaths_stage2.predict(hist_ready_d[feat_hist_d]), 0.0)
        y_pred_d = p_pos_h*mu_pos_h
    else:
        y_pred_d = np.zeros(len(hist_ready_d))
    y_true_d = hist_ready_d["Deaths"].astype(float).values
    ape_d = np.abs(y_true_d - y_pred_d) / np.maximum(1.0, np.abs(y_true_d))
    hist_ready_d["_ape_d"] = ape_d
    ape_med_d = (hist_ready_d.groupby(["Health Facility","Week"], as_index=False)
                 .agg(median_ape_d=("_ape_d","median"), n_years=("Year","nunique"))).copy()
    ape_med_d["median_ape_d"] = ape_med_d["median_ape_d"].fillna(1.0).clip(0,2)
    confidence_d = (1.0 - ape_med_d["median_ape_d"]).clip(0,1)
    weight_d = (ape_med_d["n_years"]/ape_med_d["n_years"].max()).fillna(0.5).clip(0.5,1.0)
    ape_med_d["confidence_rate_deaths"] = (0.7*confidence_d + 0.3*weight_d).clip(0,1)
    deaths_out = deaths_out.merge(ape_med_d[["Health Facility","Week","confidence_rate_deaths"]],
                                  on=["Health Facility","Week"], how="left")
    deaths_out["confidence_rate_deaths"] = deaths_out["confidence_rate_deaths"].fillna(0.5)
    deaths_out["confidence_pct_deaths"] = (100.0*deaths_out["confidence_rate_deaths"]).round(1)

    # Thresholds for cases
    thr = (db.groupby(["Health Facility","Week"], as_index=False)
           .agg(mean_cases=("Cases","mean"), std_cases=("Cases","std")))
    thr["std_cases"] = thr["std_cases"].fillna(0.0)
    thr["alert_threshold"]  = thr["mean_cases"]
    thr["action_threshold"] = thr["mean_cases"] + 1.5*thr["std_cases"]
    cases_out = cases_out.merge(thr[["Health Facility","Week","alert_threshold","action_threshold"]],
                                on=["Health Facility","Week"], how="left")
    return db, cases_out, deaths_out, hist_ready, y_true_hist, y_pred_hist, hist_ready_d, y_true_d, y_pred_d

# Run forecast
try:
    db, cases_res, deaths_res, hist_ready, y_true_hist, y_pred_hist, hist_ready_d, y_true_d, y_pred_d = run_forecast(
        mode_choice, FORECAST_YEARS, FORECAST_WEEKS, df_hist, df_weather, FACILITY_FILTER
    )
except Exception as e:
    st.error(f"Forecast error: {e}")
    st.stop()

# =========================
# Global summary (top cards)
# =========================
def overall_summary(cases_df: pd.DataFrame) -> dict:
    agg = (cases_df.groupby("Week", as_index=False)
           .agg(Pred_Cases=("Pred_Cases","sum"),
                Alert=("alert_threshold","sum"),
                Action=("action_threshold","sum")))
    agg = _ensure_all_weeks(agg, ["Pred_Cases","Alert","Action"])
    total_pred   = int(np.nansum(agg["Pred_Cases"]))
    total_alert  = float(np.nansum(agg["Alert"]))
    total_action = float(np.nansum(agg["Action"]))
    if total_pred > total_action: status = "Outbreak"
    elif total_pred > total_alert: status = "Alerting"
    else: status = "Normal"
    return dict(total_pred=total_pred, total_alert=total_alert, total_action=total_action, status=status)

summary = overall_summary(cases_res)
bg = {"Outbreak":"#fee2e2","Alerting":"#fffbeb","Normal":"#ecfdf5"}[summary["status"]]
fg = {"Outbreak":"#991b1b","Alerting":"#92400e","Normal":"#065f46"}[summary["status"]]

with st.container():
    st.subheader("🔎 Prediction Summary (All facilities, selected horizon)")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    c1.metric("Total Predicted Cases", f"{summary['total_pred']:,}")
    c2.metric("Overall Alert Threshold (sum)", f"{summary['total_alert']:.1f}")
    c3.metric("Overall Action Threshold (sum)", f"{summary['total_action']:.1f}")
    c4.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{bg};color:{fg};text-align:center;font-weight:600'>"
        f"Status: {summary['status']}</div>", unsafe_allow_html=True
    )

st.markdown("---")

# =========================
# Tabs
# =========================
tab_overview, tab_charts, tab_maps, tab_tables, tab_settings, tab_about = st.tabs(
    ["Overview", "Forecast Charts", "Maps", "Tables", "Settings", "About & Templates"]
)

# =========================
# OVERVIEW tab
# =========================
with tab_overview:
    st.subheader("📊 Model Overview & Accuracy")

    def _regression_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum()==0:
            return {"MAE": np.nan, "RMSE": np.nan, "MAPE%": np.nan}
        e = y_true[mask] - y_pred[mask]
        mae = np.mean(np.abs(e))
        rmse = np.sqrt(np.mean(e**2))
        mape = np.mean(np.abs(e) / np.maximum(1.0, np.abs(y_true[mask]))) * 100.0
        return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}

    met_cases  = _regression_metrics(y_true_hist, y_pred_hist)
    met_deaths = _regression_metrics(y_true_d,   y_pred_d)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cases backtest (historical)**")
        st.dataframe(pd.DataFrame([met_cases]).round(2), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Deaths backtest (historical)**")
        st.dataframe(pd.DataFrame([met_deaths]).round(2), use_container_width=True, hide_index=True)

    st.markdown("**Notes**")
    st.markdown(
        "- Accuracy is computed via backtesting on available historical rows (facility × week).  \n"
        "- Cases include Alert/Action thresholds derived from historical mean and variability.  \n"
        "- Deaths use a hurdle model (if classifier + regressor are available); no thresholds."
    )

# =========================
# CHARTS tab
# =========================
def fig_cases_chart(cases_df, years):
    agg_all = (cases_df.groupby("Week", as_index=False)
               .agg(Pred_Cases=("Pred_Cases","sum"),
                    Alert=("alert_threshold","sum"),
                    Action=("action_threshold","sum")))
    agg_all = _ensure_all_weeks(agg_all, ["Pred_Cases","Alert","Action"])
    fig = go.Figure()
    fig.add_bar(x=agg_all["Week"], y=agg_all["Pred_Cases"], name="Cases",
                marker=dict(opacity=0.6),
                hovertemplate="Week %{x}<br>Cases %{y}<extra></extra>")
    fig.add_trace(go.Scatter(x=agg_all["Week"], y=agg_all["Action"], mode="lines",
                             name="Action", line=dict(width=2.5),
                             hovertemplate="Week %{x}<br>Action %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=agg_all["Week"], y=agg_all["Alert"], mode="lines",
                             name="Alert", line=dict(width=2.5),
                             hovertemplate="Week %{x}<br>Alert %{y:.1f}<extra></extra>"))

    fac_list = sorted(cases_df["Health Facility"].dropna().unique().tolist())
    blocks = []
    for fac_name in fac_list:
        sub = (cases_df[cases_df["Health Facility"]==fac_name]
               .groupby("Week", as_index=False)
               .agg(Pred_Cases=("Pred_Cases","sum"),
                    Alert=("alert_threshold","mean"),
                    Action=("action_threshold","mean")))
        sub = _ensure_all_weeks(sub, ["Pred_Cases","Alert","Action"])
        s = len(fig.data)
        fig.add_bar(x=sub["Week"], y=sub["Pred_Cases"], name=f"{fac_name} — Cases",
                    visible=False, marker=dict(opacity=0.6),
                    hovertemplate="Week %{x}<br>Cases %{y}<extra></extra>")
        fig.add_trace(go.Scatter(x=sub["Week"], y=sub["Action"], mode="lines",
                                 name=f"{fac_name} — Action", visible=False, line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=sub["Week"], y=sub["Alert"], mode="lines",
                                 name=f"{fac_name} — Alert", visible=False, line=dict(width=2.5)))
        e = len(fig.data); blocks.append((s,e))

    n = len(fig.data); buttons=[]
    vis_all = [False]*n
    for i in range(3): vis_all[i]=True
    buttons.append(dict(label="All (sum)", method="update",
                        args=[{"visible":vis_all},{"title":f"Weekly CASES & Thresholds — {', '.join(map(str,years))}"}]))
    for i, fac in enumerate(fac_list):
        v=[False]*n; s,e = blocks[i]
        for k in range(s,e): v[k]=True
        buttons.append(dict(label=fac, method="update",
                            args=[{"visible":v},{"title":f"{fac} — Weekly CASES & Thresholds"}]))
    fig.update_layout(
        title=f"Weekly CASES & Thresholds — {', '.join(map(str,years))}",
        xaxis_title="Week", yaxis_title="Count",
        updatemenus=[dict(type="dropdown", x=0.01, y=1.12, xanchor="left", showactive=True, buttons=buttons)],
        legend_title="Series", height=520
    )
    return fig

def fig_deaths_chart(deaths_df, years):
    agg_all = (deaths_df.groupby("Week", as_index=False)
               .agg(Pred_Deaths=("Pred_Deaths","sum"),
                    conf=("confidence_pct_deaths","mean")))
    agg_all = _ensure_all_weeks(agg_all, ["Pred_Deaths","conf"])
    fig = go.Figure()
    fig.add_bar(x=agg_all["Week"], y=agg_all["Pred_Deaths"], name="Deaths",
                marker=dict(opacity=0.6),
                hovertemplate=("Week %{x}<br>Deaths %{y}<br>Confidence %{customdata:.1f}%<extra></extra>"),
                customdata=agg_all["conf"].to_numpy().reshape(-1,1))
    fac_list = sorted(deaths_df["Health Facility"].dropna().unique().tolist())
    blocks=[]
    for fac in fac_list:
        sub = (deaths_df[deaths_df["Health Facility"]==fac]
               .groupby("Week", as_index=False)
               .agg(Pred_Deaths=("Pred_Deaths","sum"),
                    conf=("confidence_pct_deaths","mean")))
        sub = _ensure_all_weeks(sub, ["Pred_Deaths","conf"])
        s=len(fig.data)
        fig.add_bar(x=sub["Week"], y=sub["Pred_Deaths"], name=f"{fac} — Deaths",
                    visible=False, marker=dict(opacity=0.6),
                    hovertemplate=("Week %{x}<br>Deaths %{y}<br>Confidence %{customdata:.1f}%<extra></extra>"),
                    customdata=sub["conf"].to_numpy().reshape(-1,1))
        e=len(fig.data); blocks.append((s,e))
    n=len(fig.data); buttons=[]
    vis=[False]*n; vis[0]=True
    buttons.append(dict(label="All (sum)", method="update",
                        args=[{"visible":vis},{"title":f"Weekly DEATHS — {', '.join(map(str,years))}"}]))
    for i, fac in enumerate(fac_list):
        v=[False]*n; s,e = blocks[i]
        for k in range(s,e): v[k]=True
        buttons.append(dict(label=fac, method="update",
                            args=[{"visible":v},{"title":f"{fac} — Weekly DEATHS"}]))
    fig.update_layout(title=f"Weekly DEATHS — {', '.join(map(str,years))}",
                      xaxis_title="Week", yaxis_title="Count",
                      updatemenus=[dict(type="dropdown", x=0.01, y=1.12, xanchor="left", showactive=True, buttons=buttons)],
                      legend_title="Series", height=520)
    return fig

with tab_charts:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Cases")
        st.plotly_chart(fig_cases_chart(cases_res, FORECAST_YEARS), use_container_width=True, theme="streamlit")
    with c2:
        st.subheader("📉 Deaths")
        st.plotly_chart(fig_deaths_chart(deaths_res, FORECAST_YEARS), use_container_width=True, theme="streamlit")

# =========================
# MAPS tab (filterable)
# =========================
def build_maps(wards: gpd.GeoDataFrame, POLY_NAME_COL: str, projection: str,
               cases_src: pd.DataFrame, deaths_src: pd.DataFrame,
               fac_table: pd.DataFrame, years, months, weeks):
    cases_sub  = apply_time_filters(cases_src, years, months, weeks)
    deaths_sub = apply_time_filters(deaths_src, years, months, weeks)
    fac_gdf = gpd.GeoDataFrame(
        fac_table.copy(),
        geometry=[Point(xy) for xy in zip(fac_table["Longitude"], fac_table["Latitude"])],
        crs="EPSG:4326"
    )
    if POLY_NAME_COL not in wards.columns:
        raise ValueError(f"'{POLY_NAME_COL}' not in shapefile. Columns: {list(wards.columns)}")
    fac_join = gpd.sjoin(fac_gdf, wards[[POLY_NAME_COL, "geometry"]], how="left", predicate="within")
    fac_join = fac_join.drop(columns=["index_right"], errors="ignore")

    cases_sub  = ensure_confidence_cols(cases_sub)
    deaths_sub = ensure_confidence_cols(deaths_sub)
    if "Pred_Cases" not in cases_sub.columns and "Cases" in cases_sub.columns:
        cases_sub["Pred_Cases"] = cases_sub["Cases"]
    if "Pred_Deaths" not in deaths_sub.columns and "Deaths" in deaths_sub.columns:
        deaths_sub["Pred_Deaths"] = deaths_sub["Deaths"]

    cases_fac  = cases_sub.merge(fac_join.drop(columns="geometry"), on="Health Facility", how="left")
    deaths_fac = deaths_sub.merge(fac_join.drop(columns="geometry"), on="Health Facility", how="left")

    agg = "sum"
    cases_bub  = (cases_fac.groupby(["Health Facility","Longitude","Latitude","Ward"], as_index=False)
                  .agg(Pred_Cases=("Pred_Cases", agg), confidence_rate=("confidence_rate","mean")))
    deaths_bub = (deaths_fac.groupby(["Health Facility","Longitude","Latitude","Ward"], as_index=False)
                  .agg(Pred_Deaths=("Pred_Deaths", agg), confidence_rate=("confidence_rate","mean")))
    cases_bub["Pred_Cases"] = np.rint(np.maximum(cases_bub["Pred_Cases"],0)).astype(int)
    deaths_bub["Pred_Deaths"] = np.rint(np.maximum(deaths_bub["Pred_Deaths"],0)).astype(int)
    cases_bub["confidence_pct"]  = (100.0*cases_bub["confidence_rate"]).round(1)
    deaths_bub["confidence_pct"] = (100.0*deaths_bub["confidence_rate"]).round(1)

    cases_area  = (cases_fac.groupby([POLY_NAME_COL], as_index=False)
                   .agg(Pred_Cases=("Pred_Cases", agg), confidence_rate=("confidence_rate","mean")))
    deaths_area = (deaths_fac.groupby([POLY_NAME_COL], as_index=False)
                   .agg(Pred_Deaths=("Pred_Deaths", agg), confidence_rate=("confidence_rate","mean")))
    cases_area["confidence_pct"]  = (100.0*cases_area["confidence_rate"]).round(1)
    deaths_area["confidence_pct"] = (100.0*deaths_area["confidence_rate"]).round(1)

    g_cases  = wards.merge(cases_area,  on=POLY_NAME_COL, how="left")
    g_deaths = wards.merge(deaths_area, on=POLY_NAME_COL, how="left")

    wards["_base"] = 1.0
    gj_wards = to_geojson_serializable(wards)
    gj_gc    = to_geojson_serializable(g_cases)
    gj_gd    = to_geojson_serializable(g_deaths)

    tooltip_cases = (
        "<b>Health Facility:</b> %{customdata[0]}<br>"
        "<b>Ward Number:</b> %{customdata[1]}<br>"
        "<b>Cases:</b> %{customdata[2]:.0f}<br>"
        "<b>Confidence Rate:</b> %{customdata[3]}%<extra></extra>"
    )
    tooltip_deaths = (
        "<b>Health Facility:</b> %{customdata[0]}<br>"
        "<b>Ward Number:</b> %{customdata[1]}<br>"
        "<b>Deaths:</b> %{customdata[2]:.0f}<br>"
        "<b>Confidence Rate:</b> %{customdata[3]}%<extra></extra>"
    )

    # Choropleth CASES
    f1 = go.Figure()
    f1.add_trace(go.Choropleth(geojson=gj_wards, locations=wards[POLY_NAME_COL], z=wards["_base"],
                               featureidkey=f"properties.{POLY_NAME_COL}",
                               colorscale=[[0,"#F2F2F2"],[1,"#F2F2F2"]], showscale=False,
                               marker_line_color="#9CA3AF", marker_line_width=0.6, hoverinfo="skip"))
    f1.add_trace(go.Choropleth(geojson=gj_gc, locations=g_cases[POLY_NAME_COL], z=g_cases["Pred_Cases"],
                               featureidkey=f"properties.{POLY_NAME_COL}", colorscale="Viridis",
                               colorbar_title="Cases",
                               customdata=np.stack([
                                   g_cases[POLY_NAME_COL].fillna("N/A").values,
                                   np.full(len(g_cases),"N/A",dtype=object),
                                   g_cases["Pred_Cases"].fillna(0).values,
                                   g_cases["confidence_pct"].fillna(50).values
                               ], axis=-1),
                               hovertemplate=tooltip_cases))
    f1.update_geos(fitbounds="locations", visible=False, projection_type=projection)

    # Choropleth DEATHS
    f2 = go.Figure()
    f2.add_trace(go.Choropleth(geojson=gj_wards, locations=wards[POLY_NAME_COL], z=wards["_base"],
                               featureidkey=f"properties.{POLY_NAME_COL}",
                               colorscale=[[0,"#F2F2F2"],[1,"#F2F2F2"]], showscale=False,
                               marker_line_color="#9CA3AF", marker_line_width=0.6, hoverinfo="skip"))
    f2.add_trace(go.Choropleth(geojson=gj_gd, locations=g_deaths[POLY_NAME_COL], z=g_deaths["Pred_Deaths"],
                               featureidkey=f"properties.{POLY_NAME_COL}", colorscale="Reds",
                               colorbar_title="Deaths",
                               customdata=np.stack([
                                   g_deaths[POLY_NAME_COL].fillna("N/A").values,
                                   np.full(len(g_deaths),"N/A",dtype=object),
                                   g_deaths["Pred_Deaths"].fillna(0).values,
                                   g_deaths["confidence_pct"].fillna(50).values
                               ], axis=-1),
                               hovertemplate=tooltip_deaths))
    f2.update_geos(fitbounds="locations", visible=False, projection_type=projection)

    # Bubble CASES
    f3 = go.Figure()
    f3.add_trace(go.Choropleth(geojson=gj_wards, locations=wards[POLY_NAME_COL], z=wards["_base"],
                               featureidkey=f"properties.{POLY_NAME_COL}",
                               colorscale=[[0,"#F2F2F2"],[1,"#F2F2F2"]], showscale=False,
                               marker_line_color="#9CA3AF", marker_line_width=0.6, hoverinfo="skip"))
    if not cases_bub.empty:
        f3.add_trace(go.Scattergeo(lon=cases_bub["Longitude"], lat=cases_bub["Latitude"],
                                   mode="markers+text", text=cases_bub["Pred_Cases"].astype(str),
                                   textposition="top center",
                                   marker=dict(size=np.clip(cases_bub["Pred_Cases"].values,6,40),
                                               sizemode="diameter", color=cases_bub["Pred_Cases"],
                                               colorscale="Viridis", showscale=True, colorbar=dict(title="Cases")),
                                   customdata=np.stack([
                                       cases_bub["Health Facility"].values,
                                       cases_bub["Ward"].fillna("N/A").astype(object).values,
                                       cases_bub["Pred_Cases"].values,
                                       cases_bub["confidence_pct"].values
                                   ], axis=-1),
                                   hovertemplate=tooltip_cases))
    f3.update_geos(fitbounds="locations", visible=False, projection_type=projection)

    # Bubble DEATHS
    f4 = go.Figure()
    f4.add_trace(go.Choropleth(geojson=gj_wards, locations=wards[POLY_NAME_COL], z=wards["_base"],
                               featureidkey=f"properties.{POLY_NAME_COL}",
                               colorscale=[[0,"#F2F2F2"],[1,"#F2F2F2"]], showscale=False,
                               marker_line_color="#9CA3AF", marker_line_width=0.6, hoverinfo="skip"))
    if not deaths_bub.empty:
        f4.add_trace(go.Scattergeo(lon=deaths_bub["Longitude"], lat=deaths_bub["Latitude"],
                                   mode="markers+text", text=deaths_bub["Pred_Deaths"].astype(str),
                                   textposition="top center",
                                   marker=dict(size=np.clip(deaths_bub["Pred_Deaths"].values,6,40),
                                               sizemode="diameter", color=deaths_bub["Pred_Deaths"],
                                               colorscale="Reds", showscale=True, colorbar=dict(title="Deaths")),
                                   customdata=np.stack([
                                       deaths_bub["Health Facility"].values,
                                       deaths_bub["Ward"].fillna("N/A").astype(object).values,
                                       deaths_bub["Pred_Deaths"].values,
                                       deaths_bub["confidence_pct"].values
                                   ], axis=-1),
                                   hovertemplate=tooltip_deaths))
    f4.update_geos(fitbounds="locations", visible=False, projection_type=projection)

    return f1, f2, f3, f4

with tab_maps:
    st.subheader("🗺 Cent_Wards Maps (Filterable)")
    if wards_gdf is None:
        st.info("Upload the Cent_Wards shapefile in the sidebar to render maps.")
    else:
        # quick domains
        _cases_all = ensure_confidence_cols(cases_res.rename(columns={"confidence_pct":"confidence_pct"})).copy()
        _deaths_all = ensure_confidence_cols(deaths_res.rename(columns={"confidence_pct_deaths":"confidence_pct"})).copy()
        _cases_all = to_month(_cases_all)
        _deaths_all = to_month(_deaths_all)
        _years_all  = sorted(list(set(_cases_all["Year"]).union(_deaths_all["Year"])))
        _months_all = [m for m in range(1,13)]
        _weeks_all  = [w for w in range(1,53)]
        _fac_opts   = sorted(DEFAULT_FAC_TABLE["Health Facility"].unique().tolist())
        _ward_opts  = sorted(wards_gdf[poly_name_col].dropna().astype(str).unique().tolist())
        _max_cases  = int(max(1, np.nanmax([_cases_all.get("Pred_Cases", pd.Series([0])).max(), 0])))
        _max_deaths = int(max(1, np.nanmax([_deaths_all.get("Pred_Deaths", pd.Series([0])).max(), 0])))

        with st.form("map_filters"):
            c1, c2, c3 = st.columns(3)
            with c1:
                years_pick = st.multiselect("Years", _years_all, default=_years_all or [])
                months_pick = st.multiselect("Months (1–12)", _months_all, default=_months_all)
                weeks_pick = st.multiselect("Weeks (1–52)", _weeks_all, default=_weeks_all)
            with c2:
                fac_pick  = st.multiselect("Facilities (optional)", _fac_opts, default=_fac_opts)
                ward_pick = st.multiselect(f"Wards by {poly_name_col} (optional)", _ward_opts, default=_ward_opts)
                projection_tab = st.selectbox("Projection", ["mercator","equirectangular","natural earth"], index=0)
            with c3:
                st.markdown("**Layer toggles**")
                show_choro_cases  = st.checkbox("Choropleth: Cases",  value=True)
                show_choro_deaths = st.checkbox("Choropleth: Deaths", value=True)
                show_bub_cases    = st.checkbox("Bubbles: Cases",     value=True)
                show_bub_deaths   = st.checkbox("Bubbles: Deaths",    value=True)
                st.markdown("**Value & confidence filters**")
                min_cases, max_cases = st.slider("Cases range", 0, max(5, _max_cases), (0, _max_cases))
                min_deaths, max_deaths = st.slider("Deaths range", 0, max(5, _max_deaths), (0, _max_deaths))
                min_conf = st.slider("Min confidence (%)", 0, 100, 0)

            submitted = st.form_submit_button("Apply filters")

        # apply filters & build maps
        def _apply_maps_filters(df_like: pd.DataFrame,
                                years, months, weeks,
                                facs, wards_str: list,
                                val_col: str,
                                val_min: int, val_max: int,
                                conf_min_pct: float,
                                join_w_gdf: gpd.GeoDataFrame,
                                poly_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
            df = apply_time_filters(df_like, years, months, weeks)
            if facs:
                df = df[df["Health Facility"].isin(facs)].copy()
            fac_gdf = gpd.GeoDataFrame(
                DEFAULT_FAC_TABLE.copy(),
                geometry=[Point(xy) for xy in zip(DEFAULT_FAC_TABLE["Longitude"], DEFAULT_FAC_TABLE["Latitude"])],
                crs="EPSG:4326"
            )
            fac_join = gpd.sjoin(fac_gdf, join_w_gdf[[poly_col, "geometry"]], how="left", predicate="within").drop(columns=["index_right"], errors="ignore")
            dfj = df.merge(fac_join.drop(columns="geometry"), on="Health Facility", how="left")
            dfj = ensure_confidence_cols(dfj)
            if val_col not in dfj.columns and val_col.replace("Pred_","") in dfj.columns:
                dfj[val_col] = dfj[val_col.replace("Pred_","")]
            dfj = dfj[(dfj[val_col].between(val_min, val_max)) & (dfj["confidence_pct"] >= conf_min_pct)]
            if wards_str:
                dfj = dfj[dfj[poly_col].astype(str).isin(wards_str)]
            if val_col == "Pred_Cases":
                bub = (dfj.groupby(["Health Facility","Longitude","Latitude","Ward"], as_index=False)
                          .agg(Pred_Cases=("Pred_Cases","sum"),
                               confidence_rate=("confidence_rate","mean")))
                bub["Pred_Cases"] = np.rint(np.maximum(bub["Pred_Cases"], 0)).astype(int)
            else:
                bub = (dfj.groupby(["Health Facility","Longitude","Latitude","Ward"], as_index=False)
                          .agg(Pred_Deaths=("Pred_Deaths","sum"),
                               confidence_rate=("confidence_rate","mean")))
                bub["Pred_Deaths"] = np.rint(np.maximum(bub["Pred_Deaths"], 0)).astype(int)
            bub["confidence_pct"] = (100.0 * bub["confidence_rate"]).round(1)
            if val_col == "Pred_Cases":
                area = (dfj.groupby([poly_col], as_index=False)
                          .agg(Pred_Cases=("Pred_Cases","sum"),
                               confidence_rate=("confidence_rate","mean")))
                area["confidence_pct"] = (100.0*area["confidence_rate"]).round(1)
                g_area = join_w_gdf.merge(area, on=poly_col, how="left")
            else:
                area = (dfj.groupby([poly_col], as_index=False)
                          .agg(Pred_Deaths=("Pred_Deaths","sum"),
                               confidence_rate=("confidence_rate","mean")))
                area["confidence_pct"] = (100.0*area["confidence_rate"]).round(1)
                g_area = join_w_gdf.merge(area, on=poly_col, how="left")
            return bub, area, g_area

        cases_bub, cases_area, g_cases = _apply_maps_filters(
            cases_res.rename(columns={"confidence_pct":"confidence_pct"}),
            years_pick, months_pick, weeks_pick,
            fac_pick, ward_pick, "Pred_Cases",
            min_cases, max_cases, min_conf,
            wards_gdf.copy(), poly_name_col
        )
        deaths_bub, deaths_area, g_deaths = _apply_maps_filters(
            deaths_res.rename(columns={"confidence_pct_deaths":"confidence_pct"}),
            years_pick, months_pick, weeks_pick,
            fac_pick, ward_pick, "Pred_Deaths",
            min_deaths, max_deaths, min_conf,
            wards_gdf.copy(), poly_name_col
        )

        wards_gdf["_base"] = 1.0
        gj_wards = to_geojson_serializable(wards_gdf)
        gj_gc    = to_geojson_serializable(g_cases)
        gj_gd    = to_geojson_serializable(g_deaths)

        tooltip_cases = (
            "<b>Health Facility:</b> %{customdata[0]}<br>"
            "<b>Ward Number:</b> %{customdata[1]}<br>"
            "<b>Cases:</b> %{customdata[2]:.0f}<br>"
            "<b>Confidence Rate:</b> %{customdata[3]}%<extra></extra>"
        )
        tooltip_deaths = (
            "<b>Health Facility:</b> %{customdata[0]}<br>"
            "<b>Ward Number:</b> %{customdata[1]}<br>"
            "<b>Deaths:</b> %{customdata[2]:.0f}<br>"
            "<b>Confidence Rate:</b> %{customdata[3]}%<extra></extra>"
        )

        def _base_fig():
            f = go.Figure()
            f.add_trace(go.Choropleth(
                geojson=gj_wards,
                locations=wards_gdf[poly_name_col],
                z=wards_gdf["_base"],
                featureidkey=f"properties.{poly_name_col}",
                colorscale=[[0,"#F2F2F2"],[1,"#F2F2F2"]], showscale=False,
                marker_line_color="#9CA3AF", marker_line_width=0.6, hoverinfo="skip"
            ))
            f.update_geos(fitbounds="locations", visible=False, projection_type=projection_tab)
            return f

        figs_to_show = []

        if show_choro_cases:
            f1 = _base_fig()
            f1.add_trace(go.Choropleth(
                geojson=gj_gc,
                locations=g_cases[poly_name_col],
                z=g_cases.get("Pred_Cases", pd.Series([0]*len(g_cases))),
                featureidkey=f"properties.{poly_name_col}",
                colorscale="Viridis",
                colorbar_title="Cases",
                customdata=np.stack([
                    g_cases[poly_name_col].fillna("N/A").values,
                    np.full(len(g_cases), "N/A", dtype=object),
                    g_cases.get("Pred_Cases", pd.Series([0]*len(g_cases))).fillna(0).values,
                    g_cases.get("confidence_pct", pd.Series([np.nan]*len(g_cases))).fillna(50).values
                ], axis=-1),
                hovertemplate=tooltip_cases
            ))
            f1.update_layout(title="Choropleth — Cases (Filtered)")
            figs_to_show.append(f1)

        if show_choro_deaths:
            f2 = _base_fig()
            f2.add_trace(go.Choropleth(
                geojson=gj_gd,
                locations=g_deaths[poly_name_col],
                z=g_deaths.get("Pred_Deaths", pd.Series([0]*len(g_deaths))),
                featureidkey=f"properties.{poly_name_col}",
                colorscale="Reds",
                colorbar_title="Deaths",
                customdata=np.stack([
                    g_deaths[poly_name_col].fillna("N/A").values,
                    np.full(len(g_deaths), "N/A", dtype=object),
                    g_deaths.get("Pred_Deaths", pd.Series([0]*len(g_deaths))).fillna(0).values,
                    g_deaths.get("confidence_pct", pd.Series([np.nan]*len(g_deaths))).fillna(50).values
                ], axis=-1),
                hovertemplate=tooltip_deaths
            ))
            f2.update_layout(title="Choropleth — Deaths (Filtered)")
            figs_to_show.append(f2)

        if show_bub_cases:
            f3 = _base_fig()
            if not cases_bub.empty:
                f3.add_trace(go.Scattergeo(
                    lon=cases_bub["Longitude"], lat=cases_bub["Latitude"],
                    mode="markers+text",
                    text=cases_bub["Pred_Cases"].astype(str),
                    textposition="top center",
                    marker=dict(
                        size=np.clip(cases_bub["Pred_Cases"].values, 6, 40),
                        sizemode="diameter",
                        color=cases_bub["Pred_Cases"],
                        colorscale="Viridis", showscale=True, colorbar=dict(title="Cases")
                    ),
                    customdata=np.stack([
                        cases_bub["Health Facility"].values,
                        cases_bub["Ward"].fillna("N/A").astype(object).values,
                        cases_bub["Pred_Cases"].values,
                        cases_bub["confidence_pct"].values
                    ], axis=-1),
                    hovertemplate=tooltip_cases
                ))
            f3.update_layout(title="Bubbles — Cases (Filtered)")
            figs_to_show.append(f3)

        if show_bub_deaths:
            f4 = _base_fig()
            if not deaths_bub.empty:
                f4.add_trace(go.Scattergeo(
                    lon=deaths_bub["Longitude"], lat=deaths_bub["Latitude"],
                    mode="markers+text",
                    text=deaths_bub["Pred_Deaths"].astype(str),
                    textposition="top center",
                    marker=dict(
                        size=np.clip(deaths_bub["Pred_Deaths"].values, 6, 40),
                        sizemode="diameter",
                        color=deaths_bub["Pred_Deaths"],
                        colorscale="Reds", showscale=True, colorbar=dict(title="Deaths")
                    ),
                    customdata=np.stack([
                        deaths_bub["Health Facility"].values,
                        deaths_bub["Ward"].fillna("N/A").astype(object).values,
                        deaths_bub["Pred_Deaths"].values,
                        deaths_bub["confidence_pct"].values
                    ], axis=-1),
                    hovertemplate=tooltip_deaths
                ))
            f4.update_layout(title="Bubbles — Deaths (Filtered)")
            figs_to_show.append(f4)

        if not figs_to_show:
            st.warning("No layers selected. Enable at least one layer in the filter panel.")
        else:
            cols = st.columns(2)
            for i, fig in enumerate(figs_to_show):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# =========================
# TABLES tab (with Status)
# =========================
with tab_tables:
    st.subheader("📋 Tables with Status")

    cases_tbl = add_status_column(cases_res).copy()
    cases_tbl = cases_tbl[["Health Facility","Year","Week","Pred_Cases",
                           "alert_threshold","action_threshold","confidence_pct","Status"]]
    cases_tbl = cases_tbl.rename(columns={"confidence_pct":"confidence_%"})

    deaths_tbl = deaths_res[["Health Facility","Year","Week","Pred_Deaths","confidence_pct_deaths"]].copy()
    deaths_tbl = deaths_tbl.rename(columns={"confidence_pct_deaths":"confidence_%"})

    # pandas Styler for Status column coloring
    def _style_cases(df):
        s = df.style.apply(
            lambda row: [
                f"background-color:{status_palette(row['Status'])}; color:white; font-weight:600"
                if col == "Status" else ""
                for col in row.index
            ],
            axis=1
        )
        return s

    st.markdown("**Cases (Predictions & Thresholds)**")
    st.dataframe(_style_cases(cases_tbl), use_container_width=True)

    st.markdown("**Deaths (Predictions)**")
    st.dataframe(deaths_tbl, use_container_width=True, hide_index=True)

    if download_on:
        c1,c2 = st.columns(2)
        with c1:
            st.download_button("Download cases_with_status.csv",
                               cases_tbl.to_csv(index=False).encode("utf-8"),
                               "cases_with_status.csv", "text/csv")
        with c2:
            st.download_button("Download deaths.csv",
                               deaths_tbl.to_csv(index=False).encode("utf-8"),
                               "deaths.csv", "text/csv")

# =========================
# SETTINGS tab
# =========================
with tab_settings:
    st.subheader("⚙️ Inputs Preview")
    if df_hist is not None:
        st.write("**History CSV preview:**")
        st.dataframe(df_hist.head(30), use_container_width=True, hide_index=True)
    else:
        st.info("Upload a history CSV in the sidebar to preview.")
    if df_weather is not None:
        st.write("**Weather CSV preview:**")
        st.dataframe(df_weather.head(30), use_container_width=True, hide_index=True)
    if wards_gdf is not None:
        st.write("**Shapefile columns:**")
        st.write(list(wards_gdf.columns))

# =========================
# ABOUT & TEMPLATES tab
# =========================
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def build_history_template() -> pd.DataFrame:
    data = [
        ["ALWAYS FTC", "StAlbert_Station", 2023, 1,  6, 0, 12.4, 19.0, 31.2],
        ["ALWAYS FTC", "StAlbert_Station", 2023, 2,  3, 0,  8.1, 18.5, 30.1],
        ["Chadereka RHC", "Chadereka_Stn", 2023, 1, 2, 0, 18.3, 17.2, 29.8],
        ["Chadereka RHC", "Chadereka_Stn", 2023, 2, 5, 1,  9.0, 18.1, 31.0],
        ["St. Albert's Hospital", "StAlbert_Station", 2023, 1, 12, 0, 10.0, 17.6, 30.9],
        ["St. Albert's Hospital", "StAlbert_Station", 2023, 2, 10, 0, 14.8, 18.8, 32.2],
    ]
    return pd.DataFrame(data, columns=[
        "Health Facility","Station","Year","Week","Cases","Deaths",
        "Precipitation","MIN","MAX"
    ])

def build_weather_template() -> pd.DataFrame:
    data = [
        ["ALWAYS FTC", 2025, 1, 15.0, 18.2, 32.1],
        ["ALWAYS FTC", 2025, 2,  8.6, 18.0, 31.0],
        ["Chadereka RHC", 2025, 1, 12.3, 17.9, 30.4],
        ["St. Albert's Hospital", 2025, 1, 10.1, 17.5, 31.2],
    ]
    return pd.DataFrame(data, columns=[
        "Health Facility","Year","Week","Precipitation","MIN","MAX"
    ])

def build_facility_master_template() -> pd.DataFrame:
    data = [
        ["ALWAYS FTC",            13, 31.002302, -16.673135],
        ["Chadereka RHC",          1, 31.201840, -16.166987],
        ["Chawarura RHC",         21, 31.112028, -16.518236],
        ["CHIDIKAMWEDZI FTC",     28, 30.961926, -16.800730],
        ["CHINYANI FTC",          11, 31.205876, -16.582612],
        ["St. Albert's Hospital", 10, 31.281896, -16.475277],
    ]
    return pd.DataFrame(data, columns=["Health Facility","Ward","Longitude","Latitude"])

with tab_about:
    st.subheader("ℹ️ How to use this dashboard")

    st.markdown("""
**What this app does**

- Forecasts **weekly Cases** (with **Alert**/**Action** thresholds) and **weekly Deaths** (no thresholds).
- Computes **confidence rates** from backtesting.
- Visualizes results as charts and **filterable maps** using your *Cent_Wards* shapefile.
- Summarizes **overall status** (Normal / Alerting / Outbreak) across all facilities.

---

### ✅ Quick checklist (do these in order)

1. **Load models** — put files in `outputs_nb/`  
   - `best_cases_pipeline.skops` or `.joblib`  
   - *(optional for deaths hurdle)* `best_deaths_stage1_classifier.*` and `best_deaths_stage2_regressor.*`

2. **Choose forecast mode** (left sidebar → ⚙️ Forecast Settings)  
   - `upload_exclude_weather`: upload *history* only; ignore weather.  
   - `no_new_data`: use the uploaded history as the working dataset.  
   - `upload_include_weather`: upload *history* **and** *weather projections* (exogenous features).

3. **Upload data** (left sidebar → 📄 Upload Data)  
   - **History CSV (required)**: `Health Facility, Station, Year, Week, Cases, Deaths`  
     Optional (recommended for exogenous): `Precipitation, MIN, MAX`  
   - **Weather CSV (optional / required for include_weather)**: `Year, Week` + one of `Health Facility` **or** `Station`, and your exogenous columns.

4. **Upload shapefile** (left sidebar → 🗺 Shapefile)  
   - ZIP with `.shp, .dbf, .shx, .prj` **or** upload the parts individually.  
   - Set the **polygon name column** (default `admin3Name`). The app reprojects to **EPSG:4326**.

5. **Facilities & coordinates**  
   - Default facility master is embedded. Adjust `DEFAULT_FAC_TABLE` if needed.  
   - Columns: `Health Facility, Ward, Longitude, Latitude` (WGS84 / EPSG:4326).

6. **Tune horizon & filters**  
   - In sidebar: forecast **Year(s)** and **Week(s)**, optional **facility filter**.  
   - In **Maps** tab: filter by **Years/Months/Weeks**, Facility, Ward, value ranges, **min confidence**.

---

### 📈 Status logic
- **Row-level** (Tables):  
  `Pred_Cases > action_threshold` → **Outbreak**  
  `Pred_Cases > alert_threshold`  → **Alerting**  
  otherwise → **Normal**
- **Overall badge** (top): sums predicted cases & thresholds across facilities/weeks.

---

### 🧪 Templates (download & fill)
""")

    hist_tpl = build_history_template()
    weather_tpl = build_weather_template()
    fac_tpl = build_facility_master_template()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**History CSV template**")
        st.dataframe(hist_tpl.head(6), use_container_width=True, hide_index=True)
        st.download_button("Download history_template.csv", _csv_bytes(hist_tpl),
                           "history_template.csv", "text/csv")
    with c2:
        st.markdown("**Weather CSV template**")
        st.dataframe(weather_tpl.head(6), use_container_width=True, hide_index=True)
        st.download_button("Download weather_template.csv", _csv_bytes(weather_tpl),
                           "weather_template.csv", "text/csv")
    with c3:
        st.markdown("**Facility master template (maps)**")
        st.dataframe(fac_tpl.head(6), use_container_width=True, hide_index=True)
        st.download_button("Download facility_master_template.csv", _csv_bytes(fac_tpl),
                           "facility_master_template.csv", "text/csv")

    st.markdown("""
---

### 🧭 Column requirements & notes

**History CSV (required)**  
- Required: `Health Facility` *(str)*, `Station` *(str)*, `Year` *(int)*, `Week` *(1–53 int)*, `Cases` *(int/float)*, `Deaths` *(int/float)*  
- Optional: `Precipitation` *(mm)*, `MIN` *(°C)*, `MAX` *(°C)*

**Weather CSV (for include_weather)**  
- Keys: **either** `Health Facility` **or** `Station` + `Year`, `Week`  
- Exogenous columns should match model training (commonly `Precipitation, MIN, MAX`)

**Facility master (maps)**  
- `Health Facility`, `Ward`, `Longitude`, `Latitude` (WGS84 / EPSG:4326)

**Shapefile**  
- Must include polygon name column (default `admin3Name`) and CRS convertible to EPSG:4326.

---

### 🧰 Troubleshooting
- **Models not found** → place in `outputs_nb/` with exact names above.  
- **Wrong columns** → check template spelling/caps.  
- **No map data** → confirm polygon column name; ensure facility points fall inside polygons; widen filters.  
- **Strange totals** → check for duplicates per Facility×Year×Week and deduplicate.
""")
