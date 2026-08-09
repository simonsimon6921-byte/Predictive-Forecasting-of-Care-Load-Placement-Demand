# ============================================================
# Predictive Forecasting of Care Load & Placement Demand
# Dataset: HHS Unaccompanied Alien Children Program
# Streamlit Application
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Care Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM FRONT-END DESIGN
# ============================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 8% 0%, rgba(56,189,248,.12), transparent 25%),
      radial-gradient(circle at 92% 5%, rgba(99,102,241,.10), transparent 28%),
      #f5f7fb;
}
[data-testid="stHeader"] { background: rgba(245,247,251,.75); }
.block-container { max-width: 1450px; padding-top: 1.5rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0b1220,#17233a);
}
[data-testid="stSidebar"] * { color:#eef5ff !important; }

.hero {
    padding:32px 34px;
    border-radius:28px;
    margin-bottom:20px;
    background:
      radial-gradient(circle at 88% 15%,rgba(56,189,248,.22),transparent 25%),
      radial-gradient(circle at 65% 100%,rgba(129,140,248,.20),transparent 30%),
      linear-gradient(135deg,#0f172a,#1e3150);
    box-shadow:0 18px 45px rgba(15,23,42,.18);
}
.hero h1 {
    color:white; font-size:40px; font-weight:850;
    margin:12px 0 6px; letter-spacing:-1.2px;
}
.hero p { color:#cbd8e9; max-width:850px; font-size:16px; line-height:1.6; margin:0; }
.badge {
    display:inline-block; padding:6px 12px; border-radius:999px;
    background:rgba(255,255,255,.09);
    border:1px solid rgba(255,255,255,.16);
    color:#bdeaff; font-size:11px; font-weight:800; letter-spacing:.08em;
}
.section {
    margin:26px 0 13px; padding:14px 18px;
    border-left:4px solid #38bdf8; border-radius:13px;
    background:rgba(255,255,255,.82);
    box-shadow:0 6px 20px rgba(15,23,42,.05);
}
.section-title { font-size:21px; font-weight:800; color:#162033; }
.section-sub { font-size:13px; color:#718096; margin-top:3px; }
.kpi {
    min-height:130px; padding:20px; border-radius:20px;
    background:rgba(255,255,255,.92);
    border:1px solid rgba(148,163,184,.18);
    box-shadow:0 10px 28px rgba(15,23,42,.07);
}
.kpi-label { color:#64748b; font-size:11px; font-weight:800; letter-spacing:.08em; }
.kpi-value { color:#111827; font-size:30px; font-weight:850; margin-top:8px; }
.kpi-note { color:#64748b; font-size:12px; margin-top:4px; }
.alert-good,.alert-risk {
    padding:16px 20px; border-radius:16px; margin:10px 0 18px; font-weight:650;
}
.alert-good { background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; }
.alert-risk { background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; }
.stButton > button,.stDownloadButton > button {
    border-radius:12px !important; font-weight:700 !important; min-height:42px;
}
[data-testid="stDataFrame"] { border-radius:16px; overflow:hidden; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
.footer { text-align:center; color:#7b8799; font-size:12px; padding:25px; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "data/uac_daily_data.csv"
ICON_PATH = "assets/care_forecasting_icon.png"

DATE_COL = "Date"

CARE_COL = "Children in HHS Care"
DISCHARGE_COL = "Children discharged from HHS Care"
TRANSFER_COL = "Children transferred out of CBP custody"
CBP_COL = "Children in CBP custody"
INTAKE_COL = "Children apprehended and placed in CBP custody*"


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .small-note {
        color: #666666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def numeric_series(series):
    """
    Converts strings such as '2,484' to numeric values.
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def calculate_mape(y_true, y_pred):
    """
    MAPE excluding zero actual values.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (actual != 0)
    )

    if mask.sum() == 0:
        return np.nan

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        ) * 100
    )


def metric_values(y_true, y_pred):
    """
    Returns MAE, RMSE and MAPE.
    """
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = calculate_mape(
        y_true,
        y_pred
    )

    return mae, rmse, mape


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

@st.cache_data
def load_data():


    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program - HHS_Unaccompanied_Alien_Children_Program.csv")


    # Check exact required columns
    required_columns = [
        DATE_COL,
        CARE_COL,
        DISCHARGE_COL,
        TRANSFER_COL,
        CBP_COL,
        INTAKE_COL
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in CSV:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce"
    )

    df = df.dropna(
        subset=[DATE_COL]
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        CARE_COL,
        DISCHARGE_COL,
        TRANSFER_COL,
        CBP_COL,
        INTAKE_COL
    ]

    for col in numeric_columns:
        df[col] = numeric_series(
            df[col]
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values(DATE_COL)
        .drop_duplicates(
            subset=[DATE_COL],
            keep="last"
        )
        .set_index(DATE_COL)
    )

    # --------------------------------------------------------
    # Daily continuity
    # --------------------------------------------------------

    full_dates = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D"
    )

    df = df.reindex(full_dates)

    df.index.name = DATE_COL

    # --------------------------------------------------------
    # Interpolate missing daily observations
    # --------------------------------------------------------

    for col in numeric_columns:

        df[col] = (
            df[col]
            .interpolate(
                method="time",
                limit_direction="both"
            )
            .ffill()
            .bfill()
        )

    # --------------------------------------------------------
    # Derived indicators
    # --------------------------------------------------------

    df["Net Pressure"] = (
        df[TRANSFER_COL]
        - df[DISCHARGE_COL]
    )

    df["Care Change"] = (
        df[CARE_COL].diff()
    )

    df["7 Day Care Average"] = (
        df[CARE_COL]
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    df["14 Day Care Average"] = (
        df[CARE_COL]
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    df["7 Day Discharge Average"] = (
        df[DISCHARGE_COL]
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            CARE_COL,
            DISCHARGE_COL
        ]
    )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "net_pressure",
    "day_of_week",
    "month",
    "is_weekend"
]


def create_features(
    df,
    target_column
):

    data = df.copy()

    target = numeric_series(
        data[target_column]
    )

    data["target"] = target

    # Lag features
    data["lag_1"] = (
        data["target"].shift(1)
    )

    data["lag_7"] = (
        data["target"].shift(7)
    )

    data["lag_14"] = (
        data["target"].shift(14)
    )

    # Rolling features
    data["rolling_mean_7"] = (
        data["target"]
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    data["rolling_mean_14"] = (
        data["target"]
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    data["rolling_std_7"] = (
        data["target"]
        .rolling(
            7,
            min_periods=2
        )
        .std()
        .fillna(0)
    )

    # Net pressure
    data["net_pressure"] = (
        numeric_series(
            data[TRANSFER_COL]
        )
        - numeric_series(
            data[DISCHARGE_COL]
        )
    )

    # Calendar features
    data["day_of_week"] = (
        data.index.dayofweek
    )

    data["month"] = (
        data.index.month
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # IMPORTANT FIX
    # Do NOT use data.dropna() here.
    # Initial lag values are expected to be NaN.
    # --------------------------------------------------------

    data[FEATURE_COLUMNS] = (
        data[FEATURE_COLUMNS]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .ffill()
        .bfill()
    )

    # Only target is required to be valid
    data = data.dropna(
        subset=["target"]
    )

    return data


# ============================================================
# TRAIN FORECASTING MODELS
# ============================================================

@st.cache_resource
def train_models(
    df,
    target_column
):

    data = create_features(
        df,
        target_column
    )

    if data.empty:
        raise ValueError(
            "Feature engineering produced 0 rows."
        )

    X = data[
        FEATURE_COLUMNS
    ].copy()

    y = data[
        "target"
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.ffill().bfill()

    valid_rows = (
        y.notna()
        & np.isfinite(y)
        & X.notna().all(axis=1)
    )

    X = X.loc[
        valid_rows
    ]

    y = y.loc[
        valid_rows
    ]

    if len(X) < 20:
        raise ValueError(
            f"Only {len(X)} valid rows remain. "
            "At least 20 observations are required."
        )

    # --------------------------------------------------------
    # Strict chronological split
    # --------------------------------------------------------

    split = int(
        len(X) * 0.80
    )

    split = max(
        1,
        min(
            split,
            len(X) - 1
        )
    )

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    rf_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_prediction = (
        rf_model.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gb_model.fit(
        X_train,
        y_train
    )

    gb_prediction = (
        gb_model.predict(
            X_test
        )
    )

    # --------------------------------------------------------
    # Naive persistence
    # --------------------------------------------------------

    naive_prediction = (
        y_test.shift(1)
    )

    if len(
        naive_prediction
    ) > 0:

        naive_prediction.iloc[0] = (
            y_train.iloc[-1]
        )

    # --------------------------------------------------------
    # ARIMA
    # --------------------------------------------------------

    try:

        arima_model = ARIMA(
            y_train,
            order=(2, 1, 1)
        )

        arima_result = (
            arima_model.fit()
        )

        arima_prediction = np.asarray(
            arima_result.forecast(
                steps=len(y_test)
            ),
            dtype=float
        )

    except Exception:

        arima_result = None

        arima_prediction = np.repeat(
            y_train.iloc[-1],
            len(y_test)
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model_predictions = {
        "Naive": np.asarray(
            naive_prediction
        ),
        "ARIMA": arima_prediction,
        "Random Forest": rf_prediction,
        "Gradient Boosting": gb_prediction
    }

    evaluation_rows = []

    for name, prediction in (
        model_predictions.items()
    ):

        mae, rmse, mape = metric_values(
            y_test,
            prediction
        )

        evaluation_rows.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })

    results = pd.DataFrame(
        evaluation_rows
    )

    return {
        "rf_model": rf_model,
        "gb_model": gb_model,
        "arima_result": arima_result,
        "data": data,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "rf_prediction": rf_prediction,
        "gb_prediction": gb_prediction,
        "arima_prediction": arima_prediction,
        "naive_prediction": np.asarray(
            naive_prediction
        ),
        "results": results
    }


# ============================================================
# FUTURE FEATURE BUILDER
# ============================================================

def future_feature_row(
    history,
    df,
    future_date
):

    history = list(
        map(
            float,
            history
        )
    )

    return pd.DataFrame([
        {
            "lag_1": history[-1],

            "lag_7": (
                history[-7]
                if len(history) >= 7
                else history[-1]
            ),

            "lag_14": (
                history[-14]
                if len(history) >= 14
                else history[-1]
            ),

            "rolling_mean_7": np.mean(
                history[-7:]
            ),

            "rolling_mean_14": np.mean(
                history[-14:]
            ),

            "rolling_std_7": np.std(
                history[-7:]
            ),

            "net_pressure": float(
                df["Net Pressure"].iloc[-1]
            ),

            "day_of_week": (
                future_date.dayofweek
            ),

            "month": (
                future_date.month
            ),

            "is_weekend": int(
                future_date.dayofweek >= 5
            )
        }
    ])


# ============================================================
# RECURSIVE MACHINE LEARNING FORECAST
# ============================================================

def recursive_ml_forecast(
    model,
    df,
    horizon
):

    history = (
        df[CARE_COL]
        .astype(float)
        .tolist()
    )

    predictions = []

    future_dates = pd.date_range(
        start=(
            df.index[-1]
            + pd.Timedelta(days=1)
        ),
        periods=horizon,
        freq="D"
    )

    for future_date in future_dates:

        X_future = future_feature_row(
            history,
            df,
            future_date
        )

        prediction = float(
            model.predict(
                X_future[
                    FEATURE_COLUMNS
                ]
            )[0]
        )

        prediction = max(
            0,
            prediction
        )

        predictions.append(
            prediction
        )

        history.append(
            prediction
        )

    return pd.DataFrame(
        {
            "Forecast": predictions
        },
        index=future_dates
    )


# ============================================================
# RANDOM FOREST UNCERTAINTY
# ============================================================

def random_forest_interval(
    model,
    df,
    horizon
):

    history = (
        df[CARE_COL]
        .astype(float)
        .tolist()
    )

    lower = []
    upper = []

    future_dates = pd.date_range(
        start=(
            df.index[-1]
            + pd.Timedelta(days=1)
        ),
        periods=horizon,
        freq="D"
    )

    for future_date in future_dates:

        X_future = future_feature_row(
            history,
            df,
            future_date
        )

        tree_values = np.array([
            tree.predict(
                X_future[
                    FEATURE_COLUMNS
                ]
            )[0]
            for tree in model.estimators_
        ])

        prediction = float(
            np.mean(tree_values)
        )

        lower.append(
            max(
                0,
                np.percentile(
                    tree_values,
                    5
                )
            )
        )

        upper.append(
            max(
                0,
                np.percentile(
                    tree_values,
                    95
                )
            )
        )

        history.append(
            max(
                0,
                prediction
            )
        )

    return (
        np.asarray(lower),
        np.asarray(upper)
    )


# ============================================================
# ARIMA FUTURE FORECAST
# ============================================================

def arima_forecast(
    df,
    target_column,
    horizon
):

    series = (
        df[target_column]
        .astype(float)
    )

    future_dates = pd.date_range(
        start=(
            df.index[-1]
            + pd.Timedelta(days=1)
        ),
        periods=horizon,
        freq="D"
    )

    try:

        model = ARIMA(
            series,
            order=(2, 1, 1)
        )

        result = model.fit()

        forecast_result = (
            result.get_forecast(
                steps=horizon
            )
        )

        prediction = np.asarray(
            forecast_result.predicted_mean,
            dtype=float
        )

        confidence = (
            forecast_result.conf_int(
                alpha=0.10
            )
        )

        lower = np.asarray(
            confidence.iloc[:, 0],
            dtype=float
        )

        upper = np.asarray(
            confidence.iloc[:, 1],
            dtype=float
        )

    except Exception:

        last_value = float(
            series.iloc[-1]
        )

        prediction = np.repeat(
            last_value,
            horizon
        )

        lower = (
            prediction * 0.95
        )

        upper = (
            prediction * 1.05
        )

    return pd.DataFrame(
        {
            "Forecast": np.maximum(
                prediction,
                0
            ),
            "Lower Bound": np.maximum(
                lower,
                0
            ),
            "Upper Bound": np.maximum(
                upper,
                0
            )
        },
        index=future_dates
    )


# ============================================================
# NAIVE FUTURE FORECAST
# ============================================================

def naive_forecast(
    df,
    target_column,
    horizon
):

    last_value = float(
        df[target_column].iloc[-1]
    )

    dates = pd.date_range(
        start=(
            df.index[-1]
            + pd.Timedelta(days=1)
        ),
        periods=horizon,
        freq="D"
    )

    return pd.DataFrame(
        {
            "Forecast": np.repeat(
                last_value,
                horizon
            ),
            "Lower Bound": np.repeat(
                last_value,
                horizon
            ),
            "Upper Bound": np.repeat(
                last_value,
                horizon
            )
        },
        index=dates
    )


# ============================================================
# APPLICATION START
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        "Unable to load the dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CURRENT OPERATIONAL INDICATORS
# ============================================================

latest_row = df.iloc[-1]

current_care = float(
    latest_row[CARE_COL]
)

current_transfers = float(
    latest_row[TRANSFER_COL]
)

current_discharges = float(
    latest_row[DISCHARGE_COL]
)

current_pressure = float(
    latest_row["Net Pressure"]
)

current_intake = float(
    latest_row[INTAKE_COL]
)


# ============================================================
# PREMIUM HERO
# ============================================================
import base64

if os.path.exists(ICON_PATH):
    with open(ICON_PATH, "rb") as f:
        icon64 = base64.b64encode(f.read()).decode()
    icon_html = (
        f'<img src="data:image/png;base64,{icon64}" '
        'style="width:72px;height:72px;object-fit:contain;border-radius:18px;'
        'background:rgba(255,255,255,.10);padding:8px;margin-right:18px;">'
    )
else:
    icon_html = '<div style="font-size:55px;margin-right:18px;">🏥</div>'

st.markdown(
    f"""
    <div class="hero">
      <div style="display:flex;align-items:center;">
        {icon_html}
        <div>
          <div class="badge">AI • FORECASTING • DECISION INTELLIGENCE</div>
          <h1>Predictive Care Load</h1>
          <p>
            Forecast future HHS care demand, anticipate placement pressure,
            compare forecasting models, and identify capacity risk before it happens.
          </p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div style="display:flex;gap:9px;flex-wrap:wrap;margin-bottom:18px;">
      <span style="padding:7px 12px;border-radius:999px;background:#e0f2fe;color:#075985;font-size:11px;font-weight:800;">LIVE ANALYTICS</span>
      <span style="padding:7px 12px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:11px;font-weight:800;">TIME-SERIES ML</span>
      <span style="padding:7px 12px;border-radius:999px;background:#ecfdf5;color:#047857;font-size:11px;font-weight:800;">CAPACITY MONITOR</span>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    if os.path.exists(
        ICON_PATH
    ):
        st.image(
            ICON_PATH,
            width=130
        )

    st.header(
        "Forecast Controls"
    )

    forecast_horizon = st.slider(
        "Forecast Horizon",
        min_value=7,
        max_value=30,
        value=14,
        step=1
    )

    model_name = st.selectbox(
        "Forecast Model",
        [
            "Random Forest",
            "Gradient Boosting",
            "ARIMA",
            "Naive"
        ]
    )

    capacity_default = int(
        np.ceil(
            df[CARE_COL].max() * 1.10
        )
    )

    capacity = st.number_input(
        "Capacity Threshold",
        min_value=1,
        value=max(
            1,
            capacity_default
        ),
        step=50
    )

    st.divider()

    st.caption(
        f"Dataset rows: {len(df):,}"
    )

    st.caption(
        f"Start: {df.index.min().date()}"
    )

    st.caption(
        f"End: {df.index.max().date()}"
    )


# ============================================================
# DATASET DIAGNOSTIC
# ============================================================

with st.expander(
    "🔎 Dataset Diagnostic"
):

    st.write(
        "Dataset shape:",
        df.shape
    )

    st.write(
        "Date range:",
        df.index.min().date(),
        "to",
        df.index.max().date()
    )

    diagnostic = pd.DataFrame({
        "Data Type":
            df.dtypes.astype(str),

        "Missing Values":
            df.isna().sum(),

        "Non-Missing":
            df.notna().sum()
    })

    st.dataframe(
        diagnostic,
        use_container_width=True
    )

    st.write(
        "Dataset preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# TRAIN CARE LOAD MODELS
# ============================================================

try:

    with st.spinner(
        "Training care-load forecasting models..."
    ):

        care_package = train_models(
            df,
            CARE_COL
        )

except Exception as e:

    st.error(
        "Care-load model training failed."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREMIUM KPI STRIP
# ============================================================
st.markdown(
    """
    <div class="section">
      <div class="section-title">Operational Snapshot</div>
      <div class="section-sub">Latest indicators from the reporting dataset.</div>
    </div>
    """,
    unsafe_allow_html=True
)

kpi_cols = st.columns(5)
items = [
    ("HHS CARE LOAD", f"{current_care:,.0f}", "Active children"),
    ("CBP TRANSFERS", f"{current_transfers:,.0f}", "Flow into HHS"),
    ("DISCHARGES", f"{current_discharges:,.0f}", "Placement exits"),
    ("NET PRESSURE", f"{current_pressure:,.0f}", "Transfers − discharges"),
    ("CBP INTAKE", f"{current_intake:,.0f}", "Daily intake")
]
for col, (label, value, note) in zip(kpi_cols, items):
    with col:
        st.markdown(
            f"""
            <div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# HISTORICAL CARE LOAD
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">📈 Historical HHS Care Load</div>
      <div class="section-sub">Historical care-load trend</div>
    </div>
    """,
    unsafe_allow_html=True
)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df[CARE_COL],
        mode="lines",
        name="HHS Care Load"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["7 Day Care Average"],
        mode="lines",
        name="7-Day Average"
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Children",
    hovermode="x unified",
    height=450
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FLOW ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">🔄 Transfers, Discharges & Net Pressure</div>
      <div class="section-sub">Flow dynamics</div>
    </div>
    """,
    unsafe_allow_html=True
)

flow_fig = go.Figure()

flow_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df[TRANSFER_COL],
        mode="lines",
        name="Transfers"
    )
)

flow_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df[DISCHARGE_COL],
        mode="lines",
        name="Discharges"
    )

)

flow_fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Net Pressure"],
        mode="lines",
        name="Net Pressure"
    )
)

flow_fig.add_hline(
    y=0,
    line_dash="dash"
)

flow_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Children",
    hovermode="x unified",
    height=450
)

st.plotly_chart(
    flow_fig,
    use_container_width=True
)


# ============================================================
# CARE LOAD FORECAST
# ============================================================

if model_name == "Random Forest":

    care_forecast = (
        recursive_ml_forecast(
            care_package[
                "rf_model"
            ],
            df,
            forecast_horizon
        )
    )

    lower, upper = (
        random_forest_interval(
            care_package[
                "rf_model"
            ],
            df,
            forecast_horizon
        )
    )

    care_forecast[
        "Lower Bound"
    ] = lower

    care_forecast[
        "Upper Bound"
    ] = upper

elif model_name == "Gradient Boosting":

    care_forecast = (
        recursive_ml_forecast(
            care_package[
                "gb_model"
            ],
            df,
            forecast_horizon
        )
    )

    care_forecast[
        "Lower Bound"
    ] = (
        care_forecast[
            "Forecast"
        ] * 0.95
    )

    care_forecast[
        "Upper Bound"
    ] = (
        care_forecast[
            "Forecast"
        ] * 1.05
    )

elif model_name == "ARIMA":

    care_forecast = (
        arima_forecast(
            df,
            CARE_COL,
            forecast_horizon
        )
    )

else:

    care_forecast = (
        naive_forecast(
            df,
            CARE_COL,
            forecast_horizon
        )
    )


# ============================================================
# CARE FORECAST CHART
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">🔮 Future Care Load Forecast</div>
      <div class="section-sub">Future care-load forecast</div>
    </div>
    """,
    unsafe_allow_html=True
)

forecast_fig = go.Figure()

recent = df.tail(60)

forecast_fig.add_trace(
    go.Scatter(
        x=recent.index,
        y=recent[CARE_COL],
        mode="lines",
        name="Historical"
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=care_forecast.index,
        y=care_forecast[
            "Forecast"
        ],
        mode="lines+markers",
        name=f"{model_name} Forecast"
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=care_forecast.index,
        y=care_forecast[
            "Upper Bound"
        ],
        mode="lines",
        line=dict(
            dash="dot"
        ),
        name="Upper Bound"
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=care_forecast.index,
        y=care_forecast[
            "Lower Bound"
        ],
        mode="lines",
        line=dict(
            dash="dot"
        ),
        name="Lower Bound"
    )
)

forecast_fig.add_hline(
    y=capacity,
    line_dash="dash",
    annotation_text="Capacity Threshold"
)

forecast_fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Children in HHS Care",
    hovermode="x unified",
    height=500
)

st.plotly_chart(
    forecast_fig,
    use_container_width=True
)


# ============================================================
# CARE FORECAST KPIs
# ============================================================

care_peak = float(
    care_forecast[
        "Forecast"
    ].max()
)

care_average = float(
    care_forecast[
        "Forecast"
    ].mean()
)

care_end = float(
    care_forecast[
        "Forecast"
    ].iloc[-1]
)

capacity_breach = (
    care_forecast[
        "Forecast"
    ]
    >= capacity
)

if capacity_breach.any():

    breach_date = (
        care_forecast.index[
            np.where(
                capacity_breach
            )[0][0]
        ]
    )

    surge_lead_time = (
        breach_date
        - df.index[-1]
    ).days

else:

    breach_date = None
    surge_lead_time = None


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Forecast Peak",
        f"{care_peak:,.0f}"
    )

with c2:
    st.metric(
        "Forecast Average",
        f"{care_average:,.0f}"
    )

with c3:
    st.metric(
        "End Forecast",
        f"{care_end:,.0f}"
    )

with c4:

    if surge_lead_time is None:
        st.metric(
            "Surge Lead Time",
            "No Breach"
        )
    else:
        st.metric(
            "Surge Lead Time",
            f"{surge_lead_time} Days"
        )


# ============================================================
# CAPACITY ALERT
# ============================================================

if breach_date is not None:

    st.markdown(
        f'<div class="alert-risk">⚠️ <b>Capacity stress alert:</b> '
        f'{model_name} forecasts a potential threshold breach on '
        f'{breach_date.strftime("%Y-%m-%d")}, approximately '
        f'{surge_lead_time} days ahead.</div>',
        unsafe_allow_html=True
    )

else:

    st.markdown(
        '<div class="alert-good">✓ <b>Capacity outlook:</b> '
        'no threshold breach is forecast during the selected horizon.</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DISCHARGE DEMAND FORECAST
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">📦 Discharge / Placement Demand Forecast</div>
      <div class="section-sub">Placement demand forecast</div>
    </div>
    """,
    unsafe_allow_html=True
)

try:

    discharge_package = train_models(
        df,
        DISCHARGE_COL
    )

    if model_name == "Random Forest":

        discharge_forecast = (
            recursive_ml_forecast(
                discharge_package[
                    "rf_model"
                ],
                df,
                forecast_horizon
            )
        )

        discharge_forecast[
            "Lower Bound"
        ] = (
            discharge_forecast[
                "Forecast"
            ] * 0.90
        )

        discharge_forecast[
            "Upper Bound"
        ] = (
            discharge_forecast[
                "Forecast"
            ] * 1.10
        )

    elif model_name == "Gradient Boosting":

        discharge_forecast = (
            recursive_ml_forecast(
                discharge_package[
                    "gb_model"
                ],
                df,
                forecast_horizon
            )
        )

        discharge_forecast[
            "Lower Bound"
        ] = (
            discharge_forecast[
                "Forecast"
            ] * 0.90
        )

        discharge_forecast[
            "Upper Bound"
        ] = (
            discharge_forecast[
                "Forecast"
            ] * 1.10
        )

    elif model_name == "ARIMA":

        discharge_forecast = (
            arima_forecast(
                df,
                DISCHARGE_COL,
                forecast_horizon
            )
        )

    else:

        discharge_forecast = (
            naive_forecast(
                df,
                DISCHARGE_COL,
                forecast_horizon
            )
        )

    discharge_forecast[
        "Forecast"
    ] = discharge_forecast[
        "Forecast"
    ].clip(lower=0)

    discharge_forecast[
        "Lower Bound"
    ] = discharge_forecast[
        "Lower Bound"
    ].clip(lower=0)

    discharge_forecast[
        "Upper Bound"
    ] = discharge_forecast[
        "Upper Bound"
    ].clip(lower=0)

    discharge_fig = go.Figure()

    discharge_fig.add_trace(
        go.Scatter(
            x=df.tail(60).index,
            y=df.tail(60)[
                DISCHARGE_COL
            ],
            mode="lines",
            name="Historical Discharges"
        )
    )

    discharge_fig.add_trace(
        go.Scatter(
            x=discharge_forecast.index,
            y=discharge_forecast[
                "Forecast"
            ],
            mode="lines+markers",
            name="Forecast Discharges"
        )
    )

    discharge_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Children Discharged",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(
        discharge_fig,
        use_container_width=True
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.metric(
            "Peak Discharge Demand",
            f"{discharge_forecast['Forecast'].max():,.0f}"
        )

    with d2:
        st.metric(
            "Average Discharge Demand",
            f"{discharge_forecast['Forecast'].mean():,.0f}"
        )

    with d3:
        st.metric(
            "Total Forecast Demand",
            f"{discharge_forecast['Forecast'].sum():,.0f}"
        )

except Exception as e:

    st.warning(
        "Discharge forecasting could not be generated."
    )

    st.exception(e)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">🤖 Care-Load Model Comparison</div>
      <div class="section-sub">Forecast model comparison</div>
    </div>
    """,
    unsafe_allow_html=True
)

results = (
    care_package[
        "results"
    ]
    .copy()
)

st.dataframe(
    results.style.format(
        {
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "MAPE (%)": "{:,.2f}"
        }
    ),
    use_container_width=True
)

best_model_row = results.loc[
    results["MAE"].idxmin()
]

st.info(
    f"Best validation model by MAE: "
    f"**{best_model_row['Model']}** "
    f"with MAE "
    f"**{best_model_row['MAE']:,.2f}**."
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

comparison_fig = go.Figure()

comparison_fig.add_trace(
    go.Bar(
        x=results["Model"],
        y=results["MAE"],
        name="MAE"
    )
)

comparison_fig.update_layout(
    title="Model Comparison — MAE",
    xaxis_title="Model",
    yaxis_title="MAE",
    height=400
)

st.plotly_chart(
    comparison_fig,
    use_container_width=True
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">🔍 Random Forest Feature Importance</div>
      <div class="section-sub">Feature importance</div>
    </div>
    """,
    unsafe_allow_html=True
)

importance = pd.DataFrame(
    {
        "Feature": FEATURE_COLUMNS,
        "Importance":
            care_package[
                "rf_model"
            ].feature_importances_
    }
).sort_values(
    "Importance",
    ascending=True
)

importance_fig = go.Figure()

importance_fig.add_trace(
    go.Bar(
        x=importance[
            "Importance"
        ],
        y=importance[
            "Feature"
        ],
        orientation="h"
    )
)

importance_fig.update_layout(
    title="Forecasting Feature Importance",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=450
)

st.plotly_chart(
    importance_fig,
    use_container_width=True
)


# ============================================================
# FORECAST TABLE
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">📅 Care Load Forecast Details</div>
      <div class="section-sub">Forecast details</div>
    </div>
    """,
    unsafe_allow_html=True
)

forecast_table = (
    care_forecast
    .reset_index()
    .rename(
        columns={
            "index": "Date"
        }
    )
)

forecast_table[
    "Forecast"
] = forecast_table[
    "Forecast"
].round(0)

forecast_table[
    "Lower Bound"
] = forecast_table[
    "Lower Bound"
].round(0)

forecast_table[
    "Upper Bound"
] = forecast_table[
    "Upper Bound"
].round(0)

forecast_table[
    "Capacity Risk"
] = np.where(
    forecast_table[
        "Forecast"
    ] >= capacity,
    "High",
    "Normal"
)

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    label="📥 Download Care Forecast CSV",
    data=forecast_table.to_csv(
        index=False
    ),
    file_name="care_load_forecast.csv",
    mime="text/csv"
)


# ============================================================
# DATA INSIGHTS
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">💡 Key Data Insights</div>
      <div class="section-sub">Key data insights</div>
    </div>
    """,
    unsafe_allow_html=True
)

highest_care_date = (
    df[CARE_COL].idxmax()
)

highest_care_value = float(
    df[CARE_COL].max()
)

highest_discharge_date = (
    df[DISCHARGE_COL].idxmax()
)

highest_discharge_value = float(
    df[DISCHARGE_COL].max()
)

average_pressure = float(
    df["Net Pressure"].mean()
)

i1, i2, i3 = st.columns(3)

with i1:

    st.metric(
        "Historical Peak HHS Care",
        f"{highest_care_value:,.0f}"
    )

    st.caption(
        highest_care_date.strftime(
            "%Y-%m-%d"
        )
    )

with i2:

    st.metric(
        "Historical Peak Discharges",
        f"{highest_discharge_value:,.0f}"
    )

    st.caption(
        highest_discharge_date.strftime(
            "%Y-%m-%d"
        )
    )

with i3:

    st.metric(
        "Average Net Pressure",
        f"{average_pressure:,.2f}"
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section">
      <div class="section-title">📝 Executive Summary</div>
      <div class="section-sub">Executive summary</div>
    </div>
    """,
    unsafe_allow_html=True
)

if current_pressure > 0:

    pressure_message = (
        "Transfers are currently greater than "
        "discharges, indicating positive net pressure "
        "on the HHS care load."
    )

elif current_pressure < 0:

    pressure_message = (
        "Discharges are currently greater than "
        "transfers, indicating negative net pressure "
        "on the HHS care load."
    )

else:

    pressure_message = (
        "Transfers and discharges are currently balanced."
    )


if breach_date is not None:

    capacity_message = (
        f"The selected model forecasts a capacity "
        f"threshold breach on "
        f"{breach_date.strftime('%Y-%m-%d')}, "
        f"providing approximately "
        f"{surge_lead_time} days of lead time."
    )

else:

    capacity_message = (
        "No capacity threshold breach is forecast "
        "during the selected forecast horizon."
    )


st.info(
    f"""
**Current HHS Care Load:** {current_care:,.0f}

**Selected Forecast Model:** {model_name}

**Forecast Horizon:** {forecast_horizon} days

**Forecast Peak:** {care_peak:,.0f}

**Forecast Average:** {care_average:,.0f}

**Current Net Pressure:** {current_pressure:,.0f}

**Operational Interpretation:** {pressure_message}

**Capacity Assessment:** {capacity_message}
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
      <b>Predictive Care Load & Placement Demand</b><br>
      AI Forecasting • Time-Series Analytics • Capacity Risk Intelligence<br><br>
      Forecasts are analytical estimates and should be reviewed with appropriate
      operational and domain expertise.
    </div>
    """,
    unsafe_allow_html=True
)
