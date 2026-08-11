from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException  # type: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[reportMissingImports]


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "uac_daily_data.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_care_load_model.pkl"


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Predictive Care Load & Placement Demand API",
    description=(
        "FastAPI service for UAC care-load analytics, "
        "forecasting and operational indicators."
    ),
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program - HHS_Unaccompanied_Alien_Children_Program.csv")

    df.columns = df.columns.str.strip()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date"]
        )

        df = df.sort_values(
            "Date"
        ).reset_index(drop=True)

    return df


# ============================================================
# LOAD DATASET AT STARTUP
# ============================================================

try:

    df = load_dataset()

    dataset_status = "loaded"

except Exception as e:

    df = pd.DataFrame()

    dataset_status = f"error: {str(e)}"


# ============================================================
# LOAD MODEL
# ============================================================

model = None

if MODEL_PATH.exists():

    try:

        if MODEL_PATH.stat().st_size > 0:

            model = joblib.load(
                MODEL_PATH
            )

    except Exception as e:

        print(
            f"Model loading failed: {e}"
        )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "application": (
            "Predictive Care Load "
            "& Placement Demand API"
        ),
        "version": "1.0.0",
        "dataset_status": dataset_status,
        "model_loaded": model is not None,
        "documentation": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "dataset_loaded": not df.empty,
        "model_loaded": model is not None
    }


# ============================================================
# DATASET INFORMATION
# ============================================================

@app.get("/dataset/info")
def dataset_info():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded."
        )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "start_date": (
            df["Date"].min().strftime("%Y-%m-%d")
            if "Date" in df.columns
            else None
        ),
        "end_date": (
            df["Date"].max().strftime("%Y-%m-%d")
            if "Date" in df.columns
            else None
        )
    }


# ============================================================
# DATASET PREVIEW
# ============================================================

@app.get("/dataset/preview")
def dataset_preview(
    limit: int = 10
):

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded."
        )

    limit = max(
        1,
        min(limit, 100)
    )

    preview = df.head(
        limit
    ).copy()

    if "Date" in preview.columns:

        preview["Date"] = (
            preview["Date"]
            .dt.strftime("%Y-%m-%d")
        )

    preview = preview.astype(
        object
    ).where(
        pd.notna(preview),
        None
    )

    return {
        "rows": preview.to_dict(
            orient="records"
        )
    }


# ============================================================
# LATEST DATA
# ============================================================

@app.get("/dataset/latest")
def latest_data():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded."
        )

    row = df.iloc[-1].copy()

    if "Date" in row.index:

        row["Date"] = (
            row["Date"]
            .strftime("%Y-%m-%d")
        )

    return row.to_dict()


# ============================================================
# CARE LOAD SUMMARY
# ============================================================

@app.get("/analytics/summary")
def analytics_summary():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded."
        )

    required_columns = [
        "Children in HHS Care",
        "Children transferred out of CBP custody",
        "Children discharged from HHS Care"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Required columns are missing.",
                "missing_columns": missing
            }
        )

    latest = df.iloc[-1]

    current_care = float(
        latest["Children in HHS Care"]
    )

    transfers = float(
        latest[
            "Children transferred out of CBP custody"
        ]
    )

    discharges = float(
        latest[
            "Children discharged from HHS Care"
        ]
    )

    net_pressure = (
        transfers - discharges
    )

    return {
        "date": (
            latest["Date"].strftime("%Y-%m-%d")
            if "Date" in latest.index
            else None
        ),
        "hhs_care_load": current_care,
        "transfers_to_hhs": transfers,
        "discharges": discharges,
        "net_pressure": net_pressure
    }


# ============================================================
# DATASET STATISTICS
# ============================================================

@app.get("/analytics/statistics")
def statistics():

    if df.empty:

        raise HTTPException(
            status_code=500,
            detail="Dataset is not loaded."
        )

    numeric_df = df.select_dtypes(
        include="number"
    )

    return {
        "statistics": (
            numeric_df
            .describe()
            .round(2)
            .to_dict()
        )
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model/info")
def model_info():

    if model is None:

        return {
            "model_loaded": False,
            "message": (
                "Random Forest model is not available."
            )
        }

    return {
        "model_loaded": True,
        "model_type": type(model).__name__,
        "model_file": str(MODEL_PATH)
    }
