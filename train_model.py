import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



# -----------------------------
# Load dataset
# -----------------------------
healthcare = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program - HHS_Unaccompanied_Alien_Children_Program.csv")

print("Dataset shape:", healthcare.shape)
print("Columns:")
print(healthcare.columns.tolist())


# -----------------------------
# Clean column names
# -----------------------------
healthcare.columns =(healthcare.columns.str.strip().str.lower().str.replace(" ","_").str.replace("-","_"))
print(healthcare.columns)

# -----------------------------
# Convert Date
# -----------------------------
healthcare["date"] = pd.to_datetime(healthcare["date"])
healthcare =healthcare.sort_values("date")
healthcare = healthcare.set_index("date")
healthcare.head()

# -----------------------------
# Target column
# -----------------------------
target = "children_in_hhs_care"
X = healthcare.drop(columns=[target])
y = healthcare[target]
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.80, test_size=0.20)

# -----------------------------
# Create lag features
# -----------------------------
healthcare["core_load_change"] = ( healthcare ["children_in_hhs_care"].diff)
healthcare["lag_1"] = (healthcare["children_in_hhs_care"].shift(1))
healthcare["lag_7"] = healthcare["children_in_hhs_care"].shift(7)
healthcare["lag_14"] = (healthcare["children_in_hhs_care"].shift(14))
healthcare["children_in_hhs_care"] = pd.to_numeric(
    healthcare["children_in_hhs_care"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

healthcare["core_load_change"] = healthcare["children_in_hhs_care"].diff()
healthcare["lag_1"] = healthcare["children_in_hhs_care"].shift(1)
healthcare["lag_7"] = healthcare["children_in_hhs_care"].shift(7)
healthcare["lag_14"] = healthcare["children_in_hhs_care"].shift(14)

healthcare["rolling_mean_7"] = healthcare["children_in_hhs_care"].rolling(7).mean()
healthcare["rolling_mean_14"] = (healthcare["children_in_hhs_care"].rolling(14).mean())
healthcare["rolling_mean_7"] = (healthcare["children_in_hhs_care"].rolling(7).std())

# -----------------------------
# Flow features
# -----------------------------
transfer_col = "Children transferred out of CBP custody"

healthcare["net_pressure"] = (healthcare["children_transferred_out_of_cbp_custody"] - healthcare["children_discharged_from_hhs_care"])

# -----------------------------
# Calendar features
# -----------------------------
healthcare["day_of_week"] = healthcare.index.dayofweek
healthcare["month"] = healthcare.index.month
healthcare["day"] = healthcare.index.day
healthcare["is_weekend"] = (healthcare["day_of_week"]>=5).astype(int)

# -----------------------------
# Feature columns
# -----------------------------
features = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14", "rolling_std_7", "net_pressure", "day_of_week","month","is_weekend"]


# -----------------------------
# Remove missing values
# -----------------------------
healthcare["rolling_mean_7"] = (
    healthcare[target].rolling(window=7).mean()
)
healthcare["rolling_std_7"] = (
    healthcare[target].rolling(window=7).std()
)

healthcare = healthcare.dropna(subset=features + [target])

X = healthcare[features]
y = healthcare[target]

split_index = int(len(healthcare) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
y = healthcare[target]

split_index = int(len(healthcare) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# -----------------------------
# Time-based split
# -----------------------------
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# Train Random Forest
# -----------------------------
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "healthcare_model.joblib")

joblib.dump(
    model,
    MODEL_PATH
)

print()
print("================================")
print("MODEL TRAINING SUCCESSFUL")
print("================================")
print("Model saved:", MODEL_PATH)
print("Features:", features)