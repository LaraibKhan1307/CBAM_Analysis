import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/CBAM.csv")
textile_share_of_trade_co2 = 0.54   # user-provided default; change if you have better estimate
CBAM_PRICE_USD_PER_T = 97.75        # user-provided constant

# trade_co2 assumed in million tonnes (MtCO2)
df["textile_trade_emissions_mton_B"] = df["trade_co2"] * textile_share_of_trade_co2
df["textile_trade_emissions_tonnes_B"] = df["textile_trade_emissions_mton_B"] * 1e6
df["CBAM_Cost_textile_B_USD"] = df["textile_trade_emissions_tonnes_B"] * CBAM_PRICE_USD_PER_T
df["CBAM_share_of_exports_B"] = df["CBAM_Cost_textile_B_USD"] / df["exports"]
df["emission_intensity_t_per_USD_tradeB"] = df["textile_trade_emissions_tonnes_B"] / df["exports"]

# log transforms
df["ln_exports"] = np.log(df["exports"].replace(0, np.nan))
df["ln_gdp"] = np.log(df["gdp"].replace(0, np.nan))



features = [
    "emission_intensity_t_per_USD_tradeB",
    "energy_per_gdp",
    "ln_gdp",
    "CBAM_share_of_exports_B"
]
target = "ln_exports"

# -------------------------------
# Create lags (same as your code)
# -------------------------------
df = df.sort_values("year")

for v in features + [target]:
    df[f"{v}_lag1"] = df[v].shift(1)
    df[f"{v}_lag2"] = df[v].shift(2)

df_ml = df.dropna().copy()

X_cols = []
for v in features:
    X_cols += [f"{v}_lag1", f"{v}_lag2"]

X = df_ml[X_cols]
y = df_ml[target]

# -------------------------------
# Models
# -------------------------------
models = {
    "lasso": Lasso(alpha=0.001, max_iter=5000),
    "knn": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    "svm": SVR(kernel="rbf", C=10, epsilon=0.05),
    "random_forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        random_state=42
    )
}

# -------------------------------
# Train & Save
# -------------------------------
for name, model in models.items():
    model.fit(X, y)
    joblib.dump(model, f"models/{name}.pkl")

print("✅ Models trained and saved successfully.")
