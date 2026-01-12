# =========================================================
# CBAM – EXPORTS – EMISSIONS DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample

st.set_page_config(layout="wide", page_title="CBAM Impact Dashboard")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
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
    return df

df = load_data()

# =========================================================
# SIDEBAR CONTROLS (GLOBAL)
# =========================================================
st.sidebar.header("⚙️ Global Assumptions")

cbam_price = st.sidebar.slider(
    "CBAM Carbon Price (USD / tCO₂)",
    min_value=50,
    max_value=150,
    value=98,
    step=5
)

textile_share = st.sidebar.slider(
    "Textile Share of Trade CO₂",
    min_value=0.3,
    max_value=0.7,
    value=0.54,
    step=0.05
)

gdp_growth = st.sidebar.slider(
    "Annual GDP Growth (log points)",
    0.01, 0.06, 0.03, 0.005
)

energy_eff = st.sidebar.slider(
    "Energy Intensity Improvement (%)",
    0.0, 5.0, 1.0, 0.25
) / 100

emission_eff = st.sidebar.slider(
    "Emission Intensity Improvement (%)",
    0.0, 5.0, 2.0, 0.25
) / 100

# =========================================================
# RECOMPUTE CBAM WITH SLIDERS
# =========================================================
df = df.copy()
df["textile_trade_emissions_t"] = df["trade_co2"] * textile_share * 1e6
df["CBAM_cost"] = df["textile_trade_emissions_t"] * cbam_price
df["CBAM_share"] = df["CBAM_cost"] / df["exports"]

# =========================================================
# TABS
# =========================================================
tabs = st.tabs([
    "📊 Data Overview",
    "🔥 CBAM Cost",
    "📈 Correlations",
    "📉 Econometrics",
    "🎲 Monte Carlo",
    "🤖 ML Models",
    "🔮 Forecasts",
    "📥 Downloads"
])

# =========================================================
# TAB 1: DATA OVERVIEW
# =========================================================
with tabs[0]:
    st.title("Pakistan Exports & Emissions Overview")

    col1, col2 = st.columns(2)

    fig = px.line(df, x="year", y="exports", title="Exports Over Time")
    col1.plotly_chart(fig, use_container_width=True)

    fig = px.line(df, x="year", y="trade_co2", title="Trade CO₂ Over Time")
    col2.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.tail(10))

# =========================================================
# TAB 2: CBAM COST
# =========================================================
with tabs[1]:
    st.title("CBAM Cost Calculator")

    fig = px.line(
        df,
        x="year",
        y="CBAM_cost",
        title="CBAM Cost (Textile Exports)"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        df,
        x="year",
        y="CBAM_share",
        title="CBAM Share of Exports"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3: CORRELATIONS
# =========================================================
with tabs[2]:
    st.title("Correlation Analysis")

    corr_vars = [
        "exports",
        "CBAM_cost",
        "CBAM_share",
        "energy_per_gdp",
        "emission_intensity_t_per_USD_tradeB",
        "ln_gdp"
    ]

    corr = df[corr_vars].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4: ECONOMETRICS
# =========================================================
with tabs[3]:
    st.title("Econometric Models")

    reg_df = df.dropna(subset=[
        "ln_exports",
        "CBAM_share",
        "energy_per_gdp",
        "emission_intensity_t_per_USD_tradeB",
        "ln_gdp"
    ])

    X = reg_df[[
        "emission_intensity_t_per_USD_tradeB",
        "energy_per_gdp",
        "ln_gdp",
        "CBAM_share"
    ]]
    X = sm.add_constant(X)
    y = reg_df["ln_exports"]

    model = sm.OLS(y, X).fit(cov_type="HC1")

    st.text(model.summary())

    st.subheader("ADF Tests")
    st.write("ln_exports p-value:", adfuller(reg_df["ln_exports"])[1])
    st.write("CBAM_share p-value:", adfuller(reg_df["CBAM_share"])[1])


with tabs[5]:
    st.title("ML Model Evaluation with Lag Feature")
    
    features = [
        "emission_intensity_t_per_USD_tradeB",
        "energy_per_gdp",
        "ln_gdp",
        "CBAM_share"
    ]

    df_ml = df.dropna(subset=features + ["ln_exports"]).copy()
    
    # Create lag feature
    df_ml["ln_exports_lag1"] = df_ml["ln_exports"].shift(1)
    df_ml = df_ml.dropna()

    ml_features = features + ["ln_exports_lag1"]
    X = df_ml[ml_features]
    y = df_ml["ln_exports"]

    # Train/test split
    split = int(len(df_ml) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Models
    models = {
        "OLS": LinearRegression(),
        "LASSO": Lasso(alpha=0.001),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    rows = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        rows.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred)
        })
    st.dataframe(pd.DataFrame(rows))

# =========================================================
# TAB 7: FORECASTS TO 2030 USING PREVIOUS YEAR EXPORT AS FEATURE
# =========================================================
with tabs[6]:
    st.title("Dynamic Multi-Feature Forecasts to 2030")

    # Historical dataframe
    df_hist = df.dropna().copy()

    # Features to predict (including exports)
    target_features = [
        "exports",
        "CBAM_cost",
        "CBAM_share",
        "energy_per_gdp",
        "emission_intensity_t_per_USD_tradeB",
        "ln_gdp"
    ]

    # Create lagged features for training (lag-1)
    df_ml = df_hist.copy()
    for f in target_features:
        df_ml[f"{f}_lag1"] = df_ml[f].shift(1)
    df_ml = df_ml.dropna()

    # Training data: predict each feature from all lagged features
    models = {}
    for f in target_features:
        X_train = df_ml[[f"{feat}_lag1" for feat in target_features]]
        y_train = df_ml[f]
        m = RandomForestRegressor(n_estimators=500, random_state=42)
        m.fit(X_train, y_train)
        models[f] = m

    # Recursive forecast
    last_row = df_hist.iloc[-1].copy()
    forecast_years = list(range(int(last_row["year"]) + 1, 2031))
    forecasted = []

    for yr in forecast_years:
        X_pred = pd.DataFrame([{f"{feat}_lag1": last_row[feat] for feat in target_features}])
        new_row = {}
        for f in target_features:
            pred = models[f].predict(X_pred)[0]
            new_row[f] = pred
        forecasted.append(new_row)
        last_row.update(new_row)  # update last_row for next iteration

    # Extract exports
    export_preds = [row["exports"] for row in forecasted]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=export_preds,
        mode="lines+markers",
        name="Predicted Exports"
    ))
    fig.update_layout(
        title="Predicted Exports (Recursive Dynamic Forecast)",
        xaxis_title="Year",
        yaxis_title="Exports (USD)",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 8: DOWNLOADS
# =========================================================
with tabs[7]:
    st.title("Download Processed Data")

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="cbam_dashboard_output.csv"
    )
