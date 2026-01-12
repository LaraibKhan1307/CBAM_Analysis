import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="CBAM Impact on Exports",
    layout="wide"
)

st.title("CBAM Impact on Pakistan’s Textile Exports")

# ---------------------------
# Load / create data
# ---------------------------
# Replace this with your real dataframe loading
df = pd.DataFrame({
    "year": np.arange(2010, 2023),
    "exports": np.linspace(12e9, 20e9, 13),
    "textile_trade_emissions_mton_B": np.linspace(15, 22, 13)
})

# ---------------------------
# Tabs
# ---------------------------
tab_overview, tab_analysis, tab_mc = st.tabs(
    ["Overview", "Main Analysis", "CBAM Sensitivity"]
)

# ======================================================
# OVERVIEW TAB
# ======================================================
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.write(
        """
        This dashboard evaluates the impact of the EU Carbon Border Adjustment
        Mechanism (CBAM) on Pakistan’s textile exports using historical trade
        and emissions proxy data.
        """
    )

# ======================================================
# MAIN ANALYSIS TAB
# ======================================================
with tab_analysis:
    st.subheader("Exports Over Time")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["year"], df["exports"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Exports (USD)")
    ax.set_title("Textile Exports")
    ax.grid(True)

    st.pyplot(fig)

# ======================================================
# CBAM SENSITIVITY TAB (FIXED)
# ======================================================
with tab_mc:
    st.subheader("CBAM Cost Sensitivity Analysis")

    # ---- Slider (this alone does NOTHING visually) ----
    cbam_price = st.slider(
        "CBAM price (USD per ton CO₂)",
        min_value=50,
        max_value=150,
        value=100,
        step=10
    )

    # ---- Deterministic CBAM calculation ----
    df_tmp = df.copy()
    df_tmp["CBAM_cost_USD"] = (
        df_tmp["textile_trade_emissions_mton_B"] * 1e6 * cbam_price
    )

    df_tmp["CBAM_loss_pct"] = (
        100 * df_tmp["CBAM_cost_USD"] / df_tmp["exports"]
    )

    # ---- Metrics ----
    col1, col2 = st.columns(2)
    col1.metric(
        "Average CBAM Cost (USD bn)",
        f"{df_tmp['CBAM_cost_USD'].mean() / 1e9:.2f}"
    )
    col2.metric(
        "Average Export Loss (%)",
        f"{df_tmp['CBAM_loss_pct'].mean():.2f}"
    )

    # ---- Plot ----
    st.subheader("Export Loss Share Over Time")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(
        df_tmp["year"],
        df_tmp["CBAM_loss_pct"],
        marker="o"
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Loss (% of exports)")
    ax2.set_title(f"CBAM Impact (Price = {cbam_price} USD/tCO₂)")
    ax2.grid(True)

    st.pyplot(fig2)

    # ---- Table ----
    st.subheader("Year-wise CBAM Impact")
    st.dataframe(
        df_tmp[[
            "year",
            "exports",
            "textile_trade_emissions_mton_B",
            "CBAM_cost_USD",
            "CBAM_loss_pct"
        ]]
    )

# ---------------------------
# Footer
# ---------------------------
st.caption("Deterministic CBAM sensitivity – no Monte Carlo simulation")
