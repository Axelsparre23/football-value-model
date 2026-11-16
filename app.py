"""Streamlit web interface."""
import streamlit as st
from pathlib import Path
import numpy as np
from data_manager import download_data, LEAGUES
from football_value_model_v3 import run_backtest

st.set_page_config(page_title="Football Value Model", layout="wide")
st.title("Football Value Model")
st.caption("Historical football forecasting and simulated evaluation only.")
with st.sidebar:
    st.header("Model settings")
    first_season = st.selectbox("First season", [2010, 2012, 2014, 2016, 2018, 2020], index=3)
    min_train = st.slider("Minimum training matches per league", 150, 800, 300, 50)
    min_edge = st.slider("Minimum model edge", 0.00, 0.20, 0.05, 0.01)
    refit = st.slider("Refit every N matchdays", 1, 30, 12)
    refresh = st.button("Refresh current season data")
if "data" not in st.session_state or refresh:
    with st.spinner("Downloading and caching historical data..."):
        st.session_state.data = download_data(Path("data/cache"), first_start_year=first_season)
data = st.session_state.data
st.success(f"Loaded {len(data):,} matches across {data.League.nunique()} leagues.")
st.write("Leagues:", ", ".join(LEAGUES[k] for k in sorted(data.League.unique())))
st.info("First run downloads available historical CSVs. Later runs use the local cache.")
if st.button("Run walk-forward backtest", type="primary"):
    with st.spinner("Fitting chronological models and evaluating out-of-sample forecasts..."):
        result = run_backtest(data, min_train=min_train, refit_matchdays=refit, min_edge=min_edge)
    if result.empty:
        st.warning("No predictions. Try a lower minimum training size or check data availability.")
    else:
        bets = result[result.Stake > 0]
        y = result.Actual.map({"Home":0,"Draw":1,"Away":2}).to_numpy()
        p = result[["PHome","PDraw","PAway"]].to_numpy()
        onehot = np.eye(3)[y]
        brier = float(np.mean(np.sum((p-onehot)**2, axis=1)))
        logloss = float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-9,1))))
        profit = float(bets.Profit.sum()) if len(bets) else 0.0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Forecasts", f"{len(result):,}")
        c2.metric("Simulated selections", f"{len(bets):,}")
        c3.metric("Simulated net units", f"{profit:+.2f}")
        c4.metric("Brier score", f"{brier:.4f}")
        st.metric("Log loss (lower is better)", f"{logloss:.4f}")
        st.dataframe(result.sort_values("Date", ascending=False), use_container_width=True)
        st.download_button("Download predictions CSV", result.to_csv(index=False), "predictions.csv", "text/csv")
        if len(bets):
            st.download_button("Download simulated selections CSV", bets.to_csv(index=False), "simulated_selections.csv", "text/csv")
st.caption("Model outputs are estimates, not guarantees of value or future profitability.")
