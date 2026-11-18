"""Prospective 40-match study evaluation dashboard."""
from pathlib import Path
import streamlit as st
import pandas as pd

from research.prospective_evaluation import (
    load_log, scored_rows, metric_table, calibration_table, paired_brier_bootstrap
)

st.set_page_config(page_title="Prospective Study Evaluation", page_icon="📊", layout="wide")
st.title("Prospective model evaluation")
st.caption("Read-only evaluation of the pre-match forecasts. The original forecast log is never changed here.")

root = Path(__file__).resolve().parents[1]
log_path = root / "research" / "prospective_50_match_log.csv"
try:
    log = load_log(log_path)
    scored = scored_rows(log)
except Exception as exc:
    st.error(f"Could not load the prospective forecast log: {exc}")
    st.stop()

total = len(log)
completed = len(scored)
st.warning("No results have been added yet. This dashboard will populate as verified final scores are recorded after the matches finish." if completed == 0 else "Results are available. Check source notes and odds timestamps before interpreting the comparison.")
a, b, c = st.columns(3)
a.metric("Forecasts recorded", total)
b.metric("Results scored", completed)
c.metric("Awaiting results", max(0, total - completed))

st.subheader("Scoring rules")
st.write("Brier, log loss and ranked probability score (RPS) are proper scoring rules; lower is better. Accuracy is the share of correct top-probability outcomes; higher is better. Strict market comparison requires a named single bookmaker, source and timestamp. It uses closing no-vig probabilities only when all closing provenance fields are recorded; otherwise it may use a fully timestamped pre-match snapshot.")
st.caption("Synthetic best-price aggregates and unidentified operators are excluded from the strict market comparison because their 1X2 prices may not come from one bookmaker. Their recorded probabilities remain visible for audit, but are not treated as a clean market benchmark.")
valid_market = int(scored["Market_p"].notna().sum()) if "Market_p" in scored else 0
recorded_market = int(scored["Recorded_market_p"].notna().sum()) if "Recorded_market_p" in scored else 0
st.caption(f"Market data coverage: {valid_market} strict single-operator snapshots; {recorded_market - valid_market} recorded snapshots excluded from strict comparison.")

metrics = metric_table(scored)
st.subheader("Model comparison")
if metrics.empty:
    st.info("The comparison table will appear after at least one completed match has valid stored probabilities.")
else:
    display = metrics.copy()
    for col in ("brier", "log_loss", "rps", "accuracy"):
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    st.dataframe(display, hide_index=True, use_container_width=True)

    st.subheader("Calibration")
    model = st.selectbox("Model", ["Elo", "Dixon-Coles", "Market"])
    calibration = calibration_table(scored, model)
    if calibration.empty:
        st.info(f"No valid {model} probabilities are available for calibration yet.")
    else:
        st.dataframe(calibration, hide_index=True, use_container_width=True)
        chart = calibration.pivot(index="Bin", columns="Outcome", values="Observed frequency")
        st.line_chart(chart)

    st.subheader("Paired Brier comparison")
    comparison = st.selectbox("Compare", ["Dixon-Coles vs Market", "Elo vs Market", "Dixon-Coles vs Elo"])
    first, second = comparison.split(" vs ")
    interval = paired_brier_bootstrap(scored, first, second)
    if interval is None:
        st.info("At least two matches with valid forecasts from both selected models are needed.")
    else:
        st.write(f"Paired sample: {interval['matches']} matches.")
        st.write(f"Mean Brier difference ({first} − {second}): {interval['mean_difference']:.4f}")
        st.write(f"Bootstrap 95% interval: [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")
        if interval["ci_low"] <= 0 <= interval["ci_high"]:
            st.caption("The interval includes zero; this sample does not establish a clear difference.")
        else:
            st.caption(f"The interval excludes zero in favour of {interval['interpretation']} on this sample.")

if completed:
    st.subheader("Recorded outcomes and data quality")
    view_cols = [c for c in ["match_id", "competition", "kickoff_local", "home_team", "away_team",
                             "final_home_goals", "final_away_goals", "actual_outcome", "market_snapshot",
                             "market_source_status", "closing_bookmaker_name", "closing_odds_source",
                             "closing_odds_timestamp_utc", "bookmaker_name", "odds_source",
                             "odds_timestamp_utc"] if c in scored]
    st.dataframe(scored[view_cols], hide_index=True, use_container_width=True)

st.subheader("Study integrity")
st.write("The dashboard does not tune or alter Elo, Dixon–Coles, their probabilities, or the forecast CSV. It scores only recorded pre-match probabilities against completed outcomes. Small-sample uncertainty will be shown rather than hidden.")
