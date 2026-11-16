"""Quick football match predictor."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from data_manager import download_data, LEAGUES
from football_value_model_v3 import DixonColesModel, dixon_coles_tau

st.set_page_config(page_title="Quick Match Predictor", page_icon="⚽", layout="centered")
st.title("⚽ Quick Match Predictor")
st.caption("Choose two teams to estimate home win, draw and away win probabilities.")

@st.cache_data(ttl=21600, show_spinner=False)
def load_league(league_code):
    # Load only the selected league and recent seasons, not all five leagues.
    return download_data(
        Path("data/cache"),
        first_start_year=2022,
        leagues=(league_code,),
        refresh_current=True,
    )

@st.cache_resource(show_spinner=False)
def fit_league_model(league_code, data_signature, data):
    # The signature invalidates the cached model when the dataset changes.
    model = DixonColesModel(half_life=365.0, reg=0.04, xg_weight=0.0)
    return model.fit(data, pd.Timestamp.now().normalize())

league_code = st.selectbox(
    "League",
    options=list(LEAGUES.keys()),
    format_func=lambda code: LEAGUES[code],
)

with st.spinner(f"Loading recent {LEAGUES[league_code]} results..."):
    try:
        data = load_league(league_code)
    except Exception as exc:
        st.error(f"Could not load match data: {exc}")
        st.stop()

if data.empty:
    st.error("No match data is available for this league yet.")
    st.stop()

# Prefer teams seen in the latest 12 months; fall back to all teams in the data.
recent = data[data["Date"] >= data["Date"].max() - pd.Timedelta(days=365)]
teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"]))
if len(teams) < 2:
    teams = sorted(set(data["HomeTeam"]) | set(data["AwayTeam"]))

st.caption(f"Using {len(data):,} historical matches. Latest result: {data['Date'].max().date()}.")

left, right = st.columns(2)
with left:
    home_team = st.selectbox("Home team", teams, index=0)
with right:
    away_options = [team for team in teams if team != home_team]
    away_team = st.selectbox("Away team", away_options, index=0)

if st.button("Predict match", type="primary", use_container_width=True):
    if home_team == away_team:
        st.warning("Choose two different teams.")
        st.stop()

    with st.spinner("Calculating probabilities..."):
        try:
            signature = f"{league_code}-{len(data)}-{data['Date'].max().date()}"
            model = fit_league_model(league_code, signature, data)
            probabilities, (home_xg, away_xg) = model.probabilities(home_team, away_team)

            # Most likely exact score from the Dixon-Coles-adjusted score matrix.
            scores = []
            rho = model.rho
            for h in range(8):
                for a in range(8):
                    probability = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                    if h <= 1 and a <= 1:
                        probability *= dixon_coles_tau(h, a, home_xg, away_xg, rho)
                    scores.append((probability, h, a))
            _, predicted_home_goals, predicted_away_goals = max(scores)

        except KeyError:
            st.error("One of these teams has too little recent history in this league to produce a reliable estimate.")
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    names = [home_team, "Draw", away_team]
    best_index = int(np.argmax(probabilities))
    st.subheader(f"Most likely: {names[best_index]}")
    st.markdown(f"## {probabilities[best_index]:.1%}")
    st.caption("Estimated probability of the most likely outcome — not a guarantee.")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team} win", f"{probabilities[0]:.1%}")
    c2.metric("Draw", f"{probabilities[1]:.1%}")
    c3.metric(f"{away_team} win", f"{probabilities[2]:.1%}")

    st.divider()
    st.metric("Most likely score", f"{predicted_home_goals} – {predicted_away_goals}")
    st.caption(f"Expected goals: {home_team} {home_xg:.2f} · {away_team} {away_xg:.2f}")

with st.expander("How this prediction works"):
    st.write(
        "A Dixon–Coles goal model estimates each team's attacking and defensive strength "
        "from recent league results, applies a home advantage and converts expected goals "
        "into win/draw/loss probabilities. Recent matches receive more weight. "
        "This version does not yet include injuries, confirmed lineups, suspensions or live match events."
    )

st.caption("Research estimates only. Probabilities can be wrong, especially for teams with limited recent data.")
