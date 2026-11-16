"""Fast football match calculator: Dixon-Coles + dynamic Elo + transparent team news."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from data_manager import download_data, LEAGUES
from football_value_model_v3 import DixonColesModel, dixon_coles_tau
from advanced_model import (
    elo_history, elo_probabilities, blended_probabilities, parse_player_lines,
    player_impact, absence_impact, team_profile, rest_days,
)

from model_research import chronological_ensemble_evaluation

st.set_page_config(page_title="Football Match Calculator", page_icon="⚽", layout="wide")
st.title("Football Match Calculator")
st.caption("Premier League · La Liga · Bundesliga · Serie A · Ligue 1 · Allsvenskan")

@st.cache_data(ttl=21600, show_spinner=False)
def load_league(code):
    return download_data(Path("data/cache"), first_start_year=2010, leagues=(code,), refresh_current=True)

@st.cache_resource(show_spinner=False)
def fit_model(code, signature, data):
    half_life = 240 if code == "SWE" else 300
    return DixonColesModel(half_life=half_life, reg=0.04, xg_weight=0.0).fit(data, pd.Timestamp.now().normalize())

@st.cache_data(ttl=3600, show_spinner=False)
def get_elo(data):
    return elo_history(data)

league_code = st.selectbox("League", list(LEAGUES), format_func=lambda x: LEAGUES[x])
with st.spinner(f"Loading {LEAGUES[league_code]} results and available match statistics..."):
    try:
        data = load_league(league_code)
    except Exception as exc:
        st.error(f"Data could not be loaded: {exc}")
        st.stop()

if data.empty:
    st.error("No completed matches were found for this league.")
    st.stop()

latest_date = data.Date.max()
first_date = data.Date.min()
recent_teams = data[data.Date >= latest_date - pd.Timedelta(days=450)]
teams = sorted(set(recent_teams.HomeTeam) | set(recent_teams.AwayTeam))
if len(teams) < 2:
    teams = sorted(set(data.HomeTeam) | set(data.AwayTeam))
st.caption(f"{len(data):,} completed matches · coverage: {first_date.year}–{latest_date.year} · latest result: {latest_date.date()}")

left, right = st.columns(2)
with left:
    home = st.selectbox("Home team", teams, index=0)
with right:
    away = st.selectbox("Away team", [x for x in teams if x != home], index=0)

st.subheader("Lineups and player availability")
st.caption("Enter one player per line: Name | rating (1–10) | position. Example: Alex Smith | 7.4 | ST. Ratings and availability are manual inputs, not an automatic feed.")
l1, l2 = st.columns(2)
with l1:
    st.markdown(f"**{home}**")
    home_status = st.selectbox("Lineup status", ["Not entered", "Expected", "Confirmed"], key="hs")
    home_formation = st.selectbox("Formation", ["4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "Other / unknown"], key="hf")
    home_xi_raw = st.text_area("Starting XI — player | rating | position", height=180, key="hxi",
                               placeholder="Goalkeeper | 7.0 | GK\nCentre-back | 7.2 | CB\nStriker | 8.1 | ST")
    home_abs_raw = st.text_area("Unavailable players — player | rating | position", height=110, key="habs",
                                placeholder="Starting striker | 7.8 | ST")
with l2:
    st.markdown(f"**{away}**")
    away_status = st.selectbox("Lineup status", ["Not entered", "Expected", "Confirmed"], key="as")
    away_formation = st.selectbox("Formation", ["4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "Other / unknown"], key="af")
    away_xi_raw = st.text_area("Starting XI — player | rating | position", height=180, key="axi",
                               placeholder="Goalkeeper | 7.0 | GK\nCentre-back | 7.2 | CB\nStriker | 8.1 | ST")
    away_abs_raw = st.text_area("Unavailable players — player | rating | position", height=110, key="aabs",
                                placeholder="Starting centre-back | 7.5 | CB")

home_xi, away_xi = parse_player_lines(home_xi_raw), parse_player_lines(away_xi_raw)
home_abs, away_abs = parse_player_lines(home_abs_raw), parse_player_lines(away_abs_raw)

def team_snapshot(df, team, n=5):
    rows = df[((df.HomeTeam == team) | (df.AwayTeam == team))].sort_values("Date").tail(n)
    gf, ga, points = [], [], 0
    for _, r in rows.iterrows():
        is_home = r.HomeTeam == team
        scored = r.FTHG if is_home else r.FTAG
        conceded = r.FTAG if is_home else r.FTHG
        gf.append(float(scored)); ga.append(float(conceded))
        points += 3 if scored > conceded else (1 if scored == conceded else 0)
    return {"matches":len(rows), "gf":float(np.mean(gf)) if gf else 0.0,
            "ga":float(np.mean(ga)) if ga else 0.0, "points":points, "rows":rows}

# Use the actual competition season boundary for table position (calendar year in Sweden; July start for the European leagues).
season_start = pd.Timestamp(year=latest_date.year if league_code == "SWE" or latest_date.month >= 7 else latest_date.year - 1, month=1 if league_code == "SWE" else 7, day=1)
profile_data = data[data.Date >= season_start].copy()
with st.expander("Recent form and data-derived team profiles", expanded=True):
    ph, pa = st.columns(2)
    for col, team in [(ph, home), (pa, away)]:
        snap = team_snapshot(data, team, 5)
        prof = team_profile(profile_data, team)
        with col:
            st.markdown(f"**{team}**")
            st.write(f"Rolling-year position: **{prof['rank']}** · {prof['points']} points · GD {prof['gd']:+d}")
            st.write(f"Points behind this table's leader: {prof['points_behind_leader']} · points above bottom: {prof['points_above_bottom']}")
            st.write(f"Last 5: {snap['points']}/15 points · {snap['gf']:.2f} scored and {snap['ga']:.2f} conceded per match")
            st.write(f"Season goals: {prof['gf']:.1f} for, {prof['ga']:.1f} against")
            if prof["shots"] is not None:
                st.write(f"Shots per match, last 10: {prof['shots']:.1f}")
            else:
                st.write("Shots: unavailable in source")
            if prof["sot"] is not None:
                st.write(f"Shots on target per match, last 10: {prof['sot']:.1f}")
                if prof["shots"] and prof["shots"] > 0:
                    st.write(f"Shots-on-target rate: {prof['sot']/prof['shots']:.1%}")
            else:
                st.write("Shots on target: unavailable in source")
            league_shots = profile_data[["HS","AS"]].stack().mean() if {"HS","AS"}.issubset(profile_data.columns) else np.nan
            if prof["shots"] is not None and pd.notna(league_shots):
                descriptor = "above" if prof["shots"] > league_shots * 1.08 else ("below" if prof["shots"] < league_shots * 0.92 else "around")
                st.caption(f"Shot volume is {descriptor} the league's available-match average ({league_shots:.1f}).")
            st.caption("Profile describes observed output; it does not claim to identify a coach's exact tactical system.")

with st.expander("Model settings and match context"):
    elo_weight = st.slider("Elo contribution to final probabilities", 0.0, 0.50, 0.25, 0.05,
                           help="The remainder comes from the Dixon–Coles goal model. This is a transparent blend, not yet empirically optimised.")
    kickoff = st.date_input("Expected match date", value=(latest_date + pd.Timedelta(days=1)).date(), min_value=latest_date.date())

if st.button("Calculate match prediction", type="primary", use_container_width=True):
    if home == away:
        st.error("Choose two different teams.")
        st.stop()
    try:
        signature = f"{league_code}-{len(data)}-{latest_date.date()}"
        model = fit_model(league_code, signature, data)
        dc_probs, (base_hg, base_ag) = model.probabilities(home, away)
        elo_rows, elo_ratings = get_elo(data)
        elo_diff = elo_ratings.get(home, 1500.0) + 65.0 - elo_ratings.get(away, 1500.0)
        ep = elo_probabilities(elo_diff)
        combined = blended_probabilities(dc_probs, ep, elo_weight)

        h_att, h_def, h_known = player_impact(home_xi)
        a_att, a_def, a_known = player_impact(away_xi)
        h_abs_att, h_abs_def = absence_impact(home_abs)
        a_abs_att, a_abs_def = absence_impact(away_abs)
        h_rest = rest_days(data, home, kickoff)
        a_rest = rest_days(data, away, kickoff)
        rest_adjustment = 0.0
        if h_rest is not None and a_rest is not None:
            rest_adjustment = float(np.clip((h_rest-a_rest)*0.012, -0.10, 0.10))

        adj_hg = float(np.clip(base_hg + h_att - h_abs_att - a_def + a_abs_def + rest_adjustment, 0.05, 5.5))
        adj_ag = float(np.clip(base_ag + a_att - a_abs_att - h_def + h_abs_def - rest_adjustment, 0.05, 5.5))
        # Recalculate goal-model probabilities after explicit player and rest adjustments.
        goals = np.arange(11)
        matrix = np.outer(poisson.pmf(goals, adj_hg), poisson.pmf(goals, adj_ag))
        for x in range(2):
            for y in range(2):
                matrix[x,y] *= dixon_coles_tau(x,y,adj_hg,adj_ag,model.rho)
        matrix = np.maximum(matrix, 0); matrix /= matrix.sum()
        adjusted_dc = np.array([np.tril(matrix,-1).sum(), np.trace(matrix), np.triu(matrix,1).sum()])
        final_probs = blended_probabilities(adjusted_dc, ep, elo_weight)
        labels = [f"{home} win", "Draw", f"{away} win"]
        best = int(np.argmax(final_probs))

        st.divider()
        st.subheader(f"Most likely outcome: {labels[best]}")
        st.markdown(f"### {final_probs[best]:.1%}")
        c1,c2,c3 = st.columns(3)
        c1.metric(labels[0], f"{final_probs[0]:.1%}")
        c2.metric("Draw", f"{final_probs[1]:.1%}")
        c3.metric(labels[2], f"{final_probs[2]:.1%}")
        st.write(f"Adjusted expected goals: **{home} {adj_hg:.2f} – {adj_ag:.2f} {away}**")

        st.markdown("**Model components**")
        m1,m2,m3 = st.columns(3)
        m1.metric("Dixon–Coles", f"{adjusted_dc[0]:.1%} / {adjusted_dc[1]:.1%} / {adjusted_dc[2]:.1%}")
        m2.metric("Elo baseline", f"{ep[0]:.1%} / {ep[1]:.1%} / {ep[2]:.1%}")
        m3.metric("Elo difference", f"{elo_diff:+.0f}")

        st.markdown("**Player and match impact**")
        st.write(f"{home}: {h_known} rated starters entered; {len(home_abs)} unavailable players entered. {away}: {a_known} rated starters entered; {len(away_abs)} unavailable players entered.")
        st.write(f"Attack xG adjustment: {home} {h_att-h_abs_att:+.2f}; {away} {a_att-a_abs_att:+.2f}. Defensive quality: {home} {h_def:+.2f}, {away} {a_def:+.2f}; unavailable defenders weaken the respective defence by {h_abs_def:.2f} and {a_abs_def:.2f} xG.")
        st.write(f"Rest days before {kickoff}: {home} {h_rest if h_rest is not None else 'unknown'}, {away} {a_rest if a_rest is not None else 'unknown'}. Applied home xG adjustment: {rest_adjustment:+.2f}.")
        for team, xi, status, formation in [(home,home_xi,home_status,home_formation),(away,away_xi,away_status,away_formation)]:
            st.caption(f"{team}: {status} lineup · {formation} · {len(xi)} players entered" + (f" · WARNING: expected 11, received {len(xi)}" if xi and len(xi)!=11 else ""))

        st.markdown("**Why the model leans this way**")
        st.write(f"Dixon–Coles estimates goals from historical attack/defence and home advantage; dynamic Elo contributes {elo_weight:.0%} of the final probability blend.")
        st.write(f"Pre-match Elo: {home} {elo_ratings.get(home,1500.0):.0f} vs {away} {elo_ratings.get(away,1500.0):.0f}, with a 65-point home advantage.")
        st.write(f"Recent form is shown as context, not added again as a separate multiplier. Player ratings and absences alter expected goals only when you enter ratings.")
        st.caption("Player impact uses a conservative, transparent heuristic (rating relative to 6.5, position-weighted). It is not a measured player-value model. Elo blend and rest effect require out-of-sample calibration.")
        st.caption("Probabilities are estimates, not certainty. A 60% outcome still fails roughly 4 times in 10 under a perfectly calibrated model.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Historical model evaluation — tune, calibrate, then test"):
    st.write("Uses whole match dates in chronological order: 60% training, 20% validation and 20% final test. The blend weight and calibration temperature are selected on validation data only; the final block is kept separate.")
    if st.button("Run research evaluation"):
        try:
            with st.spinner("Fitting Dixon–Coles, tuning the blend on validation data, and scoring the untouched test period..."):
                result = chronological_ensemble_evaluation(data, DixonColesModel)
            st.write(f"Training: {result['train_matches']:,} · validation: {result['validation_matches']:,} · final test: {result['test_matches']:,} · test begins: {result['cutoff']}")
            st.write(f"Validation-selected Elo weight: **{result['elo_weight']:.1%}** · calibration temperature: **{result['temperature']:.3f}**")
            metrics = pd.DataFrame(
                [result["dixon_coles"], result["elo"], result["calibrated_dixon_coles"], result["calibrated_elo"], result["elo_ensemble"], result["calibrated_ensemble"]],
                index=["Dixon–Coles raw", "Elo raw", "Dixon–Coles calibrated", "Elo calibrated", "Tuned DC + Elo", "Tuned + calibrated ensemble"]
            )
            st.dataframe(metrics, use_container_width=True)
            dc, tuned, calibrated = result["calibrated_dixon_coles"], result["elo_ensemble"], result["calibrated_ensemble"]
            if calibrated["Log loss"] < dc["Log loss"] and calibrated["Brier"] < dc["Brier"] and calibrated["RPS"] < dc["RPS"]:
                st.success("The tuned and calibrated ensemble improved all three probability scores on the final test period.")
            else:
                st.warning("The tuned and calibrated ensemble did not beat Dixon–Coles on every score. Do not assume the added complexity is an improvement.")
            st.caption("Lower Brier, log loss and RPS are better. Accuracy is secondary. This is a single chronological test, not proof of future performance.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with st.expander("Data sources and limitations"):
    st.markdown(
        "- **Football-Data.co.uk:** historical results for the five major European leagues, requested from 2010/11 onward. Match-stat coverage varies by league and season.\n"
        "- **Allsvenskan historical results:** Swedish Football Dataset, 2011–2025 (CC BY 4.0; attribution: Nathan Dygant). Current-season results come from open-matchday.\n"
        "- **open-matchday:** current-season results and selected statistics where upstream providers supply them.\n"
        "- **Dynamic Elo:** sequentially updated from historical match outcomes; new results change team ratings.\n"
        "- **Player ratings and absences:** manually entered. No automatic injury, suspension, confirmed-lineup or player-rating feed is connected.\n"
        "- **Tactical profile:** derived from league position, goals, shots and shots on target when available. Possession, PPDA, xG and detailed tactical events are not consistently available across all six leagues.\n"
        "- **Backtest:** chronological 80/20 holdout compares Dixon–Coles with the Elo ensemble using Brier score, log loss and accuracy. The blend weight is currently configurable, not learned from a separate calibration set."
    )
st.caption("Research and analysis only. No bookmaker connection or bet placement. Missing source data is never treated as zero.")
