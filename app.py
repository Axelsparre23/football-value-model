"""Quick, explainable match calculator with manual team-news adjustments."""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from data_manager import download_data, LEAGUES
from football_value_model_v3 import DixonColesModel, dixon_coles_tau

st.set_page_config(page_title="Football Match Calculator", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Calculator")
st.caption("Premier League · La Liga · Bundesliga · Serie A · Ligue 1 · Allsvenskan")

@st.cache_data(ttl=21600, show_spinner=False)
def load_league(code):
    return download_data(Path("data/cache"), first_start_year=2018, leagues=(code,), refresh_current=True)

@st.cache_resource(show_spinner=False)
def fit_model(code, signature, data):
    half_life = 240 if code == "SWE" else 300
    return DixonColesModel(half_life=half_life, reg=0.04, xg_weight=0.0).fit(data, pd.Timestamp.now().normalize())

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
recent_teams = data[data.Date >= latest_date - pd.Timedelta(days=450)]
teams = sorted(set(recent_teams.HomeTeam) | set(recent_teams.AwayTeam))
if len(teams) < 2:
    teams = sorted(set(data.HomeTeam) | set(data.AwayTeam))

st.caption(f"{len(data):,} completed matches loaded · latest result: {latest_date.date()}")
a, b = st.columns(2)
with a:
    home = st.selectbox("Home team", teams, index=0)
with b:
    away = st.selectbox("Away team", [x for x in teams if x != home], index=0)

# Recent form and match-stat summaries are computed only from matches before today.
def team_snapshot(df, team, n=5):
    mask = (df.HomeTeam == team) | (df.AwayTeam == team)
    rows = df.loc[mask].sort_values("Date").tail(n)
    gf, ga, pts = [], [], []
    for _, r in rows.iterrows():
        is_home = r.HomeTeam == team
        scored = r.FTHG if is_home else r.FTAG
        conceded = r.FTAG if is_home else r.FTHG
        gf.append(float(scored)); ga.append(float(conceded))
        pts.append(3 if scored > conceded else (1 if scored == conceded else 0))
    return {"matches": len(rows), "gf": np.mean(gf) if gf else np.nan,
            "ga": np.mean(ga) if ga else np.nan, "points": sum(pts),
            "rows": rows}

def available_average(rows, col):
    return float(rows[col].mean()) if col in rows and rows[col].notna().any() else None

with st.expander("Recent form and team statistics", expanded=True):
    h5 = team_snapshot(data, home, 5)
    a5 = team_snapshot(data, away, 5)
    c1, c2 = st.columns(2)
    for col, team, snap in [(c1, home, h5), (c2, away, a5)]:
        with col:
            st.markdown(f"**{team} — last {snap['matches']} matches**")
            st.write(f"Points: {snap['points']}/15 · Goals scored: {snap['gf']:.2f}/match · Conceded: {snap['ga']:.2f}/match")
            for label, stat in [("Shots", "HS" if team == home else "AS"),
                                ("Shots on target", "HST" if team == home else "AST"),
                                ("Corners", "HC" if team == home else "AC"),
                                ("Fouls", "HF" if team == home else "AF")]:
                val = available_average(snap["rows"], stat)
                st.write(f"{label}: {val:.1f}/match" if val is not None else f"{label}: unavailable in source")

with st.expander("Team news and tactical adjustments (optional)"):
    st.warning("These are your own estimates, not automatically verified injury or tactical feeds. Use small adjustments unless you have reliable information.")
    st.markdown("**Expected-goal adjustment for absences**")
    ih, ia = st.columns(2)
    with ih:
        st.caption(f"{home} attacking impact")
        home_attack_absence = st.slider("Home attack reduction (xG)", 0.0, 0.8, 0.0, 0.05, key="haa")
        st.caption(f"{home} defensive impact")
        home_defence_absence = st.slider("Home defence weakening (opponent xG)", 0.0, 0.8, 0.0, 0.05, key="hda")
    with ia:
        st.caption(f"{away} attacking impact")
        away_attack_absence = st.slider("Away attack reduction (xG)", 0.0, 0.8, 0.0, 0.05, key="aaa")
        st.caption(f"{away} defensive impact")
        away_defence_absence = st.slider("Away defence weakening (opponent xG)", 0.0, 0.8, 0.0, 0.05, key="ada")
    st.markdown("**Tactical matchup**")
    th, ta = st.columns(2)
    with th:
        home_tactic = st.selectbox(f"{home} approach", ["No adjustment", "High press", "Low block", "Possession-focused", "Counter-attacking"], key="htac")
        home_tactic_effect = st.slider(f"{home} tactical effect on own xG", -0.4, 0.4, 0.0, 0.05, key="hte")
    with ta:
        away_tactic = st.selectbox(f"{away} approach", ["No adjustment", "High press", "Low block", "Possession-focused", "Counter-attacking"], key="atac")
        away_tactic_effect = st.slider(f"{away} tactical effect on own xG", -0.4, 0.4, 0.0, 0.05, key="ate")

if st.button("Calculate match prediction", type="primary", use_container_width=True):
    if home == away:
        st.error("Choose two different teams.")
        st.stop()
    try:
        signature = f"{league_code}-{len(data)}-{latest_date.date()}"
        model = fit_model(league_code, signature, data)
        base_probs, (base_hg, base_ag) = model.probabilities(home, away)

        # User inputs change the expected-goal rates transparently; no hidden injury/tactics weights.
        adj_hg = float(np.clip(base_hg - home_attack_absence - away_defence_absence + home_tactic_effect, 0.05, 5.5))
        adj_ag = float(np.clip(base_ag - away_attack_absence - home_defence_absence + away_tactic_effect, 0.05, 5.5))
        mat = np.outer(poisson.pmf(np.arange(11), adj_hg), poisson.pmf(np.arange(11), adj_ag))
        for x in range(2):
            for y in range(2):
                mat[x, y] *= dixon_coles_tau(x, y, adj_hg, adj_ag, model.rho)
        mat = np.maximum(mat, 0); mat /= mat.sum()
        probs = np.array([np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()])
        names = [home, "Draw", away]
        best = int(np.argmax(probs))
        exact = np.unravel_index(np.argmax(mat), mat.shape)

        st.divider()
        st.subheader(f"Most likely outcome: {names[best]}")
        st.markdown(f"### {probs[best]:.1%}")
        p1, p2, p3 = st.columns(3)
        p1.metric(f"{home} win", f"{probs[0]:.1%}")
        p2.metric("Draw", f"{probs[1]:.1%}")
        p3.metric(f"{away} win", f"{probs[2]:.1%}")
        st.metric("Most likely score", f"{exact[0]} – {exact[1]}")
        st.write(f"Expected goals after adjustments: **{home} {adj_hg:.2f} – {adj_ag:.2f} {away}**")

        st.markdown("**Why the model leans this way**")
        reasons = [
            f"Base expected goals from team attack/defence and home advantage: {home} {base_hg:.2f}, {away} {base_ag:.2f}.",
            f"Recent form (last five): {home} {h5['points']}/15 points, {away} {a5['points']}/15 points.",
        ]
        if any([home_attack_absence, home_defence_absence, away_attack_absence, away_defence_absence]):
            reasons.append("Absence adjustments entered: home attack -" + f"{home_attack_absence:.2f} xG, home defence +" +
                           f"{home_defence_absence:.2f} opponent xG; away attack -" + f"{away_attack_absence:.2f} xG, away defence +" +
                           f"{away_defence_absence:.2f} opponent xG.")
        if home_tactic != "No adjustment" or away_tactic != "No adjustment":
            reasons.append(f"Tactical assumptions: {home} = {home_tactic} ({home_tactic_effect:+.2f} own xG); {away} = {away_tactic} ({away_tactic_effect:+.2f} own xG).")
        for reason in reasons:
            st.write("• " + reason)
        st.caption("Probabilities are estimates, not certainty. A 60% outcome still fails roughly 4 times in 10 under a perfectly calibrated model.")
    except KeyError as exc:
        st.error(f"Not enough historical league data for one selected team: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Data sources, coverage and limitations"):
    st.markdown(
        "- **Football-Data.co.uk:** historical results and odds for the five major European leagues.\n"
        "- **open-matchday (GitHub):** current-season results and fixtures for all six competitions; selected match statistics where the upstream provider supplies them.\n"
        "- **Allsvenskan:** currently has results/fixtures coverage in open-matchday, but not the same depth of match statistics as the top five.\n"
        "- **Injuries and tactics:** entered manually in this version. The app does not pretend to know unverified team news.\n"
        "- **xG and player-level data:** not yet integrated as a consistent free feed across all six leagues."
    )
st.caption("Data is refreshed/cached automatically. Missing statistics remain unavailable, never treated as zero.")
