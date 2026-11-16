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
from team_news import fetch_fixture_team_news, get_api_football_key

st.set_page_config(page_title="Match Analysis", page_icon=":soccer:", layout="wide")

st.markdown("""
<style>
:root { --ink:#17251f; --muted:#68756e; --line:#e4e9e5; --paper:#f6f8f6; --accent:#176b4a; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1180px; padding-top:2.1rem; padding-bottom:3rem; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
h1 { font-size:2.35rem !important; font-weight:720 !important; }
h2 { font-size:1.45rem !important; font-weight:680 !important; }
h3 { font-size:1.08rem !important; font-weight:650 !important; }
[data-testid="stCaptionContainer"] p { color:var(--muted); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:15px 17px; }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.82rem; }
[data-testid="stMetricValue"] { color:var(--ink); font-weight:700; }
div[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
div[data-testid="stSelectbox"] label, div[data-testid="stDateInput"] label, div[data-testid="stTextArea"] label { color:#34443b; font-weight:600; }
.stButton > button[kind="primary"] { background:#176b4a; border:1px solid #176b4a; border-radius:9px; font-weight:650; min-height:2.8rem; }
.stButton > button[kind="primary"]:hover { background:#10563b; border-color:#10563b; }
.stButton > button { border-radius:9px; }
hr { border-color:var(--line); }
[data-testid="stAlert"] { border-radius:10px; }
.match-hero { padding:1.1rem 1.35rem; background:#fff; border:1px solid var(--line); border-radius:14px; margin:.6rem 0 1.1rem; }
.match-kicker { color:var(--accent); font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:750; }
.match-sub { color:var(--muted); font-size:.9rem; margin-top:.25rem; }
.small-note { color:var(--muted); font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="match-hero"><div class="match-kicker">Match intelligence · Research edition</div><h1 style="margin:.25rem 0 0">Match analysis</h1><div class="match-sub">Probability estimates, team context and model diagnostics across six leagues.</div></div>', unsafe_allow_html=True)

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

@st.cache_data(ttl=3600, show_spinner=False)
def cached_team_news(api_key, code, home_team, away_team, match_date):
    return fetch_fixture_team_news(api_key, code, home_team, away_team, match_date)

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
data_age_days = max(0, (pd.Timestamp.now().normalize() - latest_date.normalize()).days)
if data_age_days > 21:
    st.warning(f"Data warning: the latest result is {data_age_days} days old. Treat this as a historical estimate, not a current forecast, until the league data feed is updated.")


left, right = st.columns(2)
with left:
    home = st.selectbox("Home team", teams, index=0)
with right:
    away = st.selectbox("Away team", [x for x in teams if x != home], index=0)

with st.expander("Match date", expanded=False):
    default_kickoff = max((latest_date + pd.Timedelta(days=1)).date(),
                          (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).date())
    kickoff = st.date_input("Expected match date", value=default_kickoff, min_value=latest_date.date())

lookup_selection = f"{league_code}|{home}|{away}|{kickoff.isoformat()}"
if st.session_state.get("api_news_selection") != lookup_selection:
    if st.session_state.get("api_news_selection") is not None:
        for k in ("hxi","axi","habs","aabs"):
            st.session_state[k] = ""
        for k,v in (("hs","Not entered"),("as","Not entered"),("hf","Other / unknown"),("af","Other / unknown")):
            st.session_state[k] = v
        st.session_state.pop("api_news_result", None)
    st.session_state["api_news_selection"] = lookup_selection

api_key = get_api_football_key(st)
with st.expander("Automatic team news (free API-Football)", expanded=False):
    st.caption("Optional lookup for confirmed starting XIs, season-average player ratings where available, and reported injuries/suspensions.")
    if not api_key:
        st.info("Create a free API-Football key, then add API_FOOTBALL_KEY to your Codespaces environment or .streamlit/secrets.toml. Never commit the key to GitHub.")
        st.markdown("[Create a free API-Football account](https://dashboard.api-football.com/register)")
    if st.button("Fetch fixture lineups, ratings and absences", key="fetch_team_news", disabled=not bool(api_key)):
        try:
            with st.spinner("Looking up the selected fixture..."):
                news = cached_team_news(api_key, league_code, home, away, kickoff)
            for k in ("hxi","axi","habs","aabs"): st.session_state[k] = ""
            for k,v in (("hs","Not entered"),("as","Not entered"),("hf","Other / unknown"),("af","Other / unknown")):
                st.session_state[k]=v
            for prefix,item in (("h",news["home"]),("a",news["away"])):
                if item["xi"]:
                    st.session_state[prefix+"xi"]=item["xi"]
                    st.session_state[prefix+"s"]="Confirmed" if item["count"] == 11 else "Partial / incomplete"
                    st.session_state[prefix+"f"]=item["formation"] if item["formation"] in ["4-3-3","4-2-3-1","4-4-2","3-5-2","3-4-3","4-1-4-1"] else "Other / unknown"
                st.session_state[prefix+"abs"]=item["abs"]
            st.session_state["api_news_result"]=news
            st.session_state["api_news_selection"]=lookup_selection
            st.rerun()
        except Exception as exc:
            st.error(f"Team-news lookup failed: {exc}")
    if st.session_state.get("api_news_result"):
        news=st.session_state["api_news_result"]
        st.write(f"Fixture: **{news['home_name']} vs {news['away_name']}** · {news['kickoff']} · {news['status']}")
        st.write(f"Starting XI players returned: {news['home']['count']}/11 home, {news['away']['count']}/11 away.")
        st.write(f"Reported absences: {len(news['home']['abs'].splitlines())} home, {len(news['away']['abs'].splitlines())} away.")
        if news["questionable"]: st.warning("Questionable, not treated as confirmed absences: "+"; ".join(news["questionable"]))
        if not news["home"]["xi"] and not news["away"]["xi"]:
            st.info("Confirmed lineups are not published yet or are unavailable for this fixture. Enter lineups manually.")
        st.caption("Ratings are current-season averages where available, not predicted match ratings.")

with st.expander("Optional: lineups and unavailable players", expanded=False):
    st.caption("Manual inputs remain available. Enter one player per line: Name | rating (1–10) | position. Automatic data is only filled when the provider returns it.")
    l1, l2 = st.columns(2)
    with l1:
        st.markdown(f"**{home}**")
        home_status = st.selectbox("Lineup status", ["Not entered", "Expected", "Partial / incomplete", "Confirmed"], key="hs")
        home_formation = st.selectbox("Formation", ["4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "Other / unknown"], key="hf")
        home_xi_raw = st.text_area("Starting XI — player | rating | position", height=120, key="hxi",
                                   placeholder="Goalkeeper | 7.0 | GK\nCentre-back | 7.2 | CB\nStriker | 8.1 | ST")
        home_abs_raw = st.text_area("Unavailable players — player | rating | position", height=80, key="habs",
                                    placeholder="Starting striker | 7.8 | ST")
    with l2:
        st.markdown(f"**{away}**")
        away_status = st.selectbox("Lineup status", ["Not entered", "Expected", "Partial / incomplete", "Confirmed"], key="as")
        away_formation = st.selectbox("Formation", ["4-3-3", "4-2-3-1", "4-4-2", "3-5-2", "3-4-3", "4-1-4-1", "Other / unknown"], key="af")
        away_xi_raw = st.text_area("Starting XI — player | rating | position", height=120, key="axi",
                                   placeholder="Goalkeeper | 7.0 | GK\nCentre-back | 7.2 | CB\nStriker | 8.1 | ST")
        away_abs_raw = st.text_area("Unavailable players — player | rating | position", height=80, key="aabs",
                                    placeholder="Starting centre-back | 7.5 | CB")

home_xi, away_xi = parse_player_lines(home_xi_raw if "home_xi_raw" in locals() else ""), parse_player_lines(away_xi_raw if "away_xi_raw" in locals() else "")
home_abs, away_abs = parse_player_lines(home_abs_raw if "home_abs_raw" in locals() else ""), parse_player_lines(away_abs_raw if "away_abs_raw" in locals() else "")

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
with st.expander("Optional: recent form and team profiles", expanded=False):
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
        # Elo is the current default because it outperformed DC and the blend in the available Allsvenskan holdout. DC remains an independent cross-check pending validation in all leagues.
        final_probs = ep
        labels = [f"{home} win", "Draw", f"{away} win"]
        best = int(np.argmax(final_probs))

        st.divider()
        st.markdown(f'<div class="match-hero"><div class="match-kicker">Model selection · Dynamic Elo</div><h2 style="margin:.3rem 0">{labels[best]}</h2><div style="font-size:2rem;font-weight:750;color:#176b4a">{final_probs[best]:.1%}</div><div class="match-sub">Highest estimated outcome probability · not a certainty</div></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric(labels[0], f"{final_probs[0]:.1%}")
        c2.metric("Draw", f"{final_probs[1]:.1%}")
        c3.metric(labels[2], f"{final_probs[2]:.1%}")
        st.write(f"Adjusted expected goals: **{home} {adj_hg:.2f} – {adj_ag:.2f} {away}**")

        st.markdown("**Independent model cross-check**")
        m1,m2,m3 = st.columns(3)
        m1.metric("Dixon–Coles home / draw / away", f"{adjusted_dc[0]:.1%} / {adjusted_dc[1]:.1%} / {adjusted_dc[2]:.1%}")
        m2.metric("Elo home / draw / away", f"{ep[0]:.1%} / {ep[1]:.1%} / {ep[2]:.1%}")
        m3.metric("Elo strength difference", f"{elo_diff:+.0f}")

        st.markdown("**Player and match impact**")
        st.write(f"{home}: {h_known} rated starters entered; {len(home_abs)} unavailable players entered. {away}: {a_known} rated starters entered; {len(away_abs)} unavailable players entered.")
        st.write(f"Attack xG adjustment: {home} {h_att-h_abs_att:+.2f}; {away} {a_att-a_abs_att:+.2f}. Defensive quality: {home} {h_def:+.2f}, {away} {a_def:+.2f}; unavailable defenders weaken the respective defence by {h_abs_def:.2f} and {a_abs_def:.2f} xG.")
        st.write(f"Rest days before {kickoff}: {home} {h_rest if h_rest is not None else 'unknown'}, {away} {a_rest if a_rest is not None else 'unknown'}. Applied home xG adjustment: {rest_adjustment:+.2f}.")
        for team, xi, status, formation in [(home,home_xi,home_status,home_formation),(away,away_xi,away_status,away_formation)]:
            st.caption(f"{team}: {status} lineup · {formation} · {len(xi)} players entered" + (f" · WARNING: expected 11, received {len(xi)}" if xi and len(xi)!=11 else ""))

        st.markdown("**Why the model leans this way**")
        st.write("The displayed 1X2 probabilities come from dynamic Elo. Dixon–Coles independently estimates expected goals and provides a second probability estimate; it is not blended into the displayed result because it underperformed Elo in the available Allsvenskan holdout.")
        st.write(f"Pre-match Elo: {home} {elo_ratings.get(home,1500.0):.0f} vs {away} {elo_ratings.get(away,1500.0):.0f}, with a 65-point home advantage.")
        st.write(f"Recent form is shown as context, not added again as a separate multiplier. Player ratings and absences alter expected goals only when you enter ratings.")
        st.caption("Player ratings, absences and rest-day adjustments are experimental inputs to expected goals only. Their effect on 1X2 probabilities has not been validated.")
        st.caption("Probabilities are estimates, not certainty. A 60% outcome still fails roughly 4 times in 10 under a perfectly calibrated model.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

with st.expander("Advanced: historical model evaluation", expanded=False):
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
            elo_cal, ensemble_cal = result["calibrated_elo"], result["calibrated_ensemble"]
            if ensemble_cal["Log loss"] < elo_cal["Log loss"] and ensemble_cal["Brier"] < elo_cal["Brier"] and ensemble_cal["RPS"] < elo_cal["RPS"]:
                st.success("The calibrated ensemble beat calibrated Elo on all three probability scores in this test.")
            else:
                st.warning("The ensemble did not beat calibrated Elo on all three scores. Elo remains the preferred default for this league until more evidence says otherwise.")
            st.caption("Lower Brier, log loss and RPS are better. Accuracy is secondary. This is a single chronological test, not proof of future performance.")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

with st.expander("Data sources and limitations"):
    st.markdown(
        "- **Football-Data.co.uk:** historical results for the five major European leagues, requested from 2010/11 onward. Match-stat coverage varies by league and season.\n"
        "- **Allsvenskan results:** Swedish Football Dataset, 2011–2025 (CC BY 4.0; attribution: Nathan Dygant), plus current-season completed scores from the public Fantasy Allsvenskan API mirror (TopMarx/allsvenskan; MIT repository license; underlying league data remains third-party data).\n"
        "- **open-matchday:** current-season results and selected statistics where upstream providers supply them.\n"
        "- **Dynamic Elo:** sequentially updated from historical match outcomes; new results change team ratings.\n"
        "- **Player ratings and absences:** manually entered. No automatic injury, suspension, confirmed-lineup or player-rating feed is connected.\n"
        "- **Tactical profile:** derived from league position, goals, shots and shots on target when available. Possession, PPDA, xG and detailed tactical events are not consistently available across all six leagues.\n"
        "- **Backtest:** chronological 80/20 holdout compares Dixon–Coles with the Elo ensemble using Brier score, log loss and accuracy. The blend weight is currently configurable, not learned from a separate calibration set."
    )
st.caption("Research and analysis only. No bookmaker connection or bet placement. Missing source data is never treated as zero.")
