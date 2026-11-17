"""Fast football match calculator: Dixon-Coles + dynamic Elo + transparent team news."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

from data_manager import download_data, LEAGUES
from football_value_model_v3 import DixonColesModel, dixon_coles_tau
from advanced_model import elo_history, elo_probabilities, team_profile, rest_days

from model_research import chronological_ensemble_evaluation
from match_intelligence import (
    APIFootballClient, unavailable_result,
    extract_1x2_odds, compare_odds_snapshots,
)

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


def get_api_football_key():
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("API_FOOTBALL_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def load_live_match_information(code, season, match_date, home_team, away_team, api_key):
    client = APIFootballClient(api_key=api_key)
    if not client.configured:
        return unavailable_result("API_FOOTBALL_KEY is not configured.")
    fixture = client.find_fixture(code, season, match_date, home_team, away_team)
    if not fixture:
        return {
            "overall_status": "no_fixture",
            "message": "No matching fixture was returned for this date and competition.",
            "lineups": [], "absences": [], "odds": [], "team_statistics": [],
            "sources": {},
        }
    fixture_id = (fixture.get("fixture") or {}).get("id")
    if not fixture_id:
        return {
            "overall_status": "no_fixture",
            "message": "The provider response did not include a fixture ID.",
            "lineups": [], "absences": [], "odds": [], "team_statistics": [],
            "sources": {},
        }
    return client.fixture_data(int(fixture_id))


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

with st.expander("Optional: bookmaker probability benchmark", expanded=False):
    st.caption("Enter decimal 1X2 odds to compare the model with a no-margin market estimate. This is a comparison only; market odds are not blended into the prediction.")
    compare_market = st.checkbox("Enter bookmaker odds", value=False, key="compare_market")
    if compare_market:
        oh, od, oa = st.columns(3)
        with oh:
            odds_home = st.number_input(f"{home} odds", min_value=1.01, max_value=100.0, value=2.00, step=0.05, key="odds_home")
        with od:
            odds_draw = st.number_input("Draw odds", min_value=1.01, max_value=100.0, value=3.50, step=0.05, key="odds_draw")
        with oa:
            odds_away = st.number_input(f"{away} odds", min_value=1.01, max_value=100.0, value=3.50, step=0.05, key="odds_away")
        raw_market = 1.0 / np.array([odds_home, odds_draw, odds_away], dtype=float)
        market_probs = raw_market / raw_market.sum()
    else:
        market_probs = None

def team_snapshot(df, team, n=5):
    rows = df[((df.HomeTeam == team) | (df.AwayTeam == team))].sort_values("Date").tail(n)
    gf, ga, points, form = [], [], 0, []
    for _, r in rows.iterrows():
        is_home = r.HomeTeam == team
        scored = float(r.FTHG if is_home else r.FTAG)
        conceded = float(r.FTAG if is_home else r.FTHG)
        gf.append(scored); ga.append(conceded)
        result = "W" if scored > conceded else ("D" if scored == conceded else "L")
        form.append(result)
        points += 3 if result == "W" else (1 if result == "D" else 0)
    return {"matches":len(rows), "gf":float(np.mean(gf)) if gf else 0.0,
            "ga":float(np.mean(ga)) if ga else 0.0, "points":points,
            "wins":form.count("W"), "draws":form.count("D"), "losses":form.count("L"),
            "form":"–".join(form) if form else "unavailable",
            "points_per_match":points/len(rows) if len(rows) else 0.0, "rows":rows}

# Build the table and recent-form context only from matches played before the selected fixture date.
pre_match_data = data[data.Date < pd.Timestamp(kickoff)].copy()
season_start = pd.Timestamp(
    year=kickoff.year if league_code == "SWE" or kickoff.month >= 7 else kickoff.year - 1,
    month=1 if league_code == "SWE" else 7, day=1
)
profile_data = pre_match_data[pre_match_data.Date >= season_start].copy()
if profile_data.empty:
    profile_data = pre_match_data.copy()
with st.expander("Optional: recent form and team profiles", expanded=False):
    ph, pa = st.columns(2)
    for col, team in [(ph, home), (pa, away)]:
        snap = team_snapshot(pre_match_data, team, 5)
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
                if prof["shot_accuracy"] is not None:
                    st.write(f"Shots-on-target rate: {prof['shot_accuracy']:.1%}")
            else:
                st.write("Shots on target: unavailable in source")
            if prof["shots_against"] is not None:
                st.write(f"Shots conceded per match, last 10: {prof['shots_against']:.1f}")
            if prof["sot_against"] is not None:
                st.write(f"Shots on target conceded per match: {prof['sot_against']:.1f}")
            league_shots = profile_data[["HS","AS"]].stack().mean() if {"HS","AS"}.issubset(profile_data.columns) else np.nan
            if prof["shots"] is not None and pd.notna(league_shots):
                descriptor = "above" if prof["shots"] > league_shots * 1.08 else ("below" if prof["shots"] < league_shots * 0.92 else "around")
                st.caption(f"Shot volume is {descriptor} the league's available-match average ({league_shots:.1f}).")
            st.caption("Profile describes observed output; it does not claim to identify a coach's exact tactical system.")

if st.button("Calculate match prediction", type="primary", use_container_width=True):
    if home == away:
        st.error("Choose two different teams.")
        st.stop()

    progress_area = st.container()
    progress = progress_area.progress(1, text="Starting prediction...")
    status = progress_area.empty()
    try:
        status.caption("1/5 · Preparing match data")
        progress.progress(10, text="Preparing match data...")
        signature = f"{league_code}-{len(data)}-{latest_date.date()}"

        status.caption("2/5 · Fitting goal model")
        progress.progress(25, text="Fitting goal model...")
        model = fit_model(league_code, signature, data)
        dc_probs, (base_hg, base_ag) = model.probabilities(home, away)

        status.caption("3/5 · Calculating team strength")
        progress.progress(48, text="Calculating Elo ratings...")
        elo_rows, elo_ratings = get_elo(data)
        home_elo = elo_ratings.get(home, 1500.0)
        away_elo = elo_ratings.get(away, 1500.0)
        elo_diff = home_elo + 65.0 - away_elo
        final_probs = elo_probabilities(elo_diff)

        status.caption("4/5 · Checking match context")
        progress.progress(70, text="Checking rest days and match context...")
        h_rest = rest_days(data, home, kickoff)
        a_rest = rest_days(data, away, kickoff)
        rest_adjustment = 0.0
        if h_rest is not None and a_rest is not None:
            rest_adjustment = float(np.clip((h_rest-a_rest)*0.012, -0.10, 0.10))
        adj_hg = float(np.clip(base_hg + rest_adjustment, 0.05, 5.5))
        adj_ag = float(np.clip(base_ag - rest_adjustment, 0.05, 5.5))
        goals = np.arange(11)
        matrix = np.outer(poisson.pmf(goals, adj_hg), poisson.pmf(goals, adj_ag))
        for x in range(2):
            for y in range(2):
                matrix[x,y] *= dixon_coles_tau(x,y,adj_hg,adj_ag,model.rho)
        matrix = np.maximum(matrix, 0)
        matrix /= matrix.sum()
        adjusted_dc = np.array([np.tril(matrix,-1).sum(), np.trace(matrix), np.triu(matrix,1).sum()])

        status.caption("5/5 · Preparing explanation and results")
        progress.progress(90, text="Preparing prediction...")
        labels = [f"{home} win", "Draw", f"{away} win"]
        best = int(np.argmax(final_probs))
        elo_expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        draw_probability = float(np.clip(0.265 * np.exp(-abs(elo_diff) / 650.0), 0.12, 0.30))
        home_conditional = elo_expected_home
        away_conditional = 1.0 - elo_expected_home
        decisive_probability = 1.0 - draw_probability

        progress.progress(100, text="Prediction complete")
        status.empty()

        st.divider()
        st.markdown(f'<div class="match-hero"><div class="match-kicker">Model selection · Dynamic Elo</div><h2 style="margin:.3rem 0">{labels[best]}</h2><div style="font-size:2rem;font-weight:750;color:#176b4a">{final_probs[best]:.1%}</div><div class="match-sub">Highest estimated outcome probability · not a certainty</div></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric(labels[0], f"{final_probs[0]:.1%}")
        c2.metric("Draw", f"{final_probs[1]:.1%}")
        c3.metric(labels[2], f"{final_probs[2]:.1%}")
        st.write(f"Rest-adjusted expected goals: **{home} {adj_hg:.2f} – {adj_ag:.2f} {away}**")

        if market_probs is not None:
            st.markdown("**Bookmaker benchmark (no margin)**")
            bm1, bm2, bm3 = st.columns(3)
            bm1.metric(f"{home} market", f"{market_probs[0]:.1%}", f"{(final_probs[0]-market_probs[0])*100:+.1f} pp vs model")
            bm2.metric("Draw market", f"{market_probs[1]:.1%}", f"{(final_probs[1]-market_probs[1])*100:+.1f} pp vs model")
            bm3.metric(f"{away} market", f"{market_probs[2]:.1%}", f"{(final_probs[2]-market_probs[2])*100:+.1f} pp vs model")
            st.caption("Market probabilities are proportional no-vig estimates from the odds you entered. Differences are descriptive, not evidence of value or a profitable edge.")

        with st.expander("Advanced: model diagnostics and independent cross-check"):
            m1,m2,m3 = st.columns(3)
            m1.metric("Rest-adjusted Dixon–Coles H / D / A", f"{adjusted_dc[0]:.1%} / {adjusted_dc[1]:.1%} / {adjusted_dc[2]:.1%}")
            m2.metric("Dynamic Elo H / D / A", f"{final_probs[0]:.1%} / {final_probs[1]:.1%} / {final_probs[2]:.1%}")
            m3.metric("Elo strength difference", f"{elo_diff:+.0f} points")

        st.markdown("**Why the forecast looks this way: team performance**")
        # Use only matches completed before the selected fixture date.
        home_recent = team_snapshot(pre_match_data, home, 5)
        away_recent = team_snapshot(pre_match_data, away, 5)
        home_profile = team_profile(profile_data, home)
        away_profile = team_profile(profile_data, away)

        st.write(
            f"**Recent results:** {home} has {home_recent['points']}/{home_recent['matches']*3} "
            f"points from its last {home_recent['matches']} matches ({home_recent['form']}; "
            f"{home_recent['wins']} wins, {home_recent['draws']} draws, {home_recent['losses']} losses). "
            f"{away} has {away_recent['points']}/{away_recent['matches']*3} "
            f"({away_recent['form']}; {away_recent['wins']} wins, {away_recent['draws']} draws, "
            f"{away_recent['losses']} losses)."
        )
        st.write(
            f"**Recent scoring:** over those matches, {home} has scored {home_recent['gf']:.2f} "
            f"and conceded {home_recent['ga']:.2f} goals per game; {away} has scored "
            f"{away_recent['gf']:.2f} and conceded {away_recent['ga']:.2f}."
        )
        st.write(
            f"**Season performance:** {home} is {home_profile['rank']} in the available table "
            f"({home_profile['points']} points from {home_profile['played']} matches, "
            f"goal difference {home_profile['gd']:+d}); {away} is {away_profile['rank']} "
            f"({away_profile['points']} points from {away_profile['played']} matches, "
            f"goal difference {away_profile['gd']:+d})."
        )
        home_gf10 = f"{home_profile['last10_gf']:.2f}" if home_profile["last10_gf"] is not None else "unavailable"
        home_ga10 = f"{home_profile['last10_ga']:.2f}" if home_profile["last10_ga"] is not None else "unavailable"
        away_gf10 = f"{away_profile['last10_gf']:.2f}" if away_profile["last10_gf"] is not None else "unavailable"
        away_ga10 = f"{away_profile['last10_ga']:.2f}" if away_profile["last10_ga"] is not None else "unavailable"
        st.write(
            f"**Attacking and defensive output:** in their latest ten available matches, "
            f"{home} averages {home_gf10} goals scored and {home_ga10} conceded; "
            f"{away} averages {away_gf10} scored and {away_ga10} conceded."
        )
        if home_profile["shots"] is not None and away_profile["shots"] is not None:
            st.write(
                f"**Chance volume:** {home} averages {home_profile['shots']:.1f} shots "
                f"({home_profile['sot']:.1f} on target) per match in its latest ten available matches; "
                f"{away} averages {away_profile['shots']:.1f} "
                f"({away_profile['sot']:.1f} on target)."
            )
        else:
            st.caption("Shot statistics are incomplete for at least one team, so they are not used in this comparison.")

        if home_recent["points_per_match"] > away_recent["points_per_match"] + 0.2:
            form_read = f"{home} has the stronger five-match points return."
        elif away_recent["points_per_match"] > home_recent["points_per_match"] + 0.2:
            form_read = f"{away} has the stronger five-match points return."
        else:
            form_read = "The teams' five-match points returns are broadly similar."
        st.write(f"**Matchup reading:** {form_read} The scoring and table figures above show whether that recent form is supported by the broader season performance.")

        st.caption(
            f"The displayed probabilities are the model's forecast: {home} {final_probs[0]:.1%}, "
            f"draw {final_probs[1]:.1%}, {away} {final_probs[2]:.1%}. The performance figures explain "
            f"the football evidence around that forecast; they are not falsely presented as direct percentage "
            f"adjustments. Recent-form and shot-based models have not yet shown reliable out-of-sample "
            f"improvement over the current result-based baseline, so they are not given arbitrary weights."
        )
        with st.expander("Technical model calculation"):
            st.write(
                f"The underlying forecast uses the teams' pre-match ratings, built sequentially from prior "
                f"match results. The effective home/away rating difference is {elo_diff:+.0f} points "
                f"after the home-advantage adjustment. The baseline produces {final_probs[0]:.1%} / "
                f"{final_probs[1]:.1%} / {final_probs[2]:.1%} for home / draw / away."
            )
        st.write(
            f"The independent Dixon–Coles goal model gives a separate check of "
            f"{adjusted_dc[0]:.1%} / {adjusted_dc[1]:.1%} / {adjusted_dc[2]:.1%} "
            f"for home win / draw / away win. Its expected-goal estimates are adjusted only for rest-day "
            f"difference here: {home} has {h_rest if h_rest is not None else 'unknown'} rest days and "
            f"{away} has {a_rest if a_rest is not None else 'unknown'}. The applied home expected-goals "
            f"adjustment is {rest_adjustment:+.2f}."
        )
        st.caption(
            "A 60% probability does not mean a guaranteed result: even with perfect calibration, "
            "that outcome would fail about 4 times in 10. These are estimates, not certainty."
        )

        with st.expander("Match information: lineups, availability and market", expanded=False):
            api_key = get_api_football_key()
            if not api_key:
                st.info("Live match information is not connected. Add API_FOOTBALL_KEY as a secret to enable the provider. No sample data is shown.")
            else:
                season = kickoff.year if league_code == "SWE" or kickoff.month >= 7 else kickoff.year - 1
                with st.spinner("Checking provider fixture, lineups, absences and market prices..."):
                    live = load_live_match_information(
                        league_code, season, kickoff.isoformat(), home, away, api_key
                    )
                if live.get("overall_status") == "no_fixture":
                    st.warning(live.get("message", "No matching provider fixture was found."))
                elif live.get("overall_status") == "not_configured":
                    st.info("The live provider is not configured.")
                else:
                    st.caption(
                        f"Source: API-Football · retrieved {live.get('fetched_at', 'timestamp unavailable')} · "
                        "provider coverage can vary by competition and subscription."
                    )
                    if live.get("sources"):
                        source_rows = [
                            {"Data": name.replace("_", " ").title(),
                             "Status": item.get("status", "unknown"),
                             "Records": item.get("result_count", 0),
                             "Retrieved": item.get("fetched_at", "")}
                            for name, item in live["sources"].items()
                        ]
                        st.dataframe(pd.DataFrame(source_rows), hide_index=True, use_container_width=True)

                    st.markdown("**Confirmed starting lineups**")
                    lineups = live.get("lineups") or []
                    if not lineups:
                        st.info("No confirmed lineups were returned. Expected lineups are not invented or presented as confirmed.")
                    else:
                        lineup_cols = st.columns(2)
                        for idx, team_lineup in enumerate(lineups[:2]):
                            team_info = team_lineup.get("team") or {}
                            start_xi = team_lineup.get("startXI") or []
                            names = [
                                (entry.get("player") or {}).get("name", "Unknown player")
                                for entry in start_xi
                            ]
                            with lineup_cols[idx]:
                                st.write(f"**{team_info.get('name', 'Team')}** · {team_lineup.get('formation') or 'formation unavailable'}")
                                if names:
                                    st.write("\n".join(f"• {name}" for name in names))
                                else:
                                    st.caption("Starting XI unavailable.")
                    st.caption("The provider supplies confirmed lineups when released; it does not supply a reliable expected XI here.")

                    st.markdown("**Injuries and suspensions**")
                    absences = live.get("absences") or []
                    absence_rows = []
                    for item in absences:
                        player = item.get("player") or {}
                        team_info = item.get("team") or {}
                        if player.get("name"):
                            absence_rows.append({
                                "Team": team_info.get("name", "Unknown"),
                                "Player": player.get("name"),
                                "Status": player.get("type") or "Unavailable",
                                "Reason": player.get("reason") or "Not specified",
                            })
                    if absence_rows:
                        st.dataframe(pd.DataFrame(absence_rows), hide_index=True, use_container_width=True)
                    else:
                        st.caption("No absence records returned. This does not establish that both squads are injury-free.")

                    st.markdown("**Bookmaker odds and observed movement**")
                    current_odds = extract_1x2_odds(live.get("odds") or [])
                    if current_odds:
                        st.dataframe(pd.DataFrame(current_odds), hide_index=True, use_container_width=True)
                        fixture_id = live.get("fixture_id")
                        snapshot_key = f"api_football_odds_{fixture_id}"
                        previous_odds = st.session_state.get(snapshot_key, [])
                        movement = compare_odds_snapshots(previous_odds, current_odds)
                        if movement:
                            st.write("Change since the previous check in this session (decimal odds):")
                            st.dataframe(pd.DataFrame(movement), hide_index=True, use_container_width=True)
                        else:
                            st.caption("A first odds snapshot is stored for this session. Movement will appear after a later refresh if prices change.")
                        st.session_state[snapshot_key] = current_odds
                        st.caption("Movement is measured only between snapshots observed in this active app session; it is not a complete historical odds feed.")
                    else:
                        st.caption("No usable 1X2 bookmaker prices were returned for this fixture.")

                    st.markdown("**Team-season context**")
                    team_stats = live.get("team_statistics") or []
                    if team_stats:
                        for entry in team_stats:
                            stats = entry.get("statistics") or {}
                            team_name = (stats.get("team") or {}).get("name") or entry.get("side", "Team")
                            fixtures_played = (stats.get("fixtures") or {}).get("played") or {}
                            goals_for = (stats.get("goals") or {}).get("for") or {}
                            avg_for = (goals_for.get("average") or {}).get("total")
                            st.write(
                                f"**{team_name}:** {fixtures_played.get('total', 'n/a')} matches played · "
                                f"average goals scored {avg_for if avg_for is not None else 'unavailable'}"
                            )
                    else:
                        st.caption("Team-season statistics were unavailable from the provider.")
                    st.caption("These are contextual data only. They do not change the Elo prediction; no accuracy gain is claimed.")

    except Exception as exc:
        progress_area.empty()
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
        "- **Lineups and availability:** not used as model inputs. The calculator deliberately excludes starting-XI, player-rating, injury and suspension fields.\n"
        "- **Tactical profile:** derived from league position, goals, shots and shots on target when available. Possession, PPDA, xG and detailed tactical events are not consistently available across all six leagues.\n"
        "- **Backtest:** chronological 80/20 holdout compares Dixon–Coles with the Elo ensemble using Brier score, log loss and accuracy. The blend weight is currently configurable, not learned from a separate calibration set."
    )
st.caption("Research and analysis only. No bookmaker connection or bet placement. Lineups and unavailable players are excluded. Missing source data is never treated as zero.")
