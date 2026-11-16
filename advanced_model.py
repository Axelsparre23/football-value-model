"""Dynamic Elo, transparent player-impact and match-context utilities."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

def elo_history(data: pd.DataFrame, k: float = 22.0, home_advantage: float = 65.0,
                initial: float = 1500.0, carryover: float = 0.85):
    """Return pre-match Elo ratings per fixture and final ratings, in strict date order."""
    ratings = {}
    rows = []
    ordered = data.sort_values(["Date", "League"])
    # Score every fixture on a date from the same pre-match rating snapshot.
    for _, day_block in ordered.groupby(ordered.Date.dt.normalize(), sort=True):
        updates = []
        for _, match in day_block.iterrows():
            h, a = str(match.HomeTeam), str(match.AwayTeam)
            rh, ra = ratings.get(h, initial), ratings.get(a, initial)
            rows.append({"Date": match.Date, "HomeTeam": h, "AwayTeam": a,
                         "EloHome": rh, "EloAway": ra, "EloDiff": rh + home_advantage - ra})
            hg, ag = int(match.FTHG), int(match.FTAG)
            actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            expected = 1.0 / (1.0 + 10.0 ** (-(rh + home_advantage - ra) / 400.0))
            margin = abs(hg - ag)
            multiplier = math.log1p(margin) * (2.2 / (2.2 + 0.001 * abs(rh - ra))) if margin else 1.0
            change = k * multiplier * (actual - expected)
            updates.append((h, a, change))
        for h, a, change in updates:
            ratings[h] = ratings.get(h, initial) + change
            ratings[a] = ratings.get(a, initial) - change
    return pd.DataFrame(rows), ratings

def elo_probabilities(elo_diff: float, draw_base: float = 0.265):
    """Convert Elo difference into a transparent baseline 1X2 distribution."""
    draw = float(np.clip(draw_base * math.exp(-abs(elo_diff) / 650.0), 0.12, 0.30))
    decisive = 1.0 - draw
    home_conditional = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    return np.array([decisive * home_conditional, draw, decisive * (1.0-home_conditional)])

def blended_probabilities(dc_probs, elo_probs, elo_weight: float = 0.25):
    p = (1.0-elo_weight) * np.asarray(dc_probs, dtype=float) + elo_weight * np.asarray(elo_probs, dtype=float)
    return p / p.sum()

def parse_player_lines(raw: str):
    """Parse 'Name | rating 1-10 | position' lines."""
    players = []
    for line in (raw or "").splitlines():
        bits = [x.strip() for x in line.split("|")]
        if not bits or not bits[0]:
            continue
        try:
            rating = float(bits[1]) if len(bits) > 1 and bits[1] else None
        except ValueError:
            rating = None
        position = bits[2].upper() if len(bits) > 2 else "UNK"
        if rating is not None:
            rating = float(np.clip(rating, 1.0, 10.0))
        players.append({"name": bits[0], "rating": rating, "position": position})
    return players

def player_impact(players, baseline: float = 6.5, scale: float = 0.035):
    """Estimate small xG impact from manually supplied ratings; returns attack/defence shifts."""
    attack = defence = 0.0
    weights = {"GK": 0.65, "CB": 0.75, "LB": 0.70, "RB": 0.70, "DF": 0.75,
               "DM": 0.85, "CM": 0.90, "AM": 1.0, "LM": 0.95, "RM": 0.95,
               "LW": 1.05, "RW": 1.05, "ST": 1.15, "FW": 1.10, "UNK": 0.85}
    known = [p for p in players if p["rating"] is not None]
    if not known:
        return 0.0, 0.0, 0
    for p in known:
        delta = p["rating"] - baseline
        pos = p["position"].split()[0]
        weight = weights.get(pos, weights["UNK"])
        attack += delta * scale * weight
        if pos in ("GK", "CB", "LB", "RB", "DF", "DM"):
            defence += delta * scale * weight
        elif pos in ("CM", "AM", "LM", "RM"):
            defence += delta * scale * weight * 0.35
    return float(np.clip(attack, -0.35, 0.35)), float(np.clip(defence, -0.30, 0.30)), len(known)

def absence_impact(players, scale: float = 0.035):
    """Translate listed unavailable players' ratings into attack loss and defensive weakening."""
    attack_loss = defence_loss = 0.0
    weights = {"GK": 0.65, "CB": 0.75, "LB": 0.70, "RB": 0.70, "DF": 0.75,
               "DM": 0.85, "CM": 0.90, "AM": 1.0, "LM": 0.95, "RM": 0.95,
               "LW": 1.05, "RW": 1.05, "ST": 1.15, "FW": 1.10, "UNK": 0.85}
    for p in players:
        if p["rating"] is None:
            continue
        pos = p["position"].split()[0]
        value = max(0.0, p["rating"] - 6.0) * scale * weights.get(pos, weights["UNK"])
        if pos in ("GK", "CB", "LB", "RB", "DF", "DM"):
            defence_loss += value
        else:
            attack_loss += value
    return float(min(attack_loss, 0.35)), float(min(defence_loss, 0.30))

def league_table(data: pd.DataFrame):
    teams = sorted(set(data.HomeTeam) | set(data.AwayTeam))
    rows = []
    for team in teams:
        home = data[data.HomeTeam == team]
        away = data[data.AwayTeam == team]
        played = len(home) + len(away)
        gf = float(home.FTHG.sum() + away.FTAG.sum())
        ga = float(home.FTAG.sum() + away.FTHG.sum())
        wins = int(((home.FTHG > home.FTAG).sum()) + ((away.FTAG > away.FTHG).sum()))
        draws = int(((home.FTHG == home.FTAG).sum()) + ((away.FTAG == away.FTHG).sum()))
        points = 3*wins + draws
        rows.append({"Team": team, "P": played, "W": wins, "D": draws, "L": played-wins-draws,
                     "GF": gf, "GA": ga, "GD": gf-ga, "Pts": points})
    return pd.DataFrame(rows).sort_values(["Pts","GD","GF"], ascending=False).reset_index(drop=True)

def team_profile(data: pd.DataFrame, team: str):
    matches = data[(data.HomeTeam == team) | (data.AwayTeam == team)].sort_values("Date")
    last10 = matches.tail(10)
    table = league_table(data)
    rank = int(table.index[table.Team == team][0]) + 1
    row = table.iloc[rank-1]
    def avg(col):
        return float(last10[col].mean()) if col in last10 and last10[col].notna().any() else None
    shots = []
    for _, m in last10.iterrows():
        is_home = m.HomeTeam == team
        shots.append((m.get("HS") if is_home else m.get("AS"),
                      m.get("HST") if is_home else m.get("AST")))
    shot_values = [x[0] for x in shots if pd.notna(x[0])]
    sot_values = [x[1] for x in shots if pd.notna(x[1])]
    scored = [float(m.FTHG if m.HomeTeam == team else m.FTAG) for _,m in last10.iterrows()]
    conceded = [float(m.FTAG if m.HomeTeam == team else m.FTHG) for _,m in last10.iterrows()]
    last_date = matches.Date.max() if len(matches) else None
    shots_against = []
    sot_against = []
    for _, m in last10.iterrows():
        is_home = m.HomeTeam == team
        shots_against.append(m.get("AS") if is_home else m.get("HS"))
        sot_against.append(m.get("AST") if is_home else m.get("HST"))
    shots_against_values = [x for x in shots_against if pd.notna(x)]
    sot_against_values = [x for x in sot_against if pd.notna(x)]
    shots_mean = float(np.mean(shot_values)) if shot_values else None
    sot_mean = float(np.mean(sot_values)) if sot_values else None
    shots_against_mean = float(np.mean(shots_against_values)) if shots_against_values else None
    sot_against_mean = float(np.mean(sot_against_values)) if sot_against_values else None
    return {"rank":rank, "played":int(row.P), "points":int(row.Pts), "gd":int(row.GD),
            "points_behind_leader":int(table.iloc[0].Pts-row.Pts),
            "points_above_bottom":int(row.Pts-table.iloc[-1].Pts),
            "gf":float(row.GF), "ga":float(row.GA), "last10_gf":float(np.mean(scored)) if scored else None,
            "last10_ga":float(np.mean(conceded)) if conceded else None,
            "shots":shots_mean, "sot":sot_mean,
            "shots_against":shots_against_mean, "sot_against":sot_against_mean,
            "shot_accuracy":(sot_mean / shots_mean) if shots_mean and shots_mean > 0 and sot_mean is not None else None,
            "last_date":last_date, "last10":last10}

def rest_days(data: pd.DataFrame, team: str, kickoff_date):
    prior = data[((data.HomeTeam == team) | (data.AwayTeam == team)) & (data.Date < pd.Timestamp(kickoff_date))]
    if prior.empty:
        return None
    return max(0, int((pd.Timestamp(kickoff_date).normalize() - prior.Date.max().normalize()).days))

def backtest_80_20(data: pd.DataFrame, dc_model_class, rho_tau):
    """Chronological holdout: fit on earliest 80%, evaluate later matches, no future leakage."""
    from scipy.stats import poisson
    from scipy.optimize import minimize
    df = data.sort_values("Date").reset_index(drop=True)
    cut = max(1, int(len(df)*0.80))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    if len(train) < 100 or len(test) < 20:
        raise ValueError("Need at least 120 completed matches for a useful 80/20 holdout.")
    model = dc_model_class(half_life=300, reg=0.04, xg_weight=0.0).fit(train, pd.Timestamp(test.Date.min()))
    _, final_ratings = elo_history(train)
    elo_records, elo_ratings = elo_history(train)
    # Continue Elo updates through test only after each match is scored.
    elo_current = dict(final_ratings)
    dc_rows=[]; blend_rows=[]; actuals=[]
    for _, m in test.iterrows():
        h,a=m.HomeTeam,m.AwayTeam
        diff=elo_current.get(h,1500.0)+65.0-elo_current.get(a,1500.0)
        try: dc,_=model.probabilities(h,a)
        except KeyError: continue
        ep=elo_probabilities(diff)
        bp=blended_probabilities(dc,ep)
        actual=0 if m.FTHG>m.FTAG else (1 if m.FTHG==m.FTAG else 2)
        dc_rows.append(dc); blend_rows.append(bp); actuals.append(actual)
        # Update ratings only after the outcome has been predicted.
        rh,ra=elo_current.get(h,1500.0),elo_current.get(a,1500.0)
        exp=1/(1+10**(-diff/400))
        result=1.0 if actual==0 else (0.5 if actual==1 else 0.0)
        margin=abs(int(m.FTHG)-int(m.FTAG))
        mult=math.log1p(margin)*(2.2/(2.2+0.001*abs(rh-ra))) if margin else 1.0
        change=22.0*mult*(result-exp)
        elo_current[h]=rh+change; elo_current[a]=ra-change
    if not actuals:
        raise ValueError("No holdout fixtures could be scored; teams may be missing from training data.")
    y=np.asarray(actuals,dtype=int); onehot=np.eye(3)[y]
    def metrics(rows):
        p=np.clip(np.asarray(rows),1e-8,1)
        return {"Brier":float(np.mean(np.sum((p-onehot)**2,axis=1))),
                "Log loss":float(-np.mean(np.log(p[np.arange(len(y)),y]))),
                "Accuracy":float(np.mean(p.argmax(axis=1)==y)), "Matches":len(y)}
    return {"cutoff":str(test.Date.min().date()), "train_matches":len(train), "test_matches":len(y),
            "dixon_coles":metrics(dc_rows), "elo_ensemble":metrics(blend_rows)}
