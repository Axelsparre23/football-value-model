"""Dynamic Elo ratings, player impact and transparent match-context features."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def calculate_elo(data, initial=1500.0, k_factor=24.0, home_advantage=55.0):
    """Chronological Elo. Same-day fixtures are scored before any same-day updates."""
    df=data.sort_values("Date").copy()
    ratings={}
    for team in set(df.HomeTeam).union(df.AwayTeam):
        ratings[team]=float(initial)
    for day, block in df.groupby(df.Date.dt.normalize(), sort=True):
        pending=[]
        for _, row in block.iterrows():
            h,a=row.HomeTeam,row.AwayTeam
            rh,ra=ratings.get(h,initial),ratings.get(a,initial)
            exp=1/(1+10**(-((rh+home_advantage)-ra)/400))
            hg,ag=float(row.FTHG),float(row.FTAG)
            actual=1.0 if hg>ag else (0.0 if hg<ag else .5)
            margin=max(1.0, math.log1p(abs(hg-ag))*2.2) if hg!=ag else 1.0
            change=k_factor*margin*(actual-exp)
            pending.append((h,a,change))
        # Apply updates only after all matches on the date have been forecast.
        for h,a,change in pending:
            ratings[h]=ratings.get(h,initial)+change
            ratings[a]=ratings.get(a,initial)-change
    return ratings

def elo_probabilities(home_rating, away_rating, draw_rate=0.25, home_advantage=55.0):
    diff=float(home_rating)+home_advantage-float(away_rating)
    home_share=1/(1+10**(-diff/400))
    draw=float(np.clip(draw_rate*math.exp(-abs(diff)/850),.10,.34))
    decisive=1-draw
    return np.array([decisive*home_share,draw,decisive*(1-home_share)])

def blend_probabilities(dc_probs, elo_probs, elo_weight=.25):
    p=(1-elo_weight)*np.asarray(dc_probs,dtype=float)+elo_weight*np.asarray(elo_probs,dtype=float)
    return p/p.sum()

def parse_player_entries(text):
    """Lines: Player | position | rating (1-10)."""
    players=[]
    for line in (text or "").splitlines():
        bits=[x.strip() for x in line.split("|")]
        if not bits[0]: continue
        rating=None
        if len(bits)>=3:
            try: rating=float(bits[2])
            except ValueError: rating=None
        players.append({"name":bits[0],"position":bits[1] if len(bits)>1 else "UNK",
                        "rating":float(np.clip(rating,1,10)) if rating is not None else None})
    return players

def lineup_impact(text, baseline=6.5):
    """Conservative, transparent manual rating adjustment; not a learned player model."""
    players=parse_player_entries(text)
    rated=[p for p in players if p["rating"] is not None]
    if not rated: return 0.0,0.0,len(players),0
    weights={"GK":1.15,"CB":1.05,"LB":.95,"RB":.95,"DM":1.05,"CM":1.0,"AM":1.08,
             "LW":1.08,"RW":1.08,"ST":1.15,"CF":1.1}
    weighted=np.average([p["rating"] for p in rated],
                        weights=[weights.get(p["position"].upper(),1.0) for p in rated])
    delta=float(np.clip((weighted-baseline)*.075,-.30,.30))
    return delta,weighted,len(players),len(rated)

def season_start(date, league):
    d=pd.Timestamp(date)
    return pd.Timestamp(year=d.year if league=="SWE" else (d.year if d.month>=7 else d.year-1),
                        month=1 if league=="SWE" else 7, day=1)

def team_profile(data, team, league):
    """Descriptive league table, form and available shot metrics; no inferred coach intent."""
    latest=data.Date.max()
    start=season_start(latest,league)
    season=data[data.Date>=start].copy()
    if season.empty: season=data.copy()
    rows=[]
    for t in sorted(set(season.HomeTeam)|set(season.AwayTeam)):
        played=season[(season.HomeTeam==t)|(season.AwayTeam==t)]
        gf=ga=pts=0
        for _,r in played.iterrows():
            is_h=r.HomeTeam==t
            scored=float(r.FTHG if is_h else r.FTAG); conceded=float(r.FTAG if is_h else r.FTHG)
            gf+=scored; ga+=conceded; pts+=3 if scored>conceded else (1 if scored==conceded else 0)
        rows.append({"team":t,"played":len(played),"points":pts,"gf":gf,"ga":ga,"gd":gf-ga})
    table=pd.DataFrame(rows).sort_values(["points","gd","gf"],ascending=False).reset_index(drop=True)
    match=table[table.team==team]
    if match.empty: return {"available":False}
    rank=int(match.index[0])+1; rec=match.iloc[0]
    team_rows=season[(season.HomeTeam==team)|(season.AwayTeam==team)].tail(10)
    stats={}
    for label,col_h,col_a in [("shots","HS","AS"),("shots_on_target","HST","AST")]:
        vals=[]
        for _,r in team_rows.iterrows():
            col=col_h if r.HomeTeam==team else col_a
            if col in r and pd.notna(r[col]): vals.append(float(r[col]))
        stats[label]=float(np.mean(vals)) if vals else None
    return {"available":True,"rank":rank,"teams":len(table),"played":int(rec.played),
            "points":int(rec.points),"gd":int(rec.gd),"gf_pg":float(rec.gf/max(rec.played,1)),
            "ga_pg":float(rec.ga/max(rec.played,1)),"shots":stats["shots"],
            "shots_on_target":stats["shots_on_target"],"season_start":start.date().isoformat()}

def tactical_summary(profile):
    if not profile.get("available"): return ["Insufficient league data for a profile."]
    out=[f"League position: {profile['rank']} of {profile['teams']} after {profile['played']} matches.",
         f"Season scoring: {profile['gf_pg']:.2f} goals for and {profile['ga_pg']:.2f} against per match (goal difference {profile['gd']:+d})."]
    if profile["shots"] is not None:
        out.append(f"Recent shot volume: {profile['shots']:.1f} shots per match over up to 10 matches.")
    if profile["shots_on_target"] is not None:
        out.append(f"Recent shots on target: {profile['shots_on_target']:.1f} per match.")
    if profile["rank"]<=3: out.append("Table context: currently in the top three; this describes position, not a guaranteed motivation effect.")
    elif profile["rank"]>=profile["teams"]-2: out.append("Table context: currently in the bottom three; this describes position, not a guaranteed motivation effect.")
    else: out.append("Table context: mid-table position; exact incentives depend on the competition's qualification and relegation rules.")
    return out
