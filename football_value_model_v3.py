#!/usr/bin/env python3
"""Football Value Lab v3: Dixon-Coles + historical xG-informed model + leakage-safe paper backtest.

Historical data source: https://www.football-data.co.uk/
Leagues: EPL, La Liga, Bundesliga, Serie A, Ligue 1.
Research prototype only. No real-money wagering or automated betting.
"""
from __future__ import annotations
import argparse, math, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq
from scipy.stats import poisson

LEAGUES = {"E0":"Premier League", "SP1":"La Liga", "D1":"Bundesliga", "I1":"Serie A", "F1":"Ligue 1"}
ODDS_SETS = [("AvgH","AvgD","AvgA"),("B365H","B365D","B365A")]


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load user-downloaded Football-Data CSVs; no automated website scraping."""
    if not data_dir.exists(): raise FileNotFoundError(f"Create {data_dir} and place downloaded league CSVs inside it.")
    frames=[]
    for path in sorted(data_dir.glob("*.csv")):
        code=next((c for c in LEAGUES if c in path.stem.upper()),None)
        if code is None:
            print(f"Skipping {path.name}: filename must contain one of {', '.join(LEAGUES)}")
            continue
        try: raw=pd.read_csv(path,encoding="latin-1")
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}"); continue
        if {"Date","HomeTeam","AwayTeam","FTHG","FTAG"}.issubset(raw.columns):
            raw["League"]=code; frames.append(raw)
    if not frames: raise RuntimeError(f"No usable CSVs found in {data_dir}. Download league CSVs manually and ensure filenames contain E0, SP1, D1, I1 or F1.")
    d=pd.concat(frames,ignore_index=True)
    d["Date"]=pd.to_datetime(d.Date,dayfirst=True,errors="coerce")
    for c in ["FTHG","FTAG"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG"])
    return d[(d.FTHG>=0)&(d.FTAG>=0)].sort_values(["Date","League"]).reset_index(drop=True)


def _norm_team(value):
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def load_xg_file(path: Path) -> pd.DataFrame:
    """Load optional match-level xG CSV. Supports common Understat/xGTables column names."""
    x = pd.read_csv(path, encoding="utf-8-sig")
    aliases = {
        "date": ["Date", "date", "match_date", "datetime"],
        "home": ["HomeTeam", "home_team", "home", "h_team"],
        "away": ["AwayTeam", "away_team", "away", "a_team"],
        "xgh": ["HomeXG", "home_xg", "h_xG", "xGHome", "homeXG", "xg_home"],
        "xga": ["AwayXG", "away_xg", "a_xG", "xGAway", "awayXG", "xg_away"],
    }
    lower = {str(c).casefold(): c for c in x.columns}
    chosen = {}
    for key, opts in aliases.items():
        chosen[key] = next((lower[o.casefold()] for o in opts if o.casefold() in lower), None)
    missing = [k for k,v in chosen.items() if v is None]
    if missing:
        raise ValueError("xG CSV needs date, home team, away team, home xG and away xG columns. Missing: " + ", ".join(missing))
    out = pd.DataFrame({
        "Date": pd.to_datetime(x[chosen["date"]], dayfirst=True, errors="coerce").dt.normalize(),
        "_home_key": x[chosen["home"]].map(_norm_team),
        "_away_key": x[chosen["away"]].map(_norm_team),
        "HomeXG": pd.to_numeric(x[chosen["xgh"]], errors="coerce"),
        "AwayXG": pd.to_numeric(x[chosen["xga"]], errors="coerce"),
    })
    out = out.dropna(subset=["Date", "HomeXG", "AwayXG"])
    out = out[(out.HomeXG >= 0) & (out.AwayXG >= 0) & (out.HomeXG <= 8) & (out.AwayXG <= 8)]
    return out.drop_duplicates(["Date", "_home_key", "_away_key"], keep="last")


def merge_xg(matches: pd.DataFrame, xg_path: Path | None) -> pd.DataFrame:
    matches = matches.copy()
    matches["HomeXG"] = np.nan; matches["AwayXG"] = np.nan
    if xg_path is None:
        return matches
    xg = load_xg_file(xg_path)
    matches["_home_key"] = matches.HomeTeam.map(_norm_team)
    matches["_away_key"] = matches.AwayTeam.map(_norm_team)
    matches["_date_key"] = matches.Date.dt.normalize()
    xg = xg.rename(columns={"Date":"_date_key"})
    matches = matches.merge(xg.rename(columns={"Date":"_date_key"}), on=["_date_key","_home_key","_away_key"], how="left", suffixes=("", "_xg"))
    matches["HomeXG"] = matches["HomeXG_xg"].combine_first(matches["HomeXG"])
    matches["AwayXG"] = matches["AwayXG_xg"].combine_first(matches["AwayXG"])
    return matches.drop(columns=[c for c in ["_home_key","_away_key","_date_key","HomeXG_xg","AwayXG_xg"] if c in matches.columns])


def _valid_odds(row, cols):
    if not all(c in row.index for c in cols): return None
    vals=[pd.to_numeric(row[c],errors="coerce") for c in cols]
    if all(pd.notna(x) and 1.01<=float(x)<=100 for x in vals): return tuple(map(float,vals))
    return None


def odds_of(row):
    # Use closing market averages as the probability benchmark when present;
    # use closing market maxima as the simulated executable-price proxy.
    market = (_valid_odds(row,("AvgCH","AvgCD","AvgCA")) or
              _valid_odds(row,("AvgH","AvgD","AvgA")) or
              _valid_odds(row,("B365CH","B365CD","B365CA")) or
              _valid_odds(row,("B365H","B365D","B365A")))
    best = (_valid_odds(row,("MaxCH","MaxCD","MaxCA")) or
            _valid_odds(row,("B365CH","B365CD","B365CA")) or
            _valid_odds(row,("MaxH","MaxD","MaxA")) or
            _valid_odds(row,("B365H","B365D","B365A")) or market)
    if market is None or best is None: return None
    return market,best


def dixon_coles_tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    if x==0 and y==0: return 1-lh*la*rho
    if x==0 and y==1: return 1+lh*rho
    if x==1 and y==0: return 1+la*rho
    if x==1 and y==1: return 1-rho
    return 1.0


class DixonColesModel:
    def __init__(self, half_life=600.0, reg=.04, xg_weight=.35):
        self.half_life=half_life; self.reg=reg; self.xg_weight=float(np.clip(xg_weight,0,1)); self.teams=[]; self.ix={}; self.p=None; self.rho=0.0
    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp):
        teams=sorted(set(train.HomeTeam)|set(train.AwayTeam)); n=len(teams)
        if n<4: raise ValueError("Insufficient teams")
        ix={t:i for i,t in enumerate(teams)}
        hi=train.HomeTeam.map(ix).to_numpy(int); ai=train.AwayTeam.map(ix).to_numpy(int)
        hg=train.FTHG.to_numpy(int); ag=train.FTAG.to_numpy(int)
        xgh=pd.to_numeric(train.get("HomeXG", pd.Series(np.nan,index=train.index)), errors="coerce").to_numpy(float)
        xga=pd.to_numeric(train.get("AwayXG", pd.Series(np.nan,index=train.index)), errors="coerce").to_numpy(float)
        has_xg=np.isfinite(xgh)&np.isfinite(xga)
        # xG informs team strength; actual goals remain the evaluation target and DC low-score outcome.
        target_h=np.where(has_xg, (1-self.xg_weight)*hg+self.xg_weight*xgh, hg)
        target_a=np.where(has_xg, (1-self.xg_weight)*ag+self.xg_weight*xga, ag)
        self.xg_matches=int(has_xg.sum())
        age=np.maximum((as_of-train.Date).dt.total_seconds().to_numpy()/86400,0)
        w=np.exp(-math.log(2)*age/self.half_life)
        def objective(p):
            inter,home=p[:2]; attack=p[2:2+n]; defence=p[2+n:2+2*n]; rho=p[-1]
            lh=np.clip(np.exp(inter+home+attack[hi]-defence[ai]),.03,8)
            la=np.clip(np.exp(inter+attack[ai]-defence[hi]),.03,8)
            ll=lh-target_h*np.log(lh)+la-target_a*np.log(la)
            # Vectorised Dixon-Coles low-score correction. This is mathematically
            # identical to the per-match loop, but avoids Python work on every
            # optimiser evaluation (the objective is evaluated many times).
            low_score = (hg <= 1) & (ag <= 1)
            if np.any(low_score):
                x = hg[low_score]
                y = ag[low_score]
                low_lh = lh[low_score]
                low_la = la[low_score]
                tau = np.ones(len(x), dtype=float)
                m00 = (x == 0) & (y == 0)
                m01 = (x == 0) & (y == 1)
                m10 = (x == 1) & (y == 0)
                m11 = (x == 1) & (y == 1)
                tau[m00] = 1.0 - low_lh[m00] * low_la[m00] * rho
                tau[m01] = 1.0 + low_lh[m01] * rho
                tau[m10] = 1.0 + low_la[m10] * rho
                tau[m11] = 1.0 - rho
                if np.any(tau <= 0):
                    return 1e12
                ll[low_score] -= np.log(tau)
            loss=np.sum(w*ll)+self.reg*(np.sum(attack**2)+np.sum(defence**2))+100*(attack.mean()**2+defence.mean()**2)
            loss+=8*rho*rho
            return float(loss)
        p0=np.zeros(3+2*n); p0[0]=math.log(max((hg.sum()+ag.sum())/(2*len(train)),.5)); p0[1]=.12
        bounds=[(-3,2),(-.5,.8)]+[(-2,2)]*(2*n)+[(-.15,.15)]
        res=minimize(objective,p0,method="L-BFGS-B",bounds=bounds,options={"maxiter":700,"ftol":1e-9})
        if not np.isfinite(res.fun): raise RuntimeError("Dixon-Coles fit failed")
        self.teams,self.ix,self.p,self.rho=teams,ix,res.x,float(res.x[-1])
        return self
    def expected_goals(self,home,away):
        if home not in self.ix or away not in self.ix: raise KeyError("Team unseen in training data")
        n=len(self.teams); h=self.ix[home]; a=self.ix[away]; p=self.p
        return (float(np.clip(np.exp(p[0]+p[1]+p[2+h]-p[2+n+a]),.03,6)),
                float(np.clip(np.exp(p[0]+p[2+a]-p[2+n+h]),.03,6)))
    def probabilities(self,home,away,max_goals=10):
        lh,la=self.expected_goals(home,away); rho=self.rho; g=np.arange(max_goals+1)
        ph=poisson.pmf(g,lh); pa=poisson.pmf(g,la); mat=np.outer(ph,pa)
        for x in range(2):
            for y in range(2): mat[x,y]*=dixon_coles_tau(x,y,lh,la,rho)
        mat=np.maximum(mat,0); mat/=mat.sum()
        return np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()]),(lh,la)


def devig(odds):
    """Shin margin removal; proportional normalization is a safe fallback."""
    q=1/np.asarray(odds,dtype=float); total=float(q.sum())
    if not np.isfinite(total) or total<=0: raise ValueError("Invalid odds")
    def probs(z):
        if z>=1: z=1-1e-10
        return (np.sqrt(z*z+4*(1-z)*q*q/total)-z)/(2*(1-z))
    try:
        z=brentq(lambda z: float(probs(z).sum()-1),0,1-1e-9)
        p=probs(z); return p/p.sum()
    except (ValueError,ZeroDivisionError):
        return q/total


def fit_temperature(history, min_n=250):
    """Fit one scalar temperature to prior out-of-sample forecasts only."""
    if len(history)<min_n: return 1.0
    p=np.clip(np.array([r[0] for r in history]),1e-6,1); y=np.array([r[1] for r in history],int)
    def loss(log_t):
        t=np.exp(log_t); z=np.log(p)/t; z-=z.max(axis=1,keepdims=True)
        q=np.exp(z); q/=q.sum(axis=1,keepdims=True)
        return float(-np.mean(np.log(np.clip(q[np.arange(len(y)),y],1e-9,1))))
    result=minimize_scalar(loss,bounds=(math.log(.65),math.log(1.8)),method="bounded")
    return float(np.exp(result.x)) if result.success else 1.0


def apply_temperature(p,t):
    z=np.log(np.clip(p,1e-8,1))/t; z-=z.max(); q=np.exp(z); return q/q.sum()


def run_backtest(data,min_train=300,refit_matchdays=12,min_edge=.05,stake=1.,half_life=600.,xg_weight=.35):
    records=[]
    for league,raw in data.groupby("League"):
        calibration_history=[]
        df=raw.sort_values("Date").reset_index(drop=True)
        # Split by calendar matchday, never by row: no same-day result can leak into another forecast.
        days=sorted(df.Date.dt.normalize().unique())
        if len(df)<min_train: continue
        day_idx=0
        while day_idx<len(days):
            day=days[day_idx]
            train=df[df.Date.dt.normalize()<day]
            if len(train)<min_train: day_idx+=1; continue
            block_days=days[day_idx:day_idx+refit_matchdays]
            end_day=block_days[-1]
            block=df[(df.Date.dt.normalize()>=day)&(df.Date.dt.normalize()<=end_day)]
            model=DixonColesModel(half_life,xg_weight=xg_weight).fit(train,pd.Timestamp(day))
            temp=fit_temperature(calibration_history)
            for _,r in block.iterrows():
                odds_pair=odds_of(r)
                try: p,(lh,la)=model.probabilities(r.HomeTeam,r.AwayTeam)
                except (KeyError,ValueError,OverflowError): continue
                raw_p=p.copy()
                p=apply_temperature(raw_p,temp)
                actual=0 if r.FTHG>r.FTAG else (1 if r.FTHG==r.FTAG else 2)
                # Store raw out-of-sample probabilities; never recalibrate already-calibrated values.
                calibration_history.append((raw_p,actual))
                if odds_pair is None: continue
                market_odds,best_odds=odds_pair
                market=devig(market_odds); ev=p*np.asarray(best_odds)-1
                pick=int(np.argmax(ev)); edge=float(ev[pick]); take=edge>=min_edge
                won=int(pick==actual)
                records.append({"Date":r.Date.date().isoformat(),"League":league,"HomeTeam":r.HomeTeam,"AwayTeam":r.AwayTeam,
                    "Pick":["Home","Draw","Away"][pick],"Odds":best_odds[pick],"OddsSource":"market maximum/closing proxy","MarketOddsHome":market_odds[0],"MarketOddsDraw":market_odds[1],"MarketOddsAway":market_odds[2],"ModelProbability":p[pick],"EV":edge,
                    "Actual":["Home","Draw","Away"][actual],"Won":won,"Stake":stake if take else 0.,
                    "Profit":(stake*(best_odds[pick]-1) if won else -stake) if take else 0.,
                    "PHome":p[0],"PDraw":p[1],"PAway":p[2],"MarketHome":market[0],"MarketDraw":market[1],"MarketAway":market[2],
                    "MarketBrier":float(np.sum((market-np.eye(3)[actual])**2)),"LambdaHome":lh,"LambdaAway":la,"DCRho":model.rho,"Temperature":temp,"HomeXG":r.get("HomeXG",np.nan),"AwayXG":r.get("AwayXG",np.nan),"TrainingXGMatches":model.xg_matches})
            day_idx+=len(block_days)
    return pd.DataFrame(records)


def report(res,outdir):
    if res.empty: print("No predictions. Increase available data or lower minimum training size."); return
    bets=res[res.Stake>0].copy().sort_values("Date"); total=bets.Stake.sum(); profit=bets.Profit.sum()
    y=np.zeros((len(res),3)); idx=res.Actual.map({"Home":0,"Draw":1,"Away":2}).to_numpy(); y[np.arange(len(res)),idx]=1
    p=res[["PHome","PDraw","PAway"]].to_numpy(); market=res[["MarketHome","MarketDraw","MarketAway"]].to_numpy()
    brier=float(np.mean(np.sum((p-y)**2,axis=1))); mbrier=float(np.mean(np.sum((market-y)**2,axis=1)))
    logloss=float(-np.mean(np.log(np.clip(p[np.arange(len(idx)),idx],1e-9,1))))
    print("\n=== WALK-FORWARD EVALUATION — SIMULATED ONLY ===")
    print(f"Predictions with odds: {len(res):,} | Bets: {len(bets):,}")
    print(f"Simulated ROI: {profit/total:.2%}" if total else "Simulated ROI: n/a")
    print(f"Net simulated units: {profit:+.2f} | Win rate: {bets.Won.mean():.2%}" if len(bets) else "No qualifying bets")
    if len(bets):
        eq=bets.Profit.cumsum(); peak=np.maximum.accumulate(np.maximum(eq,0)); print(f"Max drawdown: {(eq-peak).min():.2f} units")
    print(f"Model Brier: {brier:.4f} | No-vig market Brier: {mbrier:.4f} | Model log loss: {logloss:.4f}")
    print("Interpretation: lower Brier/log loss is better; market comparison is the key benchmark.")
    outdir.mkdir(parents=True,exist_ok=True); res.to_csv(outdir/"all_predictions_v3.csv",index=False); bets.to_csv(outdir/"simulated_bets_v3.csv",index=False)
    if len(bets):
        z=bets.groupby("League").agg(Bets=("Profit","size"),Profit=("Profit","sum"),Staked=("Stake","sum")); z["ROI"]=z.Profit/z.Staked; z.to_csv(outdir/"league_performance_v3.csv"); print("\n",z.to_string())
    print(f"\nCSV outputs: {outdir.resolve()}")


def main():
    ap=argparse.ArgumentParser(description="Dixon-Coles football 1X2 value model — paper backtest")
    ap.add_argument("--data-dir",type=Path,default=Path("data/input")); ap.add_argument("--min-train",type=int,default=300)
    ap.add_argument("--refit-matchdays",type=int,default=12); ap.add_argument("--min-edge",type=float,default=.05)
    ap.add_argument("--stake",type=float,default=1.); ap.add_argument("--half-life",type=float,default=600.)
    ap.add_argument("--output",type=Path,default=Path("output")); ap.add_argument("--xg-weight",type=float,default=.35,help="Share of xG in blended training target when available (0 to 1)"); ap.add_argument("--xg-file",type=Path,default=None,help="Optional match-level xG CSV with Date, HomeTeam, AwayTeam, HomeXG, AwayXG")
    a=ap.parse_args(); data=merge_xg(load_data(a.data_dir),a.xg_file); print(f"Historical xG matched to {data.HomeXG.notna().sum():,} fixtures"); print(f"Loaded {len(data):,} matches across {data.League.nunique()} leagues")
    report(run_backtest(data,a.min_train,a.refit_matchdays,a.min_edge,a.stake,a.half_life,a.xg_weight),a.output)
if __name__=="__main__": main()