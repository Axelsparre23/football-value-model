"""Chronological holdout evaluation for the Dixon-Coles + Elo ensemble."""
import numpy as np
import pandas as pd
from football_value_model_v3 import DixonColesModel
from dynamic_elo import calculate_elo, elo_probabilities, blend_probabilities

def evaluate_holdout(data, min_train=350, refit_days=10):
    rows=[]
    for league, raw in data.groupby("League"):
        df=raw.sort_values("Date").reset_index(drop=True)
        days=sorted(df.Date.dt.normalize().unique())
        if len(df)<min_train+20: continue
        # Keep the final 20% of calendar matchdays as a genuinely later holdout.
        split=max(1,int(len(days)*.8))
        first_test=days[split]
        history=df[df.Date.dt.normalize()<first_test]
        if len(history)<min_train: continue
        for offset in range(split,len(days),refit_days):
            block_days=days[offset:offset+refit_days]
            start=block_days[0]
            train=df[df.Date.dt.normalize()<start]
            if len(train)<min_train: continue
            block=df[df.Date.dt.normalize().isin(block_days)]
            try:
                model=DixonColesModel(half_life=300 if league!="SWE" else 240,reg=.04,xg_weight=0).fit(train,pd.Timestamp(start))
                elo=calculate_elo(train)
            except Exception: continue
            draw_rate=float((train.FTHG==train.FTAG).mean())
            for _,r in block.iterrows():
                try:
                    dc,_=model.probabilities(r.HomeTeam,r.AwayTeam)
                    ep=elo_probabilities(elo.get(r.HomeTeam,1500),elo.get(r.AwayTeam,1500),draw_rate)
                    p=blend_probabilities(dc,ep,.25)
                except Exception: continue
                actual=0 if r.FTHG>r.FTAG else (1 if r.FTHG==r.FTAG else 2)
                one=np.eye(3)[actual]
                rows.append({"League":league,"Date":r.Date,"Brier":float(np.sum((p-one)**2)),
                             "LogLoss":float(-np.log(max(p[actual],1e-9))),
                             "PHome":p[0],"PDraw":p[1],"PAway":p[2],"Actual":actual})
    result=pd.DataFrame(rows)
    if result.empty: return {"matches":0,"brier":None,"logloss":None,"by_league":pd.DataFrame()}
    return {"matches":len(result),"brier":float(result.Brier.mean()),
            "logloss":float(result.LogLoss.mean()),
            "by_league":result.groupby("League").agg(matches=("Brier","size"),brier=("Brier","mean"),logloss=("LogLoss","mean")).reset_index(),
            "predictions":result}
