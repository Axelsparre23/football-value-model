"""Optional API-Football fixture, lineup, rating and absence lookup."""
from __future__ import annotations
from datetime import date, timedelta
from difflib import SequenceMatcher
import os
import re
import unicodedata
import requests

BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"E0":39, "SP1":140, "D1":78, "I1":135, "F1":61, "SWE":113}
POS = {"G":"GK","GOALKEEPER":"GK","D":"DF","DEFENDER":"DF",
       "M":"CM","MIDFIELDER":"CM","F":"FW","ATTACKER":"FW"}

def get_api_football_key(st=None):
    key = os.getenv("API_FOOTBALL_KEY", "").strip()
    if key:
        return key
    if st is not None:
        try:
            return str(st.secrets.get("API_FOOTBALL_KEY", "")).strip() or None
        except Exception:
            return None
    return None

def _api(endpoint, params, key):
    r = requests.get(f"{BASE_URL}/{endpoint}", params=params,
                     headers={"x-apisports-key":key}, timeout=20)
    if r.status_code in (401,403):
        raise RuntimeError("API-Football rejected the key or the free plan does not cover this season/data.")
    if r.status_code == 429:
        raise RuntimeError("API-Football daily request limit reached (free plan: 100/day).")
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(str(body["errors"]))
    return body.get("response") or []

def _season(code, d):
    return d.year if code == "SWE" or d.month >= 7 else d.year-1

def _norm(s):
    s = unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
    tokens = re.sub(r"[^a-z0-9 ]"," ",s).split()
    return " ".join(x for x in tokens if x not in {"fc","if","aif","afc","cf","fk","club"})

def _score(a,b):
    a,b=_norm(a),_norm(b)
    if not a or not b: return 0
    if a==b: return 1
    if a in b or b in a: return .92
    return SequenceMatcher(None,a,b).ratio()

def _rating_map(team_id, league_id, season, key):
    try:
        rows=_api("players",{"team":team_id,"league":league_id,"season":season,"page":1},key)
    except Exception:
        return {}
    out={}
    for row in rows:
        p=row.get("player") or {}
        stats=(row.get("statistics") or [{}])[0]
        games=stats.get("games") or {}
        raw=games.get("rating")
        try: rating=float(raw) if raw not in (None,"") else None
        except (TypeError,ValueError): rating=None
        if p.get("id") is not None:
            out[int(p["id"])]=(rating,POS.get(str(games.get("position","")).upper(),"UNK"))
    return out

def _xi_text(team_lineup, ratings):
    lines=[]
    for entry in team_lineup.get("startXI") or []:
        p=entry.get("player") or {}
        pid=p.get("id")
        rating,pos=ratings.get(int(pid),(None,"UNK")) if pid is not None else (None,"UNK")
        pos=POS.get(str(p.get("pos") or "").upper(),pos)
        if p.get("name"):
            lines.append(f"{p['name']} | {rating:.2f}" if rating is not None else f"{p['name']} | ")
            lines[-1] += f" | {pos}"
    formation=str(team_lineup.get("formation") or "Other / unknown")
    return "\n".join(lines),formation,len(lines)

def fetch_fixture_team_news(api_key, league_code, home_team, away_team, match_date):
    if league_code not in LEAGUE_IDS:
        raise RuntimeError("No API-Football league mapping exists for this competition.")
    league_id=LEAGUE_IDS[league_code]
    season=_season(league_code,match_date)
    start,end=match_date-timedelta(days=1),match_date+timedelta(days=1)
    fixtures=_api("fixtures",{"league":league_id,"season":season,
        "from":start.isoformat(),"to":end.isoformat(),"timezone":"Europe/Stockholm"},api_key)
    candidates=[]
    for f in fixtures:
        t=f.get("teams") or {}
        hs=_score(home_team,(t.get("home") or {}).get("name",""))
        aw=_score(away_team,(t.get("away") or {}).get("name",""))
        if hs>=.78 and aw>=.78: candidates.append((hs+aw,f))
    if not candidates:
        raise RuntimeError(f"No matching fixture found from {start} to {end}. Check date, league coverage and free-plan season access.")
    candidates.sort(key=lambda x:x[0],reverse=True)
    fixture=candidates[0][1]
    info=fixture.get("fixture") or {}
    fid=info.get("id")
    th=(fixture.get("teams") or {}).get("home") or {}
    ta=(fixture.get("teams") or {}).get("away") or {}
    hid,aid=th.get("id"),ta.get("id")
    if not fid or not hid or not aid: raise RuntimeError("Fixture response is missing IDs.")
    lineups=_api("fixtures/lineups",{"fixture":fid},api_key)
    injuries=_api("injuries",{"fixture":fid},api_key)
    hr=_rating_map(int(hid),league_id,season,api_key)
    ar=_rating_map(int(aid),league_id,season,api_key)
    by_id={int((x.get("team") or {}).get("id")):x for x in lineups if (x.get("team") or {}).get("id") is not None}
    hline,hform,hcount=_xi_text(by_id.get(int(hid),{}),hr)
    aline,aform,acount=_xi_text(by_id.get(int(aid),{}),ar)
    habs,aabs,questionable=[],[],[]
    for item in injuries:
        t=item.get("team") or {}; p=item.get("player") or {}
        pid=p.get("id"); name=str(p.get("name") or "").strip()
        if not name: continue
        is_home=t.get("id") is not None and int(t["id"])==int(hid)
        is_away=t.get("id") is not None and int(t["id"])==int(aid)
        if not (is_home or is_away): continue
        kind=str(p.get("type") or p.get("reason") or "Unavailable")
        if "question" in str(p.get("type","")).lower():
            questionable.append(f"{t.get('name','')}: {name} ({kind})")
            continue
        rating,pos=(hr if is_home else ar).get(int(pid),(None,"UNK")) if pid is not None else (None,"UNK")
        value=f"{name} | {rating:.2f}" if rating is not None else f"{name} | "
        (habs if is_home else aabs).append(value+f" | {pos}")
    return {
      "fixture_id":fid,"kickoff":info.get("date",""),
      "status":(info.get("status") or {}).get("long","Scheduled"),
      "home_name":th.get("name",home_team),"away_name":ta.get("name",away_team),
      "home":{"xi":hline,"formation":hform,"count":hcount,"abs": "\n".join(habs)},
      "away":{"xi":aline,"formation":aform,"count":acount,"abs":"\n".join(aabs)},
      "questionable":questionable,"season":season
    }
