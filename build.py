"""
Builds the static Investment Advisor site into ./site
Run by GitHub Actions on a schedule (see .github/workflows/update.yml), or locally:
    pip install -r requirements.txt && python build.py
    python -m http.server -d site 8000      -> http://localhost:8000
Set DATA_PROVIDER=fake to test without internet.
"""
import os, json, time, shutil, re, logging
import numpy as np

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, "site")
CACHE = os.path.join(HERE, ".cache", "store.json")
INFO_MAX_AGE = 24 * 3600      # company financials refreshed daily
NEWS_MAX_AGE = 2 * 3600       # headlines refreshed every 2 hours
PER_RUN_LIMIT = 45            # max financials / news requests per run (keeps runs short)
log = logging.getLogger("build")

def safe(k): return re.sub(r"[^A-Za-z0-9]", "_", k)

def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=lambda o: None, allow_nan=False)

def clean(o):
    """Replace NaN/inf with None so the JSON is valid."""
    if isinstance(o, float): return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return clean(float(o))
    if isinstance(o, (np.integer,)): return int(o)
    return o

def read_watchlist(store):
    path = os.path.join(HERE, "watchlist.txt")
    if not os.path.exists(path): return
    alias = {"BTC": "BTC-USD", "GOLD": "GC=F", "XAU": "GC=F", "SILVER": "SI=F", "XAG": "SI=F"}
    for line in open(path):
        s = line.split("#")[0].strip().upper()
        if not s or not re.fullmatch(r"[A-Z0-9.\-=^]{1,15}", s): continue
        if s.isdigit(): s += ".SR"
        if s.startswith("EGX:"): s = s[4:] + ".CA"
        s = alias.get(s, s)
        if s in store.assets: continue
        mkt = "SA" if s.endswith(".SR") else "EG" if s.endswith(".CA") else "US"
        store._add(s, re.sub(r"\.(SR|CA)$", "", s), s, "Other", mkt, {"SA": "SAR", "EG": "EGP"}.get(mkt, "USD"))

def build():
    t0 = time.time()
    prov = E.FakeProvider() if E.PROVIDER == "fake" else E.YahooProvider()
    st = E.Store(prov)
    read_watchlist(st)

    cache = {"info": {}, "news": {}}
    if os.path.exists(CACHE):
        try: cache = json.load(open(CACHE))
        except Exception: pass
    now = time.time()
    for y, v in cache.get("info", {}).items():
        if y in st.assets and v.get("d"): st.info[y] = v["d"]
    for y, v in cache.get("news", {}).items():
        if y in st.assets: st.news[y] = v

    # 1) prices (2 years daily, includes today's latest delayed price) + technical/risk/macro + backtest
    st.load_history()
    log.info("history loaded in %.0fs", time.time() - t0)

    # 2) company financials and news — only the stale ones, a limited number per run
    stale_info = sorted([y for y, a in st.assets.items() if a["mkt"] != "ALT" and
                         now - cache["info"].get(y, {}).get("at", 0) > INFO_MAX_AGE],
                        key=lambda y: cache["info"].get(y, {}).get("at", 0))[:PER_RUN_LIMIT]
    for y in stale_info:
        st.load_info(y)
        if y in st.info: cache["info"][y] = {"at": now, "d": st.info[y]}
        time.sleep(0.3)
    stale_news = sorted([y for y in st.assets if now - (st.news.get(y) or {}).get("at", 0) > NEWS_MAX_AGE],
                        key=lambda y: (st.news.get(y) or {}).get("at", 0))[:PER_RUN_LIMIT]
    for y in stale_news:
        st.load_news(y); cache["news"][y] = st.news[y]
        time.sleep(0.2)
    for y in st.assets: st.compute(y)
    for y in st.assets: st.compute(y)   # second pass: sector P/E medians now complete
    st.rank_all()                         # percentile ranks within each market
    log.info("financials %d, news %d refreshed", len(stale_info), len(stale_news))

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(clean(cache), open(CACHE, "w"), default=str)

    # 3) export the site
    if os.path.exists(SITE): shutil.rmtree(SITE)
    os.makedirs(os.path.join(SITE, "data", "detail"))
    shutil.copy(os.path.join(HERE, "web", "index.html"), os.path.join(SITE, "index.html"))
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    uni = [a for a in st.assets.values() if a["f"] and a["last"]]
    dump(os.path.join(SITE, "data", "universe.json"), clean(uni))
    dump(os.path.join(SITE, "data", "track.json"), clean(st.track))
    have_info = sum(1 for a in uni if a["fundOk"] or a["mkt"] == "ALT")
    egp = st.macro.get("EGP=X")
    egp_usd = E.num(egp.iloc[-1], 4) if egp is not None and len(egp) else None
    dump(os.path.join(SITE, "data", "status.json"),
         {"builtAt": time.time(), "source": prov.name, "assets": len(uni), "withFinancials": have_info,
          "egpPerUsd": egp_usd, "benchNames": E.BENCH_NAME})
    # live scorecard: log today's published top picks once a day, then measure how they did vs the market
    dump(os.path.join(SITE, "data", "forward.json"), clean(update_scorecard(st)))
    for a in uni:
        y = a["k"]; fh = st.fhist.get(y)
        if fh is None: continue
        sma = fh["c"].rolling(50, min_periods=20).mean()
        fh = fh.iloc[-760:]; sma = sma.iloc[-760:]
        R = lambda s, nd=4: [E.num(x, nd) for x in s]
        rk = lambda col, base: fh[col].fillna(fh[base]) if col in fh else fh[base]
        dump(os.path.join(SITE, "data", "detail", safe(y) + ".json"), clean({
            "dates": [d.strftime("%Y-%m-%d") for d in fh.index], "c": R(fh["c"], 5), "b": R(fh["b"], 5), "sma50": R(sma, 5),
            "dvol": R(fh["dvol"], 5), "tech": R(rk("tech_r", "tech"), 1), "val": R(rk("val_r", "val"), 1),
            "macro": R(fh["macro"], 1), "risk": R(rk("risk_r", "risk"), 1),
            "news": (st.news.get(y) or {}).get("items", [])}))
    log.info("site built: %d assets (%d with financials) in %.0fs", len(uni), have_info, time.time() - t0)
    if len(uni) < 10:
        raise RuntimeError(f"Only {len(uni)} assets loaded: the data source refused or failed.")
    # keep a copy of this good build, to fall back on if a later run can't get data
    last = os.path.join(HERE, ".cache", "site_last")
    if os.path.exists(last): shutil.rmtree(last)
    shutil.copytree(SITE, last)

PICKS = os.path.join(HERE, "history", "picks.json")

def update_scorecard(st):
    """Append today's top-10 lists (once per UTC day) to history/picks.json (committed by the workflow),
    then score every logged pick against its market index."""
    picks = []
    if os.path.exists(PICKS):
        try: picks = json.load(open(PICKS))
        except Exception: picks = []
    today = time.strftime("%Y-%m-%d", time.gmtime())
    bench_now = {m: (float(st.macro[s].iloc[-1]) if s in st.macro and len(st.macro[s]) else None) for m, s in E.BENCH.items()}
    def bench_of(a): return 1.0 if a["mkt"] == "ALT" else bench_now.get(a["mkt"])
    logged = {(p["d"], p["m"]) for p in picks}
    for lst in ("ALL", "US", "SA", "EG"):
        if (today, lst) in logged: continue
        rows = E.top_picks(st.assets, st.track, None if lst == "ALL" else lst, 10)
        for rank, (a, x) in enumerate(rows, 1):
            b = bench_of(a)
            if not a["last"] or (a["mkt"] != "ALT" and not b): continue
            picks.append({"d": today, "m": lst, "r": rank, "k": a["k"], "sym": a["sym"], "mkt": a["mkt"],
                          "px": round(a["last"], 6), "bx": round(b, 6), "dec": x["dec"], "sc": x["score"], "cf": x["conf"]})
    cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 3 * 365 * 86400))
    picks = [p for p in picks if p["d"] >= cutoff]
    os.makedirs(os.path.dirname(PICKS), exist_ok=True)
    json.dump(picks, open(PICKS, "w"), separators=(",", ":"))
    # score them
    now = time.time(); out = {}
    for lst in ("ALL", "US", "SA", "EG"):
        evald = []
        for p in picks:
            if p["m"] != lst: continue
            a = st.assets.get(p["k"])
            if not a or not a["last"]: continue
            b = bench_of(a)
            if not b or not p["bx"]: continue
            age = (now - time.mktime(time.strptime(p["d"], "%Y-%m-%d"))) / 86400
            ret = a["last"] / p["px"] - 1; bret = b / p["bx"] - 1
            evald.append({**p, "age": age, "ret": ret, "ex": ret - bret})
        def agg(rows):
            if not rows: return None
            return {"n": len(rows), "avgEx": float(np.mean([r["ex"] for r in rows])), "avgRet": float(np.mean([r["ret"] for r in rows])),
                    "beat": float(np.mean([r["ex"] > 0 for r in rows]))}
        mature = [r for r in evald if r["age"] >= 7]
        cohorts = {}
        for r in evald: cohorts.setdefault(r["d"], []).append(r)
        out[lst] = {"since": min((p["d"] for p in picks if p["m"] == lst), default=None),
                    "days": len(cohorts), "all": agg(mature),
                    "buckets": {"1–4 weeks": agg([r for r in mature if r["age"] < 31]),
                                "1–3 months": agg([r for r in mature if 31 <= r["age"] < 92]),
                                "3+ months": agg([r for r in mature if r["age"] >= 92])},
                    "cohorts": [{"d": d, **agg(rs), "top": [x["sym"] for x in sorted(rs, key=lambda x: x["r"])[:5]]}
                                for d, rs in sorted(cohorts.items(), reverse=True)[:30]],
                    "best": sorted([{k: r[k] for k in ("sym", "d", "ex", "ret")} for r in mature], key=lambda r: -r["ex"])[:5],
                    "worst": sorted([{k: r[k] for k in ("sym", "d", "ex", "ret")} for r in mature], key=lambda r: r["ex"])[:5]}
    out["bench"] = E.BENCH_NAME
    return out

def main():
    """Never leave the site broken: if this run fails (usually Yahoo throttling), republish the last
    good data with a warning instead of failing the whole update."""
    import traceback
    try:
        build()
    except Exception as e:
        log.error("BUILD FAILED: %s\n%s", e, traceback.format_exc())
        last = os.path.join(HERE, ".cache", "site_last")
        if not os.path.exists(os.path.join(last, "data", "status.json")):
            raise SystemExit(f"Build failed and there is no earlier good data to fall back on: {e}")
        if os.path.exists(SITE): shutil.rmtree(SITE)
        shutil.copytree(last, SITE)
        shutil.copy(os.path.join(HERE, "web", "index.html"), os.path.join(SITE, "index.html"))
        st = json.load(open(os.path.join(SITE, "data", "status.json")))
        st.update(lastError=str(e)[:300], lastErrorAt=time.time())
        json.dump(st, open(os.path.join(SITE, "data", "status.json"), "w"))
        log.warning("Published the last good data from %s instead.", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(st.get("builtAt", 0))))

if __name__ == "__main__":
    main()
