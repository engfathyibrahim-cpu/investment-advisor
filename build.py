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
        s = alias.get(s, s)
        if s in store.assets: continue
        sa = s.endswith(".SR")
        store._add(s, s.replace(".SR", ""), s, "Other", "SA" if sa else "US", "SAR" if sa else "USD")

def main():
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
    have_info = sum(1 for a in uni if a["fundOk"])
    dump(os.path.join(SITE, "data", "status.json"),
         {"builtAt": time.time(), "source": prov.name, "assets": len(uni), "withFinancials": have_info})
    for a in uni:
        y = a["k"]; fh = st.fhist.get(y)
        if fh is None: continue
        fh = fh.iloc[-500:]
        sma = fh["c"].rolling(50, min_periods=20).mean()
        R = lambda s, nd=4: [E.num(x, nd) for x in s]
        dump(os.path.join(SITE, "data", "detail", safe(y) + ".json"), clean({
            "dates": [d.strftime("%Y-%m-%d") for d in fh.index], "c": R(fh["c"], 5), "sma50": R(sma, 5),
            "dvol": R(fh["dvol"], 5), "tech": R(fh["tech"], 1), "val": R(fh["val"], 1),
            "macro": R(fh["macro"], 1), "risk": R(fh["risk"], 1),
            "news": (st.news.get(y) or {}).get("items", [])}))
    log.info("site built: %d assets (%d with financials) in %.0fs", len(uni), have_info, time.time() - t0)
    if len(uni) < 10:
        raise SystemExit("Too few assets loaded — data source problem, not publishing.")

if __name__ == "__main__":
    main()
