"""
Investment Advisor — live web app (free data edition)

Data: Yahoo Finance via the `yfinance` library (free; stock prices ~15 min delayed,
crypto near real-time). Set DATA_PROVIDER=fake to run with simulated data (for testing).

Run locally:   pip install -r requirements.txt && python app.py   ->  http://localhost:7860
"""
import os, json, time, math, threading, logging, random, re
from datetime import datetime, timezone, timedelta
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

PORT = int(os.environ.get("PORT", 7860))
PROVIDER = os.environ.get("DATA_PROVIDER", "yahoo").lower()
HERE = os.path.dirname(os.path.abspath(__file__))
SAR_PER_USD = 3.75
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("advisor")

# --------------------------------------------------------------------------------------
# Universe (add more here, or use "Add any stock" in the web page)
# --------------------------------------------------------------------------------------
US = [("AAPL","Apple","Technology"),("MSFT","Microsoft","Technology"),("NVDA","NVIDIA","Technology"),
      ("AMZN","Amazon","Consumer Cyclical"),("GOOGL","Alphabet","Communication Services"),("META","Meta Platforms","Communication Services"),
      ("TSLA","Tesla","Consumer Cyclical"),("AVGO","Broadcom","Technology"),("BRK-B","Berkshire Hathaway","Financial Services"),
      ("JPM","JPMorgan Chase","Financial Services"),("V","Visa","Financial Services"),("MA","Mastercard","Financial Services"),
      ("UNH","UnitedHealth","Healthcare"),("XOM","Exxon Mobil","Energy"),("CVX","Chevron","Energy"),("JNJ","Johnson & Johnson","Healthcare"),
      ("LLY","Eli Lilly","Healthcare"),("ABBV","AbbVie","Healthcare"),("MRK","Merck","Healthcare"),("PFE","Pfizer","Healthcare"),
      ("PG","Procter & Gamble","Consumer Defensive"),("KO","Coca-Cola","Consumer Defensive"),("PEP","PepsiCo","Consumer Defensive"),
      ("WMT","Walmart","Consumer Defensive"),("COST","Costco","Consumer Defensive"),("HD","Home Depot","Consumer Cyclical"),
      ("MCD","McDonald's","Consumer Cyclical"),("NKE","Nike","Consumer Cyclical"),("BAC","Bank of America","Financial Services"),
      ("ORCL","Oracle","Technology"),("CRM","Salesforce","Technology"),("AMD","AMD","Technology"),("ADBE","Adobe","Technology"),
      ("CSCO","Cisco","Technology"),("INTC","Intel","Technology"),("QCOM","Qualcomm","Technology"),("IBM","IBM","Technology"),
      ("NFLX","Netflix","Communication Services"),("DIS","Disney","Communication Services"),("TMO","Thermo Fisher","Healthcare")]
SA = [("2222","Saudi Aramco","Energy"),("1120","Al Rajhi Bank","Financial Services"),("1180","Saudi National Bank","Financial Services"),
      ("2010","SABIC","Basic Materials"),("7010","stc","Communication Services"),("2082","ACWA Power","Utilities"),
      ("1211","Ma'aden","Basic Materials"),("1150","Alinma Bank","Financial Services"),("1010","Riyad Bank","Financial Services"),
      ("1060","Saudi Awwal Bank (SAB)","Financial Services"),("1140","Bank Albilad","Financial Services"),("1050","Banque Saudi Fransi","Financial Services"),
      ("4013","Dr. Sulaiman Al Habib","Healthcare"),("4002","Mouwasat Medical","Healthcare"),("2280","Almarai","Consumer Defensive"),
      ("7203","Elm","Technology"),("2020","SABIC Agri-Nutrients","Basic Materials"),("4190","Jarir Marketing","Consumer Cyclical"),
      ("5110","Saudi Electricity","Utilities"),("2380","Petro Rabigh","Energy"),("2350","Saudi Kayan","Basic Materials"),
      ("2290","Yansab","Basic Materials"),("1111","Saudi Tadawul Group","Financial Services"),("7020","Mobily","Communication Services"),
      ("4280","Kingdom Holding","Financial Services"),("2050","Savola","Consumer Defensive"),("4200","Aldrees","Energy"),
      ("8210","Bupa Arabia","Financial Services"),("4001","Al Othaim Markets","Consumer Defensive"),("2310","Sipchem","Basic Materials"),
      ("4300","Dar Al Arkan","Real Estate"),("4030","Bahri","Industrials")]
ALT = [("BTC-USD","BTC","Bitcoin","Crypto"),("GC=F","XAU","Gold (oz)","Precious metals"),("SI=F","XAG","Silver (oz)","Precious metals")]
MACRO = {"spx":"^GSPC","ndx":"^IXIC","vix":"^VIX","tnx":"^TNX","dxy":"DX-Y.NYB","oil":"CL=F","tasi":"^TASI.SR"}

# System model weights (same as in the page) — fundamentals are excluded from the backtest
# because free sources don't provide historical fundamentals.
SYS_W = {"tech":22,"fund":24,"val":20,"macro":18,"risk":16}

# --------------------------------------------------------------------------------------
# Data providers
# --------------------------------------------------------------------------------------
def _extract(df, s, n):
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        l0, l1 = df.columns.get_level_values(0), df.columns.get_level_values(1)
        if s in l0: sub = df[s].copy()
        elif s in l1: sub = df.xs(s, axis=1, level=1).copy()
        else: return None
    else:
        if n != 1: return None
        sub = df.copy()
    if "Close" not in sub.columns: return None
    sub = sub.dropna(subset=["Close"])
    if sub.index.tz is not None:
        sub.index = sub.index.tz_convert(None)
    return sub if len(sub) else None

class YahooProvider:
    name = "Yahoo Finance"
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    def history(self, syms, period="2y", interval="1d"):
        out = {}
        for i in range(0, len(syms), 40):
            chunk = syms[i:i+40]
            try:
                df = self.yf.download(chunk, period=period, interval=interval, group_by="ticker",
                                      auto_adjust=True, threads=True, progress=False)
            except Exception as e:
                log.warning("download failed %s: %s", chunk[:3], e); continue
            for s in chunk:
                sub = _extract(df, s, len(chunk))
                if sub is not None: out[s] = sub
        return out
    def info(self, s):
        try: return self.yf.Ticker(s).info or {}
        except Exception as e:
            log.info("info failed %s: %s", s, e); return {}
    def news(self, s):
        try: return self.yf.Ticker(s).news or []
        except Exception: return []

class FakeProvider:
    """Simulated data in the same shape as Yahoo, for offline testing."""
    name = "Simulated (test mode)"
    def __init__(self):
        self.base = {}
    def _seed(self, s): return abs(hash(s)) % (2**32)
    def _start(self, s):
        if s not in self.base:
            r = random.Random(self._seed(s))
            self.base[s] = {"^VIX":16,"^TNX":4.2,"DX-Y.NYB":104,"BTC-USD":90000,"GC=F":2650,"SI=F":31}.get(s, r.uniform(15, 600))
        return self.base[s]
    def history(self, syms, period="2y", interval="1d"):
        out = {}
        for s in syms:
            r = np.random.default_rng(self._seed(s))
            if interval == "1d":
                n = 500; idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
                vol = 0.012 if not s.startswith("^") else 0.008
                drift = r.normal(0.0004, 0.0006)
            else:
                n = 390; idx = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min")
                vol = 0.0006; drift = 0
                r = np.random.default_rng(int(time.time() // 5) + self._seed(s))
            steps = r.normal(drift, vol, n)
            c = self._start(s) * np.exp(np.cumsum(steps) - steps.sum())
            out[s] = pd.DataFrame({"Open":c,"High":c*1.005,"Low":c*0.995,"Close":c,"Volume":1e6}, index=idx)
        return out
    def info(self, s):
        r = random.Random(self._seed(s) + 7)
        if "=" in s or s.startswith("^") or s.endswith("-USD"): return {}
        p = self._start(s)
        return {"trailingPE": r.uniform(8, 45), "forwardPE": r.uniform(8, 35), "revenueGrowth": r.uniform(-0.1, 0.3),
                "earningsGrowth": r.uniform(-0.3, 0.5), "returnOnEquity": r.uniform(0.02, 0.4), "profitMargins": r.uniform(0.02, 0.35),
                "debtToEquity": r.uniform(10, 220), "trailingAnnualDividendYield": r.uniform(0, 0.05), "beta": r.uniform(0.5, 1.8),
                "targetMeanPrice": p * r.uniform(0.85, 1.35), "numberOfAnalystOpinions": r.randint(3, 40),
                "recommendationKey": r.choice(["buy","hold","strong_buy","underperform"]), "marketCap": r.uniform(1e9, 3e12)}
    def news(self, s):
        heads = ["%s beats quarterly estimates, raises outlook","%s shares fall after analyst downgrade",
                 "%s announces record dividend","%s faces regulatory probe","%s expands into new markets"]
        r = random.Random(self._seed(s) + int(time.time() // 3600))
        return [{"title": r.choice(heads) % s, "link": "https://example.com", "publisher": "Test wire",
                 "providerPublishTime": int(time.time()) - r.randint(0, 200000)} for _ in range(4)]

# --------------------------------------------------------------------------------------
# Analytics engine
# --------------------------------------------------------------------------------------
def cl(x, lo=0.0, hi=100.0):
    return x.clip(lo, hi) if isinstance(x, (pd.Series, np.ndarray)) else float(min(hi, max(lo, x)))

def num(x, nd=4):
    try:
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
    except Exception:
        return None

def tech_series(c):
    sma50 = c.rolling(50, min_periods=20).mean(); sma200 = c.rolling(200, min_periods=60).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)
    mom = c / c.shift(63) - 1
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    parts = [cl(50 + (c/sma50 - 1) * 500), cl(50 + (c/sma200 - 1) * 250), cl(50 + (sma50/sma200 - 1) * 500),
             cl(100 - (rsi - 58).abs() * 2.2), cl(50 + mom * 200), cl(50 + hist * 5000)]
    score = pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(50)
    return score, {"sma50": sma50, "sma200": sma200, "rsi": rsi, "mom": mom}

def risk_series(c):
    r = np.log(c).diff()
    dvol = r.rolling(63, min_periods=20).std()
    avol = dvol * math.sqrt(252)
    dd = c / c.rolling(252, min_periods=20).max() - 1
    score = (0.7 * cl(100 - (avol - 0.15) * 150) + 0.3 * cl(100 + dd * 200)).fillna(50)
    return score, dvol, avol, dd

def valproxy_series(c):
    sma200 = c.rolling(200, min_periods=60).mean()
    return cl(50 - (c/sma200 - 1) * 150).fillna(50)

def regime(s):
    if s is None or len(s) < 30: return None
    return cl(50 + (s / s.rolling(200, min_periods=60).mean() - 1) * 300)

def falling(s, k):  # high score when series is below its 100-day average
    if s is None or len(s) < 30: return None
    return cl(50 - (s / s.rolling(100, min_periods=40).mean() - 1) * k)

def blend(parts, index):
    ok = [(w, p.reindex(index, method="ffill")) for w, p in parts if p is not None]
    if not ok: return pd.Series(50.0, index=index)
    tot = sum(w for w, _ in ok)
    return (sum(w * p for w, p in ok) / tot).fillna(50)

POS = set("beat beats surge surges soar soars jump jumps rally rallies record upgrade upgraded upgrades raise raises raised growth strong gain gains profit profits outperform bullish rise rises higher expand expands expansion win wins approval approved dividend buyback boost boosts tops exceed exceeds".split())
NEG = set("miss misses plunge plunges fall falls drop drops slump slumps downgrade downgraded downgrades cut cuts weak loss losses lawsuit probe investigation recall bearish lower decline declines warn warns warning layoffs fine fined tumble tumbles crash crashes slowdown halt sues risk risks concern concerns".split())
def headline_sent(t):
    w = re.findall(r"[a-z]+", (t or "").lower())
    p = sum(x in POS for x in w); n = sum(x in NEG for x in w)
    return 0.0 if p + n == 0 else (p - n) / (p + n)

def parse_news(items):
    out = []
    for it in items[:10]:
        c = it.get("content") if isinstance(it.get("content"), dict) else None
        if c:
            title = c.get("title"); url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url"))
            pub = (c.get("provider") or {}).get("displayName"); ts = c.get("pubDate")
        else:
            title = it.get("title"); url = it.get("link"); pub = it.get("publisher")
            t = it.get("providerPublishTime"); ts = datetime.fromtimestamp(t, timezone.utc).isoformat() if t else None
        if title:
            out.append({"title": title, "url": url, "publisher": pub, "time": ts, "sent": round(headline_sent(title), 2)})
    return out

def decide(score, sd):
    dec = "BUY" if score >= 60 else "SELL" if score <= 44 else "HOLD"
    dist = score - 60 if dec == "BUY" else 44 - score if dec == "SELL" else min(score - 44, 60 - score)
    conf = cl(48 + dist * 1.6 + (25 - sd) * 0.9, 30, 94)
    return dec, conf

def evaluate(c, dvol, i, dec, days_td):
    """Walk forward from day i: did the call work? Returns (worked, return, days held)."""
    n = min(days_td, len(c) - 1 - i)
    if n < 5: return None
    p = c[i]; v = dvol[i] if not math.isnan(dvol[i]) else 0.015; sw = v * math.sqrt(5)
    path = c[i+1:i+1+n]
    if dec == "BUY":
        tgt = p * (1 + max(0.6 * v * math.sqrt(days_td), 0.03)); stp = p * (1 - 2.2 * sw)
        for j, x in enumerate(path):
            if x >= tgt: return True, tgt/p - 1, j + 1
            if x <= stp: return False, stp/p - 1, j + 1
        r = path[-1]/p - 1; return r > 0, r, n
    if dec == "SELL":
        tgt = p * (1 - max(0.5 * v * math.sqrt(days_td), 0.03)); stp = p * (1 + 2 * sw)
        for j, x in enumerate(path):
            if x <= tgt: return True, tgt/p - 1, j + 1
            if x >= stp: return False, stp/p - 1, j + 1
        r = path[-1]/p - 1; return r < 0, r, n
    stp = p * (1 - 3 * sw); r = path[-1]/p - 1
    return bool(min(path) > stp), r, n

# --------------------------------------------------------------------------------------
# Store: holds everything, refreshed by background threads
# --------------------------------------------------------------------------------------
class Store:
    def __init__(self, prov):
        self.p = prov; self.lock = threading.RLock()
        self.assets = {}      # ysym -> public dict
        self.hist = {}        # ysym -> daily DataFrame
        self.fhist = {}       # ysym -> DataFrame of historical price-based factors
        self.info = {}; self.news = {}; self.macro = {}; self.track = {}
        self.status = {"ready": False, "phase": "starting", "progress": 0, "source": prov.name,
                       "quotesAt": None, "histAt": None}
        for s, n, sec in US: self._add(s, s, n, sec, "US", "USD")
        for s, n, sec in SA: self._add(s + ".SR", s, n, sec, "SA", "SAR")
        for y, s, n, sec in ALT: self._add(y, s, n, sec, "ALT", "USD")

    def _add(self, y, sym, name, sector, mkt, ccy):
        self.assets[y] = {"k": y, "sym": sym, "name": name, "sector": sector, "mkt": mkt, "ccy": ccy,
                          "last": None, "prev": None, "chg": None, "vol": 0.015, "f": None, "fair": None,
                          "m": {}, "fundOk": False, "ts": None}

    # ---------- loading ----------
    def load_history(self, syms=None):
        syms = syms or list(self.assets) + list(MACRO.values())
        self.status["phase"] = "Downloading 2 years of daily prices"
        h = self.p.history(syms, period="2y", interval="1d")
        with self.lock:
            for s, df in h.items():
                if s in MACRO.values(): self.macro[s] = df["Close"]
                else: self.hist[s] = df
            self.status["histAt"] = time.time()
        for s in list(self.assets):
            if s in h: self.compute(s)
        self.compute_track()

    def macro_parts(self, a):
        g = self.macro.get
        mkt, sec = a["mkt"], a["sector"]
        if mkt == "US":
            parts = [(0.6, regime(g("^GSPC"))), (0.4, cl(100 - (g("^VIX") - 12) * 3.5) if g("^VIX") is not None else None)]
        elif mkt == "SA":
            parts = [(0.6, regime(g("^TASI.SR"))), (0.4, regime(g("CL=F")))]
            if g("^TASI.SR") is None: parts.append((0.3, regime(g("^GSPC"))))
        elif a["sym"] == "BTC":
            parts = [(0.6, regime(g("^IXIC"))), (0.4, falling(g("DX-Y.NYB"), 1000))]
        else:
            parts = [(0.5, falling(g("^TNX"), 400)), (0.5, falling(g("DX-Y.NYB"), 1000))]
        if mkt != "ALT" and sec in ("Energy", "Basic Materials"):
            parts.append((0.4, regime(g("CL=F"))))
        return parts

    def macro_text(self, a):
        g = self.macro.get; bits = []
        def trend(s, name):
            if s is None or len(s) < 60: return
            sma = s.rolling(200, min_periods=60).mean().iloc[-1]
            bits.append(f"{name} {'above' if s.iloc[-1] > sma else 'below'} its 200-day average")
        if a["mkt"] == "US":
            trend(g("^GSPC"), "S&P 500")
            if g("^VIX") is not None: bits.append(f"VIX {g('^VIX').iloc[-1]:.1f} ({'calm' if g('^VIX').iloc[-1] < 18 else 'nervous' if g('^VIX').iloc[-1] < 25 else 'fearful'})")
        elif a["mkt"] == "SA":
            trend(g("^TASI.SR"), "TASI"); trend(g("CL=F"), "Oil")
        elif a["sym"] == "BTC":
            trend(g("^IXIC"), "Nasdaq"); trend(g("DX-Y.NYB"), "US dollar")
        else:
            if g("^TNX") is not None: bits.append(f"10-yr US yield {g('^TNX').iloc[-1]:.2f}%")
            trend(g("DX-Y.NYB"), "US dollar")
        if a["mkt"] != "ALT" and a["sector"] in ("Energy", "Basic Materials"): trend(g("CL=F"), "Oil")
        return "; ".join(dict.fromkeys(bits))

    def compute(self, y):
        with self.lock:
            a = self.assets.get(y); df = self.hist.get(y)
            if a is None or df is None or len(df) < 30: return
            c = df["Close"].astype(float)
            tech, tm = tech_series(c)
            risk, dvol, avol, dd = risk_series(c)
            valp = valproxy_series(c)
            macro = blend(self.macro_parts(a), c.index)
            fh = pd.DataFrame({"tech": tech, "val": valp, "macro": macro, "risk": risk, "dvol": dvol, "c": c})
            self.fhist[y] = fh
            last = a["last"] or float(c.iloc[-1])
            info = self.info.get(y, {}); m = {}
            m.update(rsi=num(tm["rsi"].iloc[-1], 1), sma50=num(tm["sma50"].iloc[-1]), sma200=num(tm["sma200"].iloc[-1]),
                     mom3m=num(tm["mom"].iloc[-1]), hi52=num(c.iloc[-252:].max()), annVol=num(avol.iloc[-1]), dd=num(dd.iloc[-1]))
            f = {"tech": float(tech.iloc[-1]), "risk": float(risk.iloc[-1])}
            # macro + news
            news = self.news.get(y)
            ns = None
            if news and news["items"]:
                ns = float(np.mean([n["sent"] for n in news["items"]]))
            f["macro"] = cl(float(macro.iloc[-1]) + (ns or 0) * 15)
            m.update(macroTxt=self.macro_text(a), newsSent=num(ns, 2), newsN=len(news["items"]) if news else 0)
            # fundamentals & valuation
            if a["mkt"] == "ALT":
                f["fund"] = cl(50 + (c.iloc[-1] / c.iloc[max(0, len(c)-253)] - 1) * 100)
                f["val"] = float(valp.iloc[-1])
                fair = float(c.rolling(200, min_periods=60).mean().iloc[-1])
                a["fundOk"] = True
            else:
                fs = []
                g = lambda k: num(info.get(k))
                rg, eg, roe, pm, de = g("revenueGrowth"), g("earningsGrowth"), g("returnOnEquity"), g("profitMargins"), g("debtToEquity")
                if rg is not None: fs.append(cl(50 + rg * 250))
                if eg is not None: fs.append(cl(50 + cl(eg, -1, 1) * 120))
                if roe is not None: fs.append(cl(roe * 400))
                if pm is not None: fs.append(cl(30 + pm * 250))
                if de is not None and a["sector"] != "Financial Services": fs.append(cl(100 - de * 0.4))
                f["fund"] = float(np.mean(fs)) if fs else 50.0
                pe = g("trailingPE") or g("forwardPE")
                spe = self.sector_pe(a)
                tgt = g("targetMeanPrice"); vs = []
                if pe and pe > 0 and spe: vs.append(cl(50 + (spe / pe - 1) * 80))
                elif pe is not None and pe <= 0: vs.append(30.0)
                if tgt: vs.append(cl(50 + (tgt / last - 1) * 200))
                vs.append(float(valp.iloc[-1]) * 0.5 + 25)
                f["val"] = float(np.mean(vs))
                if tgt: fair = 0.6 * tgt + 0.4 * last * (1 + (f["val"] - 50) / 140)
                else: fair = last * (1 + (f["val"] - 50) / 140)
                dy = g("trailingAnnualDividendYield")
                if dy is None:
                    dy = g("dividendYield"); dy = dy / 100 if (dy is not None and dy > 0.2) else dy
                m.update(pe=num(pe, 1), sectorPE=num(spe, 1), revG=rg, epsG=eg, roe=roe, margin=pm, de=num(de / 100 if de else None, 2),
                         div=num(dy), target=tgt, analysts=info.get("numberOfAnalystOpinions"), rec=info.get("recommendationKey"),
                         beta=num(info.get("beta"), 2), mcap=info.get("marketCap"))
                a["fundOk"] = bool(fs)
                if info.get("longName") or info.get("shortName"):
                    a["name"] = info.get("longName") or info.get("shortName")
                if info.get("sector"): a["sector"] = info["sector"]
            a.update(f={k: round(v, 1) for k, v in f.items()}, fair=num(fair), m=m,
                     vol=num(dvol.iloc[-1] if not math.isnan(dvol.iloc[-1]) else 0.015, 5))
            if a["last"] is None:
                a["last"] = float(c.iloc[-1]); a["prev"] = float(c.iloc[-2]) if len(c) > 1 else a["last"]
                a["chg"] = num(a["last"] / a["prev"] - 1)

    def sector_pe(self, a):
        pes = [num(self.info.get(k, {}).get("trailingPE")) for k, b in self.assets.items()
               if b["mkt"] == a["mkt"] and b["sector"] == a["sector"]]
        pes = [p for p in pes if p and 0 < p < 200]
        if len(pes) < 3:
            pes = [num(self.info.get(k, {}).get("trailingPE")) for k, b in self.assets.items() if b["mkt"] == a["mkt"]]
            pes = [p for p in pes if p and 0 < p < 200]
        return float(np.median(pes)) if pes else None

    # ---------- backtest ----------
    def hist_decisions(self, y):
        fh = self.fhist.get(y)
        if fh is None: return None
        w = {k: SYS_W[k] for k in ("tech", "val", "macro", "risk")}; tot = sum(w.values())
        X = fh[list(w)]
        score = sum(X[k] * w[k] for k in w) / tot
        sd = X.std(axis=1, ddof=0)
        return score, sd

    def compute_track(self, days_td=63):
        """Backtest on real prices. For every asset we also measure a BASELINE:
        the same buy rule applied on every day, signal or not. A signal only has
        an edge if it works more often than that baseline."""
        out = {}; edge = {}
        for mkt in ("US", "SA", "ALT"):
            stats = {d: [] for d in ("BUY", "HOLD", "SELL")}; bands = {"90%+": [], "70–89%": [], "50–69%": [], "<50%": []}
            base_all = []
            for y, a in list(self.assets.items()):
                if a["mkt"] != mkt or y not in self.fhist: continue
                sc, sd = self.hist_decisions(y); fh = self.fhist[y]
                c = fh["c"].values; dv = fh["dvol"].values
                buys, base = [], []
                for i in range(120, len(c) - 10, 3):
                    dec, conf = decide(sc.iloc[i], sd.iloc[i])
                    rb = evaluate(c, dv, i, "BUY", days_td)
                    if rb is not None: base.append(rb)
                    r = rb if dec == "BUY" else evaluate(c, dv, i, dec, days_td)
                    if r is None: continue
                    if dec == "BUY": buys.append(r)
                    if i >= 200:
                        stats[dec].append(r)
                        b = "90%+" if conf >= 90 else "70–89%" if conf >= 70 else "50–69%" if conf >= 50 else "<50%"
                        bands[b].append(r[0])
                base_all += base
                if base:
                    edge[y] = {"n": len(buys), "hit": num(np.mean([x[0] for x in buys]) if buys else None, 3),
                               "avg": num(np.mean([x[1] for x in buys]) if buys else None, 4),
                               "base": num(np.mean([x[0] for x in base]), 3), "baseAvg": num(np.mean([x[1] for x in base]), 4)}
            out[mkt] = {"calls": [{"d": d, "n": len(v), "hit": num(np.mean([x[0] for x in v]) if v else None),
                                   "avg": num(np.mean([x[1] for x in v]) if v else None)} for d, v in stats.items()],
                        "baseline": {"n": len(base_all), "hit": num(np.mean([x[0] for x in base_all]) if base_all else None),
                                     "avg": num(np.mean([x[1] for x in base_all]) if base_all else None)},
                        "bands": [{"b": b, "n": len(v), "hit": num(np.mean(v) if v else None)} for b, v in bands.items()],
                        "days": days_td}
        with self.lock:
            self.track = out
            for y, e in edge.items():
                if y in self.assets: self.assets[y]["edge"] = e

    def similar(self, y, dec, score, days):
        sc_sd = self.hist_decisions(y)
        if sc_sd is None: return []
        sc, sd = sc_sd; fh = self.fhist[y]; c = fh["c"].values; dv = fh["dvol"].values
        days_td = max(5, int(round(days * 5 / 7)))
        out = []; last_i = None
        for i in range(len(c) - 6, 60, -1):
            d, conf = decide(sc.iloc[i], sd.iloc[i])
            if d != dec or abs(sc.iloc[i] - score) > 10: continue
            if last_i is not None and last_i - i < 15: continue
            r = evaluate(c, dv, i, dec, days_td)
            if r is None: continue
            last_i = i
            out.append({"date": fh.index[i].strftime("%Y-%m-%d"), "px": num(c[i]), "score": round(float(sc.iloc[i])),
                        "worked": bool(r[0]), "ret": num(r[1]), "held": int(r[2] * 7 / 5)})
            if len(out) >= 10: break
        return out

    # ---------- live quotes ----------
    def refresh_quotes(self, syms):
        if not syms: return
        h = self.p.history(syms, period="1d", interval="1m")
        now = time.time()
        with self.lock:
            for s, df in h.items():
                a = self.assets.get(s)
                if a is None: continue
                last = float(df["Close"].iloc[-1])
                day = df.index[-1].date()
                d = self.hist.get(s)
                prev = a["prev"]
                if d is not None and len(d) > 1:
                    prev = float(d["Close"].iloc[-2]) if d.index[-1].date() >= day else float(d["Close"].iloc[-1])
                a["last"] = last; a["prev"] = prev
                a["chg"] = num(last / prev - 1) if prev else None
                a["ts"] = df.index[-1].isoformat()
            self.status["quotesAt"] = now

    def load_info(self, y):
        a = self.assets[y]
        if a["mkt"] == "ALT": return
        inf = self.p.info(y)
        if inf:
            with self.lock: self.info[y] = inf

    def load_news(self, y):
        items = parse_news(self.p.news(y))
        with self.lock: self.news[y] = {"items": items, "at": time.time()}

    def add_symbol(self, raw):
        s = raw.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-=^]{1,15}", s): return None, "Invalid symbol"
        if s.isdigit(): s += ".SR"
        alias = {"BTC": "BTC-USD", "GOLD": "GC=F", "XAU": "GC=F", "SILVER": "SI=F", "XAG": "SI=F"}
        s = alias.get(s, s)
        if s in self.assets: return s, None
        h = self.p.history([s], period="2y", interval="1d")
        if s not in h or len(h[s]) < 30: return None, f"No price history found for {s}"
        sa = s.endswith(".SR")
        with self.lock:
            self._add(s, s.replace(".SR", ""), s, "Other", "SA" if sa else "US", "SAR" if sa else "USD")
            self.hist[s] = h[s]
        self.load_info(s); self.load_news(s)
        self.compute(s)
        return s, None

# --------------------------------------------------------------------------------------
# Background workers
# --------------------------------------------------------------------------------------
def us_open(now=None):
    ny = (now or datetime.now(timezone.utc)).astimezone(_tz("America/New_York"))
    m = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and 570 <= m < 960

def sa_open(now=None):
    ri = (now or datetime.now(timezone.utc)).astimezone(_tz("Asia/Riyadh"))
    m = ri.hour * 60 + ri.minute
    return ri.weekday() in (6, 0, 1, 2, 3) and 600 <= m < 900   # Sun–Thu 10:00–15:00

def _tz(name):
    from zoneinfo import ZoneInfo
    return ZoneInfo(name)

def worker(store):
    try:
        store.load_history()
        store.status.update(ready=True, phase="Prices loaded — fetching company financials & news", infoAt=time.time())
        all_syms = list(store.assets)
        store.refresh_quotes(all_syms)
    except Exception:
        log.exception("initial load failed"); store.status["phase"] = "Initial load failed — retrying"
        time.sleep(30); return worker(store)
    threading.Thread(target=fundamentals_worker, args=(store,), daemon=True).start()
    last_stock = {"US": 0, "SA": 0}; last_hist = time.time()
    while True:
        try:
            alts = [k for k, a in store.assets.items() if a["mkt"] == "ALT"]
            store.refresh_quotes(alts)
            for mkt, is_open in (("US", us_open()), ("SA", sa_open())):
                if is_open and time.time() - last_stock[mkt] > 60:
                    store.refresh_quotes([k for k, a in store.assets.items() if a["mkt"] == mkt]); last_stock[mkt] = time.time()
            if time.time() - last_hist > 1800:   # refresh daily history & scores every 30 min
                store.load_history(); last_hist = time.time()
        except Exception:
            log.exception("refresh loop error")
        time.sleep(15)

def fundamentals_worker(store):
    while True:
        keys = list(store.assets)
        for i, y in enumerate(keys):
            try:
                if y not in store.info: store.load_info(y)
                n = store.news.get(y)
                if not n or time.time() - n["at"] > 3600: store.load_news(y)
                store.compute(y)
                store.status["progress"] = round((i + 1) / len(keys) * 100)
            except Exception:
                log.exception("fundamentals %s", y)
            time.sleep(0.6)
        # re-score everything once sector P/E medians are complete
        for y in keys: store.compute(y)
        store.status["phase"] = "Live"
        time.sleep(600)
        if time.time() - store.status.get("infoAt", time.time()) > 43200:   # refresh financials twice a day
            with store.lock: store.info.clear()
            store.status["infoAt"] = time.time()

# --------------------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------------------
STORE = None

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj, default=lambda o: None).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        s = STORE
        try:
            if u.path in ("/", "/index.html"):
                b = open(os.path.join(HERE, "static", "index.html"), "rb").read()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            if u.path == "/api/status":
                st = dict(s.status); st.update(usOpen=us_open(), saOpen=sa_open(), now=time.time()); return self._json(st)
            if u.path == "/api/universe":
                with s.lock: return self._json([a for a in s.assets.values() if a["f"]])
            if u.path == "/api/quotes":
                with s.lock:
                    return self._json({"at": s.status["quotesAt"],
                                       "q": {k: [a["last"], a["chg"], a["ts"]] for k, a in s.assets.items() if a["last"]}})
            if u.path == "/api/track":
                return self._json(s.track.get(q.get("mkt", "US"), {}))
            if u.path == "/api/detail":
                y = q.get("k")
                if y not in s.assets: return self._json({"error": "unknown"}, 404)
                if y not in s.news: s.load_news(y); s.compute(y)
                with s.lock:
                    df = s.hist.get(y); fh = s.fhist.get(y)
                    c = df["Close"].iloc[-260:] if df is not None else pd.Series(dtype=float)
                    sma = fh["c"].rolling(50, min_periods=20).mean().iloc[-260:] if fh is not None else c
                    out = {"a": s.assets[y], "dates": [d.strftime("%Y-%m-%d") for d in c.index], "close": [num(x) for x in c],
                           "sma50": [num(x) for x in sma], "news": (s.news.get(y) or {}).get("items", [])}
                out["similar"] = s.similar(y, q.get("dec", "BUY"), float(q.get("score", 60)), int(float(q.get("days", 90))))
                return self._json(out)
            if u.path == "/api/add":
                y, err = s.add_symbol(q.get("symbol", ""))
                return self._json({"k": y, "asset": s.assets.get(y)} if y else {"error": err}, 200 if y else 400)
            self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            log.exception("request failed %s", self.path)
            try: self._json({"error": str(e)}, 500)
            except Exception: pass

def main():
    global STORE
    prov = FakeProvider() if PROVIDER == "fake" else YahooProvider()
    STORE = Store(prov)
    threading.Thread(target=worker, args=(STORE,), daemon=True).start()
    log.info("Investment Advisor on http://0.0.0.0:%d  (data: %s)", PORT, prov.name)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

if __name__ == "__main__":
    main()
