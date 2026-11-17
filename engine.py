"""
Investment Advisor: analytics engine (used by build.py)

Data: Yahoo Finance via `yfinance` (free; stock prices ~15 min delayed).
Set DATA_PROVIDER=fake to run with simulated data (for testing).

Model (v2):
  * Technical = 12-month momentum excluding the last month, plus long trend, minus 1-month reversal.
  * Stock factors are ranked as percentiles WITHIN their market (and sector, for valuation):
    50 = middle of the pack, 90 = better than 90% of peers.
  * Calls are judged against the market: a BUY "worked" if it beat the S&P 500 / TASI / EGX30
    over the holding period (Bitcoin, gold, silver are judged on absolute return).
"""
import os, json, time, math, threading, logging, random, re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROVIDER = os.environ.get("DATA_PROVIDER", "yahoo").lower()
HERE = os.path.dirname(os.path.abspath(__file__))
SAR_PER_USD = 3.75
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("advisor")
HISTORY_PERIOD = "5y"

# --------------------------------------------------------------------------------------
# Universe (add more in watchlist.txt)
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

EG = [("COMI","Commercial International Bank (CIB)","Financial Services"),("HRHO","EFG Holding","Financial Services"),
      ("TMGH","Talaat Moustafa Group","Real Estate"),("SWDY","Elsewedy Electric","Industrials"),("EAST","Eastern Company","Consumer Defensive"),
      ("ETEL","Telecom Egypt","Communication Services"),("ABUK","Abu Qir Fertilizers","Basic Materials"),("FWRY","Fawry","Technology"),
      ("ORAS","Orascom Construction","Industrials"),("AMOC","Alexandria Mineral Oils","Energy"),("JUFO","Juhayna Food","Consumer Defensive"),
      ("ESRS","Ezz Steel","Basic Materials"),("MFPC","Misr Fertilizers (MOPCO)","Basic Materials"),("PHDC","Palm Hills Developments","Real Estate"),
      ("HELI","Heliopolis Housing","Real Estate"),("CIEB","Credit Agricole Egypt","Financial Services"),("ADIB","Abu Dhabi Islamic Bank Egypt","Financial Services"),
      ("EMFD","Emaar Misr","Real Estate"),("ORWE","Oriental Weavers","Consumer Cyclical"),("SKPC","Sidi Kerir Petrochemicals","Basic Materials"),
      ("CLHO","Cleopatra Hospitals","Healthcare"),("ISPH","Ibnsina Pharma","Healthcare"),("EFID","Edita Food","Consumer Defensive"),
      ("OCDI","SODIC","Real Estate"),("MASR","Madinet Masr","Real Estate"),("GBCO","GB Corp","Consumer Cyclical"),("BTFH","Beltone Holding","Financial Services")]
ALT = [("BTC-USD","BTC","Bitcoin","Crypto"),("GC=F","XAU","Gold (oz)","Precious metals"),("SI=F","XAG","Silver (oz)","Precious metals")]
MACRO = {"spx":"^GSPC","ndx":"^IXIC","vix":"^VIX","tnx":"^TNX","dxy":"DX-Y.NYB","oil":"CL=F","tasi":"^TASI.SR","egx":"^CASE30","egp":"EGP=X"}
BENCH = {"US": "^GSPC", "SA": "^TASI.SR", "EG": "^CASE30"}
BENCH_NAME = {"US": "S&P 500", "SA": "TASI", "EG": "EGX30", "ALT": "cash"}
STOCK_MKTS = ("US", "SA", "EG")
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
    def history(self, syms, period="5y", interval="1d"):
        """Download in small batches; retry missing symbols (Yahoo often throttles GitHub's servers)."""
        out = {}
        todo = list(syms)
        for attempt in range(4):
            if not todo: break
            if attempt: time.sleep(8 * attempt); log.info("retry %d for %d symbols", attempt, len(todo))
            for i in range(0, len(todo), 20):
                chunk = todo[i:i+20]
                try:
                    df = self.yf.download(chunk, period=period, interval=interval, group_by="ticker",
                                          auto_adjust=True, threads=attempt == 0, progress=False)
                except Exception as e:
                    log.warning("download failed %s: %s", chunk[:3], e); time.sleep(3); continue
                for s in chunk:
                    sub = _extract(df, s, len(chunk))
                    if sub is not None and len(sub) > 5: out[s] = sub
                time.sleep(1)
            todo = [s for s in syms if s not in out]
        if todo: log.warning("no data for %d symbols: %s", len(todo), ", ".join(todo[:30]))
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
    def __init__(self): self.base = {}
    def _seed(self, s): return abs(hash(s)) % (2**32)
    def _start(self, s):
        if s not in self.base:
            r = random.Random(self._seed(s))
            self.base[s] = {"^VIX":16,"^TNX":4.2,"DX-Y.NYB":104,"BTC-USD":90000,"GC=F":2650,"SI=F":31,"EGP=X":51}.get(s, r.uniform(15, 600))
        return self.base[s]
    def history(self, syms, period="5y", interval="1d"):
        out = {}
        for s in syms:
            r = np.random.default_rng(self._seed(s))
            n = 1250; idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
            vol = 0.012 if not s.startswith("^") else 0.008
            if s == "EGP=X": vol = 0.004
            steps = r.normal(r.normal(0.0003, 0.0006), vol, n)
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
# Analytics
# --------------------------------------------------------------------------------------
def cl(x, lo=0.0, hi=100.0):
    return x.clip(lo, hi) if isinstance(x, (pd.Series, pd.DataFrame, np.ndarray)) else float(min(hi, max(lo, x)))

def num(x, nd=4):
    try:
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)
    except Exception:
        return None

def tech_series(c):
    """Raw technical signal: 12-1 month momentum (main), long trend, minus last-month reversal.
    Returns (absolute 0-100 score, raw value, display metrics)."""
    sma50 = c.rolling(50, min_periods=20).mean(); sma200 = c.rolling(200, min_periods=60).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50)
    mom12_1 = c.shift(21) / c.shift(252) - 1
    mom6_1 = c.shift(21) / c.shift(126) - 1
    mom1 = c / c.shift(21) - 1
    trend = c / sma200 - 1
    raw = 0.6 * mom12_1.fillna(mom6_1) + 0.25 * trend - 0.15 * mom1
    score = cl(50 + raw * 120).fillna(50)
    return score, raw, {"sma50": sma50, "sma200": sma200, "rsi": rsi, "mom": c / c.shift(63) - 1, "mom12_1": mom12_1, "mom1": mom1}

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

def pct_rank(values):
    """dict key->value  ->  dict key->percentile 0-100 within the group (ties averaged). None stays None."""
    ks = [k for k, v in values.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    out = {k: None for k in values}
    if len(ks) < 3: return out
    r = pd.Series({k: values[k] for k in ks}).rank()
    for k in ks: out[k] = float((r[k] - 0.5) / len(ks) * 100)
    return out

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

def evaluate(c, b, dvol, i, dec, days_td):
    """Walk forward from day i using the page's target/stop rules.
    Returns (worked, excess return vs benchmark, days held, own return).
    BUY worked = beat the benchmark; SELL worked = lagged it; HOLD worked = not more than 3% behind."""
    n = min(days_td, len(c) - 1 - i)
    if n < 5: return None
    p = c[i]; v = dvol[i] if not math.isnan(dvol[i]) else 0.015; sw = v * math.sqrt(5)
    j_exit = n
    if dec == "BUY":
        tgt = p * (1 + max(0.6 * v * math.sqrt(days_td), 0.03)); stp = p * (1 - 2.2 * sw)
        for j in range(1, n + 1):
            if c[i + j] >= tgt or c[i + j] <= stp: j_exit = j; break
    elif dec == "SELL":
        tgt = p * (1 - max(0.5 * v * math.sqrt(days_td), 0.03)); stp = p * (1 + 2 * sw)
        for j in range(1, n + 1):
            if c[i + j] <= tgt or c[i + j] >= stp: j_exit = j; break
    ret = c[i + j_exit] / p - 1
    bret = (b[i + j_exit] / b[i] - 1) if (b is not None and b[i] and not math.isnan(b[i]) and not math.isnan(b[i + j_exit])) else 0.0
    ex = ret - bret
    worked = ex > 0 if dec == "BUY" else ex < 0 if dec == "SELL" else ex > -0.03
    return bool(worked), ex, j_exit, ret

def shrink_p(wins, n, k=60):
    """Probability of beating the index, pulled toward 50% when there are few cases."""
    return (wins + 0.5 * k) / (n + k)

def decide_p(p, base):
    """BUY only if clearly better than this market's usual chance of beating the index (and above 50%);
    SELL if clearly worse (or below 45%). Everything else is HOLD."""
    if p >= max(0.5, base + 0.04): return "BUY"
    if p <= min(0.45, base - 0.04) or p <= base - 0.06: return "SELL"
    return "HOLD"

def calib_lookup(calib, score):
    if not calib: return {"p": 0.5, "ex": 0.0, "n": 0}
    for b in calib:
        if score <= b["hi"]: return b
    return calib[-1]

# --------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------
class Store:
    def __init__(self, prov):
        self.p = prov; self.lock = threading.RLock()
        self.assets = {}; self.hist = {}; self.fhist = {}
        self.info = {}; self.news = {}; self.macro = {}; self.track = {}
        for s, n, sec in US: self._add(s, s, n, sec, "US", "USD")
        for s, n, sec in SA: self._add(s + ".SR", s, n, sec, "SA", "SAR")
        for s, n, sec in EG: self._add(s + ".CA", s, n, sec, "EG", "EGP")
        for y, s, n, sec in ALT: self._add(y, s, n, sec, "ALT", "USD")

    def _add(self, y, sym, name, sector, mkt, ccy):
        self.assets[y] = {"k": y, "sym": sym, "name": name, "sector": sector, "mkt": mkt, "ccy": ccy,
                          "last": None, "prev": None, "chg": None, "vol": 0.015, "f": None, "fAbs": None, "fair": None,
                          "m": {}, "fundOk": False, "ts": None, "edge": None}

    def load_history(self):
        syms = list(self.assets) + list(MACRO.values())
        h = self.p.history(syms, period=HISTORY_PERIOD, interval="1d")
        with self.lock:
            for s, df in h.items():
                if s in MACRO.values(): self.macro[s] = df["Close"].astype(float)
                else: self.hist[s] = df
        for s in list(self.assets):
            if s in h: self.compute(s)
        self.rank_all()
        self.compute_track()

    def bench(self, a, index):
        s = self.macro.get(BENCH.get(a["mkt"], ""))
        if s is None or a["mkt"] == "ALT": return pd.Series(1.0, index=index)
        return s.reindex(index, method="ffill")

    def macro_parts(self, a):
        g = self.macro.get
        mkt, sec = a["mkt"], a["sector"]
        if mkt == "US":
            parts = [(0.6, regime(g("^GSPC"))), (0.4, cl(100 - (g("^VIX") - 12) * 3.5) if g("^VIX") is not None else None)]
        elif mkt == "SA":
            parts = [(0.6, regime(g("^TASI.SR"))), (0.4, regime(g("CL=F")))]
            if g("^TASI.SR") is None: parts.append((0.3, regime(g("^GSPC"))))
        elif mkt == "EG":
            parts = [(0.6, regime(g("^CASE30"))), (0.4, falling(g("EGP=X"), 600))]
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
        elif a["mkt"] == "EG":
            trend(g("^CASE30"), "EGX30")
            e = g("EGP=X")
            if e is not None and len(e) > 60:
                ch = e.iloc[-1] / e.iloc[-63] - 1
                bits.append(f"Egyptian pound {'weakened' if ch > 0 else 'strengthened'} {abs(ch)*100:.1f}% vs USD in 3 months")
        elif a["sym"] == "BTC":
            trend(g("^IXIC"), "Nasdaq"); trend(g("DX-Y.NYB"), "US dollar")
        else:
            if g("^TNX") is not None: bits.append(f"10-yr US yield {g('^TNX').iloc[-1]:.2f}%")
            trend(g("DX-Y.NYB"), "US dollar")
        if a["mkt"] != "ALT" and a["sector"] in ("Energy", "Basic Materials"): trend(g("CL=F"), "Oil")
        return "; ".join(dict.fromkeys(bits))

    def compute(self, y):
        """Absolute factor scores + metrics for one asset (ranking happens in rank_all)."""
        with self.lock:
            a = self.assets.get(y); df = self.hist.get(y)
            if a is None or df is None or len(df) < 60: return
            c = df["Close"].astype(float)
            tech, traw, tm = tech_series(c)
            risk, dvol, avol, dd = risk_series(c)
            valp = valproxy_series(c)
            macro = blend(self.macro_parts(a), c.index)
            fh = pd.DataFrame({"tech": tech, "traw": traw, "val": valp, "macro": macro, "risk": risk,
                               "dvol": dvol, "c": c, "b": self.bench(a, c.index)})
            old = self.fhist.get(y)
            if old is not None:   # keep previously computed ranks if present
                for col in ("tech_r", "val_r", "risk_r"):
                    if col in old: fh[col] = old[col].reindex(fh.index)
            self.fhist[y] = fh
            last = float(c.iloc[-1])
            info = self.info.get(y, {}); m = {}
            m.update(rsi=num(tm["rsi"].iloc[-1], 1), sma50=num(tm["sma50"].iloc[-1]), sma200=num(tm["sma200"].iloc[-1]),
                     mom3m=num(tm["mom"].iloc[-1]), mom12=num(tm["mom12_1"].iloc[-1]), mom1=num(tm["mom1"].iloc[-1]),
                     hi52=num(c.iloc[-252:].max()), annVol=num(avol.iloc[-1]), dd=num(dd.iloc[-1]), bench=BENCH_NAME[a["mkt"]])
            f = {"tech": float(tech.iloc[-1]), "risk": float(risk.iloc[-1])}
            news = self.news.get(y); ns = None
            if news and news["items"]: ns = float(np.mean([n["sent"] for n in news["items"]]))
            f["macro"] = cl(float(macro.iloc[-1]) + (ns or 0) * 15)
            m.update(macroTxt=self.macro_text(a), newsSent=num(ns, 2), newsN=len(news["items"]) if news else 0)
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
                f["fund"] = float(np.mean(fs)) if fs else None
                pe = g("trailingPE") or g("forwardPE")
                spe = self.sector_pe(a)
                tgt = g("targetMeanPrice"); vs = []
                if pe and pe > 0 and spe: vs.append(cl(50 + (spe / pe - 1) * 80))
                elif pe is not None and pe <= 0: vs.append(30.0)
                if tgt: vs.append(cl(50 + (tgt / last - 1) * 200))
                vs.append(float(valp.iloc[-1]) * 0.5 + 25)
                f["val"] = float(np.mean(vs))
                fair = (0.6 * tgt + 0.4 * last * (1 + (f["val"] - 50) / 140)) if tgt else last * (1 + (f["val"] - 50) / 140)
                dy = g("trailingAnnualDividendYield")
                if dy is None:
                    dy = g("dividendYield"); dy = dy / 100 if (dy is not None and dy > 0.2) else dy
                m.update(pe=num(pe, 1), sectorPE=num(spe, 1), revG=rg, epsG=eg, roe=roe, margin=pm, de=num(de / 100 if de else None, 2),
                         div=num(dy), target=tgt, analysts=info.get("numberOfAnalystOpinions"), rec=info.get("recommendationKey"),
                         beta=num(info.get("beta"), 2), mcap=info.get("marketCap"))
                a["fundOk"] = bool(fs)
                if info.get("longName") or info.get("shortName"): a["name"] = info.get("longName") or info.get("shortName")
                if info.get("sector"): a["sector"] = info["sector"]
            a["fAbs"] = {k: (round(v, 1) if v is not None else None) for k, v in f.items()}
            if a["f"] is None or a["mkt"] == "ALT":
                a["f"] = {k: (round(v, 1) if v is not None else 50.0) for k, v in f.items()}
            a.update(fair=num(fair), m=m, vol=num(dvol.iloc[-1] if not math.isnan(dvol.iloc[-1]) else 0.015, 5))
            a["last"] = last; a["prev"] = float(c.iloc[-2]) if len(c) > 1 else last
            a["chg"] = num(a["last"] / a["prev"] - 1); a["ts"] = c.index[-1].strftime("%Y-%m-%d")

    def rank_all(self):
        """Turn stock factors into percentiles within each market (valuation within sector when possible),
        both for today and for every historical day (so the backtest uses the same model)."""
        with self.lock:
            for mkt in STOCK_MKTS:
                keys = [y for y, a in self.assets.items() if a["mkt"] == mkt and y in self.fhist and a["fAbs"]]
                if len(keys) < 5: continue
                # historical ranks (tech raw, valuation proxy, risk) across the market on each day
                for col, out in (("traw", "tech_r"), ("val", "val_r"), ("risk", "risk_r")):
                    M = pd.DataFrame({y: self.fhist[y][col] for y in keys})
                    cnt = M.notna().sum(axis=1)
                    R = M.rank(axis=1, pct=True) * 100
                    R[cnt < 5] = np.nan
                    for y in keys: self.fhist[y][out] = R[y].reindex(self.fhist[y].index)
                # today's ranks
                tech_now = pct_rank({y: num(self.fhist[y]["traw"].iloc[-1]) for y in keys})
                risk_now = pct_rank({y: self.assets[y]["fAbs"]["risk"] for y in keys})
                fund_now = pct_rank({y: self.assets[y]["fAbs"]["fund"] for y in keys})
                val_now = {}
                by_sec = {}
                for y in keys: by_sec.setdefault(self.assets[y]["sector"], []).append(y)
                mval = pct_rank({y: self.assets[y]["fAbs"]["val"] for y in keys})
                for sec, ys in by_sec.items():
                    sv = pct_rank({y: self.assets[y]["fAbs"]["val"] for y in ys}) if len(ys) >= 5 else {}
                    for y in ys: val_now[y] = 0.6 * sv[y] + 0.4 * mval[y] if sv.get(y) is not None and mval.get(y) is not None else mval.get(y)
                n = len(keys)
                for y in keys:
                    a = self.assets[y]; fa = a["fAbs"]
                    f = {"tech": tech_now[y] if tech_now[y] is not None else fa["tech"],
                         "fund": fund_now[y] if fund_now[y] is not None else 50.0,
                         "val": val_now.get(y) if val_now.get(y) is not None else fa["val"],
                         "risk": risk_now[y] if risk_now[y] is not None else fa["risk"],
                         "macro": fa["macro"]}
                    a["f"] = {k: round(float(v), 1) for k, v in f.items()}
                    a["m"]["peers"] = n

    # ---------- evidence-based model: weights, calibration, backtest ----------
    TESTED = ("tech", "val", "macro", "risk")      # factors with history; fundamentals can't be tested on free data

    def hist_X(self, y):
        fh = self.fhist.get(y)
        if fh is None: return None
        cols = {"tech": "tech_r" if "tech_r" in fh else "tech", "val": "val_r" if "val_r" in fh else "val",
                "macro": "macro", "risk": "risk_r" if "risk_r" in fh else "risk"}
        return pd.DataFrame({k: (fh[c].fillna(fh[k]) if c != k else fh[k]) for k, c in cols.items()})

    def market_weights(self, mkt, keys, H=63):
        """Weights from evidence: average cross-sectional rank correlation (IC) between each factor and the
        next 3 months' return vs the index, sampled weekly. Negative or zero ICs get no extra weight;
        the result is blended 50/50 with equal weights so one noisy period can't dominate."""
        ics = {k: [] for k in self.TESTED}
        if mkt != "ALT" and len(keys) >= 8:
            fwd = pd.DataFrame({y: (self.fhist[y]["c"].shift(-H) / self.fhist[y]["c"] - 1)
                                   - (self.fhist[y]["b"].shift(-H) / self.fhist[y]["b"] - 1) for y in keys})
            Xs = {y: self.hist_X(y) for y in keys}
            for k in self.TESTED:
                F = pd.DataFrame({y: Xs[y][k] for y in keys})
                idx = F.index.intersection(fwd.index)
                for d in idx[260:-H:5]:
                    f, r = F.loc[d], fwd.loc[d]
                    ok = f.notna() & r.notna()
                    if ok.sum() >= 8:
                        c = f[ok].rank().corr(r[ok].rank())
                        if not math.isnan(c): ics[k].append(c)
        ic = {k: (float(np.mean(v)) if v else 0.0) for k, v in ics.items()}
        pos = {k: max(v, 0) for k, v in ic.items()}; tot = sum(pos.values())
        w = {k: 0.5 * 0.25 + 0.5 * (pos[k] / tot if tot > 0 else 0.25) for k in self.TESTED}
        w["fund"] = float(np.mean(list(w.values())))         # untested: given the average weight, no more
        t = sum(w.values())
        return {k: round(v / t * 100, 1) for k, v in w.items()}, {k: round(v, 3) for k, v in ic.items()}, {k: len(v) for k, v in ics.items()}

    def compute_track(self, days_td=63):
        """Backtest judged against each market's index, with evidence-based weights and a calibration table:
        for each score band, how often did the asset beat its index over the next 3 months?"""
        out = {}; edge = {}
        for mkt in ("US", "SA", "EG", "ALT"):
            keys = [y for y, a in self.assets.items() if a["mkt"] == mkt and y in self.fhist]
            if not keys: continue
            W, IC, ICn = self.market_weights(mkt, keys, days_td)
            tw = {k: W[k] for k in self.TESTED}; tt = sum(tw.values())
            samples = []                                   # (asset, i, score, fwd excess)
            per = {}
            for y in keys:
                X = self.hist_X(y); fh = self.fhist[y]
                sc = sum(X[k] * tw[k] for k in tw) / tt
                c = fh["c"].values; b = fh["b"].values
                per[y] = (sc, c, b, fh["dvol"].values)
                for i in range(260, len(c) - days_td, 3):
                    if math.isnan(sc.iloc[i]) or not b[i] or math.isnan(b[i + days_td]): continue
                    ex = (c[i + days_td] / c[i] - 1) - (b[i + days_td] / b[i] - 1)
                    samples.append((y, i, float(sc.iloc[i]), ex))
            calib = []
            base_rate = float(np.mean([x[3] > 0 for x in samples])) if samples else 0.5
            if len(samples) >= 50:
                scores = np.array([s[2] for s in samples]); exs = np.array([s[3] for s in samples])
                edges = np.unique(np.quantile(scores, np.linspace(0, 1, 11)))
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (scores >= lo) & (scores <= hi)
                    n = int(m.sum())
                    if n == 0: continue
                    wins = float((exs[m] > 0).sum())
                    calib.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": n,
                                  "raw": round(wins / n, 3), "p": round(shrink_p(wins, n), 3), "ex": round(float(np.mean(exs[m])), 4)})
            stats = {d: [] for d in ("BUY", "HOLD", "SELL")}; bands = {"+8 pts or more": [], "+4 to +8 pts": [], "−4 to +4 pts": [], "−4 pts or less": []}
            base_all = []
            for y in keys:
                sc, c, b, dv = per[y]; buys, base = [], []
                for i in range(260, len(c) - 10, 3):
                    if math.isnan(sc.iloc[i]): continue
                    p = calib_lookup(calib, float(sc.iloc[i]))["p"] if calib else 0.5
                    dec = decide_p(p, base_rate)
                    rb = evaluate(c, b, dv, i, "BUY", days_td)
                    if rb is not None: base.append(rb)
                    r = rb if dec == "BUY" else evaluate(c, b, dv, i, dec, days_td)
                    if r is None: continue
                    if dec == "BUY": buys.append(r)
                    stats[dec].append(r)
                    d = p - base_rate
                    k = "+8 pts or more" if d >= 0.08 else "+4 to +8 pts" if d >= 0.04 else "−4 to +4 pts" if d > -0.04 else "−4 pts or less"
                    bands[k].append(bool(r[1] > 0))   # actually beat the index?
                base_all += base
                if base:
                    edge[y] = {"n": len(buys), "hit": num(np.mean([x[0] for x in buys]) if buys else None, 3),
                               "avg": num(np.mean([x[1] for x in buys]) if buys else None, 4),
                               "base": num(np.mean([x[0] for x in base]), 3), "baseAvg": num(np.mean([x[1] for x in base]), 4)}
            out[mkt] = {"calls": [{"d": d, "n": len(v), "hit": num(np.mean([x[0] for x in v]) if v else None),
                                   "avg": num(np.mean([x[1] for x in v]) if v else None)} for d, v in stats.items()],
                        "baseline": {"n": len(base_all), "hit": num(np.mean([x[0] for x in base_all]) if base_all else None),
                                     "avg": num(np.mean([x[1] for x in base_all]) if base_all else None)},
                        "bands": [{"b": k, "n": len(v), "hit": num(np.mean(v) if v else None)} for k, v in bands.items()],
                        "days": days_td, "bench": BENCH_NAME[mkt], "weights": W, "ic": IC, "icN": ICn,
                        "calib": calib, "samples": len(samples), "assets": len(keys), "base": round(base_rate, 3)}
        with self.lock:
            self.track = out
            for y, e in edge.items():
                if y in self.assets: self.assets[y]["edge"] = e

    # ---------- financials & news ----------
    def sector_pe(self, a):
        pes = [num(self.info.get(k, {}).get("trailingPE")) for k, b in self.assets.items()
               if b["mkt"] == a["mkt"] and b["sector"] == a["sector"]]
        pes = [p for p in pes if p and 0 < p < 200]
        if len(pes) < 3:
            pes = [num(self.info.get(k, {}).get("trailingPE")) for k, b in self.assets.items() if b["mkt"] == a["mkt"]]
            pes = [p for p in pes if p and 0 < p < 200]
        return float(np.median(pes)) if pes else None

    def load_info(self, y):
        if self.assets[y]["mkt"] == "ALT": return
        inf = self.p.info(y)
        if inf:
            with self.lock: self.info[y] = inf

    def load_news(self, y):
        items = parse_news(self.p.news(y))
        with self.lock: self.news[y] = {"items": items, "at": time.time()}

# --------------------------------------------------------------------------------------
# Server-side copy of the page's decision rule (system weights, "Months" horizon),
# used to log the daily top picks for the live scorecard.
# --------------------------------------------------------------------------------------
def analyse_sys(a, track):
    """Independent mode (same as the page): evidence-based weights + calibrated probability."""
    t = track.get(a["mkt"], {}); w = t.get("weights") or SYS_W; tot = sum(w.values())
    f = a["f"]; score = sum(f[k] * w[k] for k in w) / tot
    cb = calib_lookup(t.get("calib"), score); p = cb["p"]
    dec = decide_p(p, t.get("base", 0.5))
    return {"score": round(score), "dec": dec, "conf": round(p * 100), "p": p}

def top_picks(assets, track, mkt=None, n=10):
    order = {"BUY": 0, "HOLD": 1, "SELL": 2}
    rows = []
    for a in assets.values():
        if not a.get("f") or not a.get("last"): continue
        if mkt and a["mkt"] != mkt: continue
        x = analyse_sys(a, track); rows.append((a, x))
    rows.sort(key=lambda r: (order[r[1]["dec"]], -r[1]["p"], -r[1]["score"]))
    return rows[:n]
