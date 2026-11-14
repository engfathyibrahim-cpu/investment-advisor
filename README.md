# Investment Advisor

Buy / hold / sell suggestions for US stocks, Saudi (Tadawul) stocks, Bitcoin, gold and silver,
scored on technical, fundamental, valuation, market-and-news and risk factors, with entry price,
target, stop-loss, holding period, staged buy/sell plan, confidence, justification, similar past
signals and a backtest of the model's track record. Prices in USD and SAR.

How it works: GitHub Actions runs `build.py` about every 10 minutes. It downloads free Yahoo Finance
data (stock prices typically ~15 min delayed), scores everything and publishes the site to GitHub Pages.

* Track more stocks: edit `watchlist.txt`.
* Force an update now: Actions tab → "Update advisor" → Run workflow.
* Test locally without internet: `DATA_PROVIDER=fake python build.py && python -m http.server -d site 8000`

Not financial advice.
