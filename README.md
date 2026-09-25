# Market Signal Explorer

A personal research dashboard for stocks and crypto. It combines technical
indicators, news sentiment, and a small machine-learning model into a
BUY / SELL / HOLD read — plus a backtester and basic risk-management tools
(position sizing, stop-losses, diversification checks) so you can size any
trade sensibly and see realistic historical performance before trusting it.

## Read this first

- **This cannot turn $500 into $100,000 in a month — nothing legitimate
  can.** That's a 200x return in 30 days. No combination of news, technical
  indicators, "trending tech," or macro data makes that achievable
  *safely*. A tool that promised it would either be lying or one bad trade
  from wiping you out. See "Realistic expectations" below, and run the
  backtester yourself — it'll show you honestly.
- This is a decision-support and learning tool. It is not a broker, not a
  bot, and not a person. It does not place trades for you, and it holds no
  login or withdrawal permissions to any exchange or broker — it only ever
  reads public price and news data.
- Nothing here is licensed financial advice. For advice tailored to your
  situation, talk to an AFSL-licensed financial adviser.

## What it does

- Pulls price history for stocks (via Yahoo Finance) and crypto (via
  CoinGecko), for tickers/coins worldwide.
- Computes standard technical indicators: moving averages, RSI, MACD,
  Bollinger Bands, ATR.
- Scores recent news headlines for sentiment (via NewsAPI + VADER), once
  you add a free NewsAPI key.
- Trains a small, honestly-evaluated machine learning model on each asset's
  own history — tested only on data it never trained on, and reported next
  to a naive baseline, so you can see whether it's really adding anything.
- Combines the above into a single BUY/SELL/HOLD read with a plain-English
  "why" and a confidence level — never a guarantee.
- Suggests a stop-loss, take-profit, and position size based on how much
  you say you're willing to risk per trade.
- Backtests the technical rules against real history.

## What it deliberately doesn't do

- No hardcoded "war score" or "season score." Geopolitics, sector trends,
  and seasonality are real, but they're diffuse and largely already
  reflected in price action and news sentiment — faking a precise number
  for them would look sophisticated while adding noise, not accuracy.
- No automatic trade execution. You stay in control of every buy and sell.
- No backtesting of the news-sentiment component: free news APIs don't give
  enough historical depth to test that part honestly, so the backtest and
  the ML model both rely on price/technical history only.

## Setup

Requires Python 3.10+.

**macOS/Linux**
```bash
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
streamlit run app.py
```

**Windows**
```bat
setup.bat
.venv\Scripts\activate
streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501) — open it
in your browser.

### No computer? Run it as a hosted web app instead

Everything above needs a terminal, so it needs a computer. If you only have
a phone, you can host this for free instead and just open a URL from then
on -- done entirely from a mobile browser, no local install:

1. Unzip this project on your phone (most file managers can extract a
   .zip).
2. Create a GitHub account (free) if you don't have one, then create a new
   repository and upload all the extracted files to it. The repo can be
   private -- Streamlit Cloud can deploy from private repos too.
3. Go to https://share.streamlit.io, sign in with GitHub, and click
   "Create app" -> "I have an app" -> point it at your repo, branch `main`,
   file path `app.py`.
4. Before deploying, open "Advanced settings" and paste this into the
   Secrets box (with your real key):
   ```
   NEWSAPI_KEY = "your_key_here"
   ```
5. Deploy. You'll get a permanent link like
   `yourname-market-signal-explorer.streamlit.app` that opens the app in
   any mobile browser, any time -- no reinstalling anything.

`config.py` already checks Streamlit Cloud's secrets manager first and
falls back to `.env`, so the same code works either way without changes.
Note: a free app can go to sleep after a period of no visits and take a
few seconds to wake up on your next visit -- normal, not a bug. Keep the
repo/app private if you don't want the URL discoverable by others; the
free tier supports that.

### News sentiment (optional but recommended)
1. Get a free key at https://newsapi.org/register
2. Copy `.env.example` to `.env`
3. Paste your key into `NEWSAPI_KEY=`

Everything else works without this key; you'll just miss the sentiment
component. The free plan allows 100 requests/day with articles delayed up
to 24 hours — plenty for occasional personal use, but it's a "development"
key, not meant for a public production app.

## Security

- API keys live only in your local `.env` file, loaded by `python-dotenv`.
  `.env` is already in `.gitignore` — it won't be committed if you push
  this to GitHub.
- The app only ever uses read-only, public market-data endpoints. It never
  asks for exchange or broker trading/withdrawal keys — don't add any, and
  be suspicious of any "trading app" that does ask for them.
- Runs inside its own virtual environment (`.venv`), isolated from your
  system Python.
- Before pushing to GitHub, run `git status` and confirm `.env` isn't
  listed.
- If you ever turn this into a public, multi-user website instead of a
  local tool, it would need real authentication and HTTPS first — as a
  local single-user app, it doesn't.

## Realistic expectations (the "senior broker" section)

- A very good year for a professional fund manager is roughly 15-25% —
  and that's with a team, real capital, and years of infrastructure. Treat
  any plan that assumes far more than that as a red flag.
- Run the backtester and the ML model on a few tickers/coins. Notice the
  strategy rarely beats simple buy-and-hold by much, drawdowns happen, and
  the model's accuracy is usually close to its naive baseline. That's
  normal — it's also the honest answer to "how much can this really make
  me."
- Turning $500 into anything close to $100k fast usually requires leverage
  or a concentrated, all-in bet — which is also exactly how small accounts
  go to zero. If hitting your target seems to require leverage or an
  all-in position, that's the market telling you something about the
  target, not the tool.
- A more realistic goal: don't lose the $500 while you learn. Keep position
  sizes small (this tool defaults to risking 2% of capital per trade),
  diversify rather than concentrate, avoid margin/leverage/CFDs on a small
  account, and let gains compound over months, not days.
- Fees matter more on a small account. Pick a low-fee, ASIC-regulated
  Australian broker/exchange, and don't over-trade — every buy and sell has
  a cost that eats into a $500 balance fast.
- In Australia, profits from shares and crypto are generally subject to
  capital gains tax. Keep records of every trade and talk to a tax
  professional, or check ato.gov.au.
- If you start losing money while chasing the original target, the right
  move is to stop and reassess, not add more money to "catch up."

## Project layout

- `app.py` — Streamlit dashboard (entry point)
- `data_stocks.py` / `data_crypto.py` — price history fetchers
- `indicators.py` — technical indicators
- `sentiment.py` — news fetch + VADER scoring
- `ml_model.py` — small classifier + honest out-of-sample evaluation
- `signal_engine.py` — combines indicators + sentiment into BUY/SELL/HOLD
- `risk_manager.py` — position sizing, stop-loss/take-profit, diversification
- `backtester.py` — historical simulation + performance stats

## Limitations

- Signals are heuristics based on public data. They are frequently wrong.
- Backtests and the ML model show past performance only; markets change,
  and past patterns are not guaranteed to repeat. Neither models fees or
  slippage.
- News search is a simple keyword match on the ticker/symbol — for common
  words this can pull in unrelated headlines.
- CoinGecko's free tier (no key needed) allows ~30 calls/minute; NewsAPI's
  free tier allows 100 requests/day. If requests start failing, wait a bit.
- yfinance depends on Yahoo Finance's public site, which rate-limits
  aggressively at times and occasionally changes its internal API. If
  fetches start failing, run `pip install --upgrade yfinance` and try
  again after a short wait — this is a known, common, and usually
  temporary issue, not something specific to this app.
- This analyzes one asset at a time — it doesn't track a live, multi-asset
  portfolio.
