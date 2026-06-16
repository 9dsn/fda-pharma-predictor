"""Event-driven backtest
Reads your LOOCV predictions + ticker map, pulls stock prices,
and simulates a long/short trading strategy around FDA AdCom votes.
Runs in two modes: small-cap only vs everything."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROB_PATH   = "data/processed/loocv_probabilities.csv"
TICKER_MAP  = "data/processed/ticker_map.csv"
PRICE_CACHE = Path("data/raw/prices")
OUT         = Path("data/processed")
PLOT_DIR    = OUT / "plots"
BENCHMARK   = "XBI"
LONG_THRESH  = 0.65
SHORT_THRESH = 0.35


def load_events(mode="smallcap"):
    probs = pd.read_csv(PROB_PATH)
    probs["meeting_date"] = pd.to_datetime(probs["meeting_date"])
    tmap  = pd.read_csv(TICKER_MAP)
    tmap  = tmap.loc[:, ~tmap.columns.duplicated()]
    tmap["meeting_date"] = pd.to_datetime(tmap["meeting_date"])
    tmap["ticker"] = tmap["ticker"].fillna("").str.strip().str.upper()
    tmap  = tmap[tmap["ticker"] != ""]

    merged = probs.merge(
        tmap[["meeting_date", "ticker", "is_big_pharma"]],
        on="meeting_date", how="inner"
    )
    if mode == "smallcap":
        merged = merged[merged["is_big_pharma"] == 0].copy()

    merged["meeting_date"] = pd.to_datetime(merged["meeting_date"])
    events = merged.groupby(["ticker", "meeting_date"], as_index=False).agg(
        prob_yes=("prob_yes", "mean"),
        n_votes=("prob_yes", "size"),
    )
    print(f"[{mode}] {len(merged)} rows -> {len(events)} unique positions")
    return events.sort_values("meeting_date").reset_index(drop=True)


def get_prices(ticker, event_date):
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = PRICE_CACHE / f"{ticker}.csv"
    start = (event_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end   = (event_date + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
        if cached.index.min() <= pd.Timestamp(start) and cached.index.max() >= pd.Timestamp(end):
            return cached
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        result = pd.DataFrame({"Close": close})
        result.index.name = "Date"
        result.to_csv(cache_file)
        time.sleep(0.3)
        return result
    except Exception as e:
        print(f"  ! {ticker}: {e}")
        return None


def event_return(prices, event_date, entry_offset=-2, exit_offset=1):
    if prices is None or prices.empty:
        return None
    entry_target = event_date + pd.Timedelta(days=entry_offset)
    exit_target  = event_date + pd.Timedelta(days=exit_offset)
    before = prices.index[prices.index <= entry_target]
    after  = prices.index[prices.index >= exit_target]
    if len(before) == 0 or len(after) == 0:
        return None
    return (float(prices.loc[after[0], "Close"]) - float(prices.loc[before[-1], "Close"])) / float(prices.loc[before[-1], "Close"])


def run_backtest(mode):
    events = load_events(mode)
    trades = []
    for _, row in events.iterrows():
        ticker     = row["ticker"]
        event_date = row["meeting_date"]
        prob       = row["prob_yes"]

        if SHORT_THRESH <= prob <= LONG_THRESH:
            continue

        stock_ret = event_return(get_prices(ticker, event_date), event_date)
        bench_ret = event_return(get_prices(BENCHMARK, event_date), event_date)

        if stock_ret is None:
            print(f"  skip {ticker} {event_date.date()} — no price data")
            continue

        adj_return   = stock_ret - (bench_ret or 0.0)
        direction    = 1 if prob >= LONG_THRESH else -1
        trade_return = direction * adj_return

        trades.append({
            "ticker":       ticker,
            "event_date":   event_date.date(),
            "prob_yes":     round(prob, 3),
            "direction":    "long" if direction == 1 else "short",
            "stock_return": round(stock_ret, 4),
            "adj_return":   round(adj_return, 4),
            "trade_return": round(trade_return, 4),
        })
    return pd.DataFrame(trades)


def compute_metrics(trades):
    if len(trades) == 0:
        return {"error": "no trades"}
    r  = trades["trade_return"].values
    eq = np.cumprod(1 + r)
    dates = pd.to_datetime(trades["event_date"])
    years = max((dates.max() - dates.min()).days / 365.25, 0.5)
    sharpe = (r.mean() / r.std() * np.sqrt(len(r) / years)) if r.std() > 0 else 0.0
    peak   = np.maximum.accumulate(eq)
    return {
        "n_trades":          int(len(r)),
        "total_return_pct":  round((eq[-1] - 1) * 100, 2),
        "sharpe_annualized": round(sharpe, 3),
        "max_drawdown_pct":  round(float(((eq - peak) / peak).min()) * 100, 2),
        "hit_rate_pct":      round(float((r > 0).mean()) * 100, 1),
        "mean_trade_pct":    round(float(r.mean()) * 100, 2),
    }


def plot_equity(trades, mode):
    if len(trades) == 0:
        return
    eq     = np.cumprod(1 + trades["trade_return"].values)
    labels = [f"{row.ticker}\n{row.event_date}" for _, row in trades.iterrows()]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(eq) + 1), eq, marker="o", linewidth=2)
    plt.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Breakeven")
    plt.xticks(range(1, len(eq) + 1), labels, fontsize=7, rotation=15)
    plt.title(f"Equity curve — {mode}")
    plt.ylabel("Portfolio value")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"equity_{mode}.png", dpi=130)
    plt.close()


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for mode in ["smallcap", "all"]:
        print(f"\n{'='*50}\nMODE: {mode}\n{'='*50}")
        trades  = run_backtest(mode)
        metrics = compute_metrics(trades)
        trades.to_csv(OUT / f"backtest_trades_{mode}.csv", index=False)
        print(f"\n{trades.to_string(index=False)}")
        print(f"\nMetrics:\n{json.dumps(metrics, indent=2)}")
        plot_equity(trades, mode)
        summary[mode] = metrics

    with open(OUT / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    sc, al = summary["smallcap"], summary["all"]
    writeup = f"""
RESULTS
===============

Small-cap only ({sc.get('n_trades', 0)} trades):
  Total return:  {sc.get('total_return_pct', 'N/A')}%
  Sharpe ratio:  {sc.get('sharpe_annualized', 'N/A')}
  Hit rate:      {sc.get('hit_rate_pct', 'N/A')}%
  Max drawdown:  {sc.get('max_drawdown_pct', 'N/A')}%

All tradeable ({al.get('n_trades', 0)} trades):
  Total return:  {al.get('total_return_pct', 'N/A')}%
  Sharpe ratio:  {al.get('sharpe_annualized', 'N/A')}
  Hit rate:      {al.get('hit_rate_pct', 'N/A')}%
  Max drawdown:  {al.get('max_drawdown_pct', 'N/A')}%

n={sc.get('n_trades', 0)}"""
    with open(OUT / "backtest_writeup.txt", "w") as f:
        f.write(writeup)
    print(writeup)


if __name__ == "__main__":
    main()