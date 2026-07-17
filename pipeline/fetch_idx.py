#!/usr/bin/env python3
"""
fetch_idx.py — IDX Dataset Fetcher (sumber gratis)

Sumber data:
- idx.co.id Ringkasan Saham : daftar emiten, OHLC harian, volume, foreign buy/sell
- Yahoo Finance (yfinance)  : fundamental, historical price 2 tahun
- Indikator teknikal        : dihitung lokal (pandas/numpy)

Output: dataset.json
"""

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
IDX_SUMMARY_URL = (
    "https://www.idx.co.id/primary/TradingSummary/GetStockSummary"
    "?length=9999&start=0"
)
HISTORY_PERIOD = "2y"
CHECKPOINT_EVERY = 25
RETRIES = 2
TICKER_RE = re.compile(r"^[A-Z]{4}$")  # saham biasa; skip waran/preferen

FUND_FIELDS = {
    "name": "longName",
    "sector": "sector",
    "industry": "industry",
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "per": "trailingPE",
    "forward_per": "forwardPE",
    "pbv": "priceToBook",
    "peg": "trailingPegRatio",
    "ev_ebitda": "enterpriseToEbitda",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "npm": "profitMargins",
    "gpm": "grossMargins",
    "opm": "operatingMargins",
    "der": "debtToEquity",
    "current_ratio": "currentRatio",
    "quick_ratio": "quickRatio",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "eps_growth_quarterly": "earningsQuarterlyGrowth",
    "eps_ttm": "trailingEps",
    "book_value_per_share": "bookValue",
    "dividend_yield": "dividendYield",
    "operating_cash_flow": "operatingCashflow",
    "free_cash_flow": "freeCashflow",
    "beta": "beta",
}


# ----------------------------------------------------------------------------
# Util
# ----------------------------------------------------------------------------
def clean(v):
    """Konversi nilai ke tipe JSON-safe; NaN/inf -> None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(v, (int, str, bool)):
        return v
    return str(v)


def last(series):
    """Nilai terakhir non-NaN dari Series -> float | None."""
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    return clean(v)


# ----------------------------------------------------------------------------
# Sumber 1: idx.co.id — daftar emiten + foreign flow
# ----------------------------------------------------------------------------
def fetch_idx_summary():
    """Return dict {kode: row} dari Ringkasan Saham IDX, atau None jika gagal."""
    if cloudscraper is None:
        print("[!] cloudscraper tidak terpasang, skip idx.co.id")
        return None
    try:
        scraper = cloudscraper.create_scraper()
        resp = scraper.get(IDX_SUMMARY_URL, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
        result = {}
        for raw in rows:
            row = {str(k).lower(): v for k, v in raw.items()}
            code = str(row.get("stockcode", "")).strip().upper()
            if TICKER_RE.match(code):
                result[code] = row
        print(f"[i] idx.co.id OK: {len(result)} emiten")
        return result or None
    except Exception as e:
        print(f"[!] idx.co.id gagal: {e}")
        return None


def foreign_flow_from(row):
    if not row:
        return None
    fb = clean(row.get("foreignbuy"))
    fs = clean(row.get("foreignsell"))
    net = (fb - fs) if (fb is not None and fs is not None) else None
    return {
        "foreign_buy": fb,
        "foreign_sell": fs,
        "foreign_net": clean(net),
        "volume": clean(row.get("volume")),
        "value": clean(row.get("value")),
        "frequency": clean(row.get("frequency")),
    }


# ----------------------------------------------------------------------------
# Sumber 2: Yahoo Finance — fundamental + historis
# ----------------------------------------------------------------------------
def fetch_yahoo(code):
    """Return (info_dict|None, history_df|None)."""
    t = yf.Ticker(f"{code}.JK")
    info, hist = None, None
    try:
        info = t.get_info() if hasattr(t, "get_info") else t.info
        if not isinstance(info, dict) or len(info) < 3:
            info = None
    except Exception:
        info = None
    try:
        hist = t.history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            hist = None
    except Exception:
        hist = None
    return info, hist


def extract_fundamentals(info):
    if not info:
        return None
    return {ours: clean(info.get(theirs)) for ours, theirs in FUND_FIELDS.items()}


# ----------------------------------------------------------------------------
# Indikator teknikal (dihitung lokal)
# ----------------------------------------------------------------------------
def wilder_ema(series, period):
    return series.ewm(alpha=1 / period, adjust=False).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = wilder_ema(delta.clip(lower=0), period)
    loss = wilder_ema(-delta.clip(upper=0), period)
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return wilder_ema(tr, period), tr


def compute_adx(df, period=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    _, tr = compute_atr(df, period)
    atr = wilder_ema(tr, period)
    plus_di = 100 * wilder_ema(plus_dm, period) / atr
    minus_di = 100 * wilder_ema(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder_ema(dx, period)


def detect_cross(fast, slow, lookback=10):
    """1 jika fast cross di atas slow dalam lookback bar terakhir, -1 jika cross bawah, 0 lainnya."""
    if fast is None or slow is None or len(fast) < lookback + 1:
        return 0
    diff = (fast - slow).dropna()
    if len(diff) < lookback + 1:
        return 0
    recent = np.sign(diff.iloc[-(lookback + 1):])
    changes = recent.diff().dropna()
    if (changes > 0).any() and recent.iloc[-1] > 0:
        return 1
    if (changes < 0).any() and recent.iloc[-1] < 0:
        return -1
    return 0


def pct_change_from(close, days):
    if len(close) <= days:
        return None
    base = close.iloc[-days - 1]
    return clean((close.iloc[-1] - base) / base) if base else None


def compute_technicals(df):
    if df is None or len(df) < 60:
        return None

    close, volume = df["Close"], df["Volume"]
    n = len(df)

    ema = {p: close.ewm(span=p, adjust=False).mean() for p in (20, 50, 100, 200)}
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean() if n >= 200 else None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    atr, _ = compute_atr(df)
    adx = compute_adx(df)

    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    typical = (df["High"] + df["Low"] + close) / 3
    vwap20 = (typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    daily_ret = close.pct_change().dropna()
    year = daily_ret.iloc[-252:] if len(daily_ret) > 252 else daily_ret
    cummax = close.iloc[-252:].cummax()
    drawdown = ((cummax - close.iloc[-252:]) / cummax).max()

    price = last(close)
    vol_avg20 = last(volume.rolling(20).mean())

    return {
        "price": price,
        "ema20": last(ema[20]),
        "ema50": last(ema[50]),
        "ema100": last(ema[100]),
        "ema200": last(ema[200]) if n >= 200 else None,
        "sma50": last(sma50),
        "sma200": last(sma200) if sma200 is not None else None,
        "macd": last(macd_line),
        "macd_signal": last(macd_signal),
        "macd_hist": last(macd_line - macd_signal),
        "rsi14": last(compute_rsi(close)),
        "stoch_k": last(stoch_k),
        "stoch_d": last(stoch_d),
        "atr14": last(atr),
        "adx14": last(adx),
        "obv": last(obv),
        "bb_upper": last(bb_mid + 2 * bb_std),
        "bb_mid": last(bb_mid),
        "bb_lower": last(bb_mid - 2 * bb_std),
        "vwap20": last(vwap20),
        "support_20d": clean(df["Low"].iloc[-20:].min()),
        "resistance_20d": clean(df["High"].iloc[-20:].max()),
        "support_60d": clean(df["Low"].iloc[-60:].min()),
        "resistance_60d": clean(df["High"].iloc[-60:].max()),
        "golden_cross_recent": detect_cross(sma50, sma200) if sma200 is not None else 0,
        "macd_cross_recent": detect_cross(macd_line, macd_signal),
        "change_1m": pct_change_from(close, 21),
        "change_3m": pct_change_from(close, 63),
        "change_6m": pct_change_from(close, 126),
        "change_1y": pct_change_from(close, 252),
        "volatility_annual": clean(year.std() * math.sqrt(252)),
        "max_drawdown_1y": clean(drawdown),
        "volume_avg20": vol_avg20,
        "volume_ratio": clean(last(volume) / vol_avg20) if vol_avg20 else None,
        "bars": n,
    }


# ----------------------------------------------------------------------------
# Orkestrasi
# ----------------------------------------------------------------------------
def fetch_one(code, idx_row, delay):
    for attempt in range(RETRIES + 1):
        try:
            info, hist = fetch_yahoo(code)
            fund = extract_fundamentals(info)
            tech = compute_technicals(hist)
            flow = foreign_flow_from(idx_row)
            complete = all([fund, tech])
            return {
                "kode": code,
                "nama": (fund or {}).get("name") or (idx_row or {}).get("stockname"),
                "fundamental": fund,
                "technical": tech,
                "foreign_flow": flow,
                "status": "OK" if complete else "Data Tidak Cukup",
            }
        except Exception as e:
            if attempt < RETRIES:
                time.sleep(delay * (attempt + 2))
            else:
                return {"kode": code, "status": "Data Tidak Cukup", "error": str(e)}


def load_tickers(args, idx_data):
    if args.tickers:
        return [c.strip().upper() for c in args.tickers.split(",") if c.strip()]
    if args.tickers_file:
        text = Path(args.tickers_file).read_text()
        return [c for c in re.findall(r"[A-Za-z]{4}", text.upper()) if TICKER_RE.match(c)]
    if idx_data:
        return sorted(idx_data.keys())
    sys.exit(
        "[x] Tidak ada daftar emiten. idx.co.id gagal dan tidak ada --tickers/--tickers-file.\n"
        "    Unduh daftar saham dari idx.co.id > Data Pasar > Daftar Saham, "
        "simpan sebagai CSV, lalu pakai --tickers-file."
    )


def main():
    ap = argparse.ArgumentParser(description="IDX dataset fetcher (sumber gratis)")
    ap.add_argument("--output", default="dataset.json")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah emiten (testing)")
    ap.add_argument("--delay", type=float, default=0.4, help="jeda antar emiten (detik)")
    ap.add_argument("--resume", action="store_true", help="lanjut dari checkpoint")
    ap.add_argument("--tickers", help="daftar kode dipisah koma, mis. BBCA,BBRI")
    ap.add_argument("--tickers-file", help="file berisi daftar kode emiten")
    args = ap.parse_args()

    checkpoint_path = Path(args.output + ".partial.json")
    results = {}
    if args.resume and checkpoint_path.exists():
        results = json.loads(checkpoint_path.read_text())
        print(f"[i] Resume: {len(results)} emiten sudah ada di checkpoint")

    idx_data = fetch_idx_summary()
    tickers = load_tickers(args, idx_data)
    if args.limit:
        tickers = tickers[: args.limit]

    todo = [c for c in tickers if c not in results]
    print(f"[i] Total {len(tickers)} emiten, sisa {len(todo)} akan di-fetch")

    for i, code in enumerate(todo, 1):
        results[code] = fetch_one(code, (idx_data or {}).get(code), args.delay)
        status = results[code]["status"]
        print(f"[{i}/{len(todo)}] {code}: {status}")
        if i % CHECKPOINT_EVERY == 0:
            checkpoint_path.write_text(json.dumps(results))
            print(f"[i] Checkpoint tersimpan ({len(results)} emiten)")
        time.sleep(args.delay)

    ok = sum(1 for r in results.values() if r.get("status") == "OK")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "prices_fundamentals": "Yahoo Finance (yfinance, unofficial)",
            "foreign_flow_volume": "idx.co.id Ringkasan Saham" if idx_data else "tidak tersedia",
        },
        "count_total": len(results),
        "count_ok": ok,
        "stocks": results,
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False))
    checkpoint_path.unlink(missing_ok=True)
    print(f"\n[✓] Selesai: {ok}/{len(results)} OK -> {args.output}")


if __name__ == "__main__":
    main()
