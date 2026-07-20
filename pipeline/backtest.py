#!/usr/bin/env python3
"""
backtest.py — Backtest sinyal TEKNIKAL 2 tahun ke belakang.

Cakupan jujur: hanya komponen teknikal (fundamental & berita historis
tidak tersedia gratis). Universe: emiten paling likuid dari all_scores.json.
Sinyal diuji: Golden Cross, MACD cross saat uptrend, Breakout 60D + volume.
Pembanding: return rata-rata "beli acak" di universe yang sama.

Output: data/backtest.json. Jalankan manual via workflow "Backtest Teknikal".
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yfinance as yf

DATA = Path("data")
OUT = DATA / "backtest.json"
MAX_TICKERS = 250
DELAY = 0.35


def pick_tickers():
    p = DATA / "all_scores.json"
    if not p.exists():
        return []
    rows = json.loads(p.read_text())
    liq = [r for r in rows if r.get("liquid") and r.get("daily_value_avg")]
    liq.sort(key=lambda r: r["daily_value_avg"], reverse=True)
    return [r["kode"] for r in liq[:MAX_TICKERS]]


def signals_from(df):
    close, vol = df["Close"], df["Volume"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hi60 = df["High"].rolling(60).max().shift(1)
    vol20 = vol.rolling(20).mean()

    gc = sma50 > sma200
    macd_up = macd > sig
    out = []
    for i in range(200, len(df) - 20):
        if gc.iloc[i] and not gc.iloc[i - 1]:
            out.append((i, "Golden Cross"))
        if (macd_up.iloc[i] and not macd_up.iloc[i - 1]
                and close.iloc[i] > sma200.iloc[i]):
            out.append((i, "MACD Cross (uptrend)"))
        if (hi60.iloc[i] == hi60.iloc[i] and close.iloc[i] > hi60.iloc[i]
                and close.iloc[i - 1] <= hi60.iloc[i - 1]
                and vol20.iloc[i] == vol20.iloc[i]
                and vol.iloc[i] > 1.5 * vol20.iloc[i]):
            out.append((i, "Breakout 60D + volume"))
    return out


def agg(rets):
    if not rets:
        return {"n": 0}
    wins = sum(r > 0 for r in rets)
    return {
        "n": len(rets),
        "win_rate": round(100 * wins / len(rets), 1),
        "avg_return_pct": round(100 * sum(rets) / len(rets), 2),
        "median_return_pct": round(100 * float(np.median(rets)), 2),
    }


def main():
    tickers = pick_tickers()
    if not tickers:
        print("[x] data/all_scores.json tidak ada — jalankan pipeline harian dulu")
        return
    print(f"[i] Backtest {len(tickers)} emiten paling likuid, periode 2 tahun")

    stats = {}
    base5, base20 = [], []
    for n, code in enumerate(tickers, 1):
        try:
            df = yf.Ticker(f"{code}.JK").history(
                period="2y", interval="1d", auto_adjust=True)
            if df is None or len(df) < 260:
                continue
            close = df["Close"]
            fwd5 = close.pct_change(5).shift(-5).dropna()
            fwd20 = close.pct_change(20).shift(-20).dropna()
            if len(fwd5):
                base5.append(float(fwd5.mean()))
            if len(fwd20):
                base20.append(float(fwd20.mean()))
            for i, jenis in signals_from(df):
                s = stats.setdefault(jenis, {"r5": [], "r20": []})
                p0 = float(close.iloc[i])
                s["r5"].append((float(close.iloc[i + 5]) - p0) / p0)
                s["r20"].append((float(close.iloc[i + 20]) - p0) / p0)
        except Exception as e:
            print(f"[!] {code}: {e}")
        if n % 25 == 0:
            print(f"[{n}/{len(tickers)}]")
        time.sleep(DELAY)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": len(tickers),
        "period": "2y",
        "baseline_beli_acak": {
            "avg_5d_pct": round(100 * sum(base5) / len(base5), 2) if base5 else None,
            "avg_20d_pct": round(100 * sum(base20) / len(base20), 2) if base20 else None,
        },
        "signals": {
            k: {"h5": agg(v["r5"]), "h20": agg(v["r20"])}
            for k, v in stats.items()
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\n=== HASIL ===")
    print(f"Baseline beli acak: 5H {result['baseline_beli_acak']['avg_5d_pct']}% | "
          f"20H {result['baseline_beli_acak']['avg_20d_pct']}%")
    for k, v in result["signals"].items():
        print(f"{k}: n={v['h5']['n']} | 5H win {v['h5'].get('win_rate','—')}% "
              f"avg {v['h5'].get('avg_return_pct','—')}% | "
              f"20H win {v['h20'].get('win_rate','—')}% "
              f"avg {v['h20'].get('avg_return_pct','—')}%")


if __name__ == "__main__":
    main()
