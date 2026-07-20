#!/usr/bin/env python3
"""market.py — Regime pasar (IHSG) + tren komoditas & USDIDR via yfinance.

Output data/market.json: regime_score 0-100, label, tailwind per sektor.
Best-effort: kegagalan satu aset tidak menggagalkan run.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

DATA = Path("data")
DATA.mkdir(exist_ok=True)
OUT = DATA / "market.json"

ASSETS = {
    "ihsg": "^JKSE",
    "usdidr": "IDR=X",
    "oil": "CL=F",
    "gas": "NG=F",
    "gold": "GC=F",
    "copper": "HG=F",
}
# proxy komoditas per sektor yfinance (batubara/CPO tidak tersedia di Yahoo)
SECTOR_PROXY = {
    "Energy": ["oil", "gas"],
    "Basic Materials": ["gold", "copper"],
}


def stats(ticker):
    try:
        h = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        c = h["Close"].dropna()
        if len(c) < 60:
            return None
        last = float(c.iloc[-1])

        def ret(n):
            return float((c.iloc[-1] - c.iloc[-n - 1]) / c.iloc[-n - 1]) if len(c) > n else None

        ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
        return {
            "last": round(last, 2),
            "ret_1m": ret(21),
            "ret_3m": ret(63),
            "above_ma200": (last > ma200) if ma200 is not None else None,
            "vol_annual": round(float(c.pct_change().dropna().std() * math.sqrt(252)), 3),
        }
    except Exception as e:
        print(f"[!] {ticker}: {e}")
        return None


def main():
    d = {k: stats(v) for k, v in ASSETS.items()}
    ihsg = d.get("ihsg") or {}

    score = 50.0
    if ihsg.get("above_ma200") is True:
        score += 20
    elif ihsg.get("above_ma200") is False:
        score -= 20
    for key, w in (("ret_1m", 15), ("ret_3m", 15)):
        r = ihsg.get(key)
        if r is not None:
            score += max(-w, min(w, r / 0.08 * w))
    score = round(max(0.0, min(100.0, score)), 1)
    label = "bullish" if score >= 65 else ("bearish" if score < 35 else "netral")

    tail = {}
    for sec, keys in SECTOR_PROXY.items():
        rs = [d[k]["ret_3m"] for k in keys if d.get(k) and d[k].get("ret_3m") is not None]
        if rs:
            tail[sec] = round(sum(rs) / len(rs), 4)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "regime_score": score,
        "regime_label": label,
        "assets": d,
        "sector_tailwind": tail,
    }, ensure_ascii=False))
    print(f"[✓] Regime {score} ({label}) | tailwind sektor: {tail}")


if __name__ == "__main__":
    main()
