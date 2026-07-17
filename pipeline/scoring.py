#!/usr/bin/env python3
"""
scoring.py — Scoring & ranking deterministik dari data/dataset.json

Bobot asli prompt: Fundamental 40, Technical 30, News 10, Sentiment 10,
Valuation 5, Risk 5. News & Sentiment = "Data Tidak Cukup" (tidak ada
sumber gratis) -> bobot dinormalisasi otomatis ke komponen tersedia.

Output (folder data/):
  all_scores.json  seluruh emiten + skor, terurut
  top20.json       kandidat BUY teratas
  watchlist.json   hampir lolos + alasan
  avoid.json       fundamental & teknikal buruk
  categories.json  top growth / value / dividend / momentum / foreign buy
  meta.json        timestamp, sumber, disclaimer
"""

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")

WEIGHTS = {"fundamental": 40, "technical": 30, "valuation": 5, "risk": 5}
MIN_DAILY_VALUE = 1_000_000_000  # likuiditas minimum: rata-rata 1 M IDR/hari

BUY_TOTAL, BUY_FUND, BUY_TECH = 80, 75, 75
WATCH_TOTAL = 70


# ---------------------------------------------------------------- helpers ---
def scale(v, lo, hi):
    """Linear clamp ke 0..1. None-safe."""
    if v is None:
        return None
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def inv(x):
    return None if x is None else 1.0 - x


TEXT_KEYS = {"name", "sector", "industry"}


def to_num(v):
    """Paksa nilai jadi float. String angka dikonversi; 'Infinity'/teks lain -> None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except ValueError:
            return None
    return None


def sanitize(d):
    """Bersihkan dict: semua nilai non-teks dipaksa numerik atau None."""
    if not d:
        return d
    return {
        k: (v if (k in TEXT_KEYS or isinstance(v, list)) else to_num(v))
        for k, v in d.items()
    }


def weighted(parts):
    """parts: [(skor01|None, bobot)] -> 0-100. Komponen None diabaikan,
    bobot dinormalisasi ke yang tersedia."""
    total_w = sum(w for s, w in parts if s is not None)
    if total_w == 0:
        return None
    return round(100 * sum(s * w for s, w in parts if s is not None) / total_w, 1)


def sector_medians(stocks):
    buckets = {}
    for s in stocks:
        f = s.get("fundamental") or {}
        sec = f.get("sector") or "Unknown"
        b = buckets.setdefault(sec, {"per": [], "pbv": [], "roe": [], "npm": []})
        for k in b:
            v = f.get(k)
            if isinstance(v, (int, float)) and v > 0:
                b[k].append(v)
    return {
        sec: {k: (statistics.median(vs) if vs else None) for k, vs in b.items()}
        for sec, b in buckets.items()
    }


# ---------------------------------------------------------------- skor ------
def score_fundamental(f, med):
    if not f:
        return None
    roe, npm = f.get("roe"), f.get("npm")
    parts = [
        (scale(f.get("revenue_growth"), -0.10, 0.30), 12),
        (scale(f.get("earnings_growth"), -0.10, 0.40), 12),
        (scale(roe, 0.0, 0.25), 14),
        (scale(f.get("roa"), 0.0, 0.12), 6),
        (scale(npm, 0.0, 0.25), 8),
        (scale(f.get("opm"), 0.0, 0.25), 5),
        (inv(scale(f.get("der"), 0, 200)), 10),  # yfinance: DER dalam persen
        (scale(f.get("current_ratio"), 0.8, 2.5), 6),
        (1.0 if (f.get("free_cash_flow") or 0) > 0 else 0.0, 8),
        (scale(f.get("dividend_yield"), 0.0, 0.06), 4),
    ]
    if med:  # relatif terhadap median sektor
        if roe is not None and med.get("roe"):
            parts.append((scale(roe / med["roe"], 0.5, 2.0), 8))
        if npm is not None and med.get("npm"):
            parts.append((scale(npm / med["npm"], 0.5, 2.0), 7))
    return weighted(parts)


def score_technical(t):
    if not t:
        return None
    p = t.get("price")

    def above(key):
        v = t.get(key)
        if p is None or v is None:
            return None
        return 1.0 if p > v else 0.0

    rsi = t.get("rsi14")
    if rsi is None:
        rsi_score = None
    elif rsi < 30:
        rsi_score = 0.35   # oversold — potensi reversal, trend lemah
    elif rsi < 45:
        rsi_score = 0.50
    elif rsi <= 65:
        rsi_score = 1.00   # zona sehat
    elif rsi <= 75:
        rsi_score = 0.60
    else:
        rsi_score = 0.20   # overbought

    def cross(key):
        v = t.get(key)
        return 1.0 if v == 1 else (0.0 if v == -1 else 0.5)

    parts = [
        (above("ema20"), 8),
        (above("ema50"), 8),
        (above("ema100"), 6),
        (above("ema200"), 8),
        (1.0 if (t.get("macd_hist") or 0) > 0 else 0.0, 10),
        (rsi_score, 10),
        (scale(t.get("adx14"), 10, 40), 8),
        (cross("golden_cross_recent"), 8),
        (cross("macd_cross_recent"), 6),
        (scale(t.get("change_3m"), -0.15, 0.30), 8),
        (scale(t.get("volume_ratio"), 0.5, 2.0), 6),
    ]

    pats = t.get("patterns") or []
    bull = sum(p in ("Golden Cross", "Breakout", "Double Bottom", "Bull Flag") for p in pats)
    bear = sum(p in ("Death Cross", "Breakdown") for p in pats)
    if pats:
        parts.append((max(0.0, min(1.0, 0.5 + 0.25 * bull - 0.4 * bear)), 8))

    return weighted(parts)


def fair_value(f, med, price):
    """Estimasi fair value: rata-rata (EPS x PER sektor) & (BV x PBV sektor).
    PER dicap 25, PBV dicap 4 — hindari fair value liar di sektor mahal."""
    if not f or not price:
        return None
    candidates = []
    eps, bv = f.get("eps_ttm"), f.get("book_value_per_share")
    if eps and eps > 0 and med and med.get("per"):
        candidates.append(eps * min(med["per"], 25))
    if bv and bv > 0 and med and med.get("pbv"):
        candidates.append(bv * min(med["pbv"], 4))
    if not candidates:
        return None
    return round(sum(candidates) / len(candidates), 2)


def score_valuation(f, med, fv, price):
    if not f:
        return None
    parts = []
    per, pbv, peg = f.get("per"), f.get("pbv"), f.get("peg")
    if per and per > 0 and med and med.get("per"):
        parts.append((inv(scale(per / med["per"], 0.5, 1.8)), 30))
    if pbv and pbv > 0 and med and med.get("pbv"):
        parts.append((inv(scale(pbv / med["pbv"], 0.5, 1.8)), 25))
    if peg and peg > 0:
        parts.append((inv(scale(peg, 0.5, 2.5)), 20))
    if fv and price:
        mos = (fv - price) / fv  # margin of safety
        parts.append((scale(mos, -0.10, 0.40), 25))
    return weighted(parts)


def score_risk(f, t):
    """Skor tinggi = risiko rendah."""
    if not t:
        return None
    parts = [
        (inv(scale((f or {}).get("beta"), 0.6, 1.8)), 25),
        (inv(scale(t.get("volatility_annual"), 0.20, 0.80)), 30),
        (inv(scale(t.get("max_drawdown_1y"), 0.10, 0.60)), 25),
    ]
    p, v = t.get("price"), t.get("volume_avg20")
    if p and v:
        parts.append((scale(p * v, 5e8, 5e10), 20))  # nilai transaksi harian
    return weighted(parts)


# ---------------------------------------------------------------- rakit -----
def build_row(s, med_all):
    f = sanitize(dict(s.get("fundamental") or {}))
    t = sanitize(s.get("technical") or {})
    flow = sanitize(s.get("foreign_flow") or {})

    # normalisasi dividend yield: versi yfinance baru pakai persen
    dy = f.get("dividend_yield")
    if dy and dy > 1:
        f["dividend_yield"] = dy / 100

    sec = f.get("sector") or "Unknown"
    med = med_all.get(sec)
    price = t.get("price")

    fs = score_fundamental(f, med)
    ts = score_technical(t)
    fv = fair_value(f, med, price)
    vs = score_valuation(f, med, fv, price)
    rs = score_risk(f, t)

    comps = {"fundamental": fs, "technical": ts, "valuation": vs, "risk": rs}
    wsum = sum(WEIGHTS[k] for k, v in comps.items() if v is not None)
    total = (
        round(sum(v * WEIGHTS[k] for k, v in comps.items() if v is not None) / wsum, 1)
        if wsum
        else None
    )

    # likuiditas
    vol20 = t.get("volume_avg20")
    daily_value = price * vol20 if (price and vol20) else None
    liquid = bool(daily_value and daily_value >= MIN_DAILY_VALUE)

    # trade plan
    atr = t.get("atr14")
    sl = t.get("support_20d")
    if price and sl and sl >= price * 0.99:  # support nempel/di atas harga
        sl = round(price - 2 * atr, 2) if atr else None
    tp1, tp2 = t.get("resistance_20d"), t.get("resistance_60d")
    tp3 = fv if (fv and price and fv > price) else None
    upside = round(100 * (fv - price) / price, 1) if (fv and price) else None

    rr = None
    if price and sl and tp1 and price > sl:
        rr = round((tp1 - price) / (price - sl), 2)

    # confidence: kelengkapan data
    key_fields = [
        f.get("roe"), f.get("per"), f.get("revenue_growth"), f.get("der"),
        t.get("ema200"), t.get("rsi14"), t.get("adx14"), flow.get("foreign_net"),
    ]
    filled = sum(1 for x in key_fields if x is not None)
    confidence = "High" if filled >= 7 else ("Medium" if filled >= 5 else "Low")

    # rating + alasan
    reasons = []
    if total is None or fs is None or ts is None:
        rating = "Data Tidak Cukup"
    elif total >= BUY_TOTAL and fs >= BUY_FUND and ts >= BUY_TECH and liquid:
        rating = "BUY"
    elif fs < 40 and ts < 40:
        rating = "AVOID"
        reasons.append("Fundamental lemah dan teknikal bearish")
    elif total >= WATCH_TOTAL:
        rating = "WATCHLIST"
        if fs < BUY_FUND:
            reasons.append(f"Fundamental {fs} < {BUY_FUND}")
        if ts < BUY_TECH:
            reasons.append(f"Technical {ts} < {BUY_TECH}")
        if total < BUY_TOTAL:
            reasons.append(f"Total {total} < {BUY_TOTAL}")
        if not liquid:
            reasons.append("Likuiditas di bawah ambang 1 M IDR/hari")
    else:
        rating = "NEUTRAL"

    return {
        "kode": s.get("kode"),
        "nama": s.get("nama"),
        "sektor": sec,
        "harga": price,
        "market_cap": f.get("market_cap"),
        "fair_value": fv,
        "upside_pct": upside,
        "score_fundamental": fs,
        "score_technical": ts,
        "score_news": None,        # Data Tidak Cukup
        "score_sentiment": None,   # Data Tidak Cukup
        "score_valuation": vs,
        "score_risk": rs,
        "total_score": total,
        "rating": rating,
        "rating_reasons": reasons,
        "confidence": confidence,
        "liquid": liquid,
        "daily_value_avg": round(daily_value) if daily_value else None,
        "foreign_net": flow.get("foreign_net"),
        "support": t.get("support_20d"),
        "resistance": t.get("resistance_20d"),
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "take_profit_3": tp3,
        "stop_loss": sl,
        "risk_reward": rr,
        "rsi": t.get("rsi14"),
        "sma20": t.get("sma20"),
        "sma50": t.get("sma50"),
        "sma200": t.get("sma200"),
        "macd_hist": t.get("macd_hist"),
        "stochrsi_k": t.get("stochrsi_k"),
        "stochrsi_d": t.get("stochrsi_d"),
        "volume_ratio": t.get("volume_ratio"),
        "patterns": t.get("patterns") or [],
        "change_3m": t.get("change_3m"),
        "dividend_yield": f.get("dividend_yield"),
        "earnings_growth": f.get("earnings_growth"),
    }


def top_by(rows, key, n=10, extra_filter=None):
    pool = [r for r in rows if r.get(key) is not None and r["liquid"]]
    if extra_filter:
        pool = [r for r in pool if extra_filter(r)]
    pool.sort(key=lambda r: r[key], reverse=True)
    return pool[:n]


def main():
    ds = json.loads((DATA / "dataset.json").read_text())
    stocks = [s for s in ds["stocks"].values() if s.get("status") == "OK"]
    skipped = len(ds["stocks"]) - len(stocks)

    med_all = sector_medians(stocks)
    rows = [build_row(s, med_all) for s in stocks]
    rows = [r for r in rows if r["total_score"] is not None]
    rows.sort(key=lambda r: r["total_score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    buys = [r for r in rows if r["rating"] == "BUY"][:20]
    watch = [r for r in rows if r["rating"] == "WATCHLIST"][:30]
    avoid = sorted(
        [r for r in rows if r["rating"] == "AVOID"],
        key=lambda r: r["total_score"],
    )[:20]

    categories = {
        "top_growth": top_by(rows, "earnings_growth"),
        "top_value": top_by(rows, "score_valuation"),
        "top_dividend": top_by(rows, "dividend_yield"),
        "top_momentum": top_by(rows, "change_3m"),
        "top_foreign_buy": top_by(rows, "foreign_net"),
    }

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_generated_at": ds.get("generated_at"),
        "sources": ds.get("sources"),
        "count_scored": len(rows),
        "count_skipped": skipped,
        "count_buy": len(buys),
        "weights_note": (
            "News & Sentiment: Data Tidak Cukup (tanpa sumber gratis). "
            "Bobot dinormalisasi: Fundamental/Technical/Valuation/Risk = 40/30/5/5."
        ),
        "disclaimer": (
            "Hasil screening kuantitatif otomatis dari data publik, bukan "
            "nasihat keuangan. Lakukan riset mandiri sebelum berinvestasi."
        ),
    }

    (DATA / "all_scores.json").write_text(json.dumps(rows, ensure_ascii=False))
    (DATA / "top20.json").write_text(json.dumps(buys, ensure_ascii=False))
    (DATA / "watchlist.json").write_text(json.dumps(watch, ensure_ascii=False))
    (DATA / "avoid.json").write_text(json.dumps(avoid, ensure_ascii=False))
    (DATA / "categories.json").write_text(json.dumps(categories, ensure_ascii=False))
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))

    print(f"[✓] Scored {len(rows)} | BUY {len(buys)} | WATCH {len(watch)} | AVOID {len(avoid)}")


if __name__ == "__main__":
    main()
