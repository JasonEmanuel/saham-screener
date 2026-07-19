#!/usr/bin/env python3
"""
evaluate.py — Forward-test sinyal BUY dari data/history/.

Tiap run: kumpulkan semua sinyal BUY historis, hitung return aktual
5 & 20 hari bursa setelah sinyal (dari snapshot histori sendiri),
bandingkan dengan IHSG (^JKSE, best-effort), lalu tulis
data/evaluation.json. Idempotent: selalu rekalkulasi penuh.

Definisi: win = return absolut > 0. Alpha = return - return IHSG
periode sama (dilaporkan terpisah). Evaluasi 1 hari sengaja tidak
dijadikan vonis — noise.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data")
HIST = DATA / "history"
OUT = DATA / "evaluation.json"
HORIZONS = (5, 20)


def load_history():
    idx = HIST / "index.json"
    if not idx.exists():
        return [], {}
    dates = json.loads(idx.read_text())
    snaps = {}
    for d in dates:
        p = HIST / f"{d}.json"
        if p.exists():
            try:
                snaps[d] = {r["kode"]: r for r in json.loads(p.read_text())}
            except Exception:
                pass
    return [d for d in dates if d in snaps], snaps


def ihsg_closes(dates):
    """Map tanggal histori -> close IHSG terdekat pada/sebelum tanggal itu."""
    try:
        import yfinance as yf
        h = yf.Ticker("^JKSE").history(period="1y", interval="1d", auto_adjust=True)
        if h is None or h.empty:
            return None
        series = sorted(
            (ts.strftime("%Y-%m-%d"), float(c)) for ts, c in h["Close"].items()
        )
        out = {}
        for d in dates:
            best = None
            for ds, c in series:
                if ds <= d:
                    best = c
                else:
                    break
            out[d] = best
        return out
    except Exception as e:
        print(f"[!] Data IHSG gagal: {e}")
        return None


def evaluate_horizon(sig_row, i, h, dates, snaps, ihsg):
    j = i + h
    if j >= len(dates):
        return None  # belum matang
    fut = snaps[dates[j]].get(sig_row["kode"])
    if not fut or not fut.get("harga"):
        return None
    p0, p1 = sig_row["harga"], fut["harga"]
    if not p0:
        return None
    ret = round((p1 - p0) / p0 * 100, 2)
    iret = None
    if ihsg and ihsg.get(dates[i]) and ihsg.get(dates[j]):
        iret = round((ihsg[dates[j]] - ihsg[dates[i]]) / ihsg[dates[i]] * 100, 2)
    return {
        "eval_date": dates[j],
        "return_pct": ret,
        "ihsg_pct": iret,
        "alpha_pct": None if iret is None else round(ret - iret, 2),
        "win": ret > 0,
    }


def summarize(signals, key):
    done = [s[key] for s in signals if s.get(key)]
    if not done:
        return {"n": 0}
    rets = [d["return_pct"] for d in done]
    alphas = [d["alpha_pct"] for d in done if d["alpha_pct"] is not None]
    wins = sum(d["win"] for d in done)
    return {
        "n": len(done),
        "wins": wins,
        "win_rate": round(100 * wins / len(done), 1),
        "avg_return": round(sum(rets) / len(rets), 2),
        "avg_alpha": round(sum(alphas) / len(alphas), 2) if alphas else None,
    }


def failure_patterns(signals):
    done = [s for s in signals if s.get("h5")]
    winners = [s for s in done if s["h5"]["win"]]
    losers = [s for s in done if not s["h5"]["win"]]

    def avg(group, field):
        vals = [s[field] for s in group if s.get(field) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def profile(group):
        if not group:
            return None
        return {"n": len(group), "total": avg(group, "total"),
                "fund": avg(group, "fund"), "tech": avg(group, "tech")}

    return {"basis": "5 hari", "winners": profile(winners), "losers": profile(losers)}


def main():
    dates, snaps = load_history()
    print(f"[i] Histori: {len(dates)} hari bursa")

    signals = []
    for i, d in enumerate(dates):
        for kode, r in snaps[d].items():
            if r.get("rating") != "BUY" or not r.get("harga"):
                continue
            signals.append({
                "kode": kode, "signal_date": d, "harga_signal": r["harga"],
                "total": r.get("total"), "fund": r.get("fund"), "tech": r.get("tech"),
            })

    ihsg = ihsg_closes(dates) if signals else None
    for s in signals:
        i = dates.index(s["signal_date"])
        for h in HORIZONS:
            s[f"h{h}"] = evaluate_horizon(s, i, h, dates, snaps, ihsg)

    signals.sort(key=lambda s: s["signal_date"], reverse=True)
    pending = sum(1 for s in signals if not s.get("h5"))

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "IHSG (^JKSE)" if ihsg else None,
        "days_collected": len(dates),
        "total_signals": len(signals),
        "pending": pending,
        "summary": {"h5": summarize(signals, "h5"), "h20": summarize(signals, "h20")},
        "patterns": failure_patterns(signals),
        "signals": signals[:200],
    }, ensure_ascii=False))
    s5 = summarize(signals, "h5")
    print(f"[✓] {len(signals)} sinyal | matang 5H: {s5.get('n',0)} "
          f"(win-rate {s5.get('win_rate','—')}%) | menunggu: {pending}")


if __name__ == "__main__":
    main()
