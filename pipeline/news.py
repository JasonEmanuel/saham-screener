#!/usr/bin/env python3
"""
news.py — Analisa berita & sentimen emiten IDX dari RSS media finansial.

Alur: fetch RSS -> cocokkan artikel ke kode emiten -> skor keyword
(event & tone) -> gabung ke arsip 30 hari -> agregasi per emiten
dengan bobot recency -> tulis data/news.json.

Semua feed bersifat best-effort: yang mati dilewati tanpa menggagalkan run.
"""

import json
import re
import socket
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import feedparser

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

socket.setdefaulttimeout(20)

DATA = Path("data")
DATA.mkdir(exist_ok=True)
ARCHIVE = DATA / "news_archive.json"
OUT = DATA / "news.json"
TICKERS_FILE = Path("pipeline/tickers.txt")

MAX_AGE_DAYS = 30
HALF_LIFE_DAYS = 7.0  # berita 7 hari lalu bobotnya separuh berita hari ini

GN = "https://news.google.com/rss/search?q={}&hl=id&gl=ID&ceid=ID:id"
FEEDS = [
    GN.format("emiten+saham"),
    GN.format("saham+IDX+laba"),
    GN.format("saham+dividen"),
    GN.format("emiten+kontrak+OR+akuisisi+OR+ekspansi"),
    GN.format("saham+suspensi+OR+gugatan+OR+rugi"),
    GN.format("emiten+right+issue+OR+buyback"),
    GN.format("rekomendasi+saham+hari+ini"),
    GN.format("saham+bank+OR+tambang+OR+energi"),
    "https://www.antaranews.com/rss/ekonomi.xml",
]

# --- keyword event (skor News): aksi korporasi & kinerja ---
EVENT_POS = [
    "laba naik", "laba melonjak", "laba tumbuh", "laba meningkat",
    "pendapatan naik", "pendapatan tumbuh", "kinerja positif", "cetak laba",
    "kontrak baru", "raih kontrak", "menang tender", "akuisisi", "ekspansi",
    "buyback", "beli kembali saham", "bagikan dividen", "dividen",
    "kerja sama", "kemitraan", "investasi baru", "tambah kapasitas",
    "naikkan target", "upgrade", "rekor",
]
EVENT_NEG = [
    "rugi", "kerugian", "laba turun", "laba anjlok", "pendapatan turun",
    "gagal bayar", "pailit", "pkpu", "suspensi", "suspend", "delisting",
    "digugat", "gugatan", "kasus hukum", "korupsi", "denda", "sanksi",
    "phk", "tutup pabrik", "turunkan target", "downgrade",
    "unusual market activity", "peringatan bei",
]

# --- keyword tone (skor Sentiment): nada pemberitaan/pasar ---
TONE_POS = [
    "melonjak", "menguat", "melesat", "terbang", "tumbuh", "positif",
    "optimis", "cerah", "rekomendasi beli", "akumulasi", "potensi naik", "cuan",
]
TONE_NEG = [
    "anjlok", "merosot", "melemah", "tertekan", "negatif", "pesimis",
    "rekomendasi jual", "ambles", "longsor", "terjun", "rontok", "waspada",
]

# token 4-huruf yang juga kata umum: hanya dihitung kalau ditulis dalam kurung
STANDALONE_SKIP = {"BANK", "EMAS", "FILM", "DATA", "NET"}


def load_tickers():
    if not TICKERS_FILE.exists():
        return set()
    return {
        c for c in re.findall(r"[A-Z]{4}", TICKERS_FILE.read_text().upper())
    }


def match_tickers(text, tickers):
    """Return set kode emiten yang disebut dalam teks."""
    parens = set(re.findall(r"\(([A-Z]{4})(?:\.JK)?\)", text))
    tokens = set(re.findall(r"\b[A-Z]{4}\b", text))
    return (parens & tickers) | ((tokens & tickers) - STANDALONE_SKIP)


def kw_score(text, pos, neg):
    return sum(k in text for k in pos) - sum(k in text for k in neg)


def entry_date(e):
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(e, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def get_feed(url):
    """Coba cloudscraper dulu (lolos blokir), fallback feedparser langsung."""
    if cloudscraper is not None:
        try:
            resp = cloudscraper.create_scraper().get(
                url, timeout=20, headers={"User-Agent": UA})
            if resp.status_code == 200 and resp.text.strip():
                return feedparser.parse(resp.text)
        except Exception:
            pass
    return feedparser.parse(url, agent=UA)


def fetch_articles(tickers):
    articles = []
    for url in FEEDS:
        try:
            feed = get_feed(url)
            n = 0
            for e in feed.entries:
                title = getattr(e, "title", "") or ""
                summary = re.sub(r"<[^>]+>", " ", getattr(e, "summary", "") or "")
                raw = f"{title} {summary}"
                matched = match_tickers(raw, tickers)
                if not matched:
                    continue
                low = raw.lower()
                articles.append({
                    "link": getattr(e, "link", "") or title,
                    "date": entry_date(e).isoformat(),
                    "title": title[:160],
                    "tickers": sorted(matched),
                    "ev": kw_score(low, EVENT_POS, EVENT_NEG),
                    "tn": kw_score(low, TONE_POS, TONE_NEG),
                })
                n += 1
            print(f"[i] {url} -> {len(feed.entries)} artikel, {n} match emiten")
        except Exception as ex:
            print(f"[!] {url} gagal: {ex}")
    return articles


def merge_archive(new_articles):
    old = []
    if ARCHIVE.exists():
        try:
            old = json.loads(ARCHIVE.read_text())
        except Exception:
            old = []
    by_link = {a["link"]: a for a in old}
    for a in new_articles:
        by_link[a["link"]] = a
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    merged = [
        a for a in by_link.values()
        if datetime.fromisoformat(a["date"]) >= cutoff
    ]
    merged.sort(key=lambda a: a["date"], reverse=True)
    ARCHIVE.write_text(json.dumps(merged, ensure_ascii=False))
    return merged


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def aggregate(articles):
    now = datetime.now(timezone.utc)
    per = {}
    for a in articles:
        age = max(0.0, (now - datetime.fromisoformat(a["date"])).total_seconds() / 86400)
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        for k in a["tickers"]:
            d = per.setdefault(k, {"wev": 0.0, "wtn": 0.0, "w": 0.0, "n": 0, "arts": []})
            d["wev"] += w * a["ev"]
            d["wtn"] += w * a["tn"]
            d["w"] += w
            d["n"] += 1
            d["arts"].append(a)

    result = {}
    for k, d in per.items():
        if d["w"] <= 0:
            continue
        ev = d["wev"] / d["w"]   # rata-rata tertimbang, kira-kira -2..+2
        tn = d["wtn"] / d["w"]
        arts = sorted(d["arts"], key=lambda a: a["date"], reverse=True)[:3]
        result[k] = {
            "news_score": round(clamp(50 + 18 * ev), 1),
            "sentiment_score": round(clamp(50 + 18 * tn), 1),
            "article_count": d["n"],
            "headlines": [
                {"d": a["date"][:10], "t": a["title"],
                 "s": 1 if (a["ev"] + a["tn"]) > 0 else (-1 if (a["ev"] + a["tn"]) < 0 else 0)}
                for a in arts
            ],
        }
    return result


def main():
    tickers = load_tickers()
    if not tickers:
        print("[x] tickers.txt tidak ada — skip analisa berita")
        return
    print(f"[i] {len(tickers)} kode emiten dimuat")

    new_articles = fetch_articles(tickers)
    merged = merge_archive(new_articles)
    per_ticker = aggregate(merged)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_size": len(merged),
        "tickers_covered": len(per_ticker),
        "per_ticker": per_ticker,
    }, ensure_ascii=False))
    print(f"[✓] Arsip {len(merged)} artikel (30 hari), "
          f"{len(per_ticker)} emiten punya skor berita")


if __name__ == "__main__":
    main()
