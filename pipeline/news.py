#!/usr/bin/env python3
"""
news.py — Analisa berita & sentimen emiten IDX (hybrid).

- Event (skor News)      : keyword eksplisit (aksi korporasi, kinerja, masalah hukum)
- Tone (skor Sentiment)  : model transformer sentimen bahasa Indonesia;
                           fallback otomatis ke keyword kalau model gagal
- Artikel multi-emiten   : dinilai per kalimat yang menyebut emiten ybs.

Arsip 30 hari menumpuk tiap run -> data/news.json.
Semua tahap best-effort: kegagalan satu bagian tidak menggagalkan run.
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
HALF_LIFE_DAYS = 7.0

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
TONE_POS = [
    "melonjak", "menguat", "melesat", "terbang", "tumbuh", "positif",
    "optimis", "cerah", "rekomendasi beli", "akumulasi", "potensi naik", "cuan",
]
TONE_NEG = [
    "anjlok", "merosot", "melemah", "tertekan", "negatif", "pesimis",
    "rekomendasi jual", "ambles", "longsor", "terjun", "rontok", "waspada",
]

STANDALONE_SKIP = {"BANK", "EMAS", "FILM", "DATA", "NET"}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

MODEL_CANDIDATES = [
    "w11wo/indonesian-roberta-base-sentiment-classifier",
    "mdhugol/indonesia-bert-sentiment-classification",
]

_MODEL = None          # None = belum dicoba, False = gagal, objek = siap
_STATS = {"model": 0, "keyword": 0}


# ---------------------------------------------------------------- model -----
def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    for name in MODEL_CANDIDATES:
        try:
            from transformers import pipeline as hf_pipeline
            _MODEL = hf_pipeline("sentiment-analysis", model=name, top_k=None)
            print(f"[i] Model sentimen dimuat: {name}")
            return _MODEL
        except Exception as e:
            print(f"[!] Model {name} gagal: {str(e)[:120]}")
    _MODEL = False
    print("[!] Semua model gagal — tone pakai keyword")
    return _MODEL


def model_tone(text):
    """Skor tone dari model: -2..+2 (P(pos)-P(neg) dikali 2). None kalau gagal."""
    m = load_model()
    if not m:
        return None
    try:
        scores = m(text[:512])[0]
        d = {str(s["label"]).lower(): float(s["score"]) for s in scores}
        pos = d.get("positive", d.get("label_0", 0.0))
        neg = d.get("negative", d.get("label_2", 0.0))
        return round(2 * (pos - neg), 3)
    except Exception:
        return None


# ---------------------------------------------------------------- util ------
def kw_score(text, pos, neg):
    return sum(k in text for k in pos) - sum(k in text for k in neg)


def load_tickers():
    if not TICKERS_FILE.exists():
        return set()
    return {c for c in re.findall(r"[A-Z]{4}", TICKERS_FILE.read_text().upper())}


def match_tickers(text, tickers):
    parens = set(re.findall(r"\(([A-Z]{4})(?:\.JK)?\)", text))
    tokens = set(re.findall(r"\b[A-Z]{4}\b", text))
    return (parens & tickers) | ((tokens & tickers) - STANDALONE_SKIP)


def relevant_text(title, sentences, code):
    """Kalimat yang menyebut emiten ini + judul. Kosong -> judul saja."""
    pat = re.compile(rf"\b{code}\b")
    rel = [s for s in sentences if pat.search(s)]
    return (title + ". " + " ".join(rel)) if rel else title


def entry_date(e):
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(e, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def get_feed(url):
    if cloudscraper is not None:
        try:
            resp = cloudscraper.create_scraper().get(
                url, timeout=20, headers={"User-Agent": UA})
            if resp.status_code == 200 and resp.text.strip():
                return feedparser.parse(resp.text)
        except Exception:
            pass
    return feedparser.parse(url, agent=UA)


# ---------------------------------------------------------------- fetch -----
def score_article(title, summary, matched):
    """Return (ev_by, tn_by) per emiten, dengan atribusi kalimat kalau multi-emiten."""
    raw = f"{title}. {summary}"
    sentences = re.split(r"(?<=[.!?])\s+", raw)
    ev_by, tn_by = {}, {}
    for code in matched:
        text = relevant_text(title, sentences, code) if len(matched) > 1 else raw
        low = text.lower()
        ev_by[code] = kw_score(low, EVENT_POS, EVENT_NEG)
        tn = model_tone(text)
        if tn is None:
            tn = kw_score(low, TONE_POS, TONE_NEG)
            _STATS["keyword"] += 1
        else:
            _STATS["model"] += 1
        tn_by[code] = tn
    return ev_by, tn_by


def fetch_articles(tickers):
    articles = []
    for url in FEEDS:
        try:
            feed = get_feed(url)
            n = 0
            for e in feed.entries:
                title = getattr(e, "title", "") or ""
                summary = re.sub(r"<[^>]+>", " ", getattr(e, "summary", "") or "")
                matched = match_tickers(f"{title} {summary}", tickers)
                if not matched:
                    continue
                ev_by, tn_by = score_article(title, summary, matched)
                articles.append({
                    "link": getattr(e, "link", "") or title,
                    "date": entry_date(e).isoformat(),
                    "title": title[:160],
                    "tickers": sorted(matched),
                    "ev_by": ev_by,
                    "tn_by": tn_by,
                })
                n += 1
            print(f"[i] {url[:70]} -> {len(feed.entries)} artikel, {n} match")
        except Exception as ex:
            print(f"[!] {url[:70]} gagal: {ex}")
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
    merged = [a for a in by_link.values()
              if datetime.fromisoformat(a["date"]) >= cutoff]
    merged.sort(key=lambda a: a["date"], reverse=True)
    ARCHIVE.write_text(json.dumps(merged, ensure_ascii=False))
    return merged


# ------------------------------------------------------------- aggregate ----
def art_scores(a, code):
    """Skor (ev, tn) artikel utk emiten; kompatibel arsip format lama."""
    ev = (a.get("ev_by") or {}).get(code, a.get("ev", 0))
    tn = (a.get("tn_by") or {}).get(code, a.get("tn", 0))
    return ev, tn


def clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def aggregate(articles):
    now = datetime.now(timezone.utc)
    per = {}
    for a in articles:
        age = max(0.0, (now - datetime.fromisoformat(a["date"])).total_seconds() / 86400)
        w = 0.5 ** (age / HALF_LIFE_DAYS)
        for k in a["tickers"]:
            ev, tn = art_scores(a, k)
            d = per.setdefault(k, {"wev": 0.0, "wtn": 0.0, "w": 0.0, "n": 0, "arts": []})
            d["wev"] += w * ev
            d["wtn"] += w * tn
            d["w"] += w
            d["n"] += 1
            d["arts"].append(a)

    result = {}
    for k, d in per.items():
        if d["w"] <= 0:
            continue
        ev = d["wev"] / d["w"]
        tn = d["wtn"] / d["w"]
        arts = sorted(d["arts"], key=lambda a: a["date"], reverse=True)[:3]
        heads = []
        for a in arts:
            e2, t2 = art_scores(a, k)
            s = e2 + t2
            heads.append({"d": a["date"][:10], "t": a["title"],
                          "s": 1 if s > 0.15 else (-1 if s < -0.15 else 0)})
        result[k] = {
            "news_score": round(clamp(50 + 18 * ev), 1),
            "sentiment_score": round(clamp(50 + 18 * tn), 1),
            "article_count": d["n"],
            "headlines": heads,
        }
    return result


def main():
    tickers = load_tickers()
    if not tickers:
        print("[x] tickers.txt tidak ada — skip analisa berita")
        return
    print(f"[i] {len(tickers)} kode emiten dimuat")

    t0 = time.time()
    new_articles = fetch_articles(tickers)
    merged = merge_archive(new_articles)
    per_ticker = aggregate(merged)

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_size": len(merged),
        "tickers_covered": len(per_ticker),
        "tone_method": _STATS,
        "per_ticker": per_ticker,
    }, ensure_ascii=False))
    print(f"[✓] Arsip {len(merged)} artikel, {len(per_ticker)} emiten punya skor "
          f"| tone: {_STATS['model']} via model, {_STATS['keyword']} via keyword "
          f"| {int(time.time()-t0)}s")


if __name__ == "__main__":
    main()
