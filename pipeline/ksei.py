#!/usr/bin/env python3
"""ksei.py — Komposisi kepemilikan lokal/asing per emiten (file publik KSEI).

Ambil 2 bulan terakhir "Balance Pos" -> hitung % asing & % institusi +
perubahan bulanan (pp) -> data/ksei.json.
Best-effort total: KSEI bisa memblokir server GitHub; gagal = skip.
"""

import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    import cloudscraper
except ImportError:
    cloudscraper = None

DATA = Path("data")
DATA.mkdir(exist_ok=True)
OUT = DATA / "ksei.json"

PAGES = [
    "https://www.ksei.co.id/archive_download/holding_composition",
    "https://www.ksei.co.id/archive_download/holding_composition/lang/id",
]
UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")}


def http():
    return cloudscraper.create_scraper() if cloudscraper else requests.Session()


def find_zip_links(s):
    for page in PAGES:
        try:
            r = s.get(page, timeout=30, headers=UA)
            if r.status_code != 200:
                print(f"[!] {page} -> HTTP {r.status_code}")
                continue
            links = re.findall(r'href="([^"]+?\.zip)"', r.text, re.I)
            links = [l if l.startswith("http") else "https://www.ksei.co.id" + l
                     for l in links]
            bal = [l for l in links if "balance" in l.lower() or "pos" in l.lower()]
            if bal or links:
                return bal or links
        except Exception as e:
            print(f"[!] {page}: {e}")
    return []


def parse_balance(zbytes):
    """Return {kode: (total_local, total_foreign, local_id, foreign_id)}."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for name in z.namelist():
            if not name.lower().endswith((".txt", ".csv")):
                continue
            raw = z.read(name).decode("utf-8", errors="replace")
            lines = raw.splitlines()
            if len(lines) < 2:
                continue
            delim = "|" if "|" in lines[0] else (";" if ";" in lines[0] else ",")
            head = [h.strip().lower() for h in lines[0].split(delim)]

            def col(*cands):
                for cd in cands:
                    for i, h in enumerate(head):
                        if cd in h:
                            return i
                return None

            ic, it = col("code"), col("type")
            itl, itf = col("total local"), col("total foreign")
            ild, ifd = col("local id"), col("foreign id")
            if None in (ic, itl, itf):
                continue
            for ln in lines[1:]:
                p = ln.split(delim)
                if len(p) <= max(itl, itf):
                    continue
                if it is not None and "equity" not in p[it].strip().lower():
                    continue
                code = p[ic].strip().upper()
                if not re.fullmatch(r"[A-Z]{4}", code):
                    continue

                def f(i):
                    try:
                        return float(p[i].replace(",", "").strip() or 0)
                    except Exception:
                        return 0.0

                out[code] = (f(itl), f(itf),
                             f(ild) if ild is not None else 0.0,
                             f(ifd) if ifd is not None else 0.0)
    return out


def pcts(d):
    res = {}
    for k, (tl, tf, lid, fid) in d.items():
        tot = tl + tf
        if tot <= 0:
            continue
        res[k] = {
            "foreign_pct": round(100 * tf / tot, 2),
            "inst_pct": round(100 * (tot - lid - fid) / tot, 2),
        }
    return res


def main():
    s = http()
    links = find_zip_links(s)
    if not links:
        print("[!] Tidak menemukan file KSEI — skip (kemungkinan diblokir dari runner)")
        return
    links = sorted(links, reverse=True)[:2]  # nama file mengandung tanggal -> 2 terbaru
    print(f"[i] File KSEI dipakai: {links}")

    months = []
    for url in links:
        try:
            r = s.get(url, timeout=180, headers=UA)
            r.raise_for_status()
            months.append(pcts(parse_balance(r.content)))
            print(f"[i] {url.rsplit('/',1)[-1]}: {len(months[-1])} emiten")
        except Exception as e:
            print(f"[!] {url}: {e}")

    if not months or not months[0]:
        print("[!] Parsing KSEI gagal — skip")
        return

    cur, prev = months[0], (months[1] if len(months) > 1 else {})
    per = {}
    for k, v in cur.items():
        pv = prev.get(k)
        per[k] = {
            "foreign_pct": v["foreign_pct"],
            "inst_pct": v["inst_pct"],
            "foreign_delta_pp": round(v["foreign_pct"] - pv["foreign_pct"], 2) if pv else None,
            "inst_delta_pp": round(v["inst_pct"] - pv["inst_pct"], 2) if pv else None,
        }

    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "months_loaded": len(months),
        "tickers": len(per),
        "per_ticker": per,
    }, ensure_ascii=False))
    print(f"[✓] KSEI: {len(per)} emiten, {len(months)} bulan dimuat")


if __name__ == "__main__":
    main()
