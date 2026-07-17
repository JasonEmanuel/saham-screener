#!/usr/bin/env python3
"""
update_tickers.py — Refresh daftar emiten IDX via Yahoo Finance screener.

Menulis ulang pipeline/tickers.txt dengan seluruh saham region Indonesia
yang terdaftar di Yahoo Finance (ticker .JK). Exit code 1 jika gagal atau
hasil tidak masuk akal, supaya workflow fallback ke tickers.txt lama.
"""

import re
import sys
from pathlib import Path

import yfinance as yf

OUT = Path("pipeline/tickers.txt")
PAGE_SIZE = 250
MAX_OFFSET = 2000
MIN_SANE = 500  # kalau hasil < 500 emiten, anggap gagal

def main():
    codes = set()
    query = yf.EquityQuery("eq", ["region", "id"])
    offset = 0
    while offset <= MAX_OFFSET:
        try:
            resp = yf.screen(
                query,
                offset=offset,
                size=PAGE_SIZE,
                sortField="intradaymarketcap",
                sortAsc=False,
            )
        except Exception as e:
            print(f"[!] screener error di offset {offset}: {e}")
            break
        quotes = (resp or {}).get("quotes") or []
        if not quotes:
            break
        for q in quotes:
            sym = str(q.get("symbol", ""))
            if sym.endswith(".JK"):
                code = sym[:-3]
                if re.fullmatch(r"[A-Z]{4}", code):
                    codes.add(code)
        offset += PAGE_SIZE

    print(f"[i] screener menghasilkan {len(codes)} emiten")
    if len(codes) < MIN_SANE:
        print("[x] Hasil terlalu sedikit — pakai tickers.txt lama.")
        sys.exit(1)

    OUT.write_text("\n".join(sorted(codes)) + "\n")
    print(f"[✓] {OUT} diperbarui")

if __name__ == "__main__":
    main()
