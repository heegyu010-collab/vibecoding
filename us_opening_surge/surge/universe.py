"""분석 대상 종목(유니버스) - 나스닥트레이더 상장 목록(NASDAQ + NYSE + AMEX 등) 사용."""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# 워런트/유닛/권리/우선주/채권성 상품 제외
EXCLUDE_NAME = re.compile(
    r"\b(?:warrants?|units?|rights?|preferred|preference|notes? due|debentures?|subordinated|"
    r"trust preferred|%\s*(?:fixed|senior|notes))\b",
    re.I,
)


def _read(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str).fillna("")


def _clean_symbol_ok(sym: str) -> bool:
    if not sym or any(ch in sym for ch in "$^~ "):
        return False
    if "." in sym:
        return sym.split(".")[-1] in {"A", "B", "C"}  # BRK.B 같은 클래스 주식만 허용
    return True


def download_universe(include_etf: bool = False) -> list[str]:
    nas = _read(NASDAQ_URL)
    nas = nas[(nas["Test Issue"] == "N")]
    if not include_etf:
        nas = nas[nas["ETF"] == "N"]
    nas = nas[~nas["Security Name"].str.contains(EXCLUDE_NAME)]
    syms = set(nas["Symbol"])

    oth = _read(OTHER_URL)
    oth = oth[(oth["Test Issue"] == "N")]
    if not include_etf:
        oth = oth[oth["ETF"] == "N"]
    oth = oth[~oth["Security Name"].str.contains(EXCLUDE_NAME)]
    syms |= set(oth["ACT Symbol"])

    return sorted(s for s in syms if _clean_symbol_ok(s))


def load_universe(cache_dir: Path, include_etf: bool = False, universe_file: str | None = None,
                  max_age_days: int = 7) -> list[str]:
    if universe_file:
        text = Path(universe_file).read_text(encoding="utf-8-sig")
        syms = [s.strip().upper() for s in re.split(r"[\s,;]+", text) if s.strip()]
        return sorted(set(syms))

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("universe_etf.csv" if include_etf else "universe.csv")
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        return pd.read_csv(path, dtype=str)["ticker"].tolist()
    try:
        syms = download_universe(include_etf)
        pd.DataFrame({"ticker": syms}).to_csv(path, index=False)
        return syms
    except Exception as e:
        if path.exists():
            print(f"  [유니버스] 목록 갱신 실패({e}) → 기존 캐시 사용")
            return pd.read_csv(path, dtype=str)["ticker"].tolist()
        raise RuntimeError(
            f"종목 목록 다운로드 실패: {e}\n  → --universe-file 옵션으로 종목 리스트 파일을 지정하세요."
        ) from e
