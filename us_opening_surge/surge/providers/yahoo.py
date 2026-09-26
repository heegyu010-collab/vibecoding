"""야후 파이낸스(yfinance) - API 키 불필요. 1분봉은 최근 약 30일까지만 제공."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from .base import OHLCV, Provider, regular_session

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def to_yahoo(sym: str) -> str:
    return sym.replace(".", "-").replace("/", "-")


class YahooProvider(Provider):
    name = "yahoo"
    max_minute_lookback_days = 29

    def __init__(self, chunk: int = 200, pause: float = 1.0, retries: int = 3):
        import yfinance  # noqa: F401  (설치 여부 확인)

        self.chunk = chunk
        self.pause = pause
        self.retries = retries

    # ------------------------------------------------------------------ utils
    def _download(self, tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
        """yf.download 래퍼. 레이트리밋이면 쉬었다가 재시도. {원래티커: DataFrame}"""
        import yfinance as yf

        ymap = {to_yahoo(t): t for t in tickers}
        pending = list(ymap)
        out: dict[str, pd.DataFrame] = {}
        for attempt in range(self.retries + 1):
            if not pending:
                break
            try:
                raw = yf.download(
                    pending, group_by="ticker", auto_adjust=False, progress=False,
                    threads=True, **kw,
                )
            except Exception as e:  # 네트워크 오류 등
                print(f"    [yahoo] 다운로드 오류: {e} → {5 * (attempt + 1)}초 후 재시도")
                time.sleep(5 * (attempt + 1))
                continue
            errors = dict(getattr(getattr(yf, "shared", None), "_ERRORS", {}) or {})
            got = _split(raw, pending)
            for yt, df in got.items():
                out[ymap[yt]] = df
            limited = [
                yt for yt in pending
                if yt not in got and any(s in str(errors.get(yt, "")).lower() for s in ("rate", "too many"))
            ]
            if not limited:
                break
            wait = 20 * (attempt + 1)
            print(f"    [yahoo] 요청 제한(rate limit) {len(limited)}종목 → {wait}초 대기 후 재시도")
            time.sleep(wait)
            pending = limited
        return out

    # -------------------------------------------------------------- interface
    def trading_days(self, start: date, end: date) -> list[date]:
        got = self._download(["SPY"], start=start, end=end + timedelta(days=1), interval="1d")
        df = got.get("SPY")
        if df is None or df.empty:
            raise RuntimeError("야후에서 SPY 일봉을 받지 못했습니다(네트워크/차단 확인).")
        return sorted({d.date() for d in pd.DatetimeIndex(df.index)})

    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for i in range(0, len(tickers), self.chunk):
            part = tickers[i:i + self.chunk]
            got = self._download(part, start=start, end=end + timedelta(days=1), interval="1d")
            for t, df in got.items():
                df = df[OHLCV].dropna(subset=["open", "high", "low", "close"])
                if df.empty:
                    continue
                df = df.copy()
                df["date"] = [d.date() for d in pd.DatetimeIndex(df.index)]
                df["ticker"] = t
                frames.append(df.reset_index(drop=True))
            done = min(i + self.chunk, len(tickers))
            print(f"    [yahoo] 일봉 {done}/{len(tickers)} 종목", end="\r")
            time.sleep(self.pause)
        print()
        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *OHLCV])
        return pd.concat(frames, ignore_index=True)

    def minute_bars(self, tickers: list[str], day: date) -> dict[str, pd.DataFrame]:
        got = self._download(
            tickers, start=day, end=day + timedelta(days=1), interval="1m", prepost=False,
        )
        out = {}
        for t, df in got.items():
            df = regular_session(df)
            df = df[[d.date() == day for d in df.index]] if not df.empty else df
            if not df.empty:
                out[t] = df
        time.sleep(self.pause * 0.5)
        return out


def _split(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """yf.download 결과를 종목별 DataFrame(소문자 컬럼)으로 분리."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(raw.columns.get_level_values(0))
        lv1 = set(raw.columns.get_level_values(1))
        for t in tickers:
            if t in lv0:
                sub = raw[t]
            elif t in lv1:
                sub = raw.xs(t, axis=1, level=1)
            else:
                continue
            sub = _norm_cols(sub)
            if sub is not None:
                out[t] = sub
    elif len(tickers) == 1:
        sub = _norm_cols(raw)
        if sub is not None:
            out[tickers[0]] = sub
    return out


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.rename(columns=lambda c: str(c).strip().lower())
    if not set(OHLCV).issubset(df.columns):
        return None
    df = df[OHLCV].dropna(subset=["open", "high", "low", "close"], how="any")
    return df if not df.empty else None
