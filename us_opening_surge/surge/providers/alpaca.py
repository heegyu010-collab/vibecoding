"""알파카(Alpaca) 마켓데이터 - 무료 계정 API 키로 수년치 1분봉 사용 가능.

환경변수 ALPACA_API_KEY / ALPACA_SECRET_KEY 필요.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, time as dtime, timedelta

import pandas as pd
import requests

from .base import NY, OHLCV, Provider, regular_session

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


class AlpacaProvider(Provider):
    name = "alpaca"
    max_minute_lookback_days = None

    def __init__(self, feed: str = "sip", chunk: int = 100):
        self.key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
        self.secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key or not self.secret:
            raise RuntimeError(
                "알파카 API 키가 없습니다. 환경변수 ALPACA_API_KEY, ALPACA_SECRET_KEY 를 설정하세요."
            )
        self.feed = feed
        self.chunk = chunk
        self.session = requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret})

    def _bars(self, symbols: list[str], timeframe: str, start: datetime, end: datetime) -> dict[str, list]:
        out: dict[str, list] = {}
        params = {
            "symbols": ",".join(symbols), "timeframe": timeframe,
            "start": start.isoformat(), "end": end.isoformat(),
            "limit": 10000, "adjustment": "raw", "feed": self.feed, "sort": "asc",
        }
        token = None
        while True:
            if token:
                params["page_token"] = token
            for attempt in range(6):
                r = self.session.get(BARS_URL, params=params, timeout=60)
                if r.status_code == 429:
                    time.sleep(3 * (attempt + 1))
                    continue
                if r.status_code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue
                break
            if r.status_code != 200:
                raise RuntimeError(f"알파카 오류 {r.status_code}: {r.text[:300]}")
            js = r.json()
            for sym, bars in (js.get("bars") or {}).items():
                out.setdefault(sym, []).extend(bars)
            token = js.get("next_page_token")
            if not token:
                return out

    @staticmethod
    def _frame(bars: list) -> pd.DataFrame:
        df = pd.DataFrame(bars).rename(
            columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df.index = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        return df[OHLCV]

    def trading_days(self, start: date, end: date) -> list[date]:
        s = pd.Timestamp(start, tz=NY).to_pydatetime()
        e = pd.Timestamp(end + timedelta(days=1), tz=NY).to_pydatetime()
        bars = self._bars(["SPY"], "1Day", s, e).get("SPY", [])
        if not bars:
            raise RuntimeError("알파카에서 SPY 일봉을 받지 못했습니다.")
        return sorted({d.date() for d in self._frame(bars).index})

    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        s = pd.Timestamp(start, tz=NY).to_pydatetime()
        e = pd.Timestamp(end + timedelta(days=1), tz=NY).to_pydatetime()
        frames = []
        for i in range(0, len(tickers), self.chunk):
            part = tickers[i:i + self.chunk]
            for sym, bars in self._bars(part, "1Day", s, e).items():
                df = self._frame(bars)
                df["date"] = [d.date() for d in df.index]
                df["ticker"] = sym
                frames.append(df.reset_index(drop=True))
            print(f"    [alpaca] 일봉 {min(i + self.chunk, len(tickers))}/{len(tickers)} 종목", end="\r")
        print()
        if not frames:
            return pd.DataFrame(columns=["date", "ticker", *OHLCV])
        return pd.concat(frames, ignore_index=True)

    def minute_bars(self, tickers: list[str], day: date) -> dict[str, pd.DataFrame]:
        s = pd.Timestamp(datetime.combine(day, dtime(9, 30)), tz=NY).to_pydatetime()
        e = pd.Timestamp(datetime.combine(day, dtime(16, 0)), tz=NY).to_pydatetime()
        out = {}
        for i in range(0, len(tickers), self.chunk):
            for sym, bars in self._bars(tickers[i:i + self.chunk], "1Min", s, e).items():
                df = regular_session(self._frame(bars))
                if not df.empty:
                    out[sym] = df
        return out
