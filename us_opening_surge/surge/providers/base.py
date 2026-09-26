"""데이터 제공자 공통 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

NY = "America/New_York"
OHLCV = ["open", "high", "low", "close", "volume"]


class Provider(ABC):
    name: str = "base"
    # 분봉을 받을 수 있는 최대 과거 일수(달력 기준). None 이면 제한 없음
    max_minute_lookback_days: int | None = None

    @abstractmethod
    def trading_days(self, start: date, end: date) -> list[date]:
        """start~end 사이 거래일(오름차순)."""

    @abstractmethod
    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """일봉. 컬럼: date, ticker, open, high, low, close, volume (long format)."""

    @abstractmethod
    def minute_bars(self, tickers: list[str], day: date) -> dict[str, pd.DataFrame]:
        """정규장(09:30~16:00) 1분봉. index=뉴욕시간 tz-aware, 컬럼=OHLCV."""

    def universe(self) -> list[str] | None:
        """제공자 고유 유니버스가 있으면 반환 (없으면 None → 나스닥 상장목록 사용)."""
        return None


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """tz 정리 후 정규장 봉만 남긴다."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(NY)
    else:
        idx = idx.tz_convert(NY)
    df = df.copy()
    df.index = idx
    t = df.index.hour * 60 + df.index.minute
    df = df[(t >= 9 * 60 + 30) & (t < 16 * 60)]
    df = df[OHLCV].astype(float)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df.sort_index()
