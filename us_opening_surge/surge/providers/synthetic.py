"""가상 데이터 - 인터넷 없이 프로그램 동작을 확인하기 위한 용도(실제 시장 아님)."""
from __future__ import annotations

import zlib
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from .base import NY, OHLCV, Provider


class SyntheticProvider(Provider):
    name = "synthetic"

    def __init__(self, n_tickers: int = 300, seed: int = 7, news_prob: float = 0.04, fade_prob: float = 0.78):
        self.tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
        self.seed = seed
        self.news_prob = news_prob
        self.fade_prob = fade_prob

    def universe(self) -> list[str]:
        return list(self.tickers)

    def trading_days(self, start: date, end: date) -> list[date]:
        return [d.date() for d in pd.bdate_range(start, end)]

    def daily_bars(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        rows = []
        for day in self.trading_days(start, end):
            for t in tickers:
                m = self._minutes(t, day)
                rows.append((day, t, m.open.iloc[0], m.high.max(), m.low.min(), m.close.iloc[-1], m.volume.sum()))
        return pd.DataFrame(rows, columns=["date", "ticker", *OHLCV])

    def minute_bars(self, tickers: list[str], day: date) -> dict[str, pd.DataFrame]:
        return {t: self._minutes(t, day) for t in tickers}

    @lru_cache(maxsize=20000)
    def _minutes(self, t: str, day: date) -> pd.DataFrame:
        rng = np.random.default_rng(zlib.crc32(f"{self.seed}|{t}|{day}".encode()))
        tick = np.random.default_rng(zlib.crc32(f"{self.seed}|{t}".encode()))
        p0 = float(np.exp(tick.uniform(np.log(2), np.log(300))))
        sigma = tick.uniform(0.0008, 0.003)
        base_vol = tick.uniform(2e3, 5e4) * 50 / p0 * 20

        n = 390
        drift = np.zeros(n)
        vol_mult = np.ones(n)
        is_news = rng.random() < self.news_prob
        if is_news:
            R = rng.uniform(0.06, 0.35)
            x = np.arange(30) / 29
            shape = rng.integers(5)
            if shape == 0:      # 계단식
                g = R * x
            elif shape == 1:    # 초반 급등
                g = R * (1 - np.exp(-x * 7)) / (1 - np.exp(-7))
            elif shape == 2:    # 후반 급등
                g = R * x ** 3
            elif shape == 3:    # V자
                g = R * x - 0.5 * R * np.sin(np.pi * np.clip(x * 2, 0, 1))
            else:               # 급등 후 되밀림
                g = 1.8 * R * np.sin(np.pi * np.clip(x * 1.5, 0, 1) / 2) - 0.8 * R * np.clip(x * 3 - 2, 0, 1)
            drift[:30] = np.diff(np.concatenate([[0.0], np.log1p(g)]))
            post = rng.uniform(0.02, 0.10) * (-1 if rng.random() < self.fade_prob else 1)
            drift[30:90] = np.log1p(post) / 60
            sigma *= 3
            vol_mult[:30] = 12
            vol_mult[30:90] = 5
        vol_mult *= 1 + 2 * np.exp(-np.arange(n) / 20) + np.exp(-(n - 1 - np.arange(n)) / 15)

        gap = rng.normal(0, 0.01) + (rng.uniform(0.0, 0.15) if is_news and rng.random() < 0.6 else 0)
        open_ = p0 * float(np.exp(rng.normal(0, 0.05))) * (1 + gap)
        rets = drift + rng.normal(0, sigma, n)
        close = open_ * np.exp(np.cumsum(rets))
        opens = np.concatenate([[open_], close[:-1]])
        spread = np.abs(rng.normal(0, sigma * 0.7, (2, n)))
        high = np.maximum(opens, close) * (1 + spread[0])
        low = np.minimum(opens, close) * (1 - spread[1])
        volume = np.round(base_vol * vol_mult * rng.lognormal(0, 0.4, n))
        idx = pd.date_range(pd.Timestamp(f"{day} 09:30", tz=NY), periods=n, freq="1min")
        df = pd.DataFrame({"open": opens, "high": high, "low": low, "close": close, "volume": volume}, index=idx)
        # 전일 종가는 일봉 스토어에서 전일 close 로 계산되므로 여기서는 생략
        return df
