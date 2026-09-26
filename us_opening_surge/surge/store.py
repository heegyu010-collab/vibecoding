"""일봉/분봉 로컬 캐시 - 같은 날짜를 다시 분석할 때 재다운로드하지 않는다."""
from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from .providers.base import OHLCV, Provider


class DailyStore:
    """날짜별 전 종목 일봉 단면(cross-section) 캐시."""

    def __init__(self, provider: Provider, tickers: list[str], cache_dir: Path):
        self.provider = provider
        self.tickers = tickers
        self.dir = Path(cache_dir) / provider.name / "daily"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[date, pd.DataFrame] = {}

    def _path(self, d: date) -> Path:
        return self.dir / f"{d.isoformat()}.pkl"

    def _load(self, d: date) -> pd.DataFrame | None:
        if d in self._mem:
            return self._mem[d]
        p = self._path(d)
        if p.exists():
            df = pd.read_pickle(p)
            self._mem[d] = df
            return df
        return None

    def prefetch(self, days: list[date]) -> None:
        """days 중 캐시에 없는 날짜를 한 번에 받아 저장."""
        missing = sorted(d for d in set(days) if self._load(d) is None)
        if not missing:
            return
        print(f"  [일봉] {missing[0]} ~ {missing[-1]} ({len(missing)}일, {len(self.tickers)}종목) 다운로드")
        long = self.provider.daily_bars(self.tickers, missing[0], missing[-1])
        for d in missing:
            part = long[long["date"] == d]
            df = part.set_index("ticker")[OHLCV].astype(float)
            df = df[~df.index.duplicated(keep="last")]
            if df.empty:
                continue
            self._mem[d] = df
            df.to_pickle(self._path(d))

    def get(self, day: date, prev_day: date | None) -> pd.DataFrame:
        df = self._load(day)
        if df is None:
            self.prefetch([day] + ([prev_day] if prev_day else []))
            df = self._load(day)
        if df is None:
            return pd.DataFrame(columns=OHLCV + ["prevclose"])
        df = df.copy()
        prev = self._load(prev_day) if prev_day else None
        if prev is None and prev_day is not None:
            self.prefetch([prev_day])
            prev = self._load(prev_day)
        df["prevclose"] = prev["close"].reindex(df.index) if prev is not None else float("nan")
        return df


class MinuteStore:
    """날짜별 {종목: 1분봉} 캐시. 데이터가 없던 종목도 기록해 재요청하지 않는다."""

    def __init__(self, provider: Provider, cache_dir: Path):
        self.provider = provider
        self.dir = Path(cache_dir) / provider.name / "minute"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[date, dict] = {}

    def _path(self, d: date) -> Path:
        return self.dir / f"{d.isoformat()}.pkl"

    def _load(self, d: date) -> dict:
        if d not in self._mem:
            p = self._path(d)
            self._mem[d] = pickle.loads(p.read_bytes()) if p.exists() else {}
        return self._mem[d]

    def get(self, day: date, tickers: list[str]) -> dict[str, pd.DataFrame]:
        data = self._load(day)
        need = [t for t in tickers if t not in data]
        if need:
            got = self.provider.minute_bars(need, day)
            for t in need:
                data[t] = got.get(t)  # None = 데이터 없음
            self._path(day).write_bytes(pickle.dumps(data))
        return {t: data[t] for t in tickers if data.get(t) is not None}
