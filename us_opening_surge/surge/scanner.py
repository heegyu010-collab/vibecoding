"""하루 단위 '개장 후 N분 급등 상위 종목' 스캐너.

전 종목 분봉을 다 받지 않고도 정확한 상위 N 을 구하기 위해 상한(bound) 가지치기를 쓴다:
  개장 후 30분 상승률 = 10:00 가격 / 시가 - 1  ≤  당일 고가 / 시가 - 1
따라서 (일봉 고가/시가) 가 큰 순서로 분봉을 확인하다가, 남은 후보의 상한이 현재 N 위 기록보다
작아지면 더 볼 필요가 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import Settings
from .features import compute_features
from .store import DailyStore, MinuteStore


@dataclass
class DayResult:
    day: date
    records: list[dict]          # 상위 N 종목 특징
    paths: dict[str, np.ndarray] # 종목별 누적수익률 경로
    checked: int                 # 분봉을 확인한 후보 수
    exact: bool                  # 상한 조건으로 정확성 보장 여부


def scan_day(day: date, prev_day: date | None, daily: DailyStore, minute: MinuteStore,
             s: Settings) -> DayResult:
    d = daily.get(day, prev_day)
    if d.empty:
        return DayResult(day, [], {}, 0, False)

    d = d[(d["open"] >= s.min_price) & (d["volume"] * d["close"] >= s.min_daily_dollar_volume)]
    denom = d["open"] if s.base == "open" else d["prevclose"]
    ub = (d["high"] / denom - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
    cands = ub.sort_values(ascending=False)

    found: dict[str, tuple[dict, np.ndarray]] = {}
    checked = 0
    exact = False
    i = 0
    while i < len(cands):
        nth = _nth_best(found, s.top_n)
        if nth is not None and cands.iloc[i] + s.bound_tolerance < nth:
            exact = True
            break
        if checked >= s.max_candidates:
            break
        batch = list(cands.index[i:i + s.batch_size])
        i += len(batch)
        bars = minute.get(day, batch)
        checked += len(batch)
        for t in batch:
            res = compute_features(bars.get(t), s.window, float(d.at[t, "prevclose"]), s.base)
            if res is None:
                continue
            f, path = res
            if f["dollar_vol"] < s.min_window_dollar_volume or f["open"] < s.min_price:
                continue
            found[t] = (f, path)
    else:
        exact = True  # 후보를 전부 확인함

    top = sorted(found.items(), key=lambda kv: kv[1][0]["ret_w"], reverse=True)[:s.top_n]
    records, paths = [], {}
    for rank, (t, (f, path)) in enumerate(top, 1):
        records.append({"date": day, "rank": rank, "ticker": t, **f})
        paths[t] = path
    return DayResult(day, records, paths, checked, exact)


def _nth_best(found: dict, n: int) -> float | None:
    if len(found) < n:
        return None
    vals = sorted((f["ret_w"] for f, _ in found.values()), reverse=True)
    return vals[n - 1]
