"""백워드 회귀 테스트: 최근 거래일부터 하루씩 과거로 표본을 늘리며 95% 신뢰 패턴이 나올 때까지 반복."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .config import Settings
from .providers import NY, Provider
from .rules import ACTIONABLE, build_rules, evaluate
from .scanner import scan_day
from .store import DailyStore, MinuteStore
from .universe import load_universe


@dataclass
class RunResult:
    settings: Settings
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    paths: dict = field(default_factory=dict)          # (date, ticker) -> path
    final: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    day_log: list = field(default_factory=list)
    stop_reason: str = ""
    rules: list = field(default_factory=list)


def _session_closed(day: date) -> bool:
    now = pd.Timestamp.now(tz=NY)
    return day < now.date() or (day == now.date() and (now.hour, now.minute) >= (16, 15))


def run(s: Settings, provider: Provider) -> RunResult:
    res = RunResult(settings=s)
    res.rules = rules = build_rules(s.window, s.top_n)

    # 1) 거래일 목록
    end = date.fromisoformat(s.end_date) if s.end_date else pd.Timestamp.now(tz=NY).date()
    start = end - timedelta(days=int(s.max_days * 1.5) + 15)
    days_all = [d for d in provider.trading_days(start, end) if _session_closed(d)]
    if provider.max_minute_lookback_days:
        limit = pd.Timestamp.now(tz=NY).date() - timedelta(days=provider.max_minute_lookback_days)
        usable = [d for d in days_all if d >= limit]
        if len(usable) < len(days_all):
            print(f"  ※ {provider.name} 1분봉은 최근 {provider.max_minute_lookback_days}일까지만 제공 → "
                  f"최대 {len(usable)}거래일까지 회귀 가능 (더 길게: --source alpaca)")
    else:
        usable = days_all
    targets = sorted(usable, reverse=True)[: s.max_days]
    if not targets:
        raise RuntimeError("분석 가능한 거래일이 없습니다.")
    prev_of = {d: (days_all[i - 1] if i > 0 else None) for i, d in enumerate(days_all)}

    # 2) 유니버스 & 캐시
    tickers = provider.universe() or load_universe(s.cache_dir, s.include_etf, s.universe_file)
    print(f"  유니버스 {len(tickers):,}종목 | 분석 후보일 {targets[-1]} ~ {targets[0]} ({len(targets)}일)")
    daily = DailyStore(provider, tickers, s.cache_dir)
    minute = MinuteStore(provider, s.cache_dir)

    recs: list[dict] = []
    hist_frames = []
    streak: dict[str, int] = {}
    stop_cats = {"actionable": ACTIONABLE, "any": ("형태",) + ACTIONABLE, "none": ()}[s.stop_on]
    block = 20

    try:
        for i, day in enumerate(targets):
            if i % block == 0:  # 일봉은 20거래일씩 묶어서 선다운로드
                chunk = targets[i:i + block]
                need = set(chunk) | {prev_of[d] for d in chunk if prev_of.get(d)}
                daily.prefetch(sorted(need))

            dr = scan_day(day, prev_of.get(day), daily, minute, s)
            recs.extend(dr.records)
            for r in dr.records:
                res.paths[(day, r["ticker"])] = dr.paths[r["ticker"]]
            res.day_log.append({"date": day, "found": len(dr.records), "checked": dr.checked, "exact": dr.exact})

            df = pd.DataFrame(recs)
            if df.empty:
                print(f"  [{i + 1:>3}] {day} 데이터 없음")
                continue
            ev = evaluate(df, rules, s.confidence, s.target_rate, s.min_samples, s.bonferroni)
            ev["iteration"] = i + 1
            ev["days_used"] = df["date"].nunique()
            ev["oldest_date"] = day
            hist_frames.append(ev)

            conf = ev[ev["confirmed"] & ev["category"].isin(stop_cats)]
            ids = set(conf["pattern_id"])
            streak = {pid: streak.get(pid, 0) + 1 for pid in ids}
            stable = [pid for pid, c in streak.items() if c >= s.stable_days]

            top = ", ".join(f"{t}({r:+.1%})" for t, r in zip(df[df.date == day].ticker, df[df.date == day].ret_w))
            act = ev[ev["category"].isin(ACTIONABLE)]
            best = act.iloc[0] if not act.empty else None
            best_txt = (f"최고 하한 {best.ci_low:.1%} (n={best.n}) {best.pattern}" if best is not None else "")
            print(f"  [{i + 1:>3}] {day} | 후보 {dr.checked:>3}개 확인{'' if dr.exact else '(상한미확정)'} | "
                  f"확정 {len(ids)}개 | {best_txt}")
            print(f"        상위 {len(dr.records)}: {top}")

            n_days = df["date"].nunique()
            if stop_cats and n_days >= s.min_days and stable:
                res.stop_reason = (f"{n_days}거래일({day}~{targets[0]}) 표본에서 "
                                   f"{len(stable)}개 패턴이 {s.stable_days}회 연속 {s.confidence:.0%} 신뢰 기준 충족")
                break
        else:
            res.stop_reason = f"최대 {len(targets)}거래일까지 회귀했지만 기준을 만족하는 패턴이 안정적으로 나오지 않음"
    except KeyboardInterrupt:
        res.stop_reason = "사용자 중단(Ctrl+C) - 지금까지 분석한 결과로 보고서 작성"
        print("\n  " + res.stop_reason)

    res.records = pd.DataFrame(recs)
    res.history = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    if not res.records.empty:
        res.final = evaluate(res.records, rules, s.confidence, s.target_rate, s.min_samples, s.bonferroni)
    return res
