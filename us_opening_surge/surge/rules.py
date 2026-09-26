"""패턴 규칙 정의 + 신뢰구간(Wilson) 기반 판정.

패턴 종류
  - 형태     : 상위 종목들의 30분 구간 모양(서술적 패턴, 선정 기준 자체의 영향을 받음)
  - 이후흐름 : 10:00 시점에 상위 N 을 뽑았을 때 그 이후 가격 흐름 (10시에 알 수 있음 → 실전 활용 가능)
  - 조건부   : '30분 형태 조건' → '이후 흐름' (역시 10시에 판단 가능)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from .features import SHAPES, clock

ACTIONABLE = ("이후흐름", "조건부")


@dataclass
class Rule:
    id: str
    category: str
    label: str          # 참일 때 설명
    label_neg: str      # 거짓일 때 설명
    fn: Callable[[pd.DataFrame], pd.Series]
    cond: "Rule | None" = None


def _b(col: str, op: str, thr: float) -> Callable[[pd.DataFrame], pd.Series]:
    def f(df: pd.DataFrame) -> pd.Series:
        x = pd.to_numeric(df[col], errors="coerce")
        res = {">=": x >= thr, ">": x > thr, "<": x < thr, "<=": x <= thr}[op].astype(float)
        return res.where(x.notna())
    return f


def build_rules(window: int, top_n: int) -> list[Rule]:
    end, p30, p60 = clock(window), clock(window + 30), clock(window + 60)
    third, last5 = max(1, window // 3), max(1, window - 5)

    form = [
        Rule("HIGH_LAST5", "형태", f"{window}분 고점이 마지막 5분({clock(last5)}~{end})에 형성",
             f"{window}분 고점이 {clock(last5)} 이전에 형성", _b("t_high", ">=", last5)),
        Rule("HIGH_EARLY", "형태", f"{window}분 고점이 첫 {third}분 안에 형성",
             f"{window}분 고점이 {clock(third)} 이후 형성", _b("t_high", "<", third)),
        Rule("LOW_FIRST5", "형태", "구간 저점이 첫 5분 안에 형성", "구간 저점이 5분 이후 형성", _b("t_low", "<", 5)),
        Rule("HOLD_OPEN", "형태", "시가 대비 -0.5% 아래로 한 번도 안 내려감", "장중 시가 -0.5% 아래로 밀린 적 있음",
             _b("low_ret", ">=", -0.005)),
        Rule("NEAR_HIGH", "형태", f"{end} 가격이 구간 고점 대비 -3% 이내", f"{end} 가격이 구간 고점 대비 3% 넘게 밀림",
             _b("off_high", ">=", -0.03)),
        Rule("VWAP_HOLD", "형태", "VWAP 위 체류 80% 이상", "VWAP 위 체류 80% 미만", _b("vwap_above", ">=", 0.8)),
        Rule("FIRST5_UP", "형태", "첫 5분 상승 출발", "첫 5분 하락/보합 출발", _b("r5", ">", 0)),
        Rule("HALF_EARLY", "형태", f"첫 {third}분에 {window}분 상승분의 절반 이상 달성",
             f"첫 {third}분 상승이 전체의 절반 미만", _b("frac_third", ">=", 0.5)),
        Rule("VOL_FRONT", "형태", "첫 5분 거래대금 비중 30% 이상", "첫 5분 거래대금 비중 30% 미만",
             _b("vol_first5", ">=", 0.3)),
        Rule("GAP_UP", "형태", "갭상승(시가 > 전일종가) 출발", "갭하락/보합 출발", _b("gap", ">", 0)),
        Rule("GAP_UP5", "형태", "갭상승 5% 이상 출발", "갭상승 5% 미만 출발", _b("gap", ">=", 0.05)),
        Rule("DD_SMALL", "형태", "구간 내 최대 되밀림 5% 이내", "구간 내 5% 넘게 되밀린 적 있음", _b("max_dd", ">=", -0.05)),
        Rule("UP_BARS", "형태", "1분봉 양봉 비율 55% 이상", "1분봉 양봉 비율 55% 미만", _b("up_ratio", ">=", 0.55)),
        Rule("BIG_MOVE", "형태", f"{window}분 상승률 10% 이상", f"{window}분 상승률 10% 미만", _b("ret_w", ">=", 0.10)),
    ]
    shapes = [
        Rule(f"SHAPE_{k}", "형태", f"형태: {v[0]}", f"형태: {v[0]} 아님",
             (lambda code: (lambda df: (df["shape"] == code).astype(float)))(k))
        for k, v in SHAPES.items() if k != "FLAT"
    ]
    top3 = Rule("RANK_TOP3", "형태", "급등 순위 1~3위", "급등 순위 4위 이하", _b("rank", "<=", 3))

    after = [
        Rule("P30_DOWN", "이후흐름", f"{p30} 가격 < {end} 가격 (30분 내 되밀림)",
             f"{p30} 가격 ≥ {end} 가격 (30분간 유지/추가상승)", _b("post_30", "<", 0)),
        Rule("P60_DOWN", "이후흐름", f"{p60} 가격 < {end} 가격", f"{p60} 가격 ≥ {end} 가격", _b("post_60", "<", 0)),
        Rule("CLOSE_DOWN", "이후흐름", f"장마감 가격 < {end} 가격", f"장마감 가격 ≥ {end} 가격", _b("post_close", "<", 0)),
        Rule("BREAK_HIGH", "이후흐름", f"{end} 이후 장중에 {window}분 고점을 +2% 이상 돌파",
             f"{end} 이후 {window}분 고점 +2% 돌파 실패", _b("broke_high", ">=", 0.5)),
        Rule("DROP5", "이후흐름", f"{end} 이후 -5% 이상 밀린 적 있음", f"{end} 이후 -5% 이상 밀린 적 없음",
             _b("post_low", "<=", -0.05)),
        Rule("RISE5", "이후흐름", f"{end} 이후 +5% 이상 추가상승한 적 있음", f"{end} 이후 +5% 추가상승 없음",
             _b("post_high", ">=", 0.05)),
    ]

    conds = form + shapes + [top3]
    outcomes = [r for r in after if r.id in ("P30_DOWN", "P60_DOWN", "CLOSE_DOWN", "BREAK_HIGH")]
    conditional = [
        Rule(f"{c.id}->{o.id}", "조건부", f"[{c.label}] → {o.label}", f"[{c.label}] → {o.label_neg}", o.fn, cond=c)
        for c in conds for o in outcomes
    ]
    return form + shapes + [top3] + after + conditional


def wilson(k: float, n: float, z: float) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def z_values(confidence: float, m: int, bonferroni: bool) -> tuple[float, float]:
    alpha = 1 - confidence
    z_raw = NormalDist().inv_cdf(1 - alpha / 2)
    z_adj = NormalDist().inv_cdf(1 - alpha / (2 * max(1, m))) if bonferroni else z_raw
    return z_raw, z_adj


def evaluate(df: pd.DataFrame, rules: list[Rule], confidence: float, target_rate: float,
             min_samples: int, bonferroni: bool) -> pd.DataFrame:
    """모든 규칙의 적중률/신뢰구간 계산. 각 규칙은 참/거짓 중 다수 쪽을 '패턴'으로 본다."""
    z_raw, z_adj = z_values(confidence, len(rules), bonferroni)
    rows = []
    if df.empty:
        return pd.DataFrame()
    for r in rules:
        s = r.fn(df)
        mask = pd.Series(True, index=df.index)
        if r.cond is not None:
            mask = r.cond.fn(df) == 1
        s = s[mask].dropna()
        n = int(len(s))
        if n == 0:
            continue
        k = float(s.sum())
        pos = k >= n - k
        kk = k if pos else n - k
        lo_raw, _ = wilson(kk, n, z_raw)
        lo, hi = wilson(kk, n, z_adj)
        # 형태 패턴은 해당 패턴을 보인 종목들의 이후 성과, 나머지는 조건군(또는 전체)의 이후 성과
        sub = df.loc[s.index[s == (1.0 if pos else 0.0)]] if r.category == "형태" else df.loc[s.index]
        rows.append({
            "pattern_id": r.id if pos else f"NOT {r.id}",
            "category": r.category,
            "pattern": r.label if pos else r.label_neg,
            "n": n,
            "hits": int(round(kk)),
            "rate": kk / n,
            "ci_low": lo,
            "ci_high": hi,
            "ci_low_uncorrected": lo_raw,
            "confirmed": bool(n >= min_samples and lo >= target_rate),
            "avg_post_30": float(sub["post_30"].mean()) if "post_30" in sub else math.nan,
            "avg_post_close": float(sub["post_close"].mean()) if "post_close" in sub else math.nan,
            "days": int(sub["date"].nunique()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["confirmed", "ci_low"], ascending=False).reset_index(drop=True)


def pattern_hits(df: pd.DataFrame, rule: Rule, positive: bool) -> pd.Series:
    """각 종목-일에 대해 패턴 적중(1)/미적중(0)/해당없음(NaN)."""
    s = rule.fn(df)
    if rule.cond is not None:
        s = s.where(rule.cond.fn(df) == 1)
    return s if positive else (1 - s)
