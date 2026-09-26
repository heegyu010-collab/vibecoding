"""종목-일(stock-day) 단위 30분 패턴 특징 추출."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 형태 분류 코드 → (한글, 영문)
SHAPES = {
    "V_REBOUND": ("V자 반등형", "V-rebound"),
    "SPIKE_FADE": ("급등 후 되밀림형", "Spike & fade"),
    "EARLY_SPIKE": ("초반 급등 후 유지형", "Early spike & hold"),
    "LATE_SURGE": ("후반 급등형", "Late surge"),
    "STEADY": ("계단식 꾸준한 상승형", "Steady climb"),
    "CHOPPY": ("변동성 상승형", "Choppy rise"),
    "FLAT": ("보합/하락형", "Flat/down"),
}


def clock(minutes_after_open: int) -> str:
    t = 9 * 60 + 30 + minutes_after_open
    return f"{t // 60}:{t % 60:02d}"


def _minute_index(df: pd.DataFrame) -> np.ndarray:
    return (df.index.hour * 60 + df.index.minute - (9 * 60 + 30)).to_numpy()


def _price_before(close: np.ndarray, mins: np.ndarray, t: int) -> float:
    """시각 t(개장 후 분) 직전까지 마지막 종가."""
    sel = np.nonzero(mins < t)[0]
    return float(close[sel[-1]]) if len(sel) else math.nan


def classify(ret: float, peak: float, low_ret: float, t_low: int, frac_third: float, frac_2third: float,
             max_dd: float, window: int) -> str:
    if not ret > 0.005:
        return "FLAT"
    if low_ret <= -max(0.02, 0.3 * ret) and t_low <= window // 2:
        return "V_REBOUND"
    if peak - ret >= max(0.03, 0.33 * peak):
        return "SPIKE_FADE"
    if frac_third >= 0.75:
        return "EARLY_SPIKE"
    if frac_2third <= 0.4:
        return "LATE_SURGE"
    if max_dd > -max(0.02, 0.35 * ret):
        return "STEADY"
    return "CHOPPY"


def compute_features(bars: pd.DataFrame, window: int, prevclose: float, base: str = "open",
                     min_bar_ratio: float = 0.6) -> tuple[dict, np.ndarray] | None:
    """개장 후 window 분 구간의 특징과 이후 흐름(10시 이후 결과)을 계산.

    반환: (특징 dict, 시가 대비 누적수익률 경로[window]) / 데이터 부족 시 None
    """
    if bars is None or bars.empty:
        return None
    mins = _minute_index(bars)
    in_w = (mins >= 0) & (mins < window)
    if in_w.sum() < max(3, int(window * min_bar_ratio)) or mins[in_w].min() > 5:
        return None

    w = bars[in_w]
    wm = mins[in_w]
    o = float(w["open"].iloc[0])
    if not o > 0:
        return None

    # 1분 격자로 정렬(거래 없는 분은 직전 종가로 채움)
    grid = pd.RangeIndex(window)
    g = pd.DataFrame(w.to_numpy(), index=wm, columns=w.columns)
    g = g[~g.index.duplicated(keep="last")].reindex(grid)
    g["close"] = g["close"].ffill().fillna(o)
    for c in ("open", "high", "low"):
        g[c] = g[c].fillna(g["close"])
    g["volume"] = g["volume"].fillna(0.0)

    close = g["close"].to_numpy()
    high = g["high"].to_numpy()
    low = g["low"].to_numpy()
    vol = g["volume"].to_numpy()

    denom = o if base == "open" else prevclose
    if not (denom and denom > 0 and math.isfinite(denom)):
        return None
    path = close / denom - 1.0
    c_end = float(close[-1])
    ret = c_end / denom - 1.0

    t_high = int(np.argmax(high))
    t_low = int(np.argmin(low))
    H = float(high.max())
    L = float(low.min())
    peak = H / denom - 1.0
    low_ret = L / denom - 1.0
    run_max = np.maximum.accumulate(high)
    max_dd = float(np.min(low / run_max - 1.0))

    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    cumv = np.cumsum(vol)
    vwap = np.where(cumv > 0, np.cumsum(tp.to_numpy() * vol) / np.where(cumv > 0, cumv, 1), close)
    traded = vol > 0
    vwap_above = float(np.mean(close[traded] >= vwap[traded])) if traded.any() else math.nan

    dollar = float(np.sum(close * vol))
    first5 = min(5, window)
    vol_first5 = float(np.sum((close * vol)[:first5]) / dollar) if dollar > 0 else math.nan
    moves = np.abs(np.diff(np.concatenate([[o], close])))
    efficiency = float(abs(c_end - o) / moves.sum()) if moves.sum() > 0 else math.nan
    up_ratio = float(np.mean(g["close"].to_numpy() > g["open"].to_numpy()))

    third, two_third = max(1, window // 3), max(1, 2 * window // 3)
    r5 = float(path[first5 - 1])
    r_third = float(path[third - 1])
    r_2third = float(path[two_third - 1])
    gain = ret if base == "open" else c_end / o - 1.0
    frac_third = (close[third - 1] / o - 1.0) / gain if gain > 0 else math.nan
    frac_2third = (close[two_third - 1] / o - 1.0) / gain if gain > 0 else math.nan

    # 구간 종료 후 흐름 (구간 종료 시점 가격 대비)
    all_close = bars["close"].to_numpy()
    after = mins >= window
    post = {}
    for key, t in (("post_30", window + 30), ("post_60", window + 60)):
        p = _price_before(all_close, mins, t) if (mins >= t - 1).any() else math.nan
        post[key] = p / c_end - 1.0 if math.isfinite(p) else math.nan
    if after.any():
        post["post_close"] = float(all_close[-1]) / c_end - 1.0 if mins.max() >= 380 else math.nan
        post["post_high"] = float(bars["high"].to_numpy()[after].max()) / c_end - 1.0
        post["post_low"] = float(bars["low"].to_numpy()[after].min()) / c_end - 1.0
        # 1센트 돌파는 의미가 없으므로 구간 고점보다 2% 이상 높게 거래돼야 돌파로 인정
        post["broke_high"] = float(bars["high"].to_numpy()[after].max() >= H * 1.02)
    else:
        post.update(post_close=math.nan, post_high=math.nan, post_low=math.nan, broke_high=math.nan)

    shape = classify(gain, H / o - 1.0, L / o - 1.0, t_low,
                     frac_third if math.isfinite(frac_third) else 0.0,
                     frac_2third if math.isfinite(frac_2third) else 1.0, max_dd, window)

    feats = {
        "ret_w": ret,
        "gap": o / prevclose - 1.0 if prevclose and prevclose > 0 else math.nan,
        "r5": r5, "r_third": r_third, "r_2third": r_2third,
        "frac_third": frac_third,
        "peak": peak, "t_high": t_high, "low_ret": low_ret, "t_low": t_low,
        "off_high": c_end / H - 1.0,
        "max_dd": max_dd,
        "vwap_above": vwap_above,
        "close_vs_vwap": c_end / float(vwap[-1]) - 1.0 if vwap[-1] > 0 else math.nan,
        "vol_first5": vol_first5,
        "efficiency": efficiency,
        "up_ratio": up_ratio,
        "dollar_vol": dollar,
        "n_bars": int(in_w.sum()),
        "open": o, "close_w": c_end,
        "prevclose": prevclose,
        "shape": shape,
        "shape_kr": SHAPES[shape][0],
        **post,
    }
    return feats, path
