import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surge.backtest import run  # noqa: E402
from surge.config import Settings  # noqa: E402
from surge.features import compute_features  # noqa: E402
from surge.providers.base import NY  # noqa: E402
from surge.providers.synthetic import SyntheticProvider  # noqa: E402
from surge.report import write_outputs  # noqa: E402
from surge.rules import build_rules, evaluate, wilson  # noqa: E402
from surge.scanner import scan_day  # noqa: E402
from surge.store import DailyStore, MinuteStore  # noqa: E402


def test_wilson_reference_values():
    lo, hi = wilson(8, 10, 1.959964)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson(0, 0, 1.96)[0] != wilson(0, 0, 1.96)[0]  # NaN


def _bars(closes, day="2026-09-25"):
    idx = pd.date_range(pd.Timestamp(f"{day} 09:30", tz=NY), periods=len(closes), freq="1min")
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                         "volume": np.full(len(c), 1000.0)}, index=idx)


def test_features_steady_climb_and_after():
    closes = list(np.linspace(10, 12, 30)) + list(np.linspace(12, 11, 30)) + [11.0] * 330
    f, path = compute_features(_bars(closes), 30, prevclose=9.5)
    assert f["ret_w"] == pytest.approx(0.2)
    assert f["gap"] == pytest.approx(10 / 9.5 - 1)
    assert f["t_high"] == 29 and f["shape"] == "STEADY"
    assert len(path) == 30 and path[-1] == pytest.approx(0.2)
    assert f["post_30"] < 0 and f["post_close"] == pytest.approx(11 / 12 - 1)
    assert f["broke_high"] == 0.0


def test_scanner_matches_brute_force(tmp_path):
    prov = SyntheticProvider(n_tickers=120)
    s = Settings(source="synthetic", min_window_dollar_volume=0, cache_dir=tmp_path)
    tickers = prov.universe()
    daily, minute = DailyStore(prov, tickers, tmp_path), MinuteStore(prov, tmp_path)
    days = prov.trading_days(date(2026, 9, 14), date(2026, 9, 25))
    for prev, day in zip(days[:-1], days[1:]):
        daily.prefetch([prev, day])
        got = scan_day(day, prev, daily, minute, s)
        assert got.exact
        d = daily.get(day, prev)
        brute = []
        for t, bars in prov.minute_bars(tickers, day).items():
            if d.at[t, "open"] < s.min_price:
                continue
            f, _ = compute_features(bars, 30, d.at[t, "prevclose"])
            brute.append((f["ret_w"], t))
        expect = [t for _, t in sorted(brute, reverse=True)[:10]]
        assert [r["ticker"] for r in got.records] == expect


def test_rules_are_two_sided_and_bonferroni_is_stricter():
    df = pd.DataFrame({"date": [date(2026, 9, 25)] * 40, "rank": range(1, 41),
                       "post_30": [-0.01] * 34 + [0.01] * 6})
    for c in ("t_high", "t_low", "low_ret", "off_high", "vwap_above", "r5", "frac_third", "vol_first5",
              "gap", "max_dd", "up_ratio", "ret_w", "post_60", "post_close", "broke_high", "post_low",
              "post_high"):
        df[c] = np.nan
    df["shape"] = "STEADY"
    rules = build_rules(30, 10)
    a = evaluate(df, rules, 0.95, 0.7, 30, bonferroni=True).set_index("pattern_id")
    b = evaluate(df, rules, 0.95, 0.7, 30, bonferroni=False).set_index("pattern_id")
    assert a.at["P30_DOWN", "rate"] == pytest.approx(0.85)
    assert a.at["P30_DOWN", "ci_low"] < b.at["P30_DOWN", "ci_low"]


def test_end_to_end_synthetic(tmp_path):
    s = Settings(source="synthetic", max_days=8, min_days=3, stable_days=2, end_date="2026-09-25",
                 cache_dir=tmp_path / "c", output_dir=tmp_path / "o", min_window_dollar_volume=0)
    res = run(s, SyntheticProvider(n_tickers=80))
    assert not res.records.empty and res.records.groupby("date").size().max() == 10
    out = write_outputs(res, s.output_dir)
    for name in ("report.html", "report.xlsx", "daily_top.csv", "patterns_final.csv", "chart_daily.png"):
        assert (out / name).exists(), name
