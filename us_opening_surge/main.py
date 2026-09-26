"""미국주식 개장 30분 급등 상위 10종목 패턴 분석 + 95% 신뢰 패턴 백워드 회귀테스트.

예)
  python main.py                               # 야후(무료, 최근 약 20거래일)
  python main.py --source alpaca --max-days 250 # 알파카(무료 API 키, 수년치)
  python main.py --source synthetic             # 인터넷 없이 동작 확인용 가상 데이터
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from surge.backtest import run
from surge.config import Settings
from surge.providers import make_provider
from surge.report import write_outputs

HERE = Path(__file__).resolve().parent


def parse(argv=None) -> Settings:
    d = Settings()
    p = argparse.ArgumentParser(description="미장 개장 N분 급등 상위 종목 패턴 분석 (백워드 회귀 테스트)")
    p.add_argument("--source", default=d.source, choices=["yahoo", "alpaca", "synthetic"], help="데이터 소스")
    p.add_argument("--alpaca-feed", default=d.alpaca_feed, choices=["sip", "iex"])
    p.add_argument("--universe-file", default=None, help="종목 리스트 파일(없으면 나스닥/NYSE 전 종목)")
    p.add_argument("--include-etf", action="store_true", help="ETF 포함")
    p.add_argument("--top", type=int, default=d.top_n, help="일별 상위 종목 수 (기본 10)")
    p.add_argument("--window", type=int, default=d.window, help="개장 후 분석 구간(분) (기본 30)")
    p.add_argument("--base", default=d.base, choices=["open", "prevclose"], help="상승률 기준가")
    p.add_argument("--min-price", type=float, default=d.min_price, help="최소 주가($)")
    p.add_argument("--min-dollar-volume", type=float, default=d.min_window_dollar_volume,
                   help="구간 최소 거래대금($)")
    p.add_argument("--end-date", default=None, help="가장 최근 분석일 YYYY-MM-DD (기본: 직전 완료 거래일)")
    p.add_argument("--max-days", type=int, default=d.max_days, help="최대 회귀 거래일 수")
    p.add_argument("--min-days", type=int, default=d.min_days, help="최소 분석 거래일 수")
    p.add_argument("--stable-days", type=int, default=d.stable_days, help="연속 확정 횟수")
    p.add_argument("--confidence", type=float, default=d.confidence, help="신뢰수준 (기본 0.95)")
    p.add_argument("--target-rate", type=float, default=d.target_rate,
                   help="패턴 적중률 목표: 신뢰구간 하한이 이 값 이상이면 확정 (기본 0.70)")
    p.add_argument("--min-samples", type=int, default=d.min_samples, help="패턴 최소 표본 수")
    p.add_argument("--no-bonferroni", action="store_true", help="다중검정 보정 끄기(패턴이 더 쉽게 확정됨)")
    p.add_argument("--stop-on", default=d.stop_on, choices=["actionable", "any", "none"],
                   help="actionable: 10시 이후 흐름 패턴 확정 시 종료 / any: 형태 패턴 포함 / none: max-days 까지")
    p.add_argument("--cache-dir", default=str(HERE / "cache"))
    p.add_argument("--output-dir", default=str(HERE / "output"))
    a = p.parse_args(argv)
    return Settings(
        source=a.source, alpaca_feed=a.alpaca_feed, universe_file=a.universe_file, include_etf=a.include_etf,
        top_n=a.top, window=a.window, base=a.base, min_price=a.min_price,
        min_window_dollar_volume=a.min_dollar_volume, end_date=a.end_date, max_days=a.max_days,
        min_days=a.min_days, stable_days=a.stable_days, confidence=a.confidence, target_rate=a.target_rate,
        min_samples=a.min_samples, bonferroni=not a.no_bonferroni, stop_on=a.stop_on,
        cache_dir=Path(a.cache_dir), output_dir=Path(a.output_dir),
    )


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    s = parse(argv)
    t0 = time.time()
    print(f"=== 미장 개장 {s.window}분 급등 상위 {s.top_n} 패턴 분석 ({s.source}) ===")
    provider = make_provider(s)
    res = run(s, provider)
    out = write_outputs(res, s.output_dir)

    print("\n=== 결과 ===")
    print(f"종료 사유: {res.stop_reason}")
    if not res.final.empty:
        conf = res.final[res.final.confirmed]
        print(f"확정 패턴 {len(conf)}개 (신뢰수준 {s.confidence:.0%}, 하한 ≥ {s.target_rate:.0%})")
        for _, r in conf.head(15).iterrows():
            print(f"  [{r.category}] {r.pattern}  적중 {r.hits}/{r.n} = {r.rate:.1%} "
                  f"(하한 {r.ci_low:.1%})")
        if conf.empty:
            print("  확정 패턴 없음 - 가장 근접한 후보:")
            for _, r in res.final.head(5).iterrows():
                print(f"  [{r.category}] {r.pattern}  {r.hits}/{r.n} = {r.rate:.1%} (하한 {r.ci_low:.1%})")
    print(f"\n보고서: {out / 'report.html'}")
    print(f"소요 시간: {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
