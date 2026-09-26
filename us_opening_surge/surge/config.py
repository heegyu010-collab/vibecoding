"""분석 설정값."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # 데이터
    source: str = "yahoo"               # yahoo | alpaca | synthetic
    alpaca_feed: str = "sip"            # sip | iex (알파카 무료계정은 iex 만 되는 경우도 있음)
    universe_file: str | None = None    # 직접 지정한 종목 리스트 파일 (한 줄에 한 종목)
    include_etf: bool = False

    # 스캔 조건
    top_n: int = 10                     # 일별 급등 상위 N 종목
    window: int = 30                    # 개장 후 분석 구간(분)
    base: str = "open"                  # open: 시가 대비 / prevclose: 전일 종가 대비(갭 포함)
    min_price: float = 1.0              # 시가 최소 가격($)
    min_window_dollar_volume: float = 1_000_000  # 30분 거래대금 최소($) - 초저유동성 종목 제외
    max_candidates: int = 800           # 하루에 분봉을 받아볼 최대 후보 수
    batch_size: int = 50                # 분봉 요청 묶음 크기
    bound_tolerance: float = 0.01       # 일봉 상한(고가/시가)과 분봉 시가 차이 허용치

    # 회귀(백워드) 테스트
    end_date: str | None = None         # 가장 최근 분석일(YYYY-MM-DD). 없으면 직전 완료 거래일
    max_days: int = 250                 # 최대 몇 거래일까지 과거로 갈지
    min_days: int = 5                   # 최소 분석 일수
    stable_days: int = 3                # 같은 패턴이 연속 몇 번 확정돼야 멈출지
    confidence: float = 0.95            # 신뢰수준
    target_rate: float = 0.70           # 패턴 적중률 목표(신뢰구간 하한이 이 값 이상이면 확정)
    min_samples: int = 30               # 패턴 판정 최소 표본 수
    bonferroni: bool = True             # 다중검정 보정(여러 패턴을 동시에 시험하므로 권장)
    stop_on: str = "actionable"         # actionable: 10시 이후 흐름 패턴 확정 시 종료 / any / none

    # 경로
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("output")

    @property
    def min_daily_dollar_volume(self) -> float:
        # 30분 거래대금 조건을 만족하는 종목은 일 거래대금도 최소 그 절반 이상이라는 느슨한 사전 필터
        return self.min_window_dollar_volume * 0.5
