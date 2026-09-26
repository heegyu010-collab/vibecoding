"""네트워크 없이 야후/알파카 응답 형식 파싱을 검증 (모의 응답)."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surge.providers import alpaca as alp  # noqa: E402
from surge.providers import yahoo  # noqa: E402


def _yf_frame(tickers, index):
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]],
                                      names=["Ticker", "Price"])
    data = np.tile(np.arange(len(index), dtype=float)[:, None] + 10, (1, len(cols)))
    df = pd.DataFrame(data, index=index, columns=cols)
    df[(tickers[-1], "Close")] = np.nan  # 마지막 종목은 데이터 없음 흉내
    return df


def test_yahoo_minute_bars_parsing(monkeypatch):
    idx = pd.date_range("2026-09-25 13:25", periods=400, freq="1min", tz="UTC")  # 09:25 NY 부터
    fake = _yf_frame(["AAPL", "BRK-B", "DEAD"], idx)
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: fake)
    p = yahoo.YahooProvider(pause=0)
    got = p.minute_bars(["AAPL", "BRK.B", "DEAD"], date(2026, 9, 25))
    assert set(got) == {"AAPL", "BRK.B"}
    df = got["BRK.B"]
    assert str(df.index.tz) == "America/New_York"
    assert df.index[0].strftime("%H:%M") == "09:30" and df.index[-1].strftime("%H:%M") == "15:59"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_yahoo_daily_parsing(monkeypatch):
    idx = pd.DatetimeIndex(["2026-09-24", "2026-09-25"])
    fake = _yf_frame(["AAPL", "MSFT"], idx)
    fake[("MSFT", "Close")] = 5.0
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: fake)
    df = yahoo.YahooProvider(pause=0).daily_bars(["AAPL", "MSFT"], date(2026, 9, 24), date(2026, 9, 25))
    assert set(df.ticker) == {"AAPL", "MSFT"} and set(df.date) == {date(2026, 9, 24), date(2026, 9, 25)}


class _Resp:
    def __init__(self, js):
        self.status_code, self._js, self.text = 200, js, ""

    def json(self):
        return self._js


def test_alpaca_pagination_and_tz(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    p = alp.AlpacaProvider()
    bar = lambda t: {"t": t, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 100}
    pages = [
        {"bars": {"AAA": [bar("2026-09-25T13:30:00Z")]}, "next_page_token": "x"},
        {"bars": {"AAA": [bar("2026-09-25T13:31:00Z")], "BBB": [bar("2026-09-25T19:59:00Z")]},
         "next_page_token": None},
    ]
    monkeypatch.setattr(p.session, "get", lambda *a, **k: _Resp(pages.pop(0)))
    got = p.minute_bars(["AAA", "BBB"], date(2026, 9, 25))
    assert len(got["AAA"]) == 2 and got["AAA"].index[0].strftime("%H:%M") == "09:30"
    assert got["BBB"].index[0].strftime("%H:%M") == "15:59"
