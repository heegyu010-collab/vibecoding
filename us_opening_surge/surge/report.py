"""결과 저장: CSV / 엑셀 / 차트 / HTML 보고서."""
from __future__ import annotations

import base64
import html
import io
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import RunResult
from .features import SHAPES, clock
from .rules import ACTIONABLE, pattern_hits

COLS_KR = {
    "date": "날짜", "rank": "순위", "ticker": "종목", "ret_w": "구간상승률", "gap": "갭(전일종가대비)",
    "r5": "첫5분수익률", "r_third": "1/3지점수익률", "r_2third": "2/3지점수익률", "frac_third": "1/3지점달성비율",
    "peak": "구간최고수익률", "t_high": "고점시각(분)", "low_ret": "구간최저수익률", "t_low": "저점시각(분)",
    "off_high": "고점대비종료가", "max_dd": "최대되밀림", "vwap_above": "VWAP위체류비율",
    "close_vs_vwap": "종료가/VWAP", "vol_first5": "첫5분거래대금비중", "efficiency": "경로효율성",
    "up_ratio": "양봉비율", "dollar_vol": "구간거래대금($)", "n_bars": "봉개수", "open": "시가",
    "close_w": "구간종료가", "prevclose": "전일종가", "shape": "형태코드", "shape_kr": "형태",
    "post_30": "이후30분수익률", "post_60": "이후60분수익률", "post_close": "종가까지수익률",
    "post_high": "이후최고수익률", "post_low": "이후최저수익률", "broke_high": "이후고점돌파",
}
PAT_KR = {
    "pattern_id": "패턴ID", "category": "구분", "pattern": "패턴", "n": "표본수", "hits": "적중수",
    "rate": "적중률", "ci_low": "신뢰구간하한(보정)", "ci_high": "신뢰구간상한(보정)",
    "ci_low_uncorrected": "신뢰구간하한(무보정)", "confirmed": "확정", "avg_post_30": "평균이후30분수익률",
    "avg_post_close": "평균종가까지수익률", "days": "일수",
}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def write_outputs(res: RunResult, out_root: Path) -> Path:
    s = res.settings
    out = Path(out_root) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rec, fin = res.records, res.final

    if rec.empty:
        (out / "report.txt").write_text(f"분석 가능한 데이터가 없습니다.\n{res.stop_reason}\n", encoding="utf-8")
        return out

    rec = rec.sort_values(["date", "rank"], ascending=[False, True]).reset_index(drop=True)
    path_cols = [f"p{m:02d}" for m in range(s.window)]
    paths = np.vstack([res.paths[(d, t)] for d, t in zip(rec["date"], rec["ticker"])])
    rec_p = pd.concat([rec, pd.DataFrame(paths, columns=path_cols)], axis=1)

    # ------------------------------------------------------------ 일별 요약
    daily = []
    for d, g in rec.groupby("date", sort=False):
        row = {
            "날짜": d, "종목수": len(g),
            "상위종목": ", ".join(f"{t}({r:+.1%})" for t, r in zip(g.ticker, g.ret_w)),
            "평균구간상승률": g.ret_w.mean(),
            "평균갭": g.gap.mean(),
            "대표형태": g.shape_kr.mode().iat[0],
            "고점마지막5분비율": (g.t_high >= s.window - 5).mean(),
            "이후30분하락비율": (g.post_30 < 0).mean() if g.post_30.notna().any() else np.nan,
            "평균이후30분수익률": g.post_30.mean(),
            "평균종가까지수익률": g.post_close.mean(),
        }
        for k, v in SHAPES.items():
            row[f"형태:{v[0]}"] = int((g["shape"] == k).sum())
        daily.append(row)
    daily = pd.DataFrame(daily)

    confirmed = fin[fin["confirmed"]] if not fin.empty else fin
    # 확정 패턴의 일별 적중 추이
    by_id = {r.id: r for r in res.rules}
    hit_tbl = pd.DataFrame({"날짜": daily["날짜"]})
    for pid, label in zip(confirmed.get("pattern_id", []), confirmed.get("pattern", [])):
        base = pid[4:] if pid.startswith("NOT ") else pid
        h = pattern_hits(rec, by_id[base], positive=not pid.startswith("NOT "))
        per = h.groupby(rec["date"]).agg(["sum", "count"])
        hit_tbl[label[:60]] = [
            f"{int(per.at[d, 'sum'])}/{int(per.at[d, 'count'])}" if d in per.index and per.at[d, "count"] else "-"
            for d in hit_tbl["날짜"]
        ]

    # ------------------------------------------------------------ CSV / Excel
    rec_p.rename(columns=COLS_KR).to_csv(out / "daily_top.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out / "daily_summary.csv", index=False, encoding="utf-8-sig")
    fin.rename(columns=PAT_KR).to_csv(out / "patterns_final.csv", index=False, encoding="utf-8-sig")
    if not res.history.empty:
        res.history.rename(columns=PAT_KR).to_csv(out / "patterns_history.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame({
        "항목": ["데이터소스", "분석기간", "분석일수", "종목-일 표본", "구간", "기준", "신뢰수준", "목표적중률",
                 "다중검정보정", "최소표본", "종료사유", "확정패턴수"],
        "값": [s.source, f"{rec.date.min()} ~ {rec.date.max()}", rec.date.nunique(), len(rec),
               f"개장 후 {s.window}분 (09:30~{clock(s.window)})", "시가 대비" if s.base == "open" else "전일종가 대비",
               f"{s.confidence:.0%}", f"{s.target_rate:.0%}", "Bonferroni" if s.bonferroni else "없음",
               s.min_samples, res.stop_reason, len(confirmed)],
    })
    try:
        with pd.ExcelWriter(out / "report.xlsx", engine="openpyxl") as xw:
            summary.to_excel(xw, sheet_name="요약", index=False)
            confirmed.rename(columns=PAT_KR).to_excel(xw, sheet_name="확정패턴", index=False)
            fin.rename(columns=PAT_KR).to_excel(xw, sheet_name="전체패턴", index=False)
            daily.to_excel(xw, sheet_name="일별요약", index=False)
            rec.rename(columns=COLS_KR).to_excel(xw, sheet_name="일별상위종목", index=False)
            hit_tbl.to_excel(xw, sheet_name="확정패턴_일별적중", index=False)
            pd.DataFrame(res.day_log).to_excel(xw, sheet_name="스캔로그", index=False)
    except Exception as e:  # 엑셀 파일이 열려 있는 경우 등
        print(f"  [경고] 엑셀 저장 실패: {e}")

    # ------------------------------------------------------------ 차트
    plt = _setup_matplotlib()
    x = np.arange(1, s.window + 1)
    imgs = {}

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in SHAPES.items():
        m = (rec["shape"] == k).to_numpy()
        if m.sum():
            ax.plot(x, paths[m].mean(axis=0) * 100, lw=2.2, label=f"{v[0]} ({m.sum()})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("개장 후 경과(분)")
    ax.set_ylabel("시가 대비 수익률(%)")
    ax.set_title("형태별 평균 경로")
    ax.legend()
    ax.grid(alpha=0.3)
    imgs["shape"] = _png(fig)
    fig.savefig(out / "chart_shapes.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    if not res.history.empty and not fin.empty:
        top_ids = fin[fin.category.isin(ACTIONABLE)].head(6)
        fig, ax = plt.subplots(figsize=(9, 5))
        for pid, label in zip(top_ids.pattern_id, top_ids.pattern):
            h = res.history[res.history.pattern_id == pid]
            ax.plot(h.days_used, h.ci_low * 100, marker="o", ms=3, label=label[:45])
        ax.axhline(s.target_rate * 100, color="red", ls="--", label=f"목표 {s.target_rate:.0%}")
        ax.set_xlabel("누적 분석 거래일 (최근 → 과거)")
        ax.set_ylabel("신뢰구간 하한(%)")
        ax.set_title(f"{s.confidence:.0%} 신뢰구간 하한 수렴 과정 (실전 활용 패턴 상위)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        imgs["conv"] = _png(fig)
        fig.savefig(out / "chart_convergence.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    days = list(daily["날짜"])[:30]
    cols = 5
    rows = int(np.ceil(len(days) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False, sharex=True)
    for ax, d in zip(axes.flat, days):
        m = (rec["date"] == d).to_numpy()
        for p in paths[m]:
            ax.plot(x, p * 100, lw=1)
        ax.set_title(str(d), fontsize=9)
        ax.axhline(0, color="gray", lw=0.6)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(days):]:
        ax.axis("off")
    fig.suptitle(f"일별 상위 {s.top_n} 종목 {s.window}분 경로 (시가 대비 %)", y=1.0)
    fig.tight_layout()
    imgs["daily"] = _png(fig)
    fig.savefig(out / "chart_daily.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, title in ((axes[0], "post_30", f"{clock(s.window)}→{clock(s.window + 30)} 수익률"),
                           (axes[1], "post_close", f"{clock(s.window)}→장마감 수익률")):
        v = rec[col].dropna() * 100
        if len(v):
            ax.hist(v.clip(-30, 30), bins=30, color="#4a7bd0")
            ax.axvline(0, color="red", lw=1)
            ax.set_title(f"{title} (평균 {v.mean():+.2f}%, 하락 {(v < 0).mean():.0%})")
        ax.grid(alpha=0.3)
    imgs["post"] = _png(fig)
    fig.savefig(out / "chart_after.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    (out / "report.html").write_text(_html(res, rec, daily, confirmed, imgs), encoding="utf-8")
    return out


def _fmt(df: pd.DataFrame, pct: set[str]) -> str:
    df = df.copy()
    for c in df.columns:
        if c in pct:
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}" if "수익" in c else f"{v:.1%}")
    return df.to_html(index=False, escape=True, border=0, classes="t")


def _html(res: RunResult, rec, daily, confirmed, imgs) -> str:
    s = res.settings
    fin = res.final
    pct = {"적중률", "신뢰구간하한(보정)", "신뢰구간상한(보정)", "신뢰구간하한(무보정)", "평균이후30분수익률",
           "평균종가까지수익률", "평균구간상승률", "평균갭", "고점마지막5분비율", "이후30분하락비율"}
    show = ["구분", "패턴", "표본수", "적중률", "신뢰구간하한(보정)", "신뢰구간상한(보정)", "평균이후30분수익률",
            "평균종가까지수익률", "일수"]

    def pat_table(df):
        return _fmt(df.rename(columns=PAT_KR)[show], pct) if not df.empty else "<p>없음</p>"

    act = fin[fin.category.isin(ACTIONABLE)].head(25) if not fin.empty else fin
    form = fin[fin.category == "형태"].head(25) if not fin.empty else fin
    shape_share = rec["shape_kr"].value_counts(normalize=True)
    shape_txt = " · ".join(f"{k} {v:.0%}" for k, v in shape_share.items())
    img = lambda k: f'<img src="data:image/png;base64,{imgs[k]}">' if k in imgs else ""
    dcols = ["날짜", "상위종목", "평균구간상승률", "대표형태", "고점마지막5분비율", "이후30분하락비율", "평균이후30분수익률"]

    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>미장 개장 급등 패턴</title>
<style>
body{{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;margin:0 auto;max-width:1150px;padding:16px;
background:#fafafa;color:#222;line-height:1.5}}
h1{{font-size:22px}} h2{{font-size:18px;margin-top:32px;border-bottom:2px solid #333;padding-bottom:4px}}
.box{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:10px 0}}
table.t{{border-collapse:collapse;font-size:13px;width:100%;background:#fff}}
table.t th,table.t td{{border:1px solid #ddd;padding:4px 6px;text-align:left;vertical-align:top}}
table.t th{{background:#f0f0f0}} table.t td:first-child{{white-space:nowrap}} img{{max-width:100%}} .warn{{color:#a33}}
.wrap{{overflow-x:auto}}
</style></head><body>
<h1>미장 개장 {s.window}분 급등 상위 {s.top_n} 종목 패턴 분석</h1>
<div class="box">
<b>분석기간</b> {rec.date.min()} ~ {rec.date.max()} ({rec.date.nunique()}거래일, 종목-일 {len(rec)}건) ·
<b>데이터</b> {html.escape(s.source)} · <b>기준</b> {'시가' if s.base == 'open' else '전일종가'} 대비 09:30~{clock(s.window)} 상승률<br>
<b>판정 기준</b> {s.confidence:.0%} 신뢰구간 하한 ≥ {s.target_rate:.0%}, 표본 ≥ {s.min_samples},
다중검정 보정 {'Bonferroni(' + str(len(res.rules)) + '개 동시검정)' if s.bonferroni else '없음'},
{s.stable_days}회 연속 확정 시 종료<br>
<b>종료 사유</b> {html.escape(res.stop_reason)}<br>
<b>형태 분포</b> {html.escape(shape_txt)}
</div>

<h2>1. 확정된 패턴 ({len(confirmed)}개)</h2>
<div class="wrap">{pat_table(confirmed)}</div>
<p class="warn">※ '형태' 패턴은 이미 급등한 종목을 사후에 고른 것이라 선정 효과가 섞여 있습니다.
실전 매매에는 {clock(s.window)} 시점에 판단 가능한 '이후흐름'·'조건부' 패턴을 보세요. 과거 통계는 미래 수익을 보장하지 않습니다.</p>

<h2>2. 신뢰도 수렴 과정</h2>{img('conv')}

<h2>3. 실전 활용 패턴 ({clock(s.window)} 이후 흐름) 상위 25</h2>
<div class="wrap">{pat_table(act)}</div>
{img('post')}

<h2>4. 30분 형태 패턴 상위 25</h2>
<div class="wrap">{pat_table(form)}</div>
{img('shape')}

<h2>5. 일별 분석</h2>
<div class="wrap">{_fmt(daily[dcols], pct)}</div>
{img('daily')}
<p>전체 수치: report.xlsx / daily_top.csv (분별 경로 p00~p{s.window - 1:02d} 포함)</p>
</body></html>"""
