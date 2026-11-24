"""Standalone chart-based verification for gold_data.py (multi-source).

For each gold asset x timeframe x data source: fetch data, run OHLC integrity
checks, render a candlestick PNG to verification/, and print a pass/fail report.

Additional checks:
  * 4h aggregation: independently resample 1h -> 4h and compare bar-by-bar.
  * Cross-source consistency: for assets supported by >1 source, align daily
    closes by calendar date and verify they match within a small tolerance —
    proving the multi-source architecture returns consistent prices.

Usage:
    python verify_gold_data.py                 # all assets x timeframes x all sources
    python verify_gold_data.py --source ths    # only the Tonghuashun source
    python verify_gold_data.py --source all    # explicit (same as default)
Output:
    verification/<asset>_<timeframe>_<source>.png   candlestick charts
    console report
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

import gold_data

def _setup_cjk_font() -> str | None:
    """Register a CJK font straight from its file.

    The venv's matplotlib font cache does not always pick up system CJK fonts
    by name, so lookup-by-name silently falls back to DejaVu Sans and Chinese
    titles render as tofu boxes. Registering the TTF/TTC file explicitly and
    reading its real family name is reliable.
    """
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name] + [
                f for f in plt.rcParams["font.sans-serif"] if f != name
            ]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


_CJK_FONT = _setup_cjk_font()


def _build_mpf_style():
    """mplfinance style carrying CJK rcParams.

    mpf.plot() applies its own style, which overrides rcParams set globally, so
    the font must be injected through the style's ``rc`` mapping — otherwise the
    title silently falls back to DejaVu Sans and Chinese renders as boxes.
    """
    from color_style import DOWN_COLOR, UP_COLOR

    rc = {"axes.unicode_minus": False}
    if _CJK_FONT:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = [_CJK_FONT, "DejaVu Sans"]
    marketcolors = mpf.make_marketcolors(
        base_mpf_style="yahoo",
        up=UP_COLOR,
        down=DOWN_COLOR,
        edge="inherit",
        volume="inherit",
        ohlc="inherit",
    )
    try:
        return mpf.make_mpf_style(
            base_mpf_style="yahoo", marketcolors=marketcolors, rc=rc
        )
    except TypeError:
        style = mpf.make_mpf_style(
            base_mpf_style="yahoo", marketcolors=marketcolors
        )
        style["rc"] = rc
        return style


_MPF_STYLE = _build_mpf_style()

OUT_DIR = Path("verification")
OUT_DIR.mkdir(exist_ok=True)

TF_DAYS = {"1mo": 3650, "1w": 1825, "1d": 730, "4h": 180, "1h": 60, "30m": 30, "15m": 14, "5m": 7, "1m": 5}
ASSETS = ["COMEX", "SHFE_AU"]
TIMEFRAMES = ["1d", "1h", "4h"]


def render_candlestick(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        print(f"  [skip chart] {out_path.name}: no data")
        return
    d = df.tail(90).copy()
    d.index = pd.to_datetime(d["Datetime"])
    d = d[["Open", "High", "Low", "Close", "Volume"]]
    show_vol = bool("Volume" in d.columns and d["Volume"].fillna(0).gt(0).any())
    fig, _ = mpf.plot(
        d, type="candle", style=_MPF_STYLE, title=title,
        volume=show_vol, figsize=(14, 6), returnfig=True, block=False,
    )
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fmt_bool(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_one(asset: str, tf: str, source: str):
    """Fetch + validate one combo. Returns True (pass) / False (fail) / None (no data, skip)."""
    end = datetime.now()
    start = end - timedelta(days=TF_DAYS.get(tf, 365))
    df = gold_data.fetch_gold_kline(asset, tf, start, end, min_bars=0, source=source)
    report = gold_data.validate_ohlcv(df)
    info = gold_data.asset_info(asset, source=source)
    label = gold_data.SOURCE_LABELS.get(source, source)

    name = f"{info['name']}({asset}) {tf} [{info['unit']}] <{label}>"
    print(f"\n=== {name} ===")
    if report["row_count"] == 0:
        print(f"  rows: 0  -> NO DATA (source '{label}' unavailable for this asset/timeframe) [SKIP]")
        return None
    print(f"  rows: {report['row_count']}  range: {report.get('first_datetime')} -> {report.get('last_datetime')}")
    for k, v in report["checks"].items():
        print(f"    {k:32s}: {fmt_bool(bool(v))}")
    if not report["ok"] and "bad_row_sample" in report:
        print(f"    bad rows sample: {report['bad_row_sample']}")

    render_candlestick(df, f"{name}  (last {min(90, len(df))} bars)",
                       OUT_DIR / f"{asset}_{tf}_{source}.png")
    return report["ok"]


def _daily_close_series(asset: str, source: str) -> pd.Series:
    end = datetime.now()
    df = gold_data.fetch_gold_kline(asset, "1d", end - timedelta(days=730), end, min_bars=0, source=source)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.copy()
    s["date"] = pd.to_datetime(s["Datetime"]).dt.normalize()
    return s.drop_duplicates("date").set_index("date")["Close"].astype(float)


def _count_mismatches(a: pd.Series, b: pd.Series, recent_days: int, lag: int) -> tuple:
    """Align b to a with a date lag (b index shifted +lag days), compare recent window."""
    if a.empty or b.empty:
        return 0, 0, 0.0, pd.Index([])
    b_shifted = b.copy()
    b_shifted.index = b_shifted.index + pd.Timedelta(days=lag)
    common = a.index.intersection(b_shifted.index)
    if len(common) == 0:
        return 0, 0, 0.0, common
    # restrict to the most recent recent_days of the overlap
    cutoff = common.max() - pd.Timedelta(days=recent_days)
    common = common[common >= cutoff]
    if len(common) == 0:
        return 0, 0, 0.0, common
    av = a.loc[common]
    bv = b_shifted.loc[common]
    rel = ((av - bv).abs() / av.replace(0, pd.NA)).fillna(0)
    bad = int((rel > 0.005).sum())  # 0.5% relative tolerance
    return len(common), bad, float(rel.max()), common


def verify_cross_source() -> bool:
    """Compare daily closes across sources, allowing a +/-2 day date-label offset.

    Two known, benign differences are accounted for:
      1. Date-labeling offset: SHFE futures have a night session; THS stamps a
         bar on the session-start date while akshare stamps the trade date, so
         the same price can appear one calendar day apart. We search a small lag.
      2. Contract roll: akshare AU0 is a *continuous* main contract (stitched
         across rolls) while THS returns the *current* main contract (a single
         fixed code). Before that code became front-month they track different
         contracts, so history diverges. We only compare a recent window where
         both reference the same front-month contract.
    """
    print("\n=== Verify: cross-source daily close consistency (recent window, lag-aware) ===")
    all_ok = True
    recent_days = 90  # ~60 trading days; both sources track the current front-month here
    for asset in ASSETS:
        sources = gold_data.list_data_sources(asset)
        if len(sources) < 2:
            print(f"  {asset}: only {sources} -> skip cross-source")
            continue
        closes = {src: _daily_close_series(asset, src) for src in sources}
        have = {s: srs for s, srs in closes.items() if not srs.empty}
        if len(have) < 2:
            print(f"  {asset}: only {list(have)} returned data -> skip cross-source "
                  f"(other source(s) blocked/unavailable in this environment)")
            continue

        anchor = max(have, key=lambda s: len(have[s]))
        for src, srs in have.items():
            if src == anchor:
                continue
            best = None  # (lag, overlap, bad, max_rel)
            for lag in (-2, -1, 0, 1, 2):
                overlap, bad, max_rel, _ = _count_mismatches(have[anchor], srs, recent_days, lag)
                if overlap == 0:
                    continue
                if best is None or bad < best[2] or (bad == best[2] and overlap > best[1]):
                    best = (lag, overlap, bad, max_rel)
            if best is None:
                print(f"  {asset}: {anchor} vs {src} -> no overlapping recent dates")
                all_ok = False
                continue
            lag, overlap, bad, max_rel = best
            ok = bad == 0
            print(f"  {asset}: {anchor} vs {src}: recent_overlap={overlap} "
                  f"best_lag={lag:+d}d mismatches={bad} max_rel={max_rel:.4%} -> {fmt_bool(ok)}")
            if not ok:
                all_ok = False
                a = have[anchor]
                b_shifted = srs.copy(); b_shifted.index = b_shifted.index + pd.Timedelta(days=lag)
                common = a.index.intersection(b_shifted.index)
                cutoff = common.max() - pd.Timedelta(days=recent_days)
                common = common[common >= cutoff]
                rel = ((a.loc[common] - b_shifted.loc[common]).abs() / a.loc[common].replace(0, pd.NA)).fillna(0)
                for dt in common[rel > 0.005][:3]:
                    print(f"      {dt.date()}: {anchor}={a.loc[dt]:.4f}  {src}={b_shifted.loc[dt]:.4f}  "
                          f"diff={abs(a.loc[dt]-b_shifted.loc[dt]):.4f}")
    return all_ok


def verify_4h_resample() -> bool:
    """Verify 4h bars are correctly aggregated from 1h bars.

    Eastmoney/akshare resample 1h->4h (THS serves 4h natively, so it is not
    checked here). Try COMEX via eastmoney first; fall back to SHFE_AU via
    akshare (both 1h and 4h resample paths).
    """
    print("\n=== Verify: 4h aggregation from 1h is arithmetically correct ===")
    candidates = [("COMEX", "eastmoney"), ("SHFE_AU", "akshare")]
    h1 = h4 = None
    used = None
    for asset, src in candidates:
        if src not in gold_data.list_data_sources(asset):
            continue
        end = datetime.now()
        a = gold_data.fetch_gold_kline(asset, "1h", end - timedelta(days=30), end, min_bars=0, source=src)
        b = gold_data.fetch_gold_kline(asset, "4h", end - timedelta(days=30), end, min_bars=0, source=src)
        if not a.empty and not b.empty:
            h1, h4, used = a, b, f"{asset}/{src}"
            break
    if h1 is None or h4 is None:
        print("  [skip] no source returned both 1h and 4h data (eastmoney may be blocked)")
        return None
    print(f"  using {used}")

    h1c = h1.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime")
    agg = (
        h1c.resample("4h")
        .agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
             Close=("Close", "last"), Volume=("Volume", "sum"))
        .dropna(subset=["Open"])
        .reset_index()
    )
    merged = agg.merge(h4, on="Datetime", suffixes=("_expect", "_got"))
    if merged.empty:
        print("  [skip] no overlapping 4h bars")
        return None

    # Exclude the most recent bar: while the market is open it is the in-progress
    # 4h bucket, and its high/low/close change tick-by-tick. Because the 1h and
    # 4h datasets are fetched independently a few seconds apart, that one live
    # bar can legitimately differ — it is not an aggregation error.
    merged = merged.sort_values("Datetime")
    excluded = merged["Datetime"].iloc[-1]
    merged = merged.iloc[:-1]
    if merged.empty:
        print("  [skip] only one (in-progress) 4h bar available to compare")
        return None
    print(f"  excluded last in-progress bar {excluded} (live tick drift between fetches)")

    tol = 1e-6
    bad = 0
    for _, row in merged.iterrows():
        if (abs(row["Open_expect"] - row["Open_got"]) > tol
                or abs(row["High_expect"] - row["High_got"]) > tol
                or abs(row["Low_expect"] - row["Low_got"]) > tol
                or abs(row["Close_expect"] - row["Close_got"]) > tol):
            bad += 1
            if bad <= 3:
                print(f"    MISMATCH {row['Datetime']}")
    ok = bad == 0
    print(f"  compared={len(merged)} 4h bars  mismatches={bad}  -> {fmt_bool(ok)}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify gold_data multi-source collection")
    parser.add_argument("--source", default="all",
                        help="data source to verify: eastmoney|ths|akshare|all (default: all)")
    args = parser.parse_args()

    all_ok = True
    for asset in ASSETS:
        avail = gold_data.list_data_sources(asset)
        sources = avail if args.source == "all" else [args.source]
        for src in sources:
            if src not in avail:
                print(f"\n[skip] {asset}: source '{src}' not supported (available: {avail})")
                all_ok = False
                continue
            for tf in TIMEFRAMES:
                try:
                    r = check_one(asset, tf, src)
                    if r is False:
                        all_ok = False
                except Exception as e:
                    print(f"\n!!! {asset} {tf} [{src}] raised: {e}")
                    all_ok = False

    for fn in (verify_4h_resample, verify_cross_source):
        try:
            r = fn()
            if r is False:
                all_ok = False
        except Exception as e:
            print(f"\n!!! {fn.__name__} raised: {e}")
            all_ok = False

    print("\n" + ("=" * 60))
    print("OVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print(f"Charts written to: {OUT_DIR.resolve()}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
