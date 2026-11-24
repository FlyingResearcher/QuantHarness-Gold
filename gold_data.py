"""Gold (XAU) data vendor for QuantHarness-Gold — multi-source.

Three pluggable data sources are supported and can be switched at call time:

  * eastmoney (default) — direct HTTP to push2his.eastmoney.com. Covers COMEX
    (101.GC00Y, USD/oz) and SHFE main (113.aum, CNY/g). All intra-day periods.
  * ths (同花顺) — modern fuyao common_hq_aggr quote API (quota-h.10jqka.com.cn).
    Web-side only serves SHFE futures (market 65, dynamic main contract auYYMM);
    COMEX gold has no k-line page on 10jqka web, so it is unsupported here.
  * akshare — Sina-sourced. COMEX via futures_foreign_hist_sina (daily only);
    SHFE via futures_main_sina (daily) + futures_zh_min_sina (intraday).

Pricing-anchor rule (project convention): international gold is priced on
COMEX (USD/oz). Default symbol is COMEX Gold continuous. SHFE main (CNY/g) is
a secondary China-market reference. NEVER use domestic Au99.99 (CNY/g) to stand
in for USD/oz gold.

Every provider returns the unified project schema
[Datetime, Open, High, Low, Close, Volume] at the fetcher boundary, so
downstream consumers never see raw field ordering (Eastmoney is O,C,H,L).
"""
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json as _json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------------
# Data-source configuration
# ---------------------------------------------------------------------------
DATA_SOURCES: Tuple[str, ...] = ("eastmoney", "ths", "akshare")
DEFAULT_SOURCE = "eastmoney"
SOURCE_LABELS: Dict[str, str] = {
    "eastmoney": "东方财富",
    "ths": "同花顺",
    "akshare": "akshare",
}
# Environment variable that overrides the default source without touching code,
# e.g.  $env:GOLD_DATA_SOURCE = "ths"
SOURCE_ENV_VAR = "GOLD_DATA_SOURCE"

# Runtime override set via set_default_source(); process-scoped.
_DEFAULT_SOURCE_OVERRIDE: Optional[str] = None

STANDARD_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]

# ---------------------------------------------------------------------------
# Eastmoney throttle + keep-alive session (copied from refs/a_stock.py)
# ---------------------------------------------------------------------------
_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

_EM_SESSION = _requests.Session()
_EM_SESSION.headers.update({"User-Agent": _UA})
_EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_em_last_call = [0.0]


class _UrllibResp:
    """Minimal requests.Response-compatible wrapper for stdlib urllib responses."""

    def __init__(self, status, body_bytes, encoding):
        self.status_code = status
        self._body = body_bytes
        self._encoding = encoding or "utf-8"

    @property
    def text(self) -> str:
        return self._body.decode(self._encoding, errors="replace")

    def json(self):
        return _json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _em_get(url, params=None, headers=None, timeout=15, **kwargs):
    """Throttled Eastmoney GET with stdlib-urllib fallback.

    Primary path is the keep-alive requests session. In some environments the
    requests/urllib3 stack has its connection dropped by Eastmoney
    (RemoteDisconnected) after a burst of requests; stdlib urllib (http.client)
    is not affected, so it is used as an immediate fallback.
    """
    merged_headers = {
        "User-Agent": _UA,
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    if headers:
        merged_headers.update(headers)

    wait = _EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))

    try:
        resp = _EM_SESSION.get(
            url, params=params, headers=merged_headers, timeout=timeout, **kwargs
        )
        _em_last_call[0] = time.time()
        return resp
    except (_requests.ConnectionError, _requests.Timeout) as e:
        logger.debug("requests path failed (%s), falling back to urllib", e)
        try:
            _EM_SESSION.close()
        except Exception:
            pass

    qs = urllib.parse.urlencode(params or {})
    full_url = f"{url}?{qs}" if qs else url
    req = urllib.request.Request(full_url, headers=merged_headers)
    raw = urllib.request.urlopen(req, timeout=timeout)
    charset = raw.headers.get_content_charset() or "utf-8"
    body = raw.read()
    _em_last_call[0] = time.time()
    return _UrllibResp(raw.status, body, charset)


# ---------------------------------------------------------------------------
# Tonghuashun (同花顺) fuyao common_hq_aggr API
# ---------------------------------------------------------------------------
_THS_KLINE_URL = "https://quota-h.10jqka.com.cn/fuyao/common_hq_aggr/quote/v1/single_kline"
_THS_SNAPSHOT_URL = (
    "https://quota-h.10jqka.com.cn/fuyao/common_hq_aggr/quote/v1/multi_last_snapshot"
)
# JWT captured from the 10jqka futures page (goodsfu.10jqka.com.cn). Validated
# server-side; if it ever expires (HTTP 403), re-capture from the browser's
# X-Fuyao-Auth request header on any 10jqka k-line page.
_THS_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
    "eyJhdXRob3JpemVkX25hbWVzcGFjZSI6ImNvbW1vbi1ocS1hZ2dyLWNhY2hlIiwibGljZW5z"
    "ZWVfdHlwZSI6IkZST05UX0FQUCIsImxpY2Vuc2VlX25hbWVzcGFjZSI6Imh4a2xpbmUtRlVX"
    "X0NNRU9wZXJhdGlvbl9LQW1pcyJ9."
    "os5Wz-QJjLkg792_UFailT2glUUGixhY3Y7JoaLc26M"
)
_THS_HEADERS = {
    "User-Agent": _UA,
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Referer": "https://goodsfu.10jqka.com.cn/",
    "X-Fuyao-Auth": _THS_TOKEN,
    "Source-Id": "hxkline-FUW_CMEOperation_KAmis",
    "Platform": "hxkline",
    "X-Auth-Type": "ths",
    "X-Auth-Version": "1.0",
    "X-Auth-ProgId": "7047",
    "X-Auth-AppName": "AINVEST",
}
_THS_PERIOD = {
    "1m": "min_1", "5m": "min_5", "15m": "min_15", "30m": "min_30",
    "1h": "min_60", "4h": "hour_4", "1d": "day_1", "1w": "week_1", "1mo": "month_1",
}
# data_fields mapping for THS kline value rows: [ts, open, high, low, close, vol, amount]
_THS_KLINE_FIELDS = ["1", "7", "8", "9", "11", "13", "19"]
_THS_MAIN_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# akshare period mapping (SHFE intraday only; COMEX is daily-only)
# ---------------------------------------------------------------------------
_AKSHARE_MIN_PERIOD = {
    "1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
}

# ---------------------------------------------------------------------------
# Gold asset registry — per-source routing info
# ---------------------------------------------------------------------------
# Each entry: eastmoney={secid}, ths={market, variety} or None,
#             akshare={kind: foreign|domestic, symbol}.
GOLD_ASSETS: Dict[str, Dict[str, Any]] = {
    "COMEX": {
        "name": "COMEX黄金", "unit": "USD/oz", "decimal": 1,
        "eastmoney": {"secid": "101.GC00Y"},
        "ths": None,
        "akshare": {"kind": "foreign", "symbol": "GC"},
    },
    "XAUUSD": {
        "name": "XAUUSD现货黄金", "unit": "USD/oz", "decimal": 1,
        "eastmoney": {"secid": "101.GC00Y"},
        "ths": None,
        "akshare": {"kind": "foreign", "symbol": "GC"},
    },
    "AU": {
        "name": "COMEX黄金", "unit": "USD/oz", "decimal": 1,
        "eastmoney": {"secid": "101.GC00Y"},
        "ths": None,
        "akshare": {"kind": "foreign", "symbol": "GC"},
    },
    "GC": {
        "name": "COMEX黄金", "unit": "USD/oz", "decimal": 1,
        "eastmoney": {"secid": "101.GC00Y"},
        "ths": None,
        "akshare": {"kind": "foreign", "symbol": "GC"},
    },
    "SHFE_AU": {
        "name": "沪金主连", "unit": "CNY/g", "decimal": 2,
        "eastmoney": {"secid": "113.aum"},
        "ths": {"market": "65", "variety": "au"},
        "akshare": {"kind": "domestic", "symbol": "AU0"},
    },
    "AUM": {
        "name": "沪金主连", "unit": "CNY/g", "decimal": 2,
        "eastmoney": {"secid": "113.aum"},
        "ths": {"market": "65", "variety": "au"},
        "akshare": {"kind": "domestic", "symbol": "AU0"},
    },
}
DEFAULT_GOLD_ASSET = "COMEX"

# Eastmoney klt (K-line type) codes.
_INTERVAL_KLT = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "1d": 101, "1w": 102, "1mo": 103,
}
# 4h is not natively provided by eastmoney/akshare; resample from 1h bars.
# THS serves 4h natively (hour_4).
_4H_PARENT_INTERVAL = "1h"

_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2592000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_asset(symbol: Optional[str]) -> Dict[str, Any]:
    key = (symbol or "").strip().upper()
    if key in GOLD_ASSETS:
        return GOLD_ASSETS[key]
    if key:
        logger.warning("Unknown gold symbol '%s', falling back to %s", symbol, DEFAULT_GOLD_ASSET)
    return GOLD_ASSETS[DEFAULT_GOLD_ASSET]


def get_default_source() -> str:
    """Effective default source: runtime override > env var > DEFAULT_SOURCE."""
    if _DEFAULT_SOURCE_OVERRIDE:
        return _DEFAULT_SOURCE_OVERRIDE
    env = (os.environ.get(SOURCE_ENV_VAR) or "").strip().lower()
    if env in DATA_SOURCES:
        return env
    if env:
        logger.warning("Ignoring invalid %s='%s' (valid: %s)",
                       SOURCE_ENV_VAR, env, ", ".join(DATA_SOURCES))
    return DEFAULT_SOURCE


def set_default_source(source: Optional[str]) -> str:
    """Set the process-wide default source. Pass None to clear the override."""
    global _DEFAULT_SOURCE_OVERRIDE
    if source is None:
        _DEFAULT_SOURCE_OVERRIDE = None
        return get_default_source()
    src = str(source).strip().lower()
    if src not in DATA_SOURCES:
        raise ValueError(f"Unknown source '{source}'. Valid: {', '.join(DATA_SOURCES)}")
    _DEFAULT_SOURCE_OVERRIDE = src
    logger.info("Default data source set to '%s' (%s)", src, SOURCE_LABELS.get(src, src))
    return src


def sources_config(symbol: Optional[str] = None) -> Dict[str, Any]:
    """Describe the data-source configuration (for CLI / frontend / logging)."""
    meta = _resolve_asset(symbol)
    return {
        "sources": list(DATA_SOURCES),
        "labels": dict(SOURCE_LABELS),
        "default": get_default_source(),
        "env_var": SOURCE_ENV_VAR,
        "env_value": os.environ.get(SOURCE_ENV_VAR),
        "runtime_override": _DEFAULT_SOURCE_OVERRIDE,
        "asset": symbol or DEFAULT_GOLD_ASSET,
        "asset_name": meta.get("name"),
        "available": list_data_sources(symbol),
    }


def _resolve_source(source: Optional[str]) -> str:
    src = (source or get_default_source()).strip().lower()
    if src not in DATA_SOURCES:
        fallback = get_default_source()
        logger.warning("Unknown source '%s', falling back to %s", source, fallback)
        return fallback
    return src


def list_data_sources(symbol: Optional[str] = None) -> List[str]:
    """Return the data sources that actually support the given symbol."""
    meta = _resolve_asset(symbol)
    return [s for s in DATA_SOURCES if meta.get(s) is not None]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _ymd(dt) -> str:
    """Convert a date/datetime/str to YYYYMMDD for eastmoney beg/end params."""
    if dt is None:
        return ""
    if isinstance(dt, str):
        dt = dt[:10]
    return pd.to_datetime(dt).strftime("%Y%m%d")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary provider column names to the standard schema (case/name tolerant)."""
    if df is None or df.empty:
        return _empty_df()
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    def find(*candidates) -> Optional[str]:
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    col_dt = find("datetime", "date", "日期", "trade_date", "时间", "日期时间")
    col_o = find("open", "开盘价", "开盘")
    col_h = find("high", "最高价", "最高")
    col_l = find("low", "最低价", "最低")
    col_c = find("close", "收盘价", "收盘")
    col_v = find("volume", "成交量", "vol", "成交量(手)")

    missing = [n for n, c in (("Datetime", col_dt), ("Open", col_o), ("High", col_h),
                              ("Low", col_l), ("Close", col_c)) if c is None]
    if missing:
        logger.error("Cannot normalize columns %s; available=%s", missing, list(df.columns))
        return _empty_df()

    out = pd.DataFrame({
        "Datetime": df[col_dt],
        "Open": df[col_o],
        "High": df[col_h],
        "Low": df[col_l],
        "Close": df[col_c],
        "Volume": df[col_v] if col_v is not None else 0.0,
    })
    return out


def _sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Drop impossible bars where high < max(open, close), low > min(open, close),
    or high < low. A small relative epsilon (0.05% of close, min 0.01) absorbs
    floating-point rounding noise from upstream vendors (e.g. Sina COMEX feed);
    genuinely anomalous bars are removed so they never reach downstream analysis.
    Clean sources (Eastmoney/THS) drop zero rows.
    """
    if df is None or df.empty:
        return df
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    ref = c.abs().replace(0, np.nan).fillna(1.0)
    eps = np.maximum(0.01, ref * 0.0005)
    valid = (
        (h >= (np.maximum(o, c) - eps))
        & (l <= (np.minimum(o, c) + eps))
        & (h >= (l - eps))
    )
    n_bad = int((~valid).sum())
    if n_bad:
        logger.warning(
            "Sanitizing %d impossible OHLC bar(s) (%.2f%% of %d rows)",
            n_bad, 100.0 * n_bad / len(df), len(df),
        )
    return df.loc[valid].reset_index(drop=True)


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
    df = _sanitize_ohlc(df)
    df = df.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    return df[STANDARD_COLUMNS]


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1h bars into 4h bars (Open=first, High=max, Low=min, Close=last, Vol=sum)."""
    if df_1h is None or df_1h.empty:
        return _empty_df()
    df = df_1h.sort_values("Datetime").drop_duplicates("Datetime").set_index("Datetime")
    agg = df.resample("4h").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    })
    agg = agg.dropna(subset=["Open"]).reset_index()
    return agg[STANDARD_COLUMNS]


# ---------------------------------------------------------------------------
# Provider: Eastmoney
# ---------------------------------------------------------------------------
def _parse_eastmoney_klines(klines) -> pd.DataFrame:
    """Parse eastmoney klines (raw order: date, O, C, H, L, volume) -> schema."""
    rows = []
    for line in klines or []:
        parts = str(line).split(",")
        if len(parts) < 5:
            continue
        try:
            o = float(parts[1]); c = float(parts[2])
            h = float(parts[3]); l = float(parts[4])
            v = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
        except (ValueError, IndexError):
            continue
        rows.append({"Datetime": parts[0], "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})
    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    return _to_numeric_frame(df)


def _fetch_eastmoney(meta: Dict[str, Any], interval: str,
                     start_dt, end_dt) -> pd.DataFrame:
    secid = meta["eastmoney"]["secid"]
    want_4h = (interval or "").lower() == "4h"
    fetch_interval = _4H_PARENT_INTERVAL if want_4h else (interval or "1d")
    klt = _INTERVAL_KLT.get(fetch_interval, 101)

    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": str(klt),
        "fqt": "1",
        "beg": _ymd(start_dt) or "0",
        "end": _ymd(end_dt) or "20500101",
        "lmt": "10000",
    }
    try:
        r = _em_get(_KLINE_URL, params=params, timeout=15)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.error("Eastmoney kline request failed for %s @ %s: %s", secid, interval, e)
        return _empty_df()

    df = _parse_eastmoney_klines(((payload or {}).get("data") or {}).get("klines"))
    if df.empty:
        logger.warning("No kline rows returned for %s @ %s (eastmoney)", secid, interval)
        return df
    if want_4h:
        df = _resample_4h(df)
    return df


# ---------------------------------------------------------------------------
# Provider: akshare (Sina-sourced)
# ---------------------------------------------------------------------------
def _fetch_akshare(meta: Dict[str, Any], interval: str,
                   start_dt, end_dt) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare is not installed; cannot use akshare source")
        return _empty_df()

    src = meta.get("akshare")
    if not src:
        return _empty_df()

    want_4h = (interval or "").lower() == "4h"
    if want_4h:
        h1 = _fetch_akshare(meta, "1h", None, end_dt)
        return _empty_df() if h1.empty else _resample_4h(h1)

    kind = src["kind"]
    symbol = src["symbol"]
    df: Optional[pd.DataFrame] = None

    if kind == "foreign":
        if interval != "1d":
            logger.warning("akshare COMEX (%s) only supports daily; %s unavailable", symbol, interval)
            return _empty_df()
        df = ak.futures_foreign_hist_sina(symbol=symbol)
    else:  # domestic SHFE
        if interval == "1d":
            df = ak.futures_main_sina(symbol=symbol,
                                      start_date=_ymd(start_dt) or "19900101",
                                      end_date=_ymd(end_dt) or "20500101")
        elif interval in _AKSHARE_MIN_PERIOD:
            df = ak.futures_zh_min_sina(symbol=symbol, period=_AKSHARE_MIN_PERIOD[interval])
        else:
            logger.warning("akshare domestic interval '%s' unsupported", interval)
            return _empty_df()

    if df is None or df.empty:
        logger.warning("akshare returned no rows for %s @ %s", symbol, interval)
        return _empty_df()

    out = _normalize_columns(df)
    return _to_numeric_frame(out)


# ---------------------------------------------------------------------------
# Provider: Tonghuashun (同花顺) — SHFE only, dynamic main contract
# ---------------------------------------------------------------------------
def _ths_post(url: str, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(
            url, data=_json.dumps(body).encode("utf-8"),
            headers=_THS_HEADERS, method="POST",
        )
        raw = urllib.request.urlopen(req, timeout=15)
        return _json.loads(raw.read().decode("utf-8", "replace"))
    except Exception as e:
        logger.error("THS request failed: %s", e)
        return None


def _gen_contract_codes(variety: str, months: int = 24) -> List[str]:
    """Generate near-term contract codes auYYMM starting from the current month."""
    today = datetime.now()
    codes = []
    for i in range(months):
        total = today.year * 12 + (today.month - 1) + i
        y, m = divmod(total, 12)
        codes.append(f"{variety}{y % 100:02d}{m + 1:02d}")
    return codes


def _ths_main_contract(market: str, variety: str,
                       cache_seconds: int = 3600) -> Optional[str]:
    """Discover the current main (most-active) contract code for a variety."""
    now = time.time()
    cached = _THS_MAIN_CACHE.get((market, variety))
    if cached and cached["expire"] > now:
        return cached["code"]

    candidates = _gen_contract_codes(variety, months=24)
    body = {
        "code_list": [{"market": market, "codes": candidates}],
        "trade_class": "intraday",
        "data_fields": ["13"],  # volume only
        "lang": "zh_cn",
        "gpid": 0,
    }
    resp = _ths_post(_THS_SNAPSHOT_URL, body)
    if not resp:
        return cached["code"] if cached else None

    volumes: Dict[str, float] = {}
    for item in (resp.get("data") or {}).get("quote_data") or []:
        code = item.get("code")
        val = item.get("value") or []
        if code and val and val[0] is not None:
            try:
                volumes[code] = float(val[0][0] if isinstance(val[0], list) else val[0])
            except (TypeError, ValueError):
                continue

    if not volumes:
        logger.warning("THS: no volume data for variety %s market %s", variety, market)
        return cached["code"] if cached else None

    main = max(volumes, key=volumes.get)
    logger.info("THS main contract for %s (market %s): %s (vol %.0f)",
                variety, market, main, volumes[main])
    _THS_MAIN_CACHE[(market, variety)] = {"code": main, "expire": now + cache_seconds}
    return main


def _fetch_ths(meta: Dict[str, Any], interval: str,
               start_dt, end_dt) -> pd.DataFrame:
    src = meta.get("ths")
    if not src:
        logger.warning("THS source does not support asset '%s' (no COMEX k-line on 10jqka web)",
                       meta.get("name"))
        return _empty_df()

    period = _THS_PERIOD.get((interval or "1d").lower())
    if period is None:
        logger.warning("THS unsupported interval '%s'", interval)
        return _empty_df()

    main = _ths_main_contract(src["market"], src["variety"])
    if not main:
        return _empty_df()

    # Pull a generous recent window (last ~600 bars); bounds applied in _finalize.
    body = {
        "code_list": [{"market": src["market"], "codes": [main]}],
        "trade_class": "intraday",
        "time_period": period,
        "trade_date": -1,
        "begin_time": -600,
        "end_time": 0,
        "adjust_type": "forward",
        "gpid": 0,
    }
    resp = _ths_post(_THS_KLINE_URL, body)
    if not resp:
        return _empty_df()

    qd = (resp.get("data") or {}).get("quote_data") or []
    if not qd:
        logger.warning("THS: empty quote_data for %s @ %s", main, interval)
        return _empty_df()

    rows = []
    for rec in qd[0].get("value") or []:
        # value row: [ts_ms, open, high, low, close, volume, amount]
        if not rec or len(rec) < 5:
            continue
        try:
            ts = pd.to_datetime(int(rec[0]), unit="ms")
            rows.append({
                "Datetime": ts,
                "Open": float(rec[1]), "High": float(rec[2]),
                "Low": float(rec[3]), "Close": float(rec[4]),
                "Volume": float(rec[5]) if len(rec) > 5 and rec[5] is not None else 0.0,
            })
        except (ValueError, TypeError):
            continue
    df = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    return _to_numeric_frame(df)


# Provider registry
_PROVIDERS = {
    "eastmoney": _fetch_eastmoney,
    "ths": _fetch_ths,
    "akshare": _fetch_akshare,
}


# ---------------------------------------------------------------------------
# Common post-processing (end/start bounds + min_bars padding)
# ---------------------------------------------------------------------------
def _finalize(df: pd.DataFrame, interval: str, start_dt, end_dt, min_bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_df()
    df = _to_numeric_frame(df)
    if df.empty:
        return df

    # Apply end bound (inclusive).
    if end_dt is not None:
        end = pd.to_datetime(end_dt)
        df = df[df["Datetime"] <= end]

    # Apply start bound, but keep a minimum bar count for downstream analysis.
    if start_dt is not None and not df.empty:
        start = pd.to_datetime(start_dt)
        in_range = df[df["Datetime"] >= start]
        if len(in_range) >= min_bars:
            df = in_range
        elif min_bars > 0:
            df = df.tail(max(min_bars, len(in_range)))
    elif min_bars > 0 and len(df) > min_bars:
        df = df.tail(min_bars)

    df = df.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    return df[STANDARD_COLUMNS]


def _pad_window_if_needed(df: pd.DataFrame, source: str, meta: Dict[str, Any],
                          interval: str, end_dt, min_bars: int) -> pd.DataFrame:
    """If [start,end] yielded too few bars, re-fetch with no start bound."""
    if df is None or len(df) >= min_bars or min_bars <= 0:
        return df
    secs = _INTERVAL_SECONDS.get(interval, 86400)
    need_start = (pd.to_datetime(end_dt) if end_dt is not None else pd.Timestamp.now()) \
        - pd.Timedelta(seconds=secs * min_bars * 3)
    logger.info("min_bars not met (%d<%d); re-fetching %s from %s",
                len(df), min_bars, interval, need_start)
    wider = _PROVIDERS[source](meta, interval, need_start, end_dt)
    return wider if (wider is not None and len(wider) >= len(df)) else df


# ---------------------------------------------------------------------------
# Public fetcher
# ---------------------------------------------------------------------------
def fetch_gold_kline(
    symbol: str,
    interval: str,
    start_dt: Any = None,
    end_dt: Any = None,
    min_bars: int = 60,
    source: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch gold OHLCV from the selected source, normalized to the project schema.

    Args:
        symbol: friendly key in GOLD_ASSETS (default COMEX).
        interval: one of 1m/5m/15m/30m/1h/4h/1d/1w/1mo.
        start_dt / end_dt: date/datetime/str bounds. end is inclusive.
        min_bars: guarantee at least this many rows by extending the window
            backward when [start, end] is too sparse (feeds analysis tail(45)).
            Pass 0 to strictly honor [start, end].
        source: one of DATA_SOURCES ("eastmoney"/"ths"/"akshare"). None uses
            the configured default (set_default_source / $GOLD_DATA_SOURCE).

    Returns:
        DataFrame [Datetime, Open, High, Low, Close, Volume], sorted, de-duped.
        Empty DataFrame if the source does not support the asset/interval or
        the request fails.
    """
    source = _resolve_source(source)
    meta = _resolve_asset(symbol)

    if meta.get(source) is None:
        logger.error("Source '%s' does not support asset '%s' (supported: %s)",
                     source, meta.get("name"), list_data_sources(symbol))
        return _empty_df()

    df = _PROVIDERS[source](meta, interval, start_dt, end_dt)
    df = _pad_window_if_needed(df, source, meta, interval, end_dt, min_bars)
    return _finalize(df, interval, start_dt, end_dt, min_bars)


def supports_interval(symbol: str, interval: str, source: Optional[str] = None) -> bool:
    """Whether `source` can serve `interval` for `symbol`.

    Unlike list_data_sources() (asset-level), this is interval-aware: e.g. the
    akshare COMEX feed is daily-only, so it cannot serve 1h/4h.
    """
    src = _resolve_source(source)
    meta = _resolve_asset(symbol)
    if meta.get(src) is None:
        return False
    iv = (interval or "1d").strip().lower()
    if src == "eastmoney":
        return iv in _INTERVAL_KLT or iv == "4h"
    if src == "ths":
        return iv in _THS_PERIOD
    kind = (meta.get("akshare") or {}).get("kind")
    if kind == "foreign":
        return iv == "1d"
    return iv in ("1d", "4h") or iv in _AKSHARE_MIN_PERIOD


def fetch_gold_kline_auto(
    symbol: str,
    interval: str,
    start_dt: Any = None,
    end_dt: Any = None,
    min_bars: int = 60,
    preferred: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], List[Dict[str, Any]]]:
    """Fetch with automatic failover across sources capable of this asset+interval.

    The preferred source is always tried first; if it yields no rows (vendor
    outage, IP throttling, unsupported interval) the remaining capable sources
    are tried in DATA_SOURCES order. This keeps the user's explicit choice
    authoritative when it works, while avoiding a dead end when a free feed
    is temporarily unreachable.

    Returns:
        (df, used_source, attempts) — attempts is a list of
        {"source", "label", "rows"} for every source tried.
    """
    meta = _resolve_asset(symbol)
    pref = (preferred or "").strip().lower()
    candidates = [s for s in DATA_SOURCES if supports_interval(symbol, interval, s)]
    if pref in candidates:
        candidates.remove(pref)
        candidates.insert(0, pref)
    elif pref:
        logger.warning(
            "'%s' cannot serve %s @ %s; capable sources: %s",
            pref, meta.get("name"), interval, candidates or "none",
        )

    attempts: List[Dict[str, Any]] = []
    for s in candidates:
        df = fetch_gold_kline(symbol, interval, start_dt, end_dt, min_bars=min_bars, source=s)
        rows = 0 if df is None else len(df)
        attempts.append({"source": s, "label": SOURCE_LABELS.get(s, s), "rows": rows})
        if rows:
            if pref and s != pref:
                logger.warning(
                    "Source '%s' returned no data for %s @ %s; fell back to '%s'",
                    pref, meta.get("name"), interval, s,
                )
            return df, s, attempts
        logger.warning("Source '%s' returned no data for %s @ %s",
                       s, meta.get("name"), interval)
    return _empty_df(), None, attempts


# ---------------------------------------------------------------------------
# Data integrity validation (for chart-based verification)
# ---------------------------------------------------------------------------
def validate_ohlcv(df: pd.DataFrame) -> Dict[str, Any]:
    """Run structural + financial-integrity checks on an OHLCV frame."""
    report: Dict[str, Any] = {"row_count": int(len(df)), "checks": {}, "ok": True}

    if df is None or df.empty:
        report["ok"] = False
        report["checks"]["non_empty"] = False
        return report

    for col in REQUIRED_COLUMNS:
        report["checks"][f"has_{col}"] = col in df.columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["ok"] = False
        report["missing_columns"] = missing
        return report

    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    nan_cols = [name for name, s in (("Open", o), ("High", h), ("Low", l), ("Close", c))
                if s.isna().any()]
    report["checks"]["no_nan_ohlc"] = not nan_cols
    if nan_cols:
        report["ok"] = False
        report["nan_columns"] = nan_cols

    # A small relative epsilon absorbs vendor floating-point rounding noise
    # (same tolerance the sanitizer uses); genuinely anomalous bars are dropped
    # upstream and would not reach here.
    ref = c.abs().replace(0, np.nan).fillna(1.0)
    eps = np.maximum(0.01, ref * 0.0005)
    hi_bound = np.maximum(o, c) - eps
    lo_bound = np.minimum(o, c) + eps
    bad_hi = (h < hi_bound)
    bad_lo = (l > lo_bound)
    bad_hl = (h < (l - eps))
    report["checks"]["high_ge_open_close"] = int(bad_hi.sum()) == 0
    report["checks"]["low_le_open_close"] = int(bad_lo.sum()) == 0
    report["checks"]["high_ge_low"] = int(bad_hl.sum()) == 0
    if bad_hi.any() or bad_lo.any() or bad_hl.any():
        report["ok"] = False
        bad_idx = df.index[bad_hi | bad_lo | bad_hl].tolist()[:5]
        report["bad_row_sample"] = df.loc[bad_idx, REQUIRED_COLUMNS].to_dict("records")

    if "Volume" in df.columns:
        neg_vol = int((df["Volume"] < 0).sum())
        report["checks"]["volume_nonneg"] = neg_vol == 0
        if neg_vol:
            report["ok"] = False
            report["negative_volume_rows"] = neg_vol

    dt = pd.to_datetime(df["Datetime"])
    dup = int(dt.duplicated().sum())
    report["checks"]["no_duplicate_datetime"] = dup == 0
    report["checks"]["datetime_monotonic_increasing"] = bool(dt.is_monotonic_increasing)
    if dup or not dt.is_monotonic_increasing:
        report["ok"] = False
    report["first_datetime"] = str(dt.min())
    report["last_datetime"] = str(dt.max())
    return report


def asset_info(symbol: str, source: Optional[str] = None) -> Dict[str, Any]:
    """Return metadata for a gold symbol, including per-source routing info."""
    meta = _resolve_asset(symbol)
    info = {k: v for k, v in meta.items() if k in ("name", "unit", "decimal")}
    src = _resolve_source(source) if source else None
    if src:
        info["source"] = src
        info["source_label"] = SOURCE_LABELS.get(src, src)
        info["source_supported"] = meta.get(src) is not None
    info["available_sources"] = list_data_sources(symbol)
    return info


# ---------------------------------------------------------------------------
# Chart rendering (self-contained; optional matplotlib/mplfinance dependency)
# ---------------------------------------------------------------------------
_CHART_FONT_CACHE: Dict[str, Any] = {}


def _chart_font() -> Optional[str]:
    """Register a CJK font from file so Chinese titles/labels render correctly.

    Matplotlib's font cache does not reliably resolve system CJK fonts by name,
    and mplfinance's style overrides global rcParams — so the font must both be
    registered from its TTF/TTC path and injected via the style's ``rc``.
    """
    if "name" in _CHART_FONT_CACHE:
        return _CHART_FONT_CACHE["name"]
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    name = None
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if not os.path.exists(path):
            continue
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name] + [
                f for f in plt.rcParams["font.sans-serif"] if f != name
            ]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            name = None
    _CHART_FONT_CACHE["name"] = name
    return name


def _mpf_style():
    """mplfinance style carrying CJK rcParams.

    The market colors are overridden with the project-wide "红涨绿跌" rule
    (up -> red, down -> green) from ``color_style``; the yahoo base style uses
    the opposite convention, so it must not be relied on for candle colors.
    """
    import mplfinance as mpf

    from color_style import DOWN_COLOR, UP_COLOR

    cjk = _chart_font()
    rc = {"axes.unicode_minus": False}
    if cjk:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = [cjk, "DejaVu Sans"]
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


def render_chart(
    df: pd.DataFrame,
    title: Optional[str] = None,
    out_path: Optional[Any] = None,
    last_n: int = 90,
    ma: Tuple[int, ...] = (5, 20),
    show_volume: bool = True,
    return_base64: bool = True,
) -> Dict[str, Any]:
    """Render a candlestick (+ volume) chart for an OHLCV frame.

    Args:
        df: frame with the standard schema [Datetime, Open, High, Low, Close, Volume].
        title: chart title. Auto-generated from columns if omitted.
        out_path: optional PNG path; the parent directory is created if needed.
        last_n: plot only the most recent N bars (0 = all).
        ma: moving-average windows drawn over the candles.
        show_volume: draw the volume sub-panel.
        return_base64: include a base64 PNG in the result (for <img src="data:...">).

    Returns:
        dict with row_count / title / path / image_base64 / image_data_uri / error.
    """
    import base64 as _b64
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplfinance as mpf

    result: Dict[str, Any] = {"error": None, "path": None, "image_base64": None,
                              "image_data_uri": None, "row_count": 0, "title": title}
    if df is None or df.empty:
        result["error"] = "empty dataframe"
        return result

    d = df.copy()
    d["Datetime"] = pd.to_datetime(d["Datetime"])
    d = d.sort_values("Datetime").drop_duplicates("Datetime").tail(last_n or len(d))
    result["row_count"] = len(d)
    d = d.set_index("Datetime")[["Open", "High", "Low", "Close", "Volume"]]

    title = title or "Gold OHLC"
    result["title"] = title
    show_vol = bool(show_volume and "Volume" in d.columns and d["Volume"].fillna(0).gt(0).any())
    mav = tuple(m for m in ma if m and len(d) > m) or None

    fig, _ = mpf.plot(
        d, type="candle", style=_mpf_style(), title=title,
        mav=mav, volume=show_vol, figsize=(14, 6),
        returnfig=True, block=False,
    )

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=120, bbox_inches="tight")
        result["path"] = str(p)

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
        result["image_base64"] = b64
        result["image_data_uri"] = f"data:image/png;base64,{b64}"

    plt.close(fig)
    return result


def plot_gold_kline(
    symbol: str = DEFAULT_GOLD_ASSET,
    interval: str = "1d",
    start_dt: Any = None,
    end_dt: Any = None,
    source: Optional[str] = None,
    out_path: Optional[Any] = None,
    last_n: int = 90,
    ma: Tuple[int, ...] = (5, 20),
    show_volume: bool = True,
    return_base64: bool = True,
    min_bars: int = 60,
) -> Dict[str, Any]:
    """Fetch gold klines from the selected source and render a chart in one call."""
    df = fetch_gold_kline(symbol, interval, start_dt, end_dt,
                          min_bars=min_bars, source=source)
    info = asset_info(symbol, source=source)
    src_label = info.get("source_label") or SOURCE_LABELS.get(get_default_source(), "")
    if df is None or df.empty:
        return {"error": f"no data for {info.get('name')} {interval} <{src_label}>",
                "path": None, "image_base64": None, "image_data_uri": None,
                "row_count": 0, "title": None, "symbol": symbol,
                "interval": interval, "source": info.get("source")}

    last_close = float(df["Close"].iloc[-1])
    digits = info.get("decimal", 2)
    title = (f"{info.get('name', symbol)} {interval} [{info.get('unit', '')}] "
             f"<{src_label}>  最近{min(last_n or len(df), len(df))}根  最新 {last_close:.{digits}f}")
    chart = render_chart(df, title=title, out_path=out_path, last_n=last_n,
                         ma=ma, show_volume=show_volume, return_base64=return_base64)
    chart.update({"symbol": symbol, "interval": interval,
                  "source": info.get("source"), "unit": info.get("unit"),
                  "last_close": last_close,
                  "first_datetime": str(df["Datetime"].iloc[0]),
                  "last_datetime": str(df["Datetime"].iloc[-1])})
    return chart
