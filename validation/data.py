"""回测数据层：统一把各种来源整理成 [Datetime, Open, High, Low, Close, Volume]。

支持三种数据来源：
  1. live   —— 复用 gold_data 的多数据源自动降级取数（推荐）
  2. csv    —— 本地 CSV（列名与项目标准一致，Volume 可缺省）
  3. synthetic —— 确定性合成行情（离线测试 / 演示，无需网络与 API Key）

数据指纹（fingerprint）用于可复现性校验：同一份数据在任何机器上应得到相同哈希。
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig, ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STANDARD_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
REQUIRED_COLUMNS = ["Datetime", "Open", "High", "Low", "Close"]

# 周期 → 秒（用于合成数据的时间步长，与 gold_data 保持同一口径）
_INTERVAL_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1mo": 2592000,
}


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """统一列、类型、排序与去重；缺 Volume 时补 0。"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    out = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ConfigError(f"数据缺少必要列：{', '.join(missing)}")

    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[STANDARD_COLUMNS]
    out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = (
        out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
        .sort_values("Datetime")
        .drop_duplicates("Datetime", keep="last")
        .reset_index(drop=True)
    )
    out["Volume"] = out["Volume"].fillna(0.0)
    return out


def frame_fingerprint(df: pd.DataFrame) -> str:
    """对 K 线内容做稳定哈希（浮点保留 6 位，避免平台差异）。"""
    if df is None or len(df) == 0:
        return hashlib.sha256(b"empty").hexdigest()
    frame = df[STANDARD_COLUMNS].copy()
    frame["Datetime"] = pd.to_datetime(frame["Datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ("Open", "High", "Low", "Close", "Volume"):
        frame[col] = frame[col].astype(float).round(6)
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def window_digest(df: pd.DataFrame, end_index: int, lookback: int) -> str:
    """某个决策时点所用历史窗口的指纹。

    增量回测用它判断「缓存决策是否仍然对应同一段历史」——
    即使历史数据被供应商修订，指纹变化也会让缓存自动失效。
    """
    start = max(0, end_index - lookback + 1)
    chunk = df.iloc[start : end_index + 1]
    return frame_fingerprint(chunk)


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise ConfigError(f"CSV 文件不存在：{p}")
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # noqa: BLE001 - 明确转换成配置错误
        raise ConfigError(f"读取 CSV 失败：{p}（{exc}）") from exc
    return normalize_frame(df)


def synthetic_bars(
    n: int = 320,
    seed: int = 7,
    start_price: float = 2000.0,
    timeframe: str = "1d",
    start: str = "2025-01-02 00:00:00",
    volatility: float = 0.006,
) -> pd.DataFrame:
    """确定性合成 OHLCV（带趋势切换），用于离线演示与可复现测试。

    说明：同一 seed + n 在任何机器上输出完全一致（不使用全局随机状态）。
    """
    if n < 3:
        raise ConfigError("合成数据至少需要 3 根 K 线")
    rng = np.random.RandomState(int(seed))
    step = pd.Timedelta(seconds=_INTERVAL_SECONDS.get(timeframe, 86400))
    times = pd.date_range(start=pd.Timestamp(start), periods=n, freq=step)

    # 每 40 根切换一次趋势方向，制造多空交替的行情结构
    drift = np.zeros(n)
    block = 40
    for seg_start in range(0, n, block):
        drift[seg_start : seg_start + block] = rng.choice([-1.0, -0.4, 0.0, 0.4, 1.0]) * 0.0035

    shocks = rng.normal(0.0, volatility, size=n)
    log_returns = drift + shocks
    close = start_price * np.exp(np.cumsum(log_returns))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.0008, size=n - 1))

    body_high = np.maximum(open_, close)
    body_low = np.minimum(open_, close)
    spread = np.abs(rng.normal(0.0, volatility * 0.6, size=n)) + 0.0005
    high = body_high * (1.0 + spread)
    low = body_low * (1.0 - spread)
    volume = np.round(np.abs(rng.normal(120000, 25000, size=n)) + 1000, 0)

    df = pd.DataFrame(
        {
            "Datetime": times,
            "Open": np.round(open_, 4),
            "High": np.round(high, 4),
            "Low": np.round(low, 4),
            "Close": np.round(close, 4),
            "Volume": volume,
        }
    )
    return normalize_frame(df)


def slice_bounds(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """按配置的时间范围裁剪（含端点）。"""
    out = df
    if config.start:
        out = out[out["Datetime"] >= pd.to_datetime(config.start)]
    if config.end:
        out = out[out["Datetime"] <= pd.to_datetime(config.end)]
    return out.reset_index(drop=True)


def load_live_bars(config: BacktestConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """通过 gold_data 取数（含自动降级）；失败时返回空表与尝试记录。"""
    import gold_data  # 延迟导入：离线场景（CSV/合成）完全不依赖它

    df, used, attempts = gold_data.fetch_gold_kline_auto(
        config.asset,
        config.timeframe,
        start_dt=config.start,
        end_dt=config.end,
        min_bars=config.min_bars,
        preferred=config.source,
    )
    info = gold_data.asset_info(config.asset, source=used)
    meta = {
        "mode": "live",
        "asset": config.asset,
        "asset_name": info.get("name", config.asset),
        "unit": info.get("unit"),
        "source": used,
        "source_label": gold_data.SOURCE_LABELS.get(used, used) if used else None,
        "attempts": attempts,
    }
    return normalize_frame(df), meta


def load_bars(
    config: BacktestConfig,
    csv_path: Optional[str] = None,
    synthetic: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """统一入口：返回 (标准化后的 K 线, 数据元信息)。"""
    if synthetic:
        params = {"n": 320, "seed": 7, "start_price": 2000.0, "volatility": 0.006}
        params.update({k: v for k, v in synthetic.items() if k in params})
        df = synthetic_bars(timeframe=config.timeframe, **params)
        df = slice_bounds(df, config)
        meta = {
            "mode": "synthetic",
            "asset": config.asset,
            "asset_name": f"合成行情(seed={params['seed']})",
            "unit": None,
            "source": "synthetic",
            "source_label": "合成数据",
            "attempts": [],
            "synthetic_params": params,
        }
    elif csv_path:
        df = load_csv(csv_path)
        df = slice_bounds(df, config)
        meta = {
            "mode": "csv",
            "asset": config.asset,
            "asset_name": f"CSV({Path(csv_path).name})",
            "unit": None,
            "source": "csv",
            "source_label": "本地 CSV",
            "attempts": [],
            "csv_path": str(csv_path),
        }
    else:
        df, meta = load_live_bars(config)
        meta["mode"] = "live"

    if df is None or df.empty:
        raise ConfigError(
            f"未获取到可用数据（{config.asset} @ {config.timeframe}）。"
            f"尝试记录：{meta.get('attempts') or '无'}。"
            "如无网络，可使用 --csv 或 --synthetic 离线回测。"
        )
    if len(df) < int(config.warmup_bars) + 2:
        raise ConfigError(
            f"数据不足：{len(df)} 根 < warmup_bars+2（{int(config.warmup_bars) + 2}）。"
            "请扩大时间范围或改用更小周期。"
        )

    meta.update(
        {
            "rows": int(len(df)),
            "first": str(df["Datetime"].iloc[0]),
            "last": str(df["Datetime"].iloc[-1]),
            "fingerprint": frame_fingerprint(df),
        }
    )
    return df, meta
