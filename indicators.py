"""共享技术指标模块：全仓库唯一的指标计算来源。

职责边界：
  * 纯计算，不发网络请求、不读配置、不依赖 LLM。
  * 输入是标准化的 OHLCV DataFrame（列名 Open/High/Low/Close/Volume）。
  * 输出三种投影，同源于一次计算：
      - compute_series()   完整序列，供回测使用
      - latest_values()   最新值标量，供报告使用
      - tool_payload()    近期窗口列表，供 Agent 工具使用

口径约定（见 docs/adr/0001-stochastic-percent-k-convention.md）：
  * Stochastic 的 %K 是**慢速 %K**——快速 %K 的 3 周期简单移动平均；
    %D 是 %K 的 3 周期移动平均。
  * 指标键带周期后缀（rsi_14 / roc_10 / willr_14 / atr_14）。

后端分工：
  * TA-Lib 计算 RSI / MACD / Stochastic / ROC / Williams %R。
  * pandas 计算布林带、ATR 与均线——TA-Lib 的 BBANDS 用总体标准差，
    而本仓库既有口径是样本标准差，换过去会让 %B 最多偏移 1.9。

已知口径差异（相对合并前的 report_util 实现）：
  * RSI 与 MACD 改由 TA-Lib 计算。TA-Lib 用简单移动平均给 EMA 播种，
    pandas 的 ewm(adjust=False) 从首值起算，两者数值不同——这是 ADR 认可的
    统一口径，报告的 RSI 读数会随之变化。
  * ATR 不设 min_periods，与合并前一致，头部若干根即有值。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import talib

# Agent 工具返回给 LLM 的近期窗口长度。
TOOL_WINDOW = 28

# 各指标的周期。键名与序列键一一对应，周期不写在键里以外的任何地方。
RSI_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
STOCH_FASTK_PERIOD = 14
STOCH_SLOWK_PERIOD = 3
STOCH_SLOWD_PERIOD = 3
ROC_PERIOD = 10
WILLR_PERIOD = 14
ATR_PERIOD = 14
BBANDS_PERIOD = 20
BBANDS_DEVS = 2
MA_FAST_PERIOD = 5
MA_MID_PERIOD = 20
MA_SLOW_PERIOD = 60

# 取倒数第 N 个有效值时用的偏移：1 表示当前根，2 表示上一根。
CURRENT_BAR_OFFSET = 1
PREVIOUS_BAR_OFFSET = 2

# 序列键的完整集合：二十个指标，新增指标时只改这一处。
INDICATOR_KEYS = (
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "roc_10", "willr_14",
    "ma5", "ma20", "ma60",
    "bb_up", "bb_mid", "bb_lo", "bb_pct",
    "atr_14", "close", "high", "low", "volume",
)


def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均，与既有实现同口径（pandas 默认 min_periods=window）。"""
    return series.rolling(window).mean()


def _talib_indicators(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> Dict[str, pd.Series]:
    """TA-Lib 计算的五个指标：RSI / MACD / Stochastic / ROC / Williams %R。"""
    index = close.index
    close_values = close.to_numpy()
    high_values = high.to_numpy()
    low_values = low.to_numpy()

    rsi = talib.RSI(close_values, timeperiod=RSI_PERIOD)

    macd, macd_signal, macd_hist = talib.MACD(
        close_values,
        fastperiod=MACD_FAST_PERIOD,
        slowperiod=MACD_SLOW_PERIOD,
        signalperiod=MACD_SIGNAL_PERIOD,
    )

    # TA-Lib 的 STOCH 默认返回慢速 %K（快速 %K 的 SMA）与 %D（%K 的 SMA）。
    stoch_k, stoch_d = talib.STOCH(
        high_values,
        low_values,
        close_values,
        fastk_period=STOCH_FASTK_PERIOD,
        slowk_period=STOCH_SLOWK_PERIOD,
        slowd_period=STOCH_SLOWD_PERIOD,
    )

    roc = talib.ROC(close_values, timeperiod=ROC_PERIOD)
    willr = talib.WILLR(
        high_values, low_values, close_values, timeperiod=WILLR_PERIOD
    )

    return {
        "rsi_14": pd.Series(rsi, index=index),
        "macd": pd.Series(macd, index=index),
        "macd_signal": pd.Series(macd_signal, index=index),
        "macd_hist": pd.Series(macd_hist, index=index),
        "stoch_k": pd.Series(stoch_k, index=index),
        "stoch_d": pd.Series(stoch_d, index=index),
        "roc_10": pd.Series(roc, index=index),
        "willr_14": pd.Series(willr, index=index),
    }


def _pandas_indicators(
    df: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series
) -> Dict[str, pd.Series]:
    """pandas 计算的指标：布林带、ATR、均线与价格量序列。"""
    ma_fast = _sma(close, MA_FAST_PERIOD)
    ma_mid = _sma(close, MA_MID_PERIOD)
    ma_slow = _sma(close, MA_SLOW_PERIOD)

    bb_std = close.rolling(BBANDS_PERIOD).std()
    bb_up = ma_mid + BBANDS_DEVS * bb_std
    bb_lo = ma_mid - BBANDS_DEVS * bb_std
    bb_pct = (close - bb_lo) / (bb_up - bb_lo).replace(0, np.nan) * 100

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # 不设 min_periods：与合并前的实现一致，头部若干根即有值。
    atr = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    return {
        "ma5": ma_fast,
        "ma20": ma_mid,
        "ma60": ma_slow,
        "bb_up": bb_up,
        "bb_mid": ma_mid,
        "bb_lo": bb_lo,
        "bb_pct": bb_pct,
        "atr_14": atr,
        "close": close,
        "high": high,
        "low": low,
        # 成交量缺失时留空而非补零——补零会伪造一个不存在的指标。
        "volume": (
            df["Volume"].astype(float)
            if "Volume" in df.columns
            else pd.Series(np.nan, index=close.index)
        ),
    }


def compute_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """从 K 线帧算出全部指标序列。

    这是唯一的计算入口，三种投影都从它的结果派生。
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    series = {**_talib_indicators(close, high, low),
              **_pandas_indicators(df, close, high, low)}
    return {key: series[key] for key in INDICATOR_KEYS}


def latest_values(series: Dict[str, pd.Series]) -> Dict[str, Optional[float]]:
    """最新值投影：每个指标最近一个有效值，供报告使用。

    high / low 取整段区间的最高 / 最低价，与既有报告口径一致——
    报告的"市场快照"要的是区间极值，不是最后一根的高低价。
    """
    def last(key: str) -> Optional[float]:
        values = series[key].dropna()
        return float(values.iloc[-1]) if not values.empty else None

    values: Dict[str, Optional[float]] = {
        key: last(key) for key in INDICATOR_KEYS
    }
    values["high"] = float(series["high"].max())
    values["low"] = float(series["low"].min())
    # 报告需要上一根的 MACD 柱，用来判断柱体是在放大还是收缩。
    # 取的是倒数第二根有效值——与合并前的 report_util 实现同口径。
    hist = series["macd_hist"].dropna()
    values["macd_hist_prev"] = (
        float(hist.iloc[-PREVIOUS_BAR_OFFSET])
        if len(hist) >= PREVIOUS_BAR_OFFSET
        else None
    )
    return values


def tool_payload(series: Dict[str, pd.Series]) -> Dict[str, List[float]]:
    """工具载荷投影：近期窗口的列表，供 Agent 的工具返回给 LLM。

    键名沿用 Agent 既有工具的命名（rsi / macd / roc / willr），
    这样 LLM 看到的工具输出形状保持不变。
    """
    def window(key: str, size: int = TOOL_WINDOW) -> List[float]:
        return series[key].fillna(0).round(2).tolist()[-size:]

    return {
        "rsi": window("rsi_14"),
        "macd": series["macd"].fillna(0).round(2).tolist(),
        "macd_signal": window("macd_signal"),
        "macd_hist": window("macd_hist"),
        "stoch_k": window("stoch_k"),
        "stoch_d": window("stoch_d"),
        "roc": window("roc_10"),
        "willr": window("willr_14"),
    }
