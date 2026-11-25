"""指标与信号计算。

所有序列都是「因果」的（只使用当前及更早的 K 线），因此可以安全地一次性
向量化预计算，再在回放时按索引读取——既保证无未来函数，又保证回放速度。

依赖说明：指标序列本身来自仓库根目录的共享 `indicators` 模块（见 ADR-0001），
因此本模块依赖 TA-Lib。共享模块负责 RSI / MACD / Stochastic / ROC /
Williams %R / 布林带 / ATR / 均线；本模块只补充回测特有的量能、滚动区间与
趋势斜率，并把它们与共享序列拼成回放所用的同一张表。

口径：Stochastic 的 %K 是**慢速 %K**，与报告和 Agent 一致。慢速化会削短
极端区间的停留时间，所以规则的 Stochastic 阈值相应下调（见
`STOCH_OVERBOUGHT` / `STOCH_OVERSOLD`）。

信号结构与决策智能体的文档策略保持一致：
  指标（动量/超买超卖） 45% + 趋势（通道斜率） 35% + 模式（K 线/突破） 20%
最终必须给出 LONG / SHORT（与线上 HFT 决策一致，不允许 HOLD）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import indicators as shared_indicators

from .config import BacktestConfig

# Stochastic 阈值。共享模块的 %K 是慢速 %K（快速 %K 的 3 周期 SMA）；
# 平滑让极端区间停留更短，若沿用快速 %K 的 85/15，Stochastic 信号的触发频率
# 会明显下降、该分项对总分的贡献随之缩水。83/17 把触发频率拉回原水平附近
# （固定夹具上 43.09% → 44.99%；83/16 在该夹具上恰好等于原值，但未跨夹具验证，
# 所以这里沿用本票选定的 83/17，见 docs/adr/0001）。
STOCH_OVERBOUGHT = 83.0
STOCH_OVERSOLD = 17.0


def _safe(value: Any, digits: int = 6) -> Optional[float]:
    """把 numpy/float 转成 JSON 友好的 float；NaN/Inf → None。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, digits)


# ---------------------------------------------------------------------------
# 序列计算
# ---------------------------------------------------------------------------
def compute_series(df: pd.DataFrame, config: BacktestConfig) -> Dict[str, pd.Series]:
    """回放所用的全部序列：共享模块的指标 + 回测特有的量能与通道序列。

    指标本身（RSI / MACD / Stochastic / ROC / Williams %R / 布林带 / ATR /
    均线）一律取自共享 `indicators` 模块，回测因此与报告、Agent 读同一批
    数字。这里只补充共享模块不负责的东西：开盘价、量比、滚动区间与趋势斜率。
    """
    series = dict(shared_indicators.compute_series(df))
    series.update(_backtest_only_series(df, config))
    return series


def _backtest_only_series(
    df: pd.DataFrame, config: BacktestConfig
) -> Dict[str, pd.Series]:
    """共享模块不提供、但规则策略需要的序列（量能 / 通道 / 斜率）。"""
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    # 成交量相对强度（相对过去 20 根均量，且不含当前 bar）
    volume = df["Volume"].astype(float)
    vol_ratio = volume / volume.rolling(20).mean().shift(1).replace(0, np.nan)

    w = max(5, int(config.trend_window))
    roll_high = high.rolling(max(2, int(config.breakout_window))).max().shift(1)
    roll_low = low.rolling(max(2, int(config.breakout_window))).min().shift(1)

    # 滚动线性回归斜率（向量化）：以百分比表示「窗口内整体涨跌幅」
    y = close
    idx = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    sy = y.rolling(w).sum()
    syx = (idx * y).rolling(w).sum()
    # 窗口内相对横坐标 k=0..w-1 的 Σk·y 与 Σy
    sxy = syx - (idx - (w - 1)) * sy
    sx = w * (w - 1) / 2.0
    sxx = w * (w - 1) * (2 * w - 1) / 6.0
    denom = (w * sxx - sx * sx)
    slope = (w * sxy - sx * sy) / denom if denom else pd.Series(np.nan, index=close.index)
    mean_price = sy / w
    trend_slope_pct = slope * w / mean_price.replace(0, np.nan) * 100

    return {
        "open": df["Open"].astype(float),
        "vol_ratio": vol_ratio,
        "roll_high": roll_high,
        "roll_low": roll_low,
        "trend_slope_pct": trend_slope_pct,
    }


# ---------------------------------------------------------------------------
# 单 bar 打分
# ---------------------------------------------------------------------------
def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, value)))


def _indicator_signal(v: Dict[str, Optional[float]]) -> Tuple[float, Dict[str, float]]:
    """技术指标综合分（-1..1），返回 (分数, 分项分数)。

    `v` 用的是共享模块的带周期键（rsi_14 / roc_10 / willr_14 / atr_14），
    回测与报告读同一批数字。
    """
    atr = v.get("atr_14") or 0.0
    hist = v.get("macd_hist")
    macd_vote = 0.0
    if hist is not None:
        scale = max(1e-9, atr * 0.3)
        macd_vote = _clip(hist / scale)

    rsi = v.get("rsi_14")
    rsi_vote = 0.0
    if rsi is not None:
        if rsi >= 70:
            rsi_vote = -0.30
        elif rsi <= 30:
            rsi_vote = 0.30
        elif rsi > 55:
            rsi_vote = _clip((rsi - 55) / 15.0) * 0.5
        elif rsi < 45:
            rsi_vote = -_clip((45 - rsi) / 15.0) * 0.5

    roc = v.get("roc_10")
    roc_vote = _clip(roc / 5.0) if roc is not None else 0.0

    k, d = v.get("stoch_k"), v.get("stoch_d")
    stoch_vote = 0.0
    if k is not None and d is not None:
        stoch_vote = _clip((k - d) / 20.0)
        if k > STOCH_OVERBOUGHT:
            stoch_vote -= 0.3
        elif k < STOCH_OVERSOLD:
            stoch_vote += 0.3
        stoch_vote = _clip(stoch_vote)

    willr = v.get("willr_14")
    willr_vote = _clip(-(willr + 50) / 50.0) if willr is not None else 0.0

    close, ma5, ma20 = v.get("close"), v.get("ma5"), v.get("ma20")
    ma_vote = 0.0
    if None not in (close, ma5, ma20):
        if close > ma5 > ma20:
            ma_vote = 1.0
        elif close < ma5 < ma20:
            ma_vote = -1.0
        else:
            ma_vote = _clip((close - ma20) / max(1e-9, ma20) * 20.0) * 0.3

    parts = {
        "macd": macd_vote,
        "rsi": rsi_vote,
        "roc": roc_vote,
        "stoch": stoch_vote,
        "willr": willr_vote,
        "ma": ma_vote,
    }
    weights = {"macd": 0.30, "rsi": 0.20, "roc": 0.15, "stoch": 0.15, "willr": 0.10, "ma": 0.10}
    score = sum(parts[k] * weights[k] for k in parts)
    return _clip(score), {k: round(v_, 4) for k, v_ in parts.items()}


def _trend_signal(v: Dict[str, Optional[float]], slope_pct: Optional[float]) -> Tuple[float, Dict[str, float]]:
    """趋势通道斜率信号：窗口内涨跌幅 3% 视为满强度。"""
    if slope_pct is None:
        return 0.0, {"slope_pct": 0.0}
    score = _clip(slope_pct / 3.0)
    return score, {"slope_pct": round(float(slope_pct), 4)}


def _pattern_signal(
    v: Dict[str, Optional[float]], prev: Dict[str, Optional[float]]
) -> Tuple[float, List[str]]:
    """单/双根 K 线模式 + 区间突破，返回 (-1..1, 命中模式列表)。"""
    o, h, l, c = v.get("open"), v.get("high"), v.get("low"), v.get("close")
    po, pc = prev.get("open"), prev.get("close")
    if None in (o, h, l, c):
        return 0.0, []
    score = 0.0
    names: List[str] = []

    body = abs(c - o)
    rng = max(1e-9, h - l)
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l

    if None not in (po, pc):
        prev_body = abs(pc - po)
        # 看涨吞没 / 看跌吞没
        if c > o and pc < po and c >= po and o <= pc and body > prev_body:
            score += 0.6
            names.append("bullish_engulfing")
        elif c < o and pc > po and c <= po and o >= pc and body > prev_body:
            score -= 0.6
            names.append("bearish_engulfing")
        # 十字星：实体极小 → 方向中性，轻微降低置信度（用 0 分表示）
        if body / rng < 0.1:
            names.append("doji")

    # 锤子线 / 射击之星
    if lower_shadow > 2 * max(body, 1e-9) and upper_shadow < body and body > 0:
        score += 0.35
        names.append("hammer")
    if upper_shadow > 2 * max(body, 1e-9) and lower_shadow < body and body > 0:
        score -= 0.35
        names.append("shooting_star")

    # 区间突破（需成交量配合才增强）
    rh, rl = v.get("roll_high"), v.get("roll_low")
    vol_ratio = v.get("vol_ratio") or 1.0
    boost = 1.0 if vol_ratio < 1.5 else 1.3
    if rh is not None and c > rh:
        score += 0.5 * boost
        names.append("breakout_up")
    elif rl is not None and c < rl:
        score -= 0.5 * boost
        names.append("breakout_down")

    return _clip(score), names


class RuleSignalEngine:
    """确定性信号引擎：给定 K 线，输出与线上决策同口径的 LONG/SHORT。"""

    def __init__(self, df: pd.DataFrame, config: BacktestConfig):
        self.config = config
        self.series = compute_series(df, config)
        self._weights = {
            "indicator": float(config.weight_indicator),
            "trend": float(config.weight_trend),
            "pattern": float(config.weight_pattern),
        }
        self._wsum = sum(self._weights.values())

    def _row(self, i: int) -> Dict[str, Optional[float]]:
        out = {}
        for name, s in self.series.items():
            value = s.iloc[i]
            out[name] = _safe(value) if not isinstance(value, (str,)) else None
        return out

    def evaluate(self, i: int) -> Dict[str, Any]:
        v = self._row(i)
        prev = self._row(i - 1) if i > 0 else {}

        ind_score, ind_parts = _indicator_signal(v)
        trend_score, trend_parts = _trend_signal(v, v.get("trend_slope_pct"))
        pat_score, patterns = _pattern_signal(v, prev)

        raw = (
            ind_score * self._weights["indicator"]
            + trend_score * self._weights["trend"]
            + pat_score * self._weights["pattern"]
        ) / self._wsum
        score = _clip(raw)

        tie_break = abs(score) < 1e-9
        if tie_break:
            # 决策策略约定：多空信号僵持时，以趋势通道方向为最终方向
            direction = "LONG" if trend_score >= 0 else "SHORT"
        else:
            direction = "LONG" if score > 0 else "SHORT"

        confidence = _clip(abs(score) / float(self.config.confidence_scale), 0.0, 1.0)
        confidence = max(0.05, confidence)
        rr = round(min(1.8, max(1.2, 1.2 + 0.6 * confidence)), 2)

        snapshot = {
            "rsi_14": v.get("rsi_14"),
            "macd_hist": v.get("macd_hist"),
            "roc_10": v.get("roc_10"),
            "stoch_k": v.get("stoch_k"),
            "willr_14": v.get("willr_14"),
            "atr_14": v.get("atr_14"),
            "close": v.get("close"),
            "vol_ratio": v.get("vol_ratio"),
            "trend_slope_pct": v.get("trend_slope_pct"),
            "patterns": patterns,
        }

        return {
            "direction": direction,
            "score": round(float(score), 6),
            "confidence": round(float(confidence), 6),
            "risk_reward_ratio": rr,
            "tie_break": bool(tie_break),
            "sub_signals": {
                "indicator": round(float(ind_score), 4),
                "trend": round(float(trend_score), 4),
                "pattern": round(float(pat_score), 4),
            },
            "signal_parts": {
                "indicator": ind_parts,
                "trend": trend_parts,
                "patterns": patterns,
            },
            "signals": snapshot,
        }
