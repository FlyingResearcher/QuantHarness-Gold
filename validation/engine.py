"""回测引擎：逐 bar 回放决策 → 模拟成交 → 生成决策/交易明细与权益曲线。

关键原则
  * 无未来函数：第 i 根 K 线收盘时决策，第 i+1 根开盘成交；
  * 成本可配置：单边手续费 + 单边滑点；
  * 悲观撮合：同一根 K 线同时触发止损与止盈时，按止损处理；
  * 增量缓存：决策按「时间戳 + 历史窗口指纹」缓存，数据或参数变化会自动失效；
  * 异常隔离：单次决策失败只记录 ENGINE_ERROR，不中断整轮回测（可配置为 raise）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ENGINE_VERSION, BacktestConfig
from .data import window_digest
from .strategies import DecisionStrategy, build_strategy


def _clean(value: Any, digits: int = 6) -> Any:
    """递归清理 numpy / NaN / 时间类型，保证可 JSON 序列化。"""
    if isinstance(value, dict):
        return {str(k): _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not np.isfinite(f):
            return None
        return round(f, digits)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if value is pd.NaT:
        return None
    return value


@dataclass
class Decision:
    """一次决策的完整审计记录。"""

    index: int
    datetime: str
    direction: Optional[str]
    confidence: float
    score: Optional[float]
    risk_reward_ratio: float
    sub_signals: Optional[Dict[str, float]]
    signal_parts: Dict[str, Any]
    signals: Dict[str, Any]
    latency_ms: float
    strategy: str
    window_digest: str
    error: Optional[str] = None
    tie_break: bool = False
    cached: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class Trade:
    """一次模拟成交。"""

    decision_index: int
    entry_index: int
    exit_index: int
    entry_datetime: str
    exit_datetime: str
    direction: str
    entry_price: float
    exit_price: float
    stop_price: Optional[float]
    target_price: Optional[float]
    exit_reason: str  # stop / target / horizon
    holding_bars: int
    gross_return_pct: float
    net_return_pct: float
    mfe_pct: float
    mae_pct: float
    time_to_mfe_bars: int
    correct: Optional[bool]
    confidence: float
    benchmark_return_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return _clean(self.__dict__)


@dataclass
class BacktestRun:
    """一次完整回测的载体（runner 会继续填充 metrics / anomalies）。"""

    run_id: str
    config: BacktestConfig
    data_meta: Dict[str, Any]
    decisions: List[Decision]
    trades: List[Trade]
    series: Dict[str, Any]
    stats: Dict[str, Any]
    decisions_hash: str
    decision_cache: Dict[str, Dict[str, Any]]
    data_issues: List[Dict[str, Any]]
    timings: Dict[str, float]
    strategy_name: str
    engine_version: str = ENGINE_VERSION
    metrics: Dict[str, Any] = field(default_factory=dict)
    anomalies: Dict[str, Any] = field(default_factory=dict)


def _data_issue_mask(df: pd.DataFrame) -> np.ndarray:
    """逐 bar 标记数据质量问题（缺口 / 缺失 / 零振幅 / 重复时间戳）。"""
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    ohlc = df[["Open", "High", "Low", "Close"]]
    mask |= ohlc.isna().any(axis=1).to_numpy()
    mask |= (df["High"] == df["Low"]).to_numpy()
    mask |= df["Datetime"].duplicated().to_numpy()

    seconds = df["Datetime"].diff().dt.total_seconds()
    median = float(seconds.median()) if seconds.notna().any() else 0.0
    if median > 0:
        mask |= (seconds > median * 3).fillna(False).to_numpy()
    return mask


class BacktestEngine:
    """决策回放引擎。

    使用方式：
        run = BacktestEngine(config).run(bars, data_meta, cached_decisions=...)
    """

    def __init__(self, config: BacktestConfig, strategy: Optional[DecisionStrategy] = None):
        self.config = config
        self.strategy = strategy or build_strategy(config)

    # ------------------------------------------------------------------
    def run(
        self,
        bars: pd.DataFrame,
        data_meta: Optional[Dict[str, Any]] = None,
        cached_decisions: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> BacktestRun:
        started = time.perf_counter()
        n = len(bars)
        warmup = int(self.config.warmup_bars)
        if n < warmup + 2:
            raise ValueError(f"数据不足：{n} 根不足以完成 warmup={warmup} 的回放")

        self.strategy.prepare(bars)

        cache = dict(cached_decisions or {})
        decisions: List[Decision] = []
        new_cache: Dict[str, Dict[str, Any]] = {}
        issue_mask = _data_issue_mask(bars)
        issue_prefix = np.cumsum(issue_mask)
        decision_issues: List[Dict[str, Any]] = []

        computed = reused = skipped = errors = 0

        for i in range(warmup, n - 1):
            ts = pd.Timestamp(bars.iloc[i]["Datetime"])
            ts_text = ts.strftime("%Y-%m-%d %H:%M:%S")
            digest = window_digest(bars, i, warmup)
            key = f"{ts_text}|{digest}"

            cached = cache.get(key)
            if cached is not None:
                decision = Decision.from_dict(cached)
                decision.cached = True
                reused += 1
            else:
                started_decision = time.perf_counter()
                error: Optional[str] = None
                output = None
                try:
                    output = self.strategy.decide(bars, i)
                except Exception as exc:  # noqa: BLE001 - 单次失败不应中断整轮
                    if self.config.on_error == "raise":
                        raise
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started_decision) * 1000.0

                if output is None and error is None:
                    skipped += 1
                    continue
                if error is not None:
                    errors += 1
                    decision = Decision(
                        index=i,
                        datetime=ts_text,
                        direction=None,
                        confidence=0.0,
                        score=None,
                        risk_reward_ratio=0.0,
                        sub_signals=None,
                        signal_parts={},
                        signals={},
                        latency_ms=round(latency_ms, 3),
                        strategy=self.strategy.name,
                        window_digest=digest,
                        error=error,
                    )
                else:
                    lat = float((output.extra or {}).get("latency_ms", latency_ms))
                    decision = Decision(
                        index=i,
                        datetime=ts_text,
                        direction=output.direction,
                        confidence=float(output.confidence),
                        score=output.score,
                        risk_reward_ratio=float(output.risk_reward_ratio),
                        sub_signals=output.sub_signals,
                        signal_parts=output.signal_parts,
                        signals=output.signals,
                        latency_ms=round(lat, 3),
                        strategy=self.strategy.name,
                        window_digest=digest,
                        tie_break=output.tie_break,
                        extra=output.extra,
                    )
                computed += 1
                new_cache[key] = decision.to_dict()

            decisions.append(decision)

            # 记录该决策窗口内的数据质量问题（供异常归类使用）
            left = max(0, i - warmup + 1)
            window_issues = int(issue_prefix[i] - (issue_prefix[left - 1] if left > 0 else 0))
            if window_issues > 0:
                decision_issues.append(
                    {
                        "index": i,
                        "datetime": ts_text,
                        "issue_bars": window_issues,
                        "current_bar_issue": bool(issue_mask[i]),
                    }
                )

        trades = self._simulate_trades(bars, decisions)
        series = self._build_series(trades)
        stats = {
            "bars": n,
            "eligible_decisions": max(0, n - 1 - warmup),
            "decisions": len(decisions),
            "trades": len(trades),
            "computed_decisions": computed,
            "reused_decisions": reused,
            "skipped_decisions": skipped,
            "error_decisions": errors,
            "data_issue_bars": int(issue_mask.sum()),
        }
        run = BacktestRun(
            run_id=self.config.run_id,
            config=self.config,
            data_meta=data_meta or {},
            decisions=decisions,
            trades=trades,
            series=series,
            stats=stats,
            decisions_hash=_decisions_hash(decisions),
            decision_cache={**cache, **new_cache},
            data_issues=decision_issues,
            timings={"engine_s": round(time.perf_counter() - started, 3)},
            strategy_name=self.strategy.name,
        )
        return run

    # ------------------------------------------------------------------
    def _simulate_trades(self, bars: pd.DataFrame, decisions: List[Decision]) -> List[Trade]:
        cfg = self.config
        n = len(bars)
        fee = float(cfg.fee_pct)
        slip = float(cfg.slippage_pct)
        horizon = int(cfg.horizon_bars)
        trades: List[Trade] = []

        for dec in decisions:
            if dec.direction not in ("LONG", "SHORT"):
                continue
            entry_i = dec.index + 1
            if entry_i >= n:
                continue
            last_i = min(entry_i + horizon - 1, n - 1)

            entry_raw = float(bars.iloc[entry_i]["Open"])
            if not np.isfinite(entry_raw) or entry_raw <= 0:
                continue
            sign = 1.0 if dec.direction == "LONG" else -1.0
            # 入场滑点始终不利：做多买贵、做空卖便宜
            entry_adj = entry_raw * (1.0 + slip / 100.0) if sign > 0 else entry_raw * (1.0 - slip / 100.0)

            try:
                atr = dec.signals.get("atr_14") if isinstance(dec.signals, dict) else None
            except Exception:  # pragma: no cover - 仅防御性代码
                atr = None
            try:
                atr = float(atr) if atr is not None else None
            except (TypeError, ValueError):
                atr = None

            stop_price: Optional[float] = None
            target_price: Optional[float] = None
            if cfg.use_stops and atr and atr > 0:
                stop_dist = float(cfg.stop_atr_mult) * atr
                if stop_dist > 0:
                    if sign > 0:
                        stop_price = entry_raw - stop_dist
                        target_dist = (
                            float(cfg.target_atr_mult) * atr
                            if cfg.target_atr_mult > 0
                            else stop_dist * float(cfg.risk_reward)
                        )
                        target_price = entry_raw + target_dist
                    else:
                        stop_price = entry_raw + stop_dist
                        target_dist = (
                            float(cfg.target_atr_mult) * atr
                            if cfg.target_atr_mult > 0
                            else stop_dist * float(cfg.risk_reward)
                        )
                        target_price = entry_raw - target_dist

            exit_i = last_i
            exit_raw = float(bars.iloc[last_i]["Close"])
            exit_reason = "horizon"
            for j in range(entry_i, last_i + 1):
                high = float(bars.iloc[j]["High"])
                low = float(bars.iloc[j]["Low"])
                if stop_price is not None and (
                    (sign > 0 and low <= stop_price) or (sign < 0 and high >= stop_price)
                ):
                    # 同一根 K 线同时触发止损/止盈时，按悲观假设先止损
                    exit_i, exit_raw, exit_reason = j, stop_price, "stop"
                    break
                if target_price is not None and (
                    (sign > 0 and high >= target_price) or (sign < 0 and low <= target_price)
                ):
                    exit_i, exit_raw, exit_reason = j, target_price, "target"
                    break

            exit_adj = exit_raw * (1.0 - slip / 100.0) if sign > 0 else exit_raw * (1.0 + slip / 100.0)
            gross = sign * (exit_raw / entry_raw - 1.0) * 100.0
            net = sign * (exit_adj / entry_adj - 1.0) * 100.0 - 2.0 * fee

            segment = bars.iloc[entry_i : exit_i + 1]
            if sign > 0:
                mfe = (float(segment["High"].max()) / entry_adj - 1.0) * 100.0
                mae = (float(segment["Low"].min()) / entry_adj - 1.0) * 100.0
                mfe_idx = int(np.argmax(segment["High"].to_numpy()))
            else:
                mfe = (entry_adj / float(segment["Low"].min()) - 1.0) * 100.0
                mae = (entry_adj / float(segment["High"].max()) - 1.0) * 100.0
                mfe_idx = int(np.argmax(entry_adj / segment["Low"].to_numpy() - 1.0))

            bench_net = (
                ((exit_raw * (1.0 - slip / 100.0)) / (entry_raw * (1.0 + slip / 100.0)) - 1.0) * 100.0
                - 2.0 * fee
            )
            benchmark_return = 0.0 if cfg.benchmark == "flat" else bench_net

            correct: Optional[bool]
            if abs(gross) < 1e-12:
                correct = None
            else:
                correct = gross > 0

            trades.append(
                Trade(
                    decision_index=dec.index,
                    entry_index=entry_i,
                    exit_index=exit_i,
                    entry_datetime=pd.Timestamp(bars.iloc[entry_i]["Datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
                    exit_datetime=pd.Timestamp(bars.iloc[exit_i]["Datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
                    direction=dec.direction,
                    entry_price=round(entry_adj, 6),
                    exit_price=round(exit_adj, 6),
                    stop_price=round(stop_price, 6) if stop_price is not None else None,
                    target_price=round(target_price, 6) if target_price is not None else None,
                    exit_reason=exit_reason,
                    holding_bars=int(exit_i - entry_i + 1),
                    gross_return_pct=round(gross, 6),
                    net_return_pct=round(net, 6),
                    mfe_pct=round(mfe, 6),
                    mae_pct=round(mae, 6),
                    time_to_mfe_bars=int(mfe_idx),
                    correct=correct,
                    confidence=round(float(dec.confidence), 6),
                    benchmark_return_pct=round(benchmark_return, 6),
                )
            )
        return trades

    # ------------------------------------------------------------------
    @staticmethod
    def _build_series(trades: List[Trade]) -> Dict[str, Any]:
        equity = [1.0]
        benchmark = [1.0]
        exit_dt: List[str] = []
        drawdown: List[float] = [0.0]
        peak = 1.0
        for tr in trades:
            equity.append(equity[-1] * (1.0 + tr.net_return_pct / 100.0))
            benchmark.append(benchmark[-1] * (1.0 + tr.benchmark_return_pct / 100.0))
            exit_dt.append(tr.exit_datetime)
            peak = max(peak, equity[-1])
            drawdown.append((equity[-1] / peak - 1.0) * 100.0 if peak else 0.0)
        return {
            "exit_datetime": exit_dt,
            "equity": [round(v, 8) for v in equity],
            "benchmark_equity": [round(v, 8) for v in benchmark],
            "drawdown_pct": [round(v, 6) for v in drawdown],
            "trade_net_returns_pct": [tr.net_return_pct for tr in trades],
        }


def _decisions_hash(decisions: List[Decision]) -> str:
    """决策序列哈希：同一数据 + 同一参数重跑必须完全一致。"""
    payload = [
        [
            d.datetime,
            d.direction,
            round(float(d.confidence), 6),
            None if d.score is None else round(float(d.score), 6),
            round(float(d.risk_reward_ratio), 6),
            d.error or "",
        ]
        for d in decisions
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
