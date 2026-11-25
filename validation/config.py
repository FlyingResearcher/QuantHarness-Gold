"""回测配置：所有可调参数集中在此，保证「参数配置 + 可复现」有唯一来源。

设计要点：
  * 使用冻结 dataclass，避免运行过程中被意外改写；
  * `decision_fingerprint()` 只包含影响「决策结果」的参数，
    用于增量回测时判定历史决策缓存是否仍然有效；
  * 费用 / 滑点 / 止盈止损等只影响成交与绩效，不进入决策指纹，
    因此调整这些参数重跑时仍可复用已计算的决策。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

# 版本随指标口径变化递增。它参与 decision_params() → 决策指纹，
# 因此升版会让旧的增量缓存全部失效，避免新旧指标口径的决策混在一起。
ENGINE_VERSION = "1.1.0"

# 支持的回测周期（与 gold_data 的周期口径保持一致）
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1mo")
SUPPORTED_STRATEGIES = ("rule", "recorded", "llm")
SUPPORTED_BENCHMARKS = ("buy_hold", "flat")

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ConfigError(ValueError):
    """配置非法或数据不足时抛出。"""


@dataclass(frozen=True)
class BacktestConfig:
    """一次回测的完整参数。

    数据范围
        asset / timeframe / source : 与 gold_data 的资产与数据源口径一致
        start / end                : 回测区间（含端点），None 表示由数据层决定
        min_bars                   : 数据层至少返回的 K 线根数
    运行方式
        run_id / incremental       : 增量回测：复用同 run_id 下已计算的决策
    决策与成交
        strategy                   : rule（确定性规则回放）/ recorded（回放已记录决策）/ llm（真实决策图）
        horizon_bars               : 决策预测 / 持有的 K 线根数（N）
        warmup_bars                : 决策所需的最小历史窗口（≥26 才能算 MACD）
        fee_pct / slippage_pct     : 单边成本（百分比）
        use_stops                  : 是否启用 ATR 止损/止盈
        stop_atr_mult              : 止损距离 = ATR × 该倍数
        target_atr_mult            : 止盈距离；0 表示由 risk_reward 推导
        risk_reward                : 目标风险收益比
    基准
        benchmark                  : buy_hold（同区间满仓做多）/ flat（空仓）
    异常判定阈值
        error_return_threshold_pct : 方向错误的「实质性」盈亏阈值（%）
        min_confidence             : 低置信度阈值
        max_trade_loss_pct         : 单笔最大可接受亏损（%）
        max_repeat_signals         : 连续同向决策的容忍次数
        late_entry_bars            : 入场后多少根内触发止损即视为「迟滞入场」
        on_error                   : record（记录异常并继续）/ raise（立即抛错）
    """

    # --- 数据范围 ---
    asset: str = "COMEX"
    timeframe: str = "1d"
    source: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    min_bars: int = 60

    # --- 运行方式 ---
    run_id: str = "default"
    incremental: bool = False

    # --- 决策与成交 ---
    strategy: str = "rule"
    horizon_bars: int = 1
    warmup_bars: int = 30
    fee_pct: float = 0.0
    slippage_pct: float = 0.0
    use_stops: bool = True
    stop_atr_mult: float = 2.0
    target_atr_mult: float = 0.0
    risk_reward: float = 1.5

    # --- 规则策略参数（同时也作为决策指纹的一部分）---
    weight_indicator: float = 0.45
    weight_trend: float = 0.35
    weight_pattern: float = 0.20
    confidence_scale: float = 0.50
    trend_window: int = 30
    breakout_window: int = 20

    # --- 已记录决策回放 ---
    recorded_path: Optional[str] = None

    # --- LLM 回放（可选，需要 API Key；非确定性，不入 CI）---
    llm_window_bars: int = 45

    # --- 基准 ---
    benchmark: str = "buy_hold"

    # --- 异常判定 ---
    error_return_threshold_pct: float = 0.10
    min_confidence: float = 0.35
    max_trade_loss_pct: float = 2.0
    max_repeat_signals: int = 3
    late_entry_bars: int = 1

    # --- 容错 ---
    on_error: str = "record"

    # ------------------------------------------------------------------
    # 校验与序列化
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        tf = str(self.timeframe or "").strip().lower()
        if tf not in SUPPORTED_TIMEFRAMES:
            raise ConfigError(
                f"不支持的 timeframe='{self.timeframe}'，可选：{', '.join(SUPPORTED_TIMEFRAMES)}"
            )
        object.__setattr__(self, "timeframe", tf)

        strategy = str(self.strategy or "").strip().lower()
        if strategy not in SUPPORTED_STRATEGIES:
            raise ConfigError(
                f"不支持的 strategy='{self.strategy}'，可选：{', '.join(SUPPORTED_STRATEGIES)}"
            )
        object.__setattr__(self, "strategy", strategy)

        benchmark = str(self.benchmark or "").strip().lower()
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ConfigError(
                f"不支持的 benchmark='{self.benchmark}'，可选：{', '.join(SUPPORTED_BENCHMARKS)}"
            )
        object.__setattr__(self, "benchmark", benchmark)

        if self.on_error not in ("record", "raise"):
            raise ConfigError("on_error 只能是 'record' 或 'raise'")
        if not _RUN_ID_RE.match(str(self.run_id or "")):
            raise ConfigError(
                "run_id 只允许字母/数字/._-，长度 1-64，例如 'comex_1d_rule'"
            )
        if int(self.horizon_bars) < 1:
            raise ConfigError("horizon_bars 必须 ≥ 1")
        if int(self.warmup_bars) < 26:
            raise ConfigError("warmup_bars 必须 ≥ 26（MACD 需要 26 根历史）")
        if int(self.min_bars) < int(self.warmup_bars) + 2:
            raise ConfigError("min_bars 至少为 warmup_bars + 2")
        for name in ("fee_pct", "slippage_pct", "stop_atr_mult", "risk_reward"):
            if float(getattr(self, name)) < 0:
                raise ConfigError(f"{name} 不能为负数")
        if float(self.target_atr_mult) < 0:
            raise ConfigError("target_atr_mult 不能为负数")
        weights = (self.weight_indicator, self.weight_trend, self.weight_pattern)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ConfigError("指标/趋势/模式权重不能为负，且总和必须大于 0")
        if not 0 < float(self.confidence_scale):
            raise ConfigError("confidence_scale 必须为正数")
        if strategy == "recorded" and not self.recorded_path:
            raise ConfigError("strategy='recorded' 时必须提供 recorded_path")
        if self.start and self.end and str(self.start) > str(self.end):
            raise ConfigError("start 不能晚于 end")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "BacktestConfig":
        """从字典构造，忽略未知字段（便于读取历史 manifest）。"""
        if not data:
            return cls()
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    # ------------------------------------------------------------------
    # 指纹
    # ------------------------------------------------------------------
    def decision_params(self) -> Dict[str, Any]:
        """影响「决策本身」的参数子集（费用/滑点/止损不影响决策）。"""
        base = {
            "engine_version": ENGINE_VERSION,
            "strategy": self.strategy,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "warmup_bars": int(self.warmup_bars),
            "weight_indicator": round(float(self.weight_indicator), 6),
            "weight_trend": round(float(self.weight_trend), 6),
            "weight_pattern": round(float(self.weight_pattern), 6),
            "confidence_scale": round(float(self.confidence_scale), 6),
            "trend_window": int(self.trend_window),
            "breakout_window": int(self.breakout_window),
            "llm_window_bars": int(self.llm_window_bars),
        }
        if self.strategy == "recorded":
            base["recorded_path"] = str(self.recorded_path or "")
        return base

    def decision_fingerprint(self, extra: Optional[Dict[str, Any]] = None) -> str:
        payload = self.decision_params()
        if extra:
            payload.update(extra)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def annualization_factor(timeframe: str) -> float:
    """周期 → 年化因子（按金融市场常用近似交易日折算）。"""
    mapping = {
        "1m": 525600.0,
        "5m": 105120.0,
        "15m": 35040.0,
        "30m": 17520.0,
        "1h": 8760.0,
        "4h": 2190.0,
        "1d": 252.0,
        "1w": 52.0,
        "1mo": 12.0,
    }
    return mapping.get(str(timeframe).lower(), 252.0)
