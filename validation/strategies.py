"""可插拔的决策回放策略。

三种策略共用同一接口，回测引擎只依赖接口，不关心决策如何产生：

  rule     —— 确定性规则回放（默认）。复刻决策智能体文档化的策略：
              指标 45% + 趋势 35% + 形态 20%，多空僵持时以趋势方向兜底。
  recorded —— 回放系统真实产出的决策记录（CSV/JSON），用于评估线上决策质量。
  llm      —— 逐步调用真实多智能体决策图（需要 API Key，慢且非确定性，可选）。

新增策略只需实现 `prepare()` 与 `decide()`，无需改动回测引擎。
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import BacktestConfig, ConfigError
from .data import PROJECT_ROOT
from .indicators import RuleSignalEngine


@dataclass
class StrategyOutput:
    """单次决策的输出（与引擎解耦的数据契约）。"""

    direction: str  # LONG / SHORT
    confidence: float = 0.5
    score: Optional[float] = None
    risk_reward_ratio: float = 1.5
    sub_signals: Optional[Dict[str, float]] = None
    signal_parts: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    tie_break: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class DecisionStrategy(ABC):
    name = "base"
    nondeterministic = False

    @abstractmethod
    def prepare(self, bars: pd.DataFrame) -> None:
        """回放开始前的一次性初始化。"""

    @abstractmethod
    def decide(self, bars: pd.DataFrame, index: int) -> Optional[StrategyOutput]:
        """基于 bars[0..index] 给出决策；返回 None 表示该时点跳过。"""

    def fingerprint_extra(self) -> Dict[str, Any]:
        """参与决策指纹的额外信息（例如决策记录文件哈希）。"""
        return {}


# ---------------------------------------------------------------------------
# 1) 确定性规则策略
# ---------------------------------------------------------------------------
class RuleBasedDecisionStrategy(DecisionStrategy):
    name = "rule"

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._engine: Optional[RuleSignalEngine] = None

    def prepare(self, bars: pd.DataFrame) -> None:
        self._engine = RuleSignalEngine(bars, self.config)

    def decide(self, bars: pd.DataFrame, index: int) -> Optional[StrategyOutput]:
        assert self._engine is not None, "prepare() 必须先于 decide() 调用"
        out = self._engine.evaluate(index)
        return StrategyOutput(
            direction=out["direction"],
            confidence=out["confidence"],
            score=out["score"],
            risk_reward_ratio=out["risk_reward_ratio"],
            sub_signals=out["sub_signals"],
            signal_parts=out["signal_parts"],
            signals=out["signals"],
            tie_break=out["tie_break"],
        )


# ---------------------------------------------------------------------------
# 2) 已记录决策回放
# ---------------------------------------------------------------------------
_DIRECTION_MAP = {
    "LONG": "LONG", "BUY": "LONG", "BULL": "LONG", "BULLISH": "LONG", "做多": "LONG", "买入": "LONG",
    "SHORT": "SHORT", "SELL": "SHORT", "BEAR": "SHORT", "BEARISH": "SHORT", "做空": "SHORT", "卖出": "SHORT",
}


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().upper()
    if key in _DIRECTION_MAP:
        return _DIRECTION_MAP[key]
    for word, mapped in _DIRECTION_MAP.items():
        if word in key:
            return mapped
    return None


class RecordedDecisionStrategy(DecisionStrategy):
    """把系统历史决策（或人工标注结果）按时间戳对齐到 K 线后回放。"""

    name = "recorded"

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._records: Dict[pd.Timestamp, Dict[str, Any]] = {}
        self._file_hash: Optional[str] = None
        self.missing: int = 0

    def _resolve_path(self) -> Path:
        p = Path(str(self.config.recorded_path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p

    def prepare(self, bars: pd.DataFrame) -> None:
        path = self._resolve_path()
        if not path.exists():
            raise ConfigError(f"决策记录文件不存在：{path}")
        raw = path.read_bytes()
        self._file_hash = hashlib.sha256(raw).hexdigest()

        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            frame = pd.read_json(path, lines=True)
        elif suffix == ".json":
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("decisions", [])
            frame = pd.DataFrame(payload)
        else:
            frame = pd.read_csv(path)

        if frame.empty:
            raise ConfigError(f"决策记录为空：{path}")
        if "Datetime" not in frame.columns and "datetime" in frame.columns:
            frame = frame.rename(columns={"datetime": "Datetime"})
        if "Datetime" not in frame.columns:
            raise ConfigError("决策记录缺少 Datetime 列")

        dir_col = next(
            (c for c in ("decision", "direction", "signal", "action") if c in frame.columns),
            None,
        )
        if dir_col is None:
            raise ConfigError("决策记录缺少 decision/direction/signal 列")

        records: Dict[pd.Timestamp, Dict[str, Any]] = {}
        for _, row in frame.iterrows():
            ts = pd.to_datetime(row["Datetime"], errors="coerce")
            direction = _normalize_direction(row.get(dir_col))
            if pd.isna(ts) or direction is None:
                continue
            try:
                confidence = float(row.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            try:
                rr = float(row.get("risk_reward_ratio", 1.5))
            except (TypeError, ValueError):
                rr = 1.5
            records[pd.Timestamp(ts)] = {
                "direction": direction,
                "confidence": min(1.0, max(0.0, confidence)),
                "risk_reward_ratio": rr,
            }
        if not records:
            raise ConfigError("决策记录中没有可用行（需要合法 Datetime + 方向）")
        self._records = records

    def decide(self, bars: pd.DataFrame, index: int) -> Optional[StrategyOutput]:
        ts = pd.Timestamp(bars.iloc[index]["Datetime"])
        rec = self._records.get(ts)
        if rec is None:
            self.missing += 1
            return None
        return StrategyOutput(
            direction=rec["direction"],
            confidence=rec["confidence"],
            risk_reward_ratio=rec["risk_reward_ratio"],
            sub_signals=None,
            signals={"recorded": True},
            extra={"source": "recorded"},
        )

    def fingerprint_extra(self) -> Dict[str, Any]:
        # 决策指纹在 prepare 之前计算；此处按需读取文件哈希，
        # 保证「决策记录内容变化」也能让增量缓存失效。
        if self._file_hash is None and self.config.recorded_path:
            try:
                self._file_hash = hashlib.sha256(self._resolve_path().read_bytes()).hexdigest()
            except OSError:
                self._file_hash = ""
        return {"recorded_file_sha256": self._file_hash or ""}


# ---------------------------------------------------------------------------
# 3) 真实决策图回放（可选）
# ---------------------------------------------------------------------------
def parse_decision_text(raw: Any) -> Dict[str, Any]:
    """从决策图输出中解析结构化字段（与 run_gold_analysis._parse_decision 同口径）。"""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except ValueError:
            pass
    return {"raw": text}


class LLMDecisionStrategy(DecisionStrategy):
    """逐步调用 TradingGraph 真实决策（可选）。

    注意：
      * 需要可用的 LLM API Key（与线上系统同一套 .env 配置）；
      * 非确定性，不纳入自动测试与可复现断言；
      * 决策图在缺少图片时会走 OHLC/文本兜底，故这里不生成图片以节省时间。
    """

    name = "llm"
    nondeterministic = True

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._graph = None

    def prepare(self, bars: pd.DataFrame) -> None:
        from trading_graph import TradingGraph  # 延迟导入，保证 rule/recorded 离线可用

        self._graph = TradingGraph()

    def decide(self, bars: pd.DataFrame, index: int) -> Optional[StrategyOutput]:
        if self._graph is None:
            raise RuntimeError("LLM 策略尚未初始化（prepare 未调用）")
        window = bars.iloc[max(0, index - int(self.config.llm_window_bars) + 1) : index + 1]
        kline_data = {
            "Datetime": window["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "Open": window["Open"].tolist(),
            "High": window["High"].tolist(),
            "Low": window["Low"].tolist(),
            "Close": window["Close"].tolist(),
            "Volume": window["Volume"].fillna(0).tolist(),
        }
        state = {
            "kline_data": kline_data,
            "analysis_results": None,
            "messages": [],
            "time_frame": self.config.timeframe,
            "stock_name": self.config.asset,
            "pattern_image": "",
            "trend_image": "",
        }
        started = time.perf_counter()
        final_state = self._graph.graph.invoke(state)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        parsed = parse_decision_text(final_state.get("final_trade_decision"))
        direction = _normalize_direction(parsed.get("decision"))
        if direction is None:
            direction = _normalize_direction(parsed.get("raw"))
        if direction is None:
            raise ValueError(f"无法从决策输出解析方向：{str(parsed)[:200]}")

        try:
            rr = float(parsed.get("risk_reward_ratio", 1.5))
        except (TypeError, ValueError):
            rr = 1.5
        # LLM 不直接给置信度：按风险收益比在 [1.2, 1.8] 的位置做代理映射
        confidence = min(1.0, max(0.05, (rr - 1.2) / 0.6))

        return StrategyOutput(
            direction=direction,
            confidence=round(confidence, 6),
            risk_reward_ratio=rr,
            sub_signals=None,
            signals={
                "forecast_horizon": parsed.get("forecast_horizon"),
                "justification": str(parsed.get("justification") or "")[:400],
                "provenance": final_state.get("analysis_provenance"),
            },
            extra={"latency_ms": round(elapsed_ms, 3), "source": "llm"},
        )


def build_strategy(config: BacktestConfig) -> DecisionStrategy:
    if config.strategy == "rule":
        return RuleBasedDecisionStrategy(config)
    if config.strategy == "recorded":
        return RecordedDecisionStrategy(config)
    if config.strategy == "llm":
        return LLMDecisionStrategy(config)
    raise ConfigError(f"未知策略：{config.strategy}")


__all__ = [
    "DecisionStrategy",
    "StrategyOutput",
    "RuleBasedDecisionStrategy",
    "RecordedDecisionStrategy",
    "LLMDecisionStrategy",
    "build_strategy",
    "parse_decision_text",
]
