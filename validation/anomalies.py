"""异常决策自动标记与归类。

回测过程中自动捕获以下异常，并按「错误类 / 偏差类 / 数据类」归类：

  WRONG_DIRECTION  方向做反且亏损超过阈值（实质性错误）
  MARGINAL_ERROR   方向做反但幅度较小（噪声级错误）
  SIGNAL_CONFLICT  决策方向与指标/趋势/模式中的多数分项信号相反（偏差）
  LOW_CONFIDENCE   置信度低于阈值仍被执行的决策（偏差）
  RISK_BREACH      单笔净亏损超过风控上限（错误）
  LATE_ENTRY       入场后极短时间内即触发止损（迟滞入场）
  DUPLICATE_SIGNAL 连续同向决策超过容忍次数（信号迟滞 / 过度交易）
  DATA_ANOMALY     决策窗口内存在缺口 / 缺失 / 零振幅等数据问题
  ENGINE_ERROR     决策计算本身抛错（引擎已隔离，未中断整轮回测）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .engine import BacktestRun

CATEGORY_META: Dict[str, Dict[str, str]] = {
    "WRONG_DIRECTION": {"label": "方向错误", "severity": "high", "kind": "error"},
    "MARGINAL_ERROR": {"label": "小幅方向错误", "severity": "low", "kind": "error"},
    "SIGNAL_CONFLICT": {"label": "信号冲突", "severity": "medium", "kind": "deviation"},
    "LOW_CONFIDENCE": {"label": "低置信度执行", "severity": "low", "kind": "deviation"},
    "RISK_BREACH": {"label": "风控超限", "severity": "high", "kind": "error"},
    "LATE_ENTRY": {"label": "迟滞入场", "severity": "medium", "kind": "deviation"},
    "DUPLICATE_SIGNAL": {"label": "连续同向信号", "severity": "low", "kind": "deviation"},
    "DATA_ANOMALY": {"label": "数据异常", "severity": "medium", "kind": "data"},
    "ENGINE_ERROR": {"label": "引擎执行错误", "severity": "high", "kind": "error"},
}

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _conflict_count(direction: str, sub_signals: Optional[Dict[str, float]]) -> int:
    if not sub_signals:
        return 0
    if direction == "LONG":
        return sum(1 for v in sub_signals.values() if v is not None and v <= -0.05)
    return sum(1 for v in sub_signals.values() if v is not None and v >= 0.05)


def detect_anomalies(run: BacktestRun, max_samples: int = 20) -> Dict[str, Any]:
    cfg = run.config
    trade_by_decision = {t.decision_index: t for t in run.trades}
    issue_by_decision = {item["index"]: item for item in run.data_issues}

    flags: List[Dict[str, Any]] = []
    prev_direction: Optional[str] = None
    repeat = 0

    def add(code: str, dec, trade, reason: str, evidence: Dict[str, Any]) -> None:
        meta = CATEGORY_META[code]
        flags.append(
            {
                "code": code,
                "label": meta["label"],
                "severity": meta["severity"],
                "kind": meta["kind"],
                "decision_index": dec.index,
                "datetime": dec.datetime,
                "direction": dec.direction,
                "confidence": round(float(dec.confidence), 6),
                "gross_return_pct": round(trade.gross_return_pct, 6) if trade else None,
                "net_return_pct": round(trade.net_return_pct, 6) if trade else None,
                "exit_reason": trade.exit_reason if trade else None,
                "reason": reason,
                "evidence": evidence,
            }
        )

    for dec in run.decisions:
        if dec.direction is None:
            add("ENGINE_ERROR", dec, None, dec.error or "决策计算失败", {})
            prev_direction, repeat = None, 0
            continue

        trade = trade_by_decision.get(dec.index)

        # --- 连续同向信号 ---
        if dec.direction == prev_direction:
            repeat += 1
        else:
            repeat = 1
            prev_direction = dec.direction
        if repeat == int(cfg.max_repeat_signals) + 1:
            # 仅在一段连续同向信号的「首次越界」时标记一次，避免逐 bar 刷屏
            add(
                "DUPLICATE_SIGNAL",
                dec,
                trade,
                f"连续第 {repeat} 次给出 {dec.direction}",
                {"repeat": repeat, "limit": int(cfg.max_repeat_signals)},
            )

        # --- 方向错误 ---
        if trade is not None and trade.correct is False:
            if abs(trade.gross_return_pct) >= float(cfg.error_return_threshold_pct):
                add(
                    "WRONG_DIRECTION",
                    dec,
                    trade,
                    f"方向 {dec.direction} 做反，毛收益 {trade.gross_return_pct:.4f}%",
                    {
                        "gross_return_pct": trade.gross_return_pct,
                        "threshold_pct": cfg.error_return_threshold_pct,
                        "exit_reason": trade.exit_reason,
                    },
                )
            else:
                add(
                    "MARGINAL_ERROR",
                    dec,
                    trade,
                    f"方向做反但幅度仅 {trade.gross_return_pct:.4f}%",
                    {"gross_return_pct": trade.gross_return_pct},
                )

        # --- 风险超限 ---
        if trade is not None and trade.net_return_pct < -abs(float(cfg.max_trade_loss_pct)):
            add(
                "RISK_BREACH",
                dec,
                trade,
                f"单笔净亏损 {trade.net_return_pct:.4f}% 超过上限 {cfg.max_trade_loss_pct}%",
                {"net_return_pct": trade.net_return_pct, "limit_pct": cfg.max_trade_loss_pct},
            )

        # --- 迟滞入场 ---
        if (
            trade is not None
            and trade.exit_reason == "stop"
            and trade.holding_bars <= int(cfg.late_entry_bars)
        ):
            add(
                "LATE_ENTRY",
                dec,
                trade,
                f"入场后 {trade.holding_bars} 根即触发止损",
                {"holding_bars": trade.holding_bars, "limit": int(cfg.late_entry_bars)},
            )

        # --- 低置信度 ---
        if float(dec.confidence) < float(cfg.min_confidence):
            add(
                "LOW_CONFIDENCE",
                dec,
                trade,
                f"置信度 {dec.confidence:.4f} 低于阈值 {cfg.min_confidence}",
                {"confidence": dec.confidence, "threshold": cfg.min_confidence},
            )

        # --- 信号冲突（仅规则策略提供分项信号） ---
        conflicts = _conflict_count(dec.direction, dec.sub_signals)
        if conflicts >= 2:
            add(
                "SIGNAL_CONFLICT",
                dec,
                trade,
                f"{dec.direction} 决策与 {conflicts} 个分项信号方向相反",
                {"conflicts": conflicts, "sub_signals": dec.sub_signals or {}},
            )

        # --- 数据异常 ---
        issue = issue_by_decision.get(dec.index)
        if issue:
            add(
                "DATA_ANOMALY",
                dec,
                trade,
                f"决策窗口内存在 {issue['issue_bars']} 根异常 K 线",
                {
                    "issue_bars": issue["issue_bars"],
                    "current_bar_issue": issue["current_bar_issue"],
                },
            )

    # --- 汇总 ---
    by_category: Dict[str, Dict[str, Any]] = {}
    for code, meta in CATEGORY_META.items():
        count = sum(1 for f in flags if f["code"] == code)
        if count:
            by_category[code] = {**meta, "count": count}

    by_severity = {"high": 0, "medium": 0, "low": 0}
    for f in flags:
        by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1

    flagged_decisions = len({f["decision_index"] for f in flags})
    total_decisions = max(1, len(run.decisions))

    ranked = sorted(
        flags,
        key=lambda f: (
            _SEVERITY_ORDER.get(f["severity"], 9),
            -abs(f["net_return_pct"] or 0.0),
        ),
    )
    samples = ranked[: max(1, int(max_samples))]

    return {
        "total_flags": len(flags),
        "flagged_decisions": flagged_decisions,
        "flagged_ratio": round(flagged_decisions / total_decisions, 6),
        "by_category": by_category,
        "by_severity": by_severity,
        "error_count": sum(1 for f in flags if f["kind"] == "error"),
        "deviation_count": sum(1 for f in flags if f["kind"] == "deviation"),
        "data_count": sum(1 for f in flags if f["kind"] == "data"),
        "samples": samples,
        "all": flags,
    }
