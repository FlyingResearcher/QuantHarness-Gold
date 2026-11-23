"""多维度验证指标：决策准确率 / 收益表现 / 风险 / 响应时效 / 基准对比。

所有指标都是「确定性可复算」的纯函数：输入同一份决策与交易明细，输出完全一致。
空集、零波动、单笔交易等边界情况统一返回 0.0 或 None，绝不抛异常、绝不产生 NaN。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import annualization_factor
from .engine import BacktestRun, Trade


def _f(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _round(value: Any, digits: int = 6) -> Optional[float]:
    f = _f(value)
    return None if f is None else round(f, digits)


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def _std(values: Sequence[float], ddof: int = 1) -> Optional[float]:
    if len(values) < 2:
        return None
    f = float(np.std(values, ddof=ddof))
    return f if math.isfinite(f) else None


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.percentile(values, q)) if len(values) else None


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(min(hi, max(lo, value)))


# ---------------------------------------------------------------------------
# 1) 决策准确率
# ---------------------------------------------------------------------------
def accuracy_metrics(trades: List[Trade]) -> Dict[str, Any]:
    evaluated = [t for t in trades if t.correct is not None]
    n_eval = len(evaluated)
    n_trades = len(trades)

    correct_n = sum(1 for t in evaluated if t.correct)
    wrong_n = n_eval - correct_n
    tie_n = n_trades - n_eval

    long_trades = [t for t in evaluated if t.direction == "LONG"]
    short_trades = [t for t in evaluated if t.direction == "SHORT"]
    long_correct = sum(1 for t in long_trades if t.correct)
    short_correct = sum(1 for t in short_trades if t.correct)

    # 混淆矩阵：以「做多」为正类，实际上涨为正样本
    tp = long_correct
    fp = len(long_trades) - long_correct
    tn = short_correct
    fn = len(short_trades) - short_correct

    precision_long = tp / (tp + fp) if (tp + fp) else None
    precision_short = tn / (tn + fn) if (tn + fn) else None
    recall_long = tp / (tp + fn) if (tp + fn) else None
    recall_short = tn / (tn + fp) if (tn + fp) else None
    f1_long = (
        2 * precision_long * recall_long / (precision_long + recall_long)
        if precision_long and recall_long
        else None
    )
    f1_short = (
        2 * precision_short * recall_short / (precision_short + recall_short)
        if precision_short and recall_short
        else None
    )

    correct_confs = [t.confidence for t in evaluated if t.correct]
    wrong_confs = [t.confidence for t in evaluated if not t.correct]

    # 置信度校准误差（ECE，5 个等宽分箱）
    ece = None
    if n_eval:
        bins = np.linspace(0.0, 1.0, 6)
        total_err = 0.0
        for b in range(5):
            lo, hi = bins[b], bins[b + 1]
            members = [
                t for t in evaluated if lo <= t.confidence < hi or (b == 4 and t.confidence == 1.0)
            ]
            if not members:
                continue
            acc = sum(1 for t in members if t.correct) / len(members)
            conf = float(np.mean([t.confidence for t in members]))
            total_err += len(members) / n_eval * abs(acc - conf)
        ece = total_err

    balanced = (
        (recall_long + recall_short) / 2.0
        if recall_long is not None and recall_short is not None
        else None
    )
    return {
        "evaluated_trades": n_eval,
        "correct": correct_n,
        "wrong": wrong_n,
        "ties": tie_n,
        "accuracy": _round(correct_n / n_eval) if n_eval else None,
        "balanced_accuracy": _round(balanced),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_accuracy": _round(long_correct / len(long_trades)) if long_trades else None,
        "short_accuracy": _round(short_correct / len(short_trades)) if short_trades else None,
        "precision_long": _round(precision_long),
        "precision_short": _round(precision_short),
        "recall_long": _round(recall_long),
        "recall_short": _round(recall_short),
        "f1_long": _round(f1_long),
        "f1_short": _round(f1_short),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "avg_confidence_correct": _round(_mean(correct_confs)),
        "avg_confidence_wrong": _round(_mean(wrong_confs)),
        "confidence_gap": _round(
            (_mean(correct_confs) or 0.0) - (_mean(wrong_confs) or 0.0)
        ) if correct_confs or wrong_confs else None,
        "ece": _round(ece),
    }


# ---------------------------------------------------------------------------
# 2) 收益表现 + 基准对比
# ---------------------------------------------------------------------------
def performance_metrics(run: BacktestRun, periods_per_year: Optional[float] = None) -> Dict[str, Any]:
    trades = run.trades
    returns = [t.net_return_pct for t in trades]
    bench = [t.benchmark_return_pct for t in trades]

    if periods_per_year is None:
        avg_hold = float(np.mean([t.holding_bars for t in trades])) if trades else 1.0
        periods_per_year = annualization_factor(run.config.timeframe) / max(1.0, avg_hold)

    equity = 1.0
    bench_equity = 1.0
    for r, b in zip(returns, bench):
        equity *= 1.0 + r / 100.0
        bench_equity *= 1.0 + b / 100.0

    total_return = (equity - 1.0) * 100.0
    bench_return = (bench_equity - 1.0) * 100.0
    n = len(returns)

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)

    avg_ret = _mean(returns)
    std = _std(returns)
    downside = [r for r in returns if r < 0]
    down_std = _std(downside) if len(downside) >= 2 else None

    ann_return = None
    if n > 0 and periods_per_year > 0:
        try:
            ann = (equity ** (periods_per_year / n) - 1.0) * 100.0
            ann_return = ann if math.isfinite(ann) else None
        except (OverflowError, ValueError):
            ann_return = None

    sharpe = None
    if avg_ret is not None and std:
        sharpe = avg_ret / std * math.sqrt(periods_per_year)
    sortino = None
    if avg_ret is not None and down_std:
        sortino = avg_ret / down_std * math.sqrt(periods_per_year)

    # 基准对比（逐笔区间收益）
    beta = alpha = tracking_error = info_ratio = None
    if n >= 2:
        arr_s = np.asarray(returns, dtype=float)
        arr_b = np.asarray(bench, dtype=float)
        var_b = float(np.var(arr_b, ddof=1))
        cov = float(np.cov(arr_s, arr_b, ddof=1)[0, 1]) if n >= 2 else None
        if var_b > 1e-15 and cov is not None:
            beta = cov / var_b
            alpha = float(np.mean(arr_s)) - beta * float(np.mean(arr_b))
        diff = arr_s - arr_b
        diff_std = _std(diff.tolist())
        if diff_std:
            tracking_error = diff_std
            info_ratio = float(np.mean(diff)) / diff_std * math.sqrt(periods_per_year)

    mdd = _f(run.metrics.get("risk", {}).get("max_drawdown_pct")) if run.metrics else None
    if mdd is None:
        mdd = _max_drawdown(returns)

    calmar = None
    if ann_return is not None and mdd and abs(mdd) > 1e-12:
        calmar = ann_return / abs(mdd)

    return {
        "trades": n,
        "total_return_pct": _round(total_return),
        "annualized_return_pct": _round(ann_return),
        "avg_trade_return_pct": _round(avg_ret),
        "median_trade_return_pct": _round(_percentile(returns, 50)),
        "best_trade_pct": _round(max(returns)) if returns else None,
        "worst_trade_pct": _round(min(returns)) if returns else None,
        "win_rate": _round(len(wins) / n) if n else None,
        "profit_factor": _round(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else _round(float("inf"), 6)),
        "expectancy_pct": _round(avg_ret),
        "avg_win_pct": _round(_mean(wins)),
        "avg_loss_pct": _round(_mean(losses)),
        "payoff_ratio": _round((_mean(wins) or 0.0) / abs(_mean(losses))) if losses and (_mean(wins) is not None) else None,
        "sharpe": _round(sharpe),
        "sortino": _round(sortino),
        "calmar": _round(calmar),
        "periods_per_year": _round(periods_per_year, 2),
        "benchmark_total_return_pct": _round(bench_return),
        "excess_return_pct": _round(total_return - bench_return),
        "beta": _round(beta),
        "alpha_pct": _round(alpha),
        "tracking_error_pct": _round(tracking_error),
        "information_ratio": _round(info_ratio),
    }


def _max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in returns:
        equity *= 1.0 + r / 100.0
        peak = max(peak, equity)
        if peak > 0:
            mdd = min(mdd, (equity / peak - 1.0) * 100.0)
    return mdd


# ---------------------------------------------------------------------------
# 3) 风险指标
# ---------------------------------------------------------------------------
def risk_metrics(run: BacktestRun) -> Dict[str, Any]:
    trades = run.trades
    returns = [t.net_return_pct for t in trades]
    n = len(returns)
    annual = annualization_factor(run.config.timeframe)

    drawdown = run.series.get("drawdown_pct") or [0.0]
    mdd = min(drawdown) if drawdown else 0.0
    mdd_trades = 0
    trough_pos = 0
    peak_pos = 0
    if drawdown:
        trough_pos = int(np.argmin(drawdown))
        peak_pos = int(np.argmax(drawdown[: trough_pos + 1])) if trough_pos > 0 else 0
        mdd_trades = trough_pos - peak_pos

    losses = [r for r in returns if r < 0]
    down_std = _std(losses) if len(losses) >= 2 else 0.0
    p5 = _percentile(returns, 5)
    tail = [r for r in returns if p5 is not None and r <= p5]
    cvar = _mean(tail)

    vol = _std(returns)
    avg_hold = float(np.mean([t.holding_bars for t in trades])) if trades else 1.0
    periods_per_year = annual / max(1.0, avg_hold)
    consec = 0
    max_consec = 0
    for r in returns:
        if r < 0:
            consec += 1
            max_consec = max(max_consec, consec)
        else:
            consec = 0

    # 持仓暴露：被交易覆盖的 bar 数 / 可回放 bar 数
    eligible = max(1, int(run.stats.get("eligible_decisions", 1)) + 1)
    covered: set[int] = set()
    for t in trades:
        covered.update(range(t.entry_index, t.exit_index + 1))
    exposure = len(covered) / eligible

    return {
        "max_drawdown_pct": _round(mdd),
        "max_drawdown_trades": int(mdd_trades),
        "drawdown_trough_trade": trough_pos,
        "volatility_pct": _round(vol),
        "annualized_volatility_pct": _round(vol * math.sqrt(periods_per_year)) if vol is not None else None,
        "downside_deviation_pct": _round(down_std) if down_std else None,
        "var_95_pct": _round(p5),
        "cvar_95_pct": _round(cvar),
        "tail_loss_pct": _round(-p5) if p5 is not None else None,
        "max_consecutive_losses": int(max_consec),
        "max_single_trade_loss_pct": _round(min(returns)) if returns else None,
        "exposure_ratio": _round(exposure),
        "realized_payoff_ratio": _round(
            (_mean([r for r in returns if r > 0]) or 0.0) / abs(_mean(losses))
        ) if losses and any(r > 0 for r in returns) else None,
        "trades": n,
    }


# ---------------------------------------------------------------------------
# 4) 响应时效性
# ---------------------------------------------------------------------------
def timeliness_metrics(run: BacktestRun) -> Dict[str, Any]:
    latencies = [d.latency_ms for d in run.decisions]
    holdings = [t.holding_bars for t in run.trades]
    mfe_bars = [t.time_to_mfe_bars for t in run.trades]
    late = [
        t
        for t in run.trades
        if t.exit_reason == "stop" and t.holding_bars <= int(run.config.late_entry_bars)
    ]

    eligible = max(1, int(run.stats.get("eligible_decisions", 1)))
    decisions = int(run.stats.get("decisions", 0))
    reused = int(run.stats.get("reused_decisions", 0))
    ratio_to_hold = [
        (t.time_to_mfe_bars + 1) / max(1, t.holding_bars) for t in run.trades
    ]

    return {
        "avg_decision_latency_ms": _round(_mean(latencies)),
        "p50_decision_latency_ms": _round(_percentile(latencies, 50)),
        "p95_decision_latency_ms": _round(_percentile(latencies, 95)),
        "max_decision_latency_ms": _round(max(latencies)) if latencies else None,
        "execution_lag_bars": 1,
        "avg_holding_bars": _round(_mean(holdings)),
        "median_holding_bars": _round(_percentile(holdings, 50)),
        "avg_time_to_mfe_bars": _round(_mean(mfe_bars)),
        "median_time_to_mfe_bars": _round(_percentile(mfe_bars, 50)),
        "avg_time_to_mfe_ratio": _round(_mean(ratio_to_hold)),
        "late_entry_trades": len(late),
        "late_entry_ratio": _round(len(late) / len(run.trades)) if run.trades else None,
        "decision_coverage": _round(decisions / eligible),
        "cache_reuse_ratio": _round(reused / decisions) if decisions else None,
    }


# ---------------------------------------------------------------------------
# 5) 汇总 + 综合评分
# ---------------------------------------------------------------------------
def compute_metrics(run: BacktestRun) -> Dict[str, Any]:
    accuracy = accuracy_metrics(run.trades)
    performance = performance_metrics(run)
    risk = risk_metrics(run)
    timeliness = timeliness_metrics(run)

    acc = accuracy.get("accuracy")
    accuracy_score = acc * 100.0 if acc is not None else 0.0

    excess = performance.get("excess_return_pct") or 0.0
    perf_score = _clip(50.0 + excess)

    mdd = abs(risk.get("max_drawdown_pct") or 0.0)
    risk_score = _clip(100.0 - mdd * 4.0)

    late_ratio = timeliness.get("late_entry_ratio") or 0.0
    p95 = timeliness.get("p95_decision_latency_ms") or 0.0
    timeliness_score = _clip(100.0 - late_ratio * 60.0 - min(40.0, p95 / 50.0))

    composite = (
        0.40 * accuracy_score + 0.25 * perf_score + 0.20 * risk_score + 0.15 * timeliness_score
    )

    return {
        "accuracy": accuracy,
        "performance": performance,
        "risk": risk,
        "timeliness": timeliness,
        "composite": {
            "score": round(composite, 2),
            "accuracy_score": round(accuracy_score, 2),
            "performance_score": round(perf_score, 2),
            "risk_score": round(risk_score, 2),
            "timeliness_score": round(timeliness_score, 2),
            "formula": "0.40*准确率 + 0.25*收益(50+超额收益) + 0.20*风险(100-4*最大回撤%) + 0.15*时效(100-60*迟滞率-延迟惩罚)",
        },
    }
