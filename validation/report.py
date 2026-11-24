"""验证结果可视化：生成完全自包含的 HTML 报表（内联 CSS + 内联 SVG 图表）。

不依赖 CDN、不依赖 JS，可离线打开、可邮件分发、可归档复现。
同时提供 Markdown 版本，便于在终端/仓库中快速查看。
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 基础格式化
# ---------------------------------------------------------------------------
def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _pct(value: Any, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if signed else ""
    return f"{f:{sign}.{digits}f}%"


def _pct_ratio(value: Any, digits: int = 2) -> str:
    """比率（0-1）转百分比显示；None 显示为 —。"""
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _cls(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "muted"
    if f > 0:
        return "pos"
    if f < 0:
        return "neg"
    return "muted"


# ---------------------------------------------------------------------------
# SVG 图表
# ---------------------------------------------------------------------------
def _svg_line_chart(
    series: Sequence[Dict[str, Any]],
    x_labels: Sequence[str],
    width: int = 960,
    height: int = 320,
    y_suffix: str = "%",
) -> str:
    if not series or not x_labels:
        return '<div class="empty">暂无可绘制的数据</div>'

    values = [v for s in series for v in s.get("values", []) if v is not None]
    if not values:
        return '<div class="empty">暂无可绘制的数据</div>'

    left, right, top, bottom = 64, 24, 22, 44
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)
    y_min, y_max = min(values), max(values)
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def x_at(i: int) -> float:
        n = max(1, len(x_labels) - 1)
        return left + plot_w * i / n

    def y_at(v: float) -> float:
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="趋势曲线">'
    ]
    # 网格 + y 轴刻度
    for k in range(5):
        value = y_min + (y_max - y_min) * k / 4
        y = y_at(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" class="axis" text-anchor="end">'
            f"{value:.2f}{y_suffix}</text>"
        )
    # x 轴刻度（均匀抽取 6 个）
    ticks = sorted(set([round(i * (len(x_labels) - 1) / 5) for i in range(6)]))
    for idx in ticks:
        if idx >= len(x_labels):
            continue
        label = str(x_labels[idx])[:16]
        parts.append(
            f'<text x="{x_at(idx):.1f}" y="{height - 16}" class="axis" text-anchor="middle">'
            f"{_esc(label)}</text>"
        )

    # 折线
    for s in series:
        pts = []
        for i, v in enumerate(s.get("values", [])):
            if v is None:
                continue
            pts.append(f"{x_at(i):.1f},{y_at(float(v)):.1f}")
        if not pts:
            continue
        parts.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{_esc(s.get("color", "#c8a24a"))}" '
            f'stroke-width="{_esc(s.get("width", 2))}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # 图例
    lx = left
    for s in series:
        parts.append(
            f'<rect x="{lx}" y="{top - 12}" width="12" height="4" fill="{_esc(s.get("color", "#c8a24a"))}"/>'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{top - 6}" class="legend">{_esc(s.get("label", ""))}</text>'
        )
        lx += 24 + 12 * len(str(s.get("label", "")))

    parts.append("</svg>")
    return "".join(parts)


def _svg_area_chart(
    values: Sequence[Optional[float]], width: int = 960, height: int = 200, label: str = ""
) -> str:
    cleaned = [float(v) if v is not None else 0.0 for v in values]
    if not cleaned:
        return '<div class="empty">暂无可绘制的数据</div>'

    left, right, top, bottom = 64, 24, 18, 30
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)
    y_min = min(min(cleaned), 0.0)
    y_max = 0.0
    if abs(y_max - y_min) < 1e-9:
        y_min = -1.0

    def x_at(i: int) -> float:
        return left + plot_w * i / max(1, len(cleaned) - 1)

    def y_at(v: float) -> float:
        return top + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="回撤曲线">']
    for k in range(4):
        value = y_min + (y_max - y_min) * k / 3
        y = y_at(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" class="axis" text-anchor="end">'
            f"{value:.2f}%</text>"
        )
    points = [f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(cleaned)]
    area = f"{left},{y_at(0):.1f} " + " ".join(points) + f" {x_at(len(cleaned) - 1):.1f},{y_at(0):.1f}"
    parts.append(f'<polygon points="{area}" fill="rgba(214,74,74,0.18)" stroke="none"/>')
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#d64a4a" stroke-width="2"/>'
    )
    if label:
        parts.append(f'<text x="{left}" y="{top - 4}" class="legend">{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_bar_chart(items: Sequence[Dict[str, Any]], width: int = 620) -> str:
    """水平条形图：items=[{label, count, color}]。"""
    if not items:
        return '<div class="empty">未发现异常</div>'
    row_h = 30
    height = row_h * len(items) + 16
    max_count = max(1, max(int(i.get("count", 0)) for i in items))
    label_w = 150
    bar_w = width - label_w - 70

    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="异常分类">']
    for idx, item in enumerate(items):
        y = 10 + idx * row_h
        count = int(item.get("count", 0))
        w = max(2, bar_w * count / max_count)
        color = item.get("color", "#c8a24a")
        parts.append(
            f'<text x="0" y="{y + 15}" class="axis">{_esc(item.get("label", ""))}</text>'
        )
        parts.append(
            f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="16" rx="3" fill="{_esc(color)}"/>'
        )
        parts.append(
            f'<text x="{label_w + w + 8:.1f}" y="{y + 16}" class="axis">{count}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 通用组件
# ---------------------------------------------------------------------------
def _kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    tone_cls = f" {tone}" if tone else ""
    return (
        f'<div class="kpi{tone_cls}"><div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value">{_esc(value)}</div>'
        f'<div class="kpi-sub">{_esc(sub)}</div></div>'
    )


def _kv_table(rows: Sequence[Sequence[Any]]) -> str:
    body = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows
    )
    return f'<table class="kv">{body}</table>'


def _section(title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="subtitle">{_esc(subtitle)}</p>' if subtitle else ""
    return f'<section><h2>{_esc(title)}</h2>{sub}{body}</section>'


def _tone_for_score(score: Any) -> str:
    try:
        f = float(score)
    except (TypeError, ValueError):
        return ""
    if f >= 70:
        return "good"
    if f >= 45:
        return "warn"
    return "bad"


# ---------------------------------------------------------------------------
# HTML 报表
# ---------------------------------------------------------------------------
_CSS = """
:root{--gold:#b8862f;--gold-soft:#f5ecd8;--ink:#24211c;--muted:#7a746a;--line:#e6e0d5;
--good:#2f7d4f;--warn:#b07a14;--bad:#b23b3b;--up:#b23b3b;--down:#2f7d4f;--card:#fffdf8;--bg:#f6f3ec;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;font-size:14px;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 64px}
header.hero{background:linear-gradient(135deg,#241f17 0%,#3b3122 55%,#5a4626 100%);color:#f8f3e6;
border-radius:18px;padding:30px 34px;box-shadow:0 12px 30px rgba(60,45,20,.18)}
header.hero h1{margin:0 0 6px;font-size:26px;letter-spacing:.5px}
header.hero .meta{color:#d8c9a7;font-size:13px}
header.hero .badge{display:inline-block;margin-top:12px;padding:4px 12px;border-radius:999px;
background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);font-size:12px;color:#f1e6cc}
section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px 24px;margin-top:22px;
box-shadow:0 3px 12px rgba(80,65,35,.05)}
h2{margin:0 0 14px;font-size:17px;color:#3a2f1d;letter-spacing:.4px}
h2:before{content:"";display:inline-block;width:4px;height:15px;background:var(--gold);border-radius:2px;
margin-right:9px;vertical-align:-2px}
h3{font-size:14px;margin:18px 0 8px;color:#5a4a2a}
p.subtitle{margin:-8px 0 14px;color:var(--muted);font-size:12.5px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi-label{font-size:12px;color:var(--muted)}
.kpi-value{font-size:23px;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
.kpi-sub{font-size:11.5px;color:var(--muted);margin-top:2px}
.kpi.good .kpi-value{color:var(--good)} .kpi.warn .kpi-value{color:var(--warn)} .kpi.bad .kpi-value{color:var(--bad)}
table{width:100%;border-collapse:collapse;font-size:13px}
table th,table td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}
table thead th{background:var(--gold-soft);color:#5c4a24;font-weight:600;white-space:nowrap;
position:sticky;top:0}
table.kv th{width:190px;background:transparent;color:var(--muted);font-weight:500}
table tbody tr:hover{background:#faf6ec}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.pos{color:var(--up)} .neg{color:var(--down)} .muted{color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.chart{width:100%;height:auto;display:block}
.grid{stroke:#eee7d9;stroke-width:1}
.axis{fill:#8b8375;font-size:11px}
.legend{fill:#6b5c3c;font-size:11.5px}
.empty{color:var(--muted);padding:18px;text-align:center;background:#faf7f0;border-radius:8px}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--line);border-radius:10px}
.note{color:var(--muted);font-size:12px;margin-top:10px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11.5px;border:1px solid var(--line);
background:#fbf7ee;color:#6b5c3c;white-space:nowrap}
.pill.high{border-color:#e3b7b7;background:#fbeeee;color:#a13232}
.pill.medium{border-color:#e6d3a6;background:#fdf6e3;color:#8a6414}
.pill.low{border-color:#cfe0cf;background:#f0f7f0;color:#3f6b48}
.matrix td{text-align:center;font-variant-numeric:tabular-nums;font-weight:600}
.mat-legend{font-size:12px;color:var(--muted);margin-top:8px}
footer{margin-top:26px;color:var(--muted);font-size:12px;text-align:center}
code{background:#f2ecdd;padding:1px 6px;border-radius:4px;font-size:12px}
"""


def _accuracy_section(metrics: Dict[str, Any]) -> str:
    acc = metrics.get("accuracy", {})
    conf = acc.get("confusion", {})
    total = sum(int(conf.get(k, 0)) for k in ("tp", "fp", "tn", "fn")) or 1

    def shade(v: int) -> str:
        alpha = 0.08 + 0.32 * (int(v) / total)
        return f"background:rgba(184,134,47,{alpha:.3f})"

    matrix = (
        '<table class="matrix"><thead><tr><th></th><th>实际上涨</th><th>实际下跌</th></tr></thead><tbody>'
        f'<tr><th>预测 LONG</th><td style="{shade(int(conf.get("tp", 0)))}">{int(conf.get("tp", 0))}</td>'
        f'<td style="{shade(int(conf.get("fp", 0)))}">{int(conf.get("fp", 0))}</td></tr>'
        f'<tr><th>预测 SHORT</th><td style="{shade(int(conf.get("fn", 0)))}">{int(conf.get("fn", 0))}</td>'
        f'<td style="{shade(int(conf.get("tn", 0)))}">{int(conf.get("tn", 0))}</td></tr>'
        "</tbody></table>"
        '<div class="mat-legend">TP=做多且上涨，FP=做多但下跌，FN=做空但上涨，TN=做空且下跌（不计成本前方向）</div>'
    )

    rows = [
        ("方向准确率", _pct_ratio(acc.get("accuracy"), 2), "方向正确的交易占比"),
        ("平衡准确率", _pct_ratio(acc.get("balanced_accuracy"), 2), "多空两侧召回率的平均"),
        ("做多准确率", _pct_ratio(acc.get("long_accuracy"), 2), f"样本 {acc.get('long_trades', 0)} 笔"),
        ("做空准确率", _pct_ratio(acc.get("short_accuracy"), 2), f"样本 {acc.get('short_trades', 0)} 笔"),
        ("做多精确率 / 召回率", f"{_pct_ratio(acc.get('precision_long'))} / {_pct_ratio(acc.get('recall_long'))}", "LONG 类"),
        ("做空精确率 / 召回率", f"{_pct_ratio(acc.get('precision_short'))} / {_pct_ratio(acc.get('recall_short'))}", "SHORT 类"),
        ("平均置信度（正确 / 错误）", f"{_num(acc.get('avg_confidence_correct'), 3)} / {_num(acc.get('avg_confidence_wrong'), 3)}",
         f"置信度落差 {_num(acc.get('confidence_gap'), 3)}"),
        ("置信度校准误差 ECE", _num(acc.get("ece"), 4), "越小说明置信度越可信"),
        ("评估样本", f"{acc.get('evaluated_trades', 0)} 笔", f"其中方向错误 {acc.get('wrong', 0)} 笔，平局 {acc.get('ties', 0)} 笔"),
    ]
    table = "".join(
        f"<tr><th>{_esc(a)}</th><td>{_esc(b)}</td><td class='muted'>{_esc(c)}</td></tr>" for a, b, c in rows
    )
    return (
        '<div class="grid2"><div><table class="kv">'
        + table
        + "</table></div><div><h3>混淆矩阵</h3>"
        + matrix
        + "</div></div>"
    )


def _performance_section(metrics: Dict[str, Any]) -> str:
    p = metrics.get("performance", {})
    rows = [
        ("累计净收益", _pct(p.get("total_return_pct"), 2, True), _cls(p.get("total_return_pct"))),
        ("年化收益（近似）", _pct(p.get("annualized_return_pct"), 2, True), _cls(p.get("annualized_return_pct"))),
        ("平均每笔收益", _pct(p.get("avg_trade_return_pct"), 3, True), _cls(p.get("avg_trade_return_pct"))),
        ("胜率（净收益>0）", _pct_ratio(p.get("win_rate"), 2), ""),
        ("盈亏比（平均盈利/平均亏损）", _num(p.get("payoff_ratio"), 3), ""),
        ("盈利因子", _num(p.get("profit_factor"), 3), ""),
        ("期望收益 / 笔", _pct(p.get("expectancy_pct"), 3, True), _cls(p.get("expectancy_pct"))),
        ("最佳 / 最差单笔", f"{_pct(p.get('best_trade_pct'), 2, True)} / {_pct(p.get('worst_trade_pct'), 2, True)}", ""),
        ("Sharpe", _num(p.get("sharpe"), 3), ""),
        ("Sortino", _num(p.get("sortino"), 3), ""),
        ("Calmar", _num(p.get("calmar"), 3), ""),
    ]
    body = "".join(
        f"<tr><th>{_esc(a)}</th><td class='num {c}'>{_esc(b)}</td></tr>" for a, b, c in rows
    )
    return f'<table class="kv">{body}</table>'


def _risk_section(metrics: Dict[str, Any]) -> str:
    r = metrics.get("risk", {})
    rows = [
        ("最大回撤", _pct(r.get("max_drawdown_pct"), 2), _cls(-abs(float(r.get("max_drawdown_pct") or 0)))),
        ("最大回撤持续", f"{r.get('max_drawdown_trades', 0)} 笔", ""),
        ("单笔波动率", _pct(r.get("volatility_pct"), 3), ""),
        ("年化波动率", _pct(r.get("annualized_volatility_pct"), 2), ""),
        ("下行标准差", _pct(r.get("downside_deviation_pct"), 3), ""),
        ("VaR(95%)", _pct(r.get("var_95_pct"), 3), _cls(r.get("var_95_pct"))),
        ("CVaR(95%)", _pct(r.get("cvar_95_pct"), 3), _cls(r.get("cvar_95_pct"))),
        ("单笔最大亏损", _pct(r.get("max_single_trade_loss_pct"), 3), "neg"),
        ("最大连续亏损", f"{r.get('max_consecutive_losses', 0)} 笔", ""),
        ("持仓暴露比例", _pct((r.get("exposure_ratio") or 0) * 100, 1), ""),
        ("已实现盈亏比", _num(r.get("realized_payoff_ratio"), 3), ""),
    ]
    body = "".join(
        f"<tr><th>{_esc(a)}</th><td class='num {c}'>{_esc(b)}</td></tr>" for a, b, c in rows
    )
    return f'<table class="kv">{body}</table>'


def _timeliness_section(metrics: Dict[str, Any]) -> str:
    t = metrics.get("timeliness", {})
    rows = [
        ("平均决策耗时", f"{_num(t.get('avg_decision_latency_ms'), 2)} ms", ""),
        ("P50 / P95 决策耗时", f"{_num(t.get('p50_decision_latency_ms'), 1)} / {_num(t.get('p95_decision_latency_ms'), 1)} ms", ""),
        ("最大决策耗时", f"{_num(t.get('max_decision_latency_ms'), 1)} ms", ""),
        ("执行滞后", f"{t.get('execution_lag_bars', 1)} 根 K 线", "收盘决策 → 下一根开盘成交"),
        ("平均持仓", f"{_num(t.get('avg_holding_bars'), 1)} 根", ""),
        ("平均/中位到达最大浮盈", f"{_num(t.get('avg_time_to_mfe_bars'), 1)} / {_num(t.get('median_time_to_mfe_bars'), 1)} 根", ""),
        ("最大浮盈出现位置", _pct((t.get("avg_time_to_mfe_ratio") or 0) * 100, 1), "占持仓周期比例，越小越及时"),
        ("迟滞入场笔数", f"{t.get('late_entry_trades', 0)} 笔（{_pct((t.get('late_entry_ratio') or 0) * 100, 1)}）", ""),
        ("决策覆盖率", _pct((t.get("decision_coverage") or 0) * 100, 1), "有效决策 / 可回放 bar"),
        ("增量缓存复用率", _pct((t.get("cache_reuse_ratio") or 0) * 100, 1), ""),
    ]
    body = "".join(
        f"<tr><th>{_esc(a)}</th><td class='num'>{_esc(b)}</td><td class='muted'>{_esc(c)}</td></tr>"
        for a, b, c in rows
    )
    return f'<table class="kv">{body}</table>'


def _benchmark_section(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    p = metrics.get("performance", {})
    bench_label = "同区间满仓做多（买入持有）" if config.get("benchmark") == "buy_hold" else "空仓（0 收益）"
    rows = [
        ("基准定义", bench_label, ""),
        ("基准累计收益", _pct(p.get("benchmark_total_return_pct"), 2, True), _cls(p.get("benchmark_total_return_pct"))),
        ("策略累计收益", _pct(p.get("total_return_pct"), 2, True), _cls(p.get("total_return_pct"))),
        ("超额收益", _pct(p.get("excess_return_pct"), 2, True), _cls(p.get("excess_return_pct"))),
        ("Beta", _num(p.get("beta"), 3), "逐笔区间收益相对基准的敏感度"),
        ("Alpha", _pct(p.get("alpha_pct"), 3, True), _cls(p.get("alpha_pct"))),
        ("跟踪误差", _pct(p.get("tracking_error_pct"), 3), ""),
        ("信息比率", _num(p.get("information_ratio"), 3), ""),
    ]
    body = "".join(
        f"<tr><th>{_esc(a)}</th><td class='num {c}'>{_esc(b)}</td></tr>"
        for a, b, c in rows
    )
    return f'<table class="kv">{body}</table>'


def _anomaly_section(anomalies: Dict[str, Any]) -> str:
    by_cat = anomalies.get("by_category", {})
    palette = {"high": "#b23b3b", "medium": "#c08a25", "low": "#7d9b6a"}
    items = [
        {
            "label": f"{meta.get('label', code)}",
            "count": meta.get("count", 0),
            "color": palette.get(meta.get("severity", "low"), "#c8a24a"),
        }
        for code, meta in by_cat.items()
    ]
    chart = _svg_bar_chart(items)

    rows = "".join(
        f"<tr><td><span class='pill {_esc(meta.get('severity'))}'>{_esc(meta.get('kind'))}</span></td>"
        f"<td>{_esc(meta.get('label', code))}</td><td class='num'>{int(meta.get('count', 0))}</td>"
        f"<td class='muted'><code>{_esc(code)}</code></td></tr>"
        for code, meta in by_cat.items()
    )
    cat_table = (
        '<table class="scroll"><thead><tr><th>类型</th><th>异常</th><th class="num">数量</th><th>代码</th>'
        "</tr></thead><tbody>" + (rows or '<tr><td colspan="4" class="muted">未发现异常</td></tr>')
        + "</tbody></table>"
    )

    samples = anomalies.get("samples", [])
    sample_rows = "".join(
        f"<tr><td class='num'>{_esc(s.get('decision_index'))}</td><td>{_esc(s.get('datetime'))}</td>"
        f"<td><span class='pill {_esc(s.get('severity'))}'>{_esc(s.get('label'))}</span></td>"
        f"<td>{_esc(s.get('direction'))}</td>"
        f"<td class='num {_cls(s.get('net_return_pct'))}'>{_pct(s.get('net_return_pct'), 3, True)}</td>"
        f"<td>{_esc(s.get('reason'))}</td></tr>"
        for s in samples
    )
    sample_table = (
        '<div class="scroll" style="max-height:300px"><table><thead><tr><th class="num">#</th><th>时间</th>'
        '<th>异常</th><th>方向</th><th class="num">净收益</th><th>说明</th></tr></thead><tbody>'
        + (sample_rows or '<tr><td colspan="6" class="muted">未发现异常</td></tr>')
        + "</tbody></table></div>"
    )

    summary = (
        f"<p class='subtitle'>共标记 <b>{anomalies.get('total_flags', 0)}</b> 项异常，"
        f"涉及 <b>{anomalies.get('flagged_decisions', 0)}</b> 个决策"
        f"（占比 {_pct((anomalies.get('flagged_ratio') or 0) * 100, 1)}）；"
        f"错误类 {anomalies.get('error_count', 0)}、偏差类 {anomalies.get('deviation_count', 0)}、"
        f"数据类 {anomalies.get('data_count', 0)}。</p>"
    )
    return summary + '<div class="grid2"><div>' + chart + "</div><div>" + cat_table + "</div></div>" + (
        "<h3>异常样本（按严重度排序）</h3>" + sample_table
    )


def _decision_table(decisions: List[Dict[str, Any]], trades: List[Dict[str, Any]], limit: int = 500) -> str:
    trade_by_index = {t["decision_index"]: t for t in trades}
    rows: List[str] = []
    for d in decisions[:limit]:
        t = trade_by_index.get(d.get("index"))
        direction = d.get("direction") or "—"
        correct = t.get("correct") if t else None
        correct_txt = "✔" if correct is True else ("✘" if correct is False else "—")
        rows.append(
            "<tr>"
            f"<td class='num'>{_esc(d.get('index'))}</td>"
            f"<td>{_esc(d.get('datetime'))}</td>"
            f"<td>{_esc(direction)}</td>"
            f"<td class='num'>{_num(d.get('confidence'), 3)}</td>"
            f"<td class='num'>{_num(d.get('score'), 3)}</td>"
            f"<td class='num'>{_num(t.get('entry_price') if t else None, 4)}</td>"
            f"<td class='num'>{_num(t.get('exit_price') if t else None, 4)}</td>"
            f"<td>{_esc(t.get('exit_reason') if t else '—')}</td>"
            f"<td class='num'>{_esc(t.get('holding_bars') if t else '—')}</td>"
            f"<td class='num {_cls(t.get('gross_return_pct') if t else None)}'>"
            f"{_pct(t.get('gross_return_pct') if t else None, 3, True)}</td>"
            f"<td class='num {_cls(t.get('net_return_pct') if t else None)}'>"
            f"{_pct(t.get('net_return_pct') if t else None, 3, True)}</td>"
            f"<td class='num {_cls(t.get('benchmark_return_pct') if t else None)}'>"
            f"{_pct(t.get('benchmark_return_pct') if t else None, 3, True)}</td>"
            f"<td class='num'>{correct_txt}</td>"
            "</tr>"
        )
    if not rows:
        rows = ['<tr><td colspan="13" class="muted">无决策记录</td></tr>']
    note = (
        f'<p class="note">共 {len(decisions)} 条决策，表格最多展示前 {limit} 条；完整明细见 '
        "<code>decisions.jsonl</code>。</p>"
    )
    return (
        '<div class="scroll"><table><thead><tr><th class="num">#</th><th>决策时间</th><th>方向</th>'
        '<th class="num">置信度</th><th class="num">评分</th><th class="num">入场价</th>'
        '<th class="num">出场价</th><th>出场原因</th><th class="num">持有</th>'
        '<th class="num">毛收益</th><th class="num">净收益</th><th class="num">基准</th><th class="num">正确</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>" + note
    )


def render_html(payload: Dict[str, Any]) -> str:
    """把回测结果 payload 渲染为自包含 HTML。"""
    cfg = payload.get("config", {})
    data = payload.get("data", {})
    stats = payload.get("stats", {})
    metrics = payload.get("metrics", {})
    anomalies = payload.get("anomalies", {})
    series = payload.get("series", {})
    decisions = payload.get("decisions", [])
    trades = payload.get("trades", [])
    composite = metrics.get("composite", {})
    acc = metrics.get("accuracy", {})
    perf = metrics.get("performance", {})
    risk = metrics.get("risk", {})
    timeliness = metrics.get("timeliness", {})

    equity = [((v - 1.0) * 100.0) for v in series.get("equity", [])]
    bench = [((v - 1.0) * 100.0) for v in series.get("benchmark_equity", [])]
    x_labels = [""] + list(series.get("exit_datetime", []))
    n_points = max(len(equity), len(bench))
    if len(equity) < len(x_labels):
        equity = equity + [equity[-1] if equity else 0.0] * (len(x_labels) - len(equity))
    if len(bench) < len(x_labels):
        bench = bench + [bench[-1] if bench else 0.0] * (len(x_labels) - len(bench))

    equity_chart = _svg_line_chart(
        [
            {"label": "策略净值（%）", "values": equity[:n_points], "color": "#b8862f"},
            {"label": "基准净值（%）", "values": bench[:n_points], "color": "#8aa0b8"},
        ],
        x_labels[:n_points],
    )
    drawdown_chart = _svg_area_chart(
        series.get("drawdown_pct", []), label="策略回撤（%）"
    )

    score = composite.get("score")
    hero = (
        '<header class="hero">'
        "<h1>决策正确性验证报告</h1>"
        f'<div class="meta">{_esc(data.get("asset_name") or cfg.get("asset"))}'
        f'（{_esc(cfg.get("asset"))}） · {_esc(cfg.get("timeframe"))} · 策略 {_esc(payload.get("strategy"))}'
        f' · {_esc(data.get("source_label") or data.get("source"))}</div>'
        f'<div class="meta">数据区间 {_esc(data.get("first"))} → {_esc(data.get("last"))}'
        f' · {_esc(data.get("rows"))} 根 K 线 · 生成于 {_esc(payload.get("created_at"))}</div>'
        f'<div class="badge">综合评分 {_num(score, 1)} / 100 · run_id={_esc(payload.get("run_id"))}'
        f' · 决策哈希 {_esc(str(payload.get("decisions_hash"))[:12])}…</div>'
        "</header>"
    )

    kpis = (
        '<div class="kpis">'
        + _kpi("综合评分", _num(score, 1), "0-100", _tone_for_score(score))
        + _kpi("方向准确率", _pct_ratio(acc.get("accuracy"), 1), f"样本 {acc.get('evaluated_trades', 0)} 笔",
               _tone_for_score((acc.get("accuracy") or 0) * 100))
        + _kpi("累计净收益", _pct(perf.get("total_return_pct"), 2, True),
               f"超额 {_pct(perf.get('excess_return_pct'), 2, True)}", _cls(perf.get("total_return_pct")))
        + _kpi("最大回撤", _pct(risk.get("max_drawdown_pct"), 2), "峰值到谷底", "bad")
        + _kpi("Sharpe", _num(perf.get("sharpe"), 2), "逐笔年化")
        + _kpi("异常决策", _pct((anomalies.get("flagged_ratio") or 0) * 100, 1),
               f"{anomalies.get('flagged_decisions', 0)} 个", "warn")
        + _kpi("P95 决策耗时", f"{_num(timeliness.get('p95_decision_latency_ms'), 1)} ms", "响应时效")
        + "</div>"
    )

    conclusion = (
        f"方向准确率 <b>{_pct_ratio(acc.get('accuracy'), 2)}</b>（{acc.get('evaluated_trades', 0)} 笔有效样本，"
        f"{acc.get('wrong', 0)} 笔方向错误），累计净收益 <b>{_pct(perf.get('total_return_pct'), 2, True)}</b>"
        f"（基准 {_pct(perf.get('benchmark_total_return_pct'), 2, True)}），最大回撤 <b>{_pct(risk.get('max_drawdown_pct'), 2)}</b>，"
        f"共标记 <b>{anomalies.get('total_flags', 0)}</b> 项异常；"
        f"决策耗时 P95 为 <b>{_num(timeliness.get('p95_decision_latency_ms'), 1)} ms</b>。"
    )

    config_rows = [
        ("回测区间", f"{cfg.get('start') or data.get('first')} → {cfg.get('end') or data.get('last')}"),
        ("周期 / 预测跨度", f"{cfg.get('timeframe')} / 持有 {cfg.get('horizon_bars')} 根"),
        ("决策策略", cfg.get("strategy")),
        ("手续费 / 滑点（单边）", f"{_num(cfg.get('fee_pct'), 4)}% / {_num(cfg.get('slippage_pct'), 4)}%"),
        ("止损 / 止盈", f"ATR×{_num(cfg.get('stop_atr_mult'), 2)} / "
                     + (f"ATR×{_num(cfg.get('target_atr_mult'), 2)}" if (cfg.get("target_atr_mult") or 0) > 0
                        else f"风险收益比 {_num(cfg.get('risk_reward'), 2)}")),
        ("权重（指标/趋势/模式）", f"{_num(cfg.get('weight_indicator'), 2)} / "
                                f"{_num(cfg.get('weight_trend'), 2)} / {_num(cfg.get('weight_pattern'), 2)}"),
        ("异常阈值", f"方向错误≥{_pct(cfg.get('error_return_threshold_pct'), 2)}、"
                   f"低置信<{_num(cfg.get('min_confidence'), 2)}、"
                   f"风控>{_pct(cfg.get('max_trade_loss_pct'), 2)}"),
    ]

    reproduce_rows = [
        ("引擎版本", payload.get("engine_version")),
        ("run_id", payload.get("run_id")),
        ("决策序列哈希", payload.get("decisions_hash")),
        ("数据指纹", data.get("fingerprint")),
        ("决策参数指纹", payload.get("decision_fingerprint")),
        ("决策条数", f"{stats.get('decisions', 0)}（新计算 {stats.get('computed_decisions', 0)}，"
                   f"缓存复用 {stats.get('reused_decisions', 0)}）"),
        ("可复现", "是（同数据同参数重跑决策哈希一致）" if payload.get("reproducible", True) else "否（LLM 策略为非确定性）"),
    ]

    body = (
        '<div class="wrap">'
        + hero
        + '<section><h2>结论速览</h2><p>' + conclusion + "</p></section>"
        + kpis
        + _section("一、验证配置与可复现信息",
                   '<div class="grid2"><div><h3>回测配置</h3>' + _kv_table(config_rows) + "</div>"
                   '<div><h3>复现信息</h3>' + _kv_table(reproduce_rows) + "</div></div>",
                   "所有参数与指纹都会写入 manifest.json，用于结果归档与复算")
        + _section("二、决策准确率", _accuracy_section(metrics),
                   "按「毛收益方向」判定正确性，避免手续费/滑点干扰方向评估")
        + _section("三、收益表现", _performance_section(metrics),
                   "按逐笔复利计算；年化指标为近似值，仅供横向比较")
        + _section("四、风险指标", _risk_section(metrics),
                   "VaR/CVaR 为逐笔历史分位数，负数表示损失")
        + _section("五、响应时效性", _timeliness_section(metrics),
                   "决策在 K 线收盘后生成，下一根开盘执行（滞后 1 根）")
        + _section("六、基准对比", _benchmark_section(metrics, cfg),
                   "基准与策略使用同一区间、同一成本口径")
        + _section("七、异常决策自动标记", _anomaly_section(anomalies),
                   "错误类 / 偏差类 / 数据类三类，按严重度 high > medium > low 排序")
        + _section("八、净值与回撤曲线",
                   "<h3>累计收益曲线</h3>" + equity_chart
                   + "<h3>策略回撤</h3>" + drawdown_chart,
                   "横轴为每笔交易的出场时间；策略与基准均以 0% 为起点")
        + _section("九、决策明细", _decision_table(decisions, trades),
                   "方向、置信度、成交价、出场原因与盈亏逐条可追溯")
        + _section("十、方法说明",
                   "<p>1) 回放：第 i 根 K 线收盘后决策，第 i+1 根开盘成交，禁止使用未来数据；<br>"
                   "2) 撮合：同一根 K 线同时触发止损与止盈时按止损（悲观）处理；<br>"
                   "3) 成本：单边手续费 + 单边滑点，出场时同样计提；<br>"
                   "4) 效益：逐笔复利；基准为同区间满仓做多（或空仓），成本口径一致；<br>"
                   "5) 异常：方向错误、信号冲突、低置信、风控超限、迟滞入场、连续同向、数据异常、引擎错误；<br>"
                   "6) 复现：决策按「时间戳 + 历史窗口指纹」缓存，数据或参数变化会自动失效并重算。</p>")
        + f'<footer>QuantHarness-Gold · 决策正确性验证模块 v{_esc(payload.get("engine_version"))} · '
          f'{_esc(payload.get("created_at"))}</footer>'
        + "</div>"
    )

    return (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>决策正确性验证报告 · {_esc(payload.get('run_id'))}</title>"
        f"<style>{_CSS}</style></head><body>{body}</body></html>"
    )


# ---------------------------------------------------------------------------
# Markdown 报表
# ---------------------------------------------------------------------------
def render_markdown(payload: Dict[str, Any]) -> str:
    cfg = payload.get("config", {})
    data = payload.get("data", {})
    metrics = payload.get("metrics", {})
    anomalies = payload.get("anomalies", {})
    acc = metrics.get("accuracy", {})
    perf = metrics.get("performance", {})
    risk = metrics.get("risk", {})
    timeliness = metrics.get("timeliness", {})
    composite = metrics.get("composite", {})

    lines = [
        "# 决策正确性验证报告",
        "",
        f"- run_id：`{payload.get('run_id')}`",
        f"- 资产 / 周期：{data.get('asset_name')}（{cfg.get('asset')}） / {cfg.get('timeframe')}",
        f"- 数据区间：{data.get('first')} → {data.get('last')}（{data.get('rows')} 根）",
        f"- 策略：`{cfg.get('strategy')}` ｜ 持有 {cfg.get('horizon_bars')} 根 ｜ 基准 `{cfg.get('benchmark')}`",
        f"- 生成时间：{payload.get('created_at')}",
        f"- 综合评分：**{_num(composite.get('score'), 1)} / 100**",
        "",
        "## 一、决策准确率",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 方向准确率 | {_pct_ratio(acc.get('accuracy'), 2)} |",
        f"| 平衡准确率 | {_pct_ratio(acc.get('balanced_accuracy'), 2)} |",
        f"| 做多 / 做空准确率 | {_pct_ratio(acc.get('long_accuracy'))} / {_pct_ratio(acc.get('short_accuracy'))} |",
        f"| 有效样本 | {acc.get('evaluated_trades', 0)} |",
        f"| 方向错误 | {acc.get('wrong', 0)} |",
        f"| 置信度 ECE | {_num(acc.get('ece'), 4)} |",
        "",
        "## 二、收益表现",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 累计净收益 | {_pct(perf.get('total_return_pct'), 2, True)} |",
        f"| 年化收益（近似） | {_pct(perf.get('annualized_return_pct'), 2, True)} |",
        f"| 胜率 | {_pct_ratio(perf.get('win_rate'), 2)} |",
        f"| 盈亏比 | {_num(perf.get('payoff_ratio'), 3)} |",
        f"| 盈利因子 | {_num(perf.get('profit_factor'), 3)} |",
        f"| Sharpe / Sortino | {_num(perf.get('sharpe'), 3)} / {_num(perf.get('sortino'), 3)} |",
        f"| 基准收益 / 超额 | {_pct(perf.get('benchmark_total_return_pct'), 2, True)} / {_pct(perf.get('excess_return_pct'), 2, True)} |",
        "",
        "## 三、风险指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 最大回撤 | {_pct(risk.get('max_drawdown_pct'), 2)} |",
        f"| VaR(95%) / CVaR(95%) | {_pct(risk.get('var_95_pct'), 3)} / {_pct(risk.get('cvar_95_pct'), 3)} |",
        f"| 年化波动率 | {_pct(risk.get('annualized_volatility_pct'), 2)} |",
        f"| 最大连续亏损 | {risk.get('max_consecutive_losses', 0)} 笔 |",
        f"| 持仓暴露 | {_pct((risk.get('exposure_ratio') or 0) * 100, 1)} |",
        "",
        "## 四、响应时效性",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 平均 / P95 决策耗时 | {_num(timeliness.get('avg_decision_latency_ms'), 2)} / "
        f"{_num(timeliness.get('p95_decision_latency_ms'), 2)} ms |",
        f"| 平均持仓 | {_num(timeliness.get('avg_holding_bars'), 1)} 根 |",
        f"| 迟滞入场 | {timeliness.get('late_entry_trades', 0)} 笔"
        f"（{_pct((timeliness.get('late_entry_ratio') or 0) * 100, 1)}） |",
        f"| 决策覆盖率 | {_pct((timeliness.get('decision_coverage') or 0) * 100, 1)} |",
        "",
        "## 五、异常决策",
        "",
        f"共标记 **{anomalies.get('total_flags', 0)}** 项，涉及 **{anomalies.get('flagged_decisions', 0)}** "
        f"个决策（{_pct((anomalies.get('flagged_ratio') or 0) * 100, 1)}）。",
        "",
        "| 异常 | 类型 | 数量 |",
        "|---|---|---:|",
    ]
    for code, meta in anomalies.get("by_category", {}).items():
        lines.append(f"| {meta.get('label')}（`{code}`） | {meta.get('kind')} | {meta.get('count')} |")

    lines += [
        "",
        "## 六、复现信息",
        "",
        f"- 引擎版本：`{payload.get('engine_version')}`",
        f"- 决策序列哈希：`{payload.get('decisions_hash')}`",
        f"- 数据指纹：`{data.get('fingerprint')}`",
        f"- 决策参数指纹：`{payload.get('decision_fingerprint')}`",
        f"- 新计算 / 缓存复用：{payload.get('stats', {}).get('computed_decisions', 0)} / "
        f"{payload.get('stats', {}).get('reused_decisions', 0)}",
        "",
        "> 本报告由 QuantHarness-Gold 决策正确性验证模块自动生成，仅供研究参考。",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(payload: Dict[str, Any], out_dir) -> Dict[str, str]:
    """把 HTML / Markdown 报表写入 run 目录，返回相对路径。"""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "report.html"
    md_path = out / "report.md"
    html_path.write_text(render_html(payload), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return {"html": str(html_path), "markdown": str(md_path)}
