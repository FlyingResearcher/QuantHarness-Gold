"""黄金分析报告工具集（从 run_gold_analysis.py 抽离）。

职责边界：
  * 纯计算 / 纯字符串拼接，不发任何网络请求、不读配置。
  * 输入统一是 gold_data 产出的 kline_data（dict of list）或 df。
  * 输出是技术指标 dict、Markdown 文本片段。

放在独立模块而不是 static_util.py，是因为 static_util.py 只负责
matplotlib/mplfinance 出图；把指标计算与报告拼装塞进去会破坏职责单一。

指标数值取自共享的 `indicators` 模块——报告与 Agent 因此读的是同一批数字。
"""
import base64
import json
from datetime import datetime

import pandas as pd

import gold_data
import indicators

KLINE_COLUMNS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# 数据转换
# ---------------------------------------------------------------------------

def build_kline_dict(df):
    """转成 graph 需要的 dict：Datetime 用字符串以支持 JSON 序列化。"""
    out = {}
    for col in KLINE_COLUMNS:
        if col not in df.columns:
            continue
        if col == "Datetime":
            out[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        elif col == "Volume":
            out[col] = df[col].fillna(0).tolist()
        else:
            out[col] = df[col].tolist()
    return out


def _kline_to_df(kline_data):
    """把 kline_data 转成指标模块要的 OHLCV 帧。"""
    df = pd.DataFrame({k: v for k, v in kline_data.items() if k in ["Open", "High", "Low", "Close", "Volume"]})
    return df.astype(float)


# ---------------------------------------------------------------------------
# 技术指标计算（来自共享 indicators 模块，不依赖 LLM）
# ---------------------------------------------------------------------------

def compute_indicators(kline_data):
    """返回 dict：每个指标的最新值。

    这是报告侧的适配器：把报告的输入形状（kline_data 字典）转成共享
    `indicators` 模块要的帧，再取它算出的最新值。指标数值一律来自该模块——
    报告与 Agent 因此读的是同一批数字，%K 口径不可能再分叉
    （见 docs/adr/0001-stochastic-percent-k-convention.md）。
    """
    df = _kline_to_df(kline_data)
    return indicators.latest_values(indicators.compute_series(df))


def _indicator_interpretation(vals):
    """把指标数值翻译成人类可读的解读文本。"""
    lines = []
    close = vals["close"]

    # RSI
    rsi = vals["rsi_14"]
    if rsi is not None:
        if rsi >= 70:
            rsi_text = f"{rsi:.1f} → **超买区**（>70），短期可能过热回落"
        elif rsi <= 30:
            rsi_text = f"{rsi:.1f} → **超卖区**（<30），注意反弹修复风险"
        elif rsi >= 55:
            rsi_text = f"{rsi:.1f} → 偏强（55-70），多头主导"
        elif rsi <= 45:
            rsi_text = f"{rsi:.1f} → 偏弱（30-45），空头主导"
        else:
            rsi_text = f"{rsi:.1f} → 中性（45-55），方向不明"
        lines.append(f"- **RSI(14)**：{rsi_text}")

    # MACD
    macd, sig, hist, hist_prev = vals["macd"], vals["macd_signal"], vals["macd_hist"], vals["macd_hist_prev"]
    if macd is not None and sig is not None:
        cross = ""
        if macd > sig:
            cross = "**金叉**（多头信号）"
        else:
            cross = "**死叉**（空头信号）"
        hist_dir = ""
        if hist is not None and hist_prev is not None:
            if hist > 0 and hist > hist_prev:
                hist_dir = "，红柱放大（上涨加速）"
            elif hist > 0 and hist < hist_prev:
                hist_dir = "，红柱收缩（上涨减速）"
            elif hist < 0 and hist < hist_prev:
                hist_dir = "，绿柱放大（下跌加速）"
            elif hist < 0 and hist > hist_prev:
                hist_dir = "，绿柱收缩（下跌减速）"
        lines.append(
            f"- **MACD(12,26,9)**：DIF={macd:.2f}，DEA={sig:.2f}，MACD柱={hist:.2f}（{cross}{hist_dir}）"
        )

    # Stochastic
    sk, sd = vals["stoch_k"], vals["stoch_d"]
    if sk is not None and sd is not None:
        if sk >= 80:
            st_txt = f"%K={sk:.1f} 进入 **超买区**（>80）"
        elif sk <= 20:
            st_txt = f"%K={sk:.1f} 进入 **超卖区**（<20）"
        else:
            st_txt = f"%K={sk:.1f}，%D={sd:.1f}"
        lines.append(f"- **Stochastic(14)**：{st_txt}")

    # ROC
    roc = vals["roc_10"]
    if roc is not None:
        if roc > 5:
            roc_txt = f"{roc:+.2f}% → **上涨动能强劲**"
        elif roc < -5:
            roc_txt = f"{roc:+.2f}% → **下跌动能强劲**"
        else:
            roc_txt = f"{roc:+.2f}% → 动能一般"
        lines.append(f"- **ROC(10)**：{roc_txt}")

    # Williams %R
    wr = vals["willr_14"]
    if wr is not None:
        if wr <= -80:
            wr_txt = f"{wr:.1f} → **超卖区**（<-80）"
        elif wr >= -20:
            wr_txt = f"{wr:.1f} → **超买区**（>-20）"
        else:
            wr_txt = f"{wr:.1f} → 中性区域"
        lines.append(f"- **Williams %R(14)**：{wr_txt}")

    # 均线
    ma5, ma20, ma60 = vals["ma5"], vals["ma20"], vals["ma60"]
    ma_lines = ["- **均线系统**："]
    if close is not None and ma5 is not None and ma20 is not None:
        if close > ma5 > ma20:
            ma_lines.append("  - 价格 > MA5 > MA20，**多头排列**")
        elif close < ma5 < ma20:
            ma_lines.append("  - 价格 < MA5 < MA20，**空头排列**")
        else:
            ma_lines.append("  - 均线交织，方向不明朗")
    if ma60 is not None and close is not None:
        pos = "站上" if close > ma60 else "跌破"
        ma_lines.append(f"  - 价格 {pos} MA60（{ma60:.2f}），长期趋势 {'偏多' if close > ma60 else '偏空'}")
    lines.extend(ma_lines)

    # Bollinger
    bb_up, bb_mid, bb_lo, bb_pct = vals["bb_up"], vals["bb_mid"], vals["bb_lo"], vals["bb_pct"]
    if bb_up is not None and bb_lo is not None and bb_pct is not None:
        if close >= bb_up:
            bb_txt = f"触及上轨，可能超买回落（%B={bb_pct:.1f}%）"
        elif close <= bb_lo:
            bb_txt = f"触及下轨，可能超卖反弹（%B={bb_pct:.1f}%）"
        elif bb_pct > 80:
            bb_txt = f"接近上轨，偏强（%B={bb_pct:.1f}%）"
        elif bb_pct < 20:
            bb_txt = f"接近下轨，偏弱（%B={bb_pct:.1f}%）"
        else:
            bb_txt = f"带宽运行，中性（%B={bb_pct:.1f}%）"
        lines.append(f"- **Bollinger Bands(20,2)**：上轨={bb_up:.2f}，中轨={bb_mid:.2f}，下轨={bb_lo:.2f}，{bb_txt}")

    # ATR
    atr = vals["atr_14"]
    if atr is not None:
        atr_pct = atr / close * 100 if close else None
        lines.append(
            f"- **ATR(14)**：{atr:.4f}"
            + (f"（日均波动 {atr_pct:.2f}%）" if atr_pct else "")
            + "，用于设置止损 / 仓位规模参考"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _parse_decision(raw):
    """从 final_trade_decision 字符串中尽量提取结构化字段。"""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except ValueError:
            pass
    return {"raw": raw}


def _market_summary(kline_data):
    closes = kline_data.get("Close", [])
    opens = kline_data.get("Open", [])
    highs = kline_data.get("High", [])
    lows = kline_data.get("Low", [])
    dates = kline_data.get("Datetime", [])
    if not closes:
        return {}
    last_close = closes[-1]
    first_open = opens[0] if opens else closes[0]
    period_return = (last_close - first_open) / first_open * 100 if first_open else None
    hi = max(highs) if highs else last_close
    lo = min(lows) if lows else last_close
    amp = (hi - lo) / lo * 100 if lo else None
    return {
        "bars": len(closes),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "first_open": first_open,
        "last_close": last_close,
        "high": hi,
        "low": lo,
        "period_return_pct": period_return,
        "amplitude_pct": amp,
    }


def _save_b64_png(b64_str, out_dir, stamp, label):
    """把 base64 图片存成 PNG，返回相对路径（给 Markdown 嵌入）。"""
    if not b64_str:
        return None
    try:
        data = base64.b64decode(b64_str)
    except Exception:
        return None
    sub = out_dir / "assets"
    sub.mkdir(parents=True, exist_ok=True)
    fname = f"{stamp}_{label}.png"
    path = sub / fname
    path.write_bytes(data)
    # 返回相对于报告 .md 的路径
    return f"assets/{fname}"


PROVENANCE_LABELS = {
    "llm": ("🤖 大模型（LLM）", True),
    "vision_llm": ("🤖 大模型视觉（Vision LLM）", True),
    "llm_text": ("🤖 大模型（纯文本）", True),
    "local_fallback": ("⚙️ 程序化（本地计算）", False),
    "ohlc_fallback": ("⚙️ 程序化（OHLC 规则）", False),
    "unavailable": ("⚠️ 不可用（中性处理）", False),
}

PROVENANCE_STAGE_NAMES = {
    "indicator": "技术指标分析",
    "pattern": "模式识别",
    "trend": "趋势分析",
    "decision": "最终交易决策",
}

PROVENANCE_SHORT = {
    "llm": "🤖 大模型",
    "vision_llm": "🤖 大模型视觉",
    "llm_text": "🤖 大模型文本",
    "local_fallback": "⚙️ 程序化",
    "ohlc_fallback": "⚙️ 程序化 OHLC",
    "unavailable": "⚠️ 不可用",
}


def _provenance_report(prov, models):
    """生成「模型与算法溯源」章节文本。"""
    lines = ["## 七、模型与算法溯源\n"]
    if not prov:
        lines.append("_未采集到执行溯源信息。_\n")
        return "\n".join(lines)

    llm_stages, prog_stages = [], []
    for key in ("indicator", "pattern", "trend", "decision"):
        item = prov.get(key) or {}
        label, used_llm = PROVENANCE_LABELS.get(
            item.get("method"), (item.get("method") or "未知", False)
        )
        stage_name = PROVENANCE_STAGE_NAMES[key]
        if used_llm:
            model = item.get("model") or "—"
            provider = item.get("provider")
            shown = f"{stage_name}（`{model}`" + (f"，{provider}）" if provider else "）")
            llm_stages.append(shown)
        else:
            prog_stages.append(stage_name)

    if llm_stages:
        lines.append(
            f"> **使用大模型的环节**：" + "、".join(llm_stages)
        )
    if prog_stages:
        lines.append(
            f"> **未使用大模型（程序化）的环节**：{'、'.join(prog_stages)}"
        )
    lines.append("")

    lines.append("| 阶段 | 执行方式 | 是否使用大模型 | 具体模型 | 说明 |")
    lines.append("|---|---|---|---|---|")
    for key in ("indicator", "pattern", "trend", "decision"):
        item = prov.get(key) or {}
        label, used_llm = PROVENANCE_LABELS.get(
            item.get("method"), (item.get("method") or "未知", False)
        )
        model = item.get("model")
        provider = item.get("provider")
        if used_llm and model:
            model_cell = f"`{model}`" + (f"（{provider}）" if provider else "")
        elif model:
            model_cell = f"`{model}`（调用失败，未采用）"
        else:
            model_cell = "—（未调用模型）"
        lines.append(
            f"| {PROVENANCE_STAGE_NAMES[key]} | {label} | "
            f"{'是' if used_llm else '否'} | {model_cell} | {item.get('detail') or '—'} |"
        )
    lines.append("")

    if models:
        lines.append("**本次运行配置的模型**：")
        lines.append(
            f"- 分析（工具/文本）模型：`{models.get('agent_llm_model', '—')}`"
            f"（provider: `{models.get('agent_llm_provider', '—')}`）"
        )
        lines.append(
            f"- 视觉/决策模型：`{models.get('graph_llm_model', '—')}`"
            f"（provider: `{models.get('graph_llm_provider', '—')}`）"
        )
        lines.append("")
    return "\n".join(lines)


def generate_markdown_report(
    *, args, info, used_source, attempts,
    kline_data, final_state,
    pre_dt, analysis_dt, post_dt,
    pattern_image_b64=None, trend_image_b64=None,
    report_dir=None, stamp=None, models=None,
) -> str:
    """组装一份 Markdown 格式的完整运行报告。"""
    decision_obj = _parse_decision(final_state.get("final_trade_decision"))
    market = _market_summary(kline_data)
    provenance = final_state.get("analysis_provenance") or {}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 保存图片 ----
    pattern_rel = None
    trend_rel = None
    if report_dir and stamp:
        pattern_rel = _save_b64_png(pattern_image_b64, report_dir, stamp, "pattern")
        trend_rel = _save_b64_png(trend_image_b64, report_dir, stamp, "trend")

    # ---- 本地计算指标 ----
    ind = compute_indicators(kline_data)
    ind_text = _indicator_interpretation(ind)

    # 决策格式化
    decision_line = (decision_obj.get("decision") or "").upper()
    risk_reward = decision_obj.get("risk_reward_ratio", "—")
    horizon = decision_obj.get("forecast_horizon", "—")
    justification = decision_obj.get("justification", "") or decision_obj.get("raw", "")

    def _prov_tag(stage):
        item = provenance.get(stage) or {}
        method = item.get("method")
        model = item.get("model")
        if method in ("llm", "vision_llm", "llm_text") and model:
            return f"🤖 {model} 视觉" if method == "vision_llm" else f"🤖 {model}"
        return PROVENANCE_SHORT.get(method, "未知")

    report = []

    # ==================================================================
    # 标题 + 元信息
    # ==================================================================
    report.append(f"# 黄金多智能体完整分析报告\n")
    report.append(f"> 生成时间 **{ts}**  |  资产 **{info['name']}（{args.asset}）**  |  周期 **{args.timeframe}**  |  "
                  f"数据源 **{gold_data.SOURCE_LABELS.get(used_source, used_source)}**"
                  f"{'（自动降级）' if used_source != (args.source or '') else ''}")
    report.append(f"> 分析耗时：取数 {pre_dt:.1f}s / 分析 {analysis_dt:.1f}s / 总计 {post_dt:.1f}s")
    if attempts:
        tried = " → ".join(
            f"{gold_data.SOURCE_LABELS.get(a['source'], a['source'])}({a['rows']}根)"
            for a in attempts
        )
        report.append(f"> 数据源尝试链路：{tried}")
    if provenance:
        report.append(
            "> 执行方式："
            f"技术指标 **{_prov_tag('indicator')}** ｜ "
            f"模式识别 **{_prov_tag('pattern')}** ｜ "
            f"趋势分析 **{_prov_tag('trend')}** ｜ "
            f"最终决策 **{_prov_tag('decision')}**"
        )
    report.append("")

    # ==================================================================
    # 一、市场快照
    # ==================================================================
    report.append("## 一、市场快照\n")
    if market:
        report.append("| 指标 | 值 |")
        report.append("|---|---:|")
        report.append(f"| K 线区间 | {market['first_date']} → {market['last_date']} |")
        report.append(f"| 根数 | {market['bars']} |")
        report.append(f"| 最新收盘 | {market['last_close']:.4f} |")
        report.append(f"| 首根开盘 | {market['first_open']:.4f} |")
        report.append(f"| 区间最高 | {market['high']:.4f} |")
        report.append(f"| 区间最低 | {market['low']:.4f} |")
        if market["period_return_pct"] is not None:
            chg = f"{market['period_return_pct']:+.2f}%"
            report.append(f"| 区间涨跌幅 | {chg} |")
        if market["amplitude_pct"] is not None:
            report.append(f"| 区间振幅 | {market['amplitude_pct']:.2f}% |")
        if ind["atr_14"] and ind["close"]:
            report.append(f"| ATR(14) 日均波动 | {ind['atr_14']:.2f}（{ind['atr_14']/ind['close']*100:.2f}%）|")
    report.append("")

    # ==================================================================
    # 二、图表
    # ==================================================================
    report.append("## 二、K 线图表\n")
    if pattern_rel:
        report.append("### 模式识别用 K 线\n")
        report.append(f"![K 线模式图]({pattern_rel})\n")
    else:
        report.append("_（模式图未生成）_\n")
    if trend_rel:
        report.append("### 带趋势线与支撑/阻力的 K 线\n")
        report.append(f"![趋势分析图]({trend_rel})\n")
    else:
        report.append("_（趋势图未生成）_\n")

    # ==================================================================
    # 三、技术指标详解
    # ==================================================================
    report.append("## 三、技术指标详解\n")

    # 3.1 数值总览表
    report.append("### 3.1 指标数值一览\n")
    report.append("| 指标 | 最新值 | 解读窗口 |")
    report.append("|---|---:|---|")
    if ind["rsi_14"] is not None:
        report.append(f"| RSI(14) | {ind['rsi_14']:.2f} | 超买 >70 / 超卖 <30 |")
    if ind["macd"] is not None:
        report.append(
            f"| MACD(12,26,9) | DIF={ind['macd']:.4f}<br>DEA={ind['macd_signal']:.4f}<br>柱={ind['macd_hist']:.4f} | 金叉/死叉 信号反转 |"
        )
    if ind["stoch_k"] is not None:
        report.append(
            f"| Stochastic(14) | %K={ind['stoch_k']:.2f}<br>%D={ind['stoch_d']:.2f} | 超买 >80 / 超卖 <20 |"
        )
    if ind["roc_10"] is not None:
        report.append(f"| ROC(10) | {ind['roc_10']:+.2f}% | 0 轴为动能分界 |")
    if ind["willr_14"] is not None:
        report.append(f"| Williams %R(14) | {ind['willr_14']:.2f} | 超买 >-20 / 超卖 <-80 |")
    if ind["ma5"] is not None:
        ma60_str = f"{ind['ma60']:.2f}" if ind["ma60"] is not None else "N/A"
        report.append(
            f"| 均线 | MA5={ind['ma5']:.2f}<br>MA20={ind['ma20']:.2f}<br>MA60={ma60_str} | 多头/空头排列 |"
        )
    if ind["bb_up"] is not None:
        report.append(
            f"| Bollinger(20,2) | 上={ind['bb_up']:.2f}<br>中={ind['bb_mid']:.2f}<br>下={ind['bb_lo']:.2f} | %B 反映相对位置 |"
        )
    if ind["atr_14"] is not None:
        report.append(f"| ATR(14) | {ind['atr_14']:.4f} | 波动性度量，仓位/止损参考 |")
    report.append("")

    # 3.2 逐指标解读
    report.append("### 3.2 逐指标解读\n")
    report.append(ind_text)
    report.append("")

    # 3.3 指标速查表
    report.append("### 3.3 指标速查（关键阈值）\n")
    report.append("| 指标 | 公式要点 | 看涨 | 看跌 | 备注 |")
    report.append("|---|---|---|---|---|")
    report.append("| **RSI** | 14 日涨幅跌幅相对强度，归一化到 0–100 | >50 偏强 | <50 偏弱 | 70/30 是传统超买/超卖阈值 |")
    report.append("| **MACD** | 12 日 EMA − 26 日 EMA；9 日 EMA 作信号线 | DIF 上穿 DEA（金叉） | DIF 下穿 DEA（死叉） | 柱体方向反映动能增减 |")
    report.append("| **Stochastic** | 快速 %K = (C − L14) / (H14 − L14) × 100；%K = SMA3(快速 %K)（慢速 %K）；%D = SMA3(%K) | %K < 20 且上穿 %D | %K > 80 且下穿 %D | 震荡市最有效 |")
    report.append("| **ROC** | (C − C₁₀) / C₁₀ × 100 | >0 动能向上 | <0 动能向下 | 0 轴为关键 |")
    report.append("| **Williams %R** | (H14 − C) / (H14 − L14) × −100 | < −80 超卖反弹 | > −20 超买回落 | 与 Stochastic 互补镜像 |")
    report.append("| **MA 组合** | 简单移动平均 | 价格 > MA5 > MA20 多头排列 | 价格 < MA5 < MA20 空头排列 | MA60 常作牛熊分界 |")
    report.append("| **Bollinger** | MA20 ± 2σ | 触及下轨 +%B < 0 可能反弹 | 触及上轨 +%B > 100 可能回落 | 带宽收窄预示突破 |")
    report.append("| **ATR** | TR 的 14 日 EMA | — | — | 止损位 ≈ 入场价 ± 2·ATR |")
    report.append("")

    # 3.4 Agent 原始技术指标报告
    report.append(f"### 3.4 Agent 原始技术指标报告（{_prov_tag('indicator')}）\n")
    report.append((final_state.get("indicator_report") or "_未生成_").strip())
    report.append("")

    # ==================================================================
    # 四、模式识别报告
    # ==================================================================
    report.append(f"## 四、模式识别报告（{_prov_tag('pattern')}）\n")
    report.append((final_state.get("pattern_report") or "_未生成_").strip())
    report.append("")

    # ==================================================================
    # 五、趋势分析报告
    # ==================================================================
    report.append(f"## 五、趋势分析报告（{_prov_tag('trend')}）\n")
    report.append((final_state.get("trend_report") or "_未生成_").strip())
    report.append("")

    # ==================================================================
    # 六、最终交易决策
    # ==================================================================
    report.append(f"## 六、最终交易决策（{_prov_tag('decision')}）\n")
    # 颜色规范：红涨绿跌 —— 做多/看涨用红，做空/看跌用绿
    badge = {"BUY": "🔴 **买入 / 做多**",
             "SELL": "🟢 **卖出 / 做空**",
             "HOLD": "⚪ **观望 / 持有**"}.get(decision_line, decision_line or "—")
    report.append(f"### {badge}\n")
    report.append(f"| 项 | 值 |")
    report.append(f"|---|---|")
    report.append(f"| 结论 | {badge.replace('**', '')} |")
    report.append(f"| 风险收益比 | 1 : {risk_reward} |")
    report.append(f"| 预期周期 | {horizon} |")
    report.append(f"| 最新收盘价 | {market.get('last_close', '—')} |")

    if market and decision_line:
        last_close = market["last_close"]
        try:
            rr = float(risk_reward)
        except (TypeError, ValueError):
            rr = None
        if rr and rr > 0 and decision_line == "SELL":
            entry = last_close
            stop = entry + ind["atr_14"] * 2 if ind["atr_14"] else entry * 1.02
            target = entry - (stop - entry) * rr
            report.append("")
            report.append("### 参考交易计划（仅研究，非建议）\n")
            report.append(f"- **入场价**：{entry:.2f}（当前价或下一根确认价）")
            if ind["atr_14"]:
                report.append(f"- **止损**：{stop:.2f}（当前价上方 2·ATR ≈ {ind['atr_14']*2:.2f}）")
            report.append(f"- **止盈**：{target:.2f}（风险 × {rr}）")
        elif rr and rr > 0 and decision_line == "BUY":
            entry = last_close
            stop = entry - ind["atr_14"] * 2 if ind["atr_14"] else entry * 0.98
            target = entry + (entry - stop) * rr
            report.append("")
            report.append("### 参考交易计划（仅研究，非建议）\n")
            report.append(f"- **入场价**：{entry:.2f}")
            if ind["atr_14"]:
                report.append(f"- **止损**：{stop:.2f}（当前价下方 2·ATR ≈ {ind['atr_14']*2:.2f}）")
            report.append(f"- **止盈**：{target:.2f}（风险 × {rr}）")

    if justification:
        report.append("")
        report.append("### 决策依据\n")
        report.append(justification.strip())
    report.append("")

    # ==================================================================
    # 七、模型与算法溯源
    # ==================================================================
    report.append(_provenance_report(provenance, models))
    report.append("")

    # ==================================================================
    # 八、方法说明
    # ==================================================================
    report.append("## 八、方法说明\n")
    report.append("本报告由 QuantHarness-Gold 多智能体系统自动生成：")
    report.append("1. **取数**：通过 `gold_data` 从指定数据源（东方财富 / 同花顺 / AKShare）拉取 OHLCV，支持自动降级。")
    report.append("2. **图表**：`static_util` 基于 matplotlib 生成模式识别图与带趋势线/支撑阻力的 K 线图。")
    report.append("3. **技术指标**：本报告 **三、技术指标详解** 中的数值由共享的 `indicators` 模块本地计算"
                  "（RSI / MACD / Stochastic / ROC / Williams %R 用 TA-Lib，布林带 / ATR / 均线用 pandas，确定性结果）；"
                  " Agent 原始报告则由 LLM 调用工具（compute_rsi / compute_macd / …）后归纳生成，二者数值应一致或极接近。")
    report.append("4. **Agent 流程**：indicator → pattern → trend → decision，四个 agent 通过 LangGraph 状态机串联执行。")
    report.append("5. **是否使用大模型**：见 **七、模型与算法溯源**；未使用大模型的环节由本地规则/指标计算产出，报告中已逐段标注。")
    report.append("6. **风险提示**：本报告仅供研究参考，不构成任何投资建议。期货/杠杆交易存在高风险。")
    report.append("")
    report.append("---")
    report.append(f"*Generated at {ts} by QuantHarness-Gold*")
    return "\n".join(report)
