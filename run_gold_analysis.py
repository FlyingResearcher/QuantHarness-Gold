"""直接通过 TradingGraph 运行黄金多智能体分析（不经过 web_interface）。

用法：
    uv run python run_gold_analysis.py                 # 默认 COMEX / 1d
    uv run python run_gold_analysis.py SHFE_AU 4h ths  # 沪金 4小时 / 同花顺
    uv run python run_gold_analysis.py COMEX 1d akshare --bars 120
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import gold_data
import static_util
from report_util import (
    PROVENANCE_LABELS,
    PROVENANCE_STAGE_NAMES,
    build_kline_dict,
    generate_markdown_report,
)
from trading_graph import TradingGraph


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run gold analysis via TradingGraph")
    parser.add_argument("asset", nargs="?", default="COMEX")
    parser.add_argument("timeframe", nargs="?", default="1d")
    parser.add_argument("source", nargs="?", default=None,
                        help="eastmoney / ths / akshare（默认取配置，失败自动降级）")
    parser.add_argument("--bars", type=int, default=60, help="送入模型的最大 K 线根数")
    parser.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                        help="是否生成 Markdown 报告（默认开启）")
    parser.add_argument("--report-dir", type=str, default="reports",
                        help="报告输出目录（默认 reports/）")
    args = parser.parse_args()

    t0 = time.time()
    df, used, attempts = gold_data.fetch_gold_kline_auto(
        args.asset, args.timeframe, min_bars=args.bars, preferred=args.source
    )
    if df is None or df.empty:
        raise SystemExit(f"取数失败：{args.asset} {args.timeframe} 尝试过 {attempts}")

    info = gold_data.asset_info(args.asset, source=used)
    print(f"[data] {info['name']} {args.timeframe} 源={gold_data.SOURCE_LABELS.get(used, used)} "
          f"共 {len(df)} 根，{df['Datetime'].iloc[0]} -> {df['Datetime'].iloc[-1]} "
          f"({time.time() - t0:.1f}s)")

    kline_data = build_kline_dict(df)
    pattern_image = static_util.generate_kline_image(kline_data)["pattern_image"]
    trend_image = static_util.generate_trend_image(kline_data)["trend_image"]

    trading_graph = TradingGraph()
    initial_state = {
        "kline_data": kline_data,
        "analysis_results": None,
        "messages": [],
        "time_frame": args.timeframe,
        "stock_name": info["name"],
        "pattern_image": pattern_image,
        "trend_image": trend_image,
    }

    t1 = time.time()
    final_state = trading_graph.graph.invoke(initial_state)
    t2 = time.time()
    analysis_dt = t2 - t1
    pre_dt = t1 - t0
    total_dt = t2 - t0

    print(f"\n[analysis] 耗时 {analysis_dt:.1f}s\n" + "=" * 60)
    print("== 最终交易决策 ==")
    print(final_state.get("final_trade_decision"))
    print("\n== 技术指标报告 ==")
    print(final_state.get("indicator_report"))
    print("\n== 模式识别报告 ==")
    print(final_state.get("pattern_report"))
    print("\n== 趋势分析报告 ==")
    print(final_state.get("trend_report"))

    provenance = final_state.get("analysis_provenance") or {}
    if provenance:
        print("\n== 执行方式（是否使用大模型）==")
        for stage, name in PROVENANCE_STAGE_NAMES.items():
            item = provenance.get(stage) or {}
            label, used_llm = PROVENANCE_LABELS.get(item.get("method"), ("未知", False))
            model = item.get("model")
            model_txt = f" <- {model}" if (used_llm and model) else ""
            print(f"  {name}: {label}{model_txt}  -> {'使用大模型' if used_llm else '未使用大模型'}")

    if args.report:
        out_dir = Path(args.report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{stamp}_{args.asset}_{args.timeframe}.md"
        out_path = out_dir / fname

        models_info = {
            "agent_llm_provider": trading_graph.config.get("agent_llm_provider"),
            "agent_llm_model": trading_graph.config.get("agent_llm_model"),
            "graph_llm_provider": trading_graph.config.get("graph_llm_provider"),
            "graph_llm_model": trading_graph.config.get("graph_llm_model"),
        }

        md = generate_markdown_report(
            args=args, info=info, used_source=used, attempts=attempts,
            kline_data=kline_data, final_state=final_state,
            pre_dt=pre_dt, analysis_dt=analysis_dt, post_dt=total_dt,
            pattern_image_b64=pattern_image, trend_image_b64=trend_image,
            report_dir=out_dir, stamp=stamp, models=models_info,
        )
        out_path.write_text(md, encoding="utf-8")
        print(f"\n[report] 已生成完整 Markdown 报告：{out_path}")
        if (out_dir / "assets").exists():
            pngs = list((out_dir / "assets").glob(f"{stamp}_*.png"))
            for p in pngs:
                print(f"         配图：{p}")


if __name__ == "__main__":
    main()
