"""命令行入口：`python -m validation.cli <run|list|show|check|serve>`。

示例：
    python -m validation.cli run --asset COMEX --timeframe 1d --run-id comex_1d
    python -m validation.cli run --csv record.csv --timeframe 1d --run-id demo_rule
    python -m validation.cli run --synthetic 400 --run-id demo_synthetic --incremental
    python -m validation.cli check --synthetic 400 --run-id reproducibility
    python -m validation.cli list
    python -m validation.cli show comex_1d
    python -m validation.cli serve --port 5001
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from .config import BacktestConfig
from .runner import check_reproducible, list_runs, load_run, run_backtest


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asset", default="COMEX", help="黄金资产代码（COMEX / SHFE_AU / ...）")
    parser.add_argument("--timeframe", default="1d", help="K 线周期（1m/5m/15m/30m/1h/4h/1d/1w/1mo）")
    parser.add_argument("--source", default=None, help="数据源 eastmoney / ths / akshare")
    parser.add_argument("--start", default=None, help="开始时间，如 2025-01-01")
    parser.add_argument("--end", default=None, help="结束时间，如 2025-06-30")
    parser.add_argument("--csv", default=None, help="离线 CSV 文件路径（优先于在线取数）")
    parser.add_argument("--synthetic", type=int, default=0, help="生成 N 根确定性合成 K 线（离线演示）")
    parser.add_argument("--synthetic-seed", type=int, default=7)
    parser.add_argument("--strategy", default="rule", choices=("rule", "recorded", "llm"))
    parser.add_argument("--recorded-path", default=None, help="已记录决策 CSV/JSON（strategy=recorded）")
    parser.add_argument("--horizon", type=int, default=1, help="预测 / 持有 K 线根数")
    parser.add_argument("--warmup", type=int, default=30, help="决策所需最小历史窗口")
    parser.add_argument("--min-bars", type=int, default=60)
    parser.add_argument("--fee", type=float, default=0.0, help="单边手续费 %%")
    parser.add_argument("--slippage", type=float, default=0.0, help="单边滑点 %%")
    parser.add_argument("--stop-atr", type=float, default=2.0, help="止损 = ATR × 倍数")
    parser.add_argument("--target-atr", type=float, default=0.0, help="止盈 = ATR × 倍数（0 表示用 risk-reward）")
    parser.add_argument("--risk-reward", type=float, default=1.5)
    parser.add_argument("--no-stops", action="store_true", help="关闭止损/止盈，仅按持有期平仓")
    parser.add_argument("--benchmark", default="buy_hold", choices=("buy_hold", "flat"))
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--error-threshold", type=float, default=0.10, help="实质性方向错误的收益阈值 %%")
    parser.add_argument("--max-trade-loss", type=float, default=2.0, help="单笔风控上限 %%")
    parser.add_argument("--run-id", default="default")
    parser.add_argument("--incremental", action="store_true", help="复用同 run_id 下已计算的决策")
    parser.add_argument("--runs-dir", default=None, help="结果输出目录（默认 validation_runs/）")


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        asset=args.asset,
        timeframe=args.timeframe,
        source=args.source,
        start=args.start,
        end=args.end,
        min_bars=args.min_bars,
        run_id=args.run_id,
        incremental=args.incremental,
        strategy=args.strategy,
        horizon_bars=args.horizon,
        warmup_bars=args.warmup,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
        use_stops=not args.no_stops,
        stop_atr_mult=args.stop_atr,
        target_atr_mult=args.target_atr,
        risk_reward=args.risk_reward,
        benchmark=args.benchmark,
        recorded_path=args.recorded_path,
        min_confidence=args.min_confidence,
        error_return_threshold_pct=args.error_threshold,
        max_trade_loss_pct=args.max_trade_loss,
    )


def _synthetic_params(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.synthetic and args.synthetic > 0:
        return {"n": args.synthetic, "seed": args.synthetic_seed}
    return None


def _print_summary(payload: Dict[str, Any]) -> None:
    m = payload.get("metrics", {})
    acc, perf = m.get("accuracy", {}), m.get("performance", {})
    risk, time_m = m.get("risk", {}), m.get("timeliness", {})
    anom = payload.get("anomalies", {})
    print("=" * 72)
    print(f"run_id     : {payload.get('run_id')} （策略 {payload.get('strategy')}）")
    print(f"数据       : {payload.get('data', {}).get('asset_name')} "
          f"{payload.get('data', {}).get('first')} → {payload.get('data', {}).get('last')} "
          f"({payload.get('data', {}).get('rows')} 根)")
    print(f"决策/交易  : {payload.get('stats', {}).get('decisions')} / {len(payload.get('trades', []))}")
    print(f"综合评分   : {m.get('composite', {}).get('score')} / 100")
    accuracy = acc.get("accuracy")
    print(f"方向准确率 : {accuracy * 100:.2f}%" if accuracy is not None else "方向准确率 : —")
    print(f"累计净收益 : {perf.get('total_return_pct')}%  （基准 {perf.get('benchmark_total_return_pct')}%，"
          f"超额 {perf.get('excess_return_pct')}%）")
    print(f"最大回撤   : {risk.get('max_drawdown_pct')}%   Sharpe: {perf.get('sharpe')}")
    print(f"决策耗时   : avg {time_m.get('avg_decision_latency_ms')} ms / p95 {time_m.get('p95_decision_latency_ms')} ms")
    print(f"异常标记   : {anom.get('total_flags')} 项 / 涉及 {anom.get('flagged_decisions')} 个决策")
    print(f"决策哈希   : {str(payload.get('decisions_hash'))[:16]}…  可复现: {payload.get('reproducible')}")
    print(f"报告       : validation_runs/{payload.get('run_id')}/report.html")
    print("=" * 72)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m validation.cli",
        description="QuantHarness-Gold 决策正确性验证（回测）工具",
    )
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="执行一次回测验证")
    _add_common(run_parser)
    run_parser.add_argument("--json", action="store_true", help="以 JSON 输出完整结果")

    list_parser = sub.add_parser("list", help="列出历史回测")
    list_parser.add_argument("--runs-dir", default=None)

    show_parser = sub.add_parser("show", help="查看某次回测结果")
    show_parser.add_argument("run_id")
    show_parser.add_argument("--runs-dir", default=None)
    show_parser.add_argument("--json", action="store_true")

    check_parser = sub.add_parser("check", help="可复现性校验（连跑两次比对决策哈希）")
    _add_common(check_parser)

    serve_parser = sub.add_parser("serve", help="启动可视化 Web 页面")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5001)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run":
        cfg = _build_config(args)
        payload = run_backtest(
            cfg, csv_path=args.csv, synthetic=_synthetic_params(args), runs_dir=args.runs_dir
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_summary(payload)
        return 0

    if args.command == "list":
        runs = list_runs(args.runs_dir)
        if not runs:
            print("暂无历史回测。先运行：python -m validation.cli run --synthetic 400")
            return 0
        for r in runs:
            print(f"- {r['run_id']:<24} {r.get('strategy'):<8} {r.get('asset')}/{r.get('timeframe')} "
                  f"{r.get('created_at')}  hash={str(r.get('decisions_hash'))[:12]}…")
        return 0

    if args.command == "show":
        payload = load_run(args.run_id, args.runs_dir)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_summary(payload)
        return 0

    if args.command == "check":
        cfg = _build_config(args)
        result = check_reproducible(
            cfg, csv_path=args.csv, synthetic=_synthetic_params(args)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("reproducible") else 2

    if args.command == "serve":
        from .web_app import create_app

        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
