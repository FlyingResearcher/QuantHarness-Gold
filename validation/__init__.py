"""决策正确性验证模块（回测 / Decision-Correctness Validation）。

该包与主决策系统（trading_graph / decision_agent）完全解耦：
  * 不修改、不导入 LangGraph 决策图即可独立完成「历史数据 → 决策回放 → 指标评估」；
  * 仅通过可插拔策略接口（rule / recorded / llm）复用决策逻辑；
  * 输出可复现的 manifest + JSONL 决策明细 + 自包含 HTML 报表。

常用入口：
    from validation import run_backtest, BacktestConfig

    result = run_backtest(BacktestConfig(asset="COMEX", timeframe="1d"))
"""

from .config import BacktestConfig, ConfigError, ENGINE_VERSION
from .engine import BacktestEngine, BacktestRun, Decision, Trade
from .runner import list_runs, load_run, run_backtest

__all__ = [
    "BacktestConfig",
    "ConfigError",
    "ENGINE_VERSION",
    "BacktestEngine",
    "BacktestRun",
    "Decision",
    "Trade",
    "run_backtest",
    "load_run",
    "list_runs",
]

__version__ = ENGINE_VERSION
