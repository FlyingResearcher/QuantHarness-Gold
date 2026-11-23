"""回测引擎测试：无未来函数、成交模拟、增量复用、边界与异常输入。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from validation.config import BacktestConfig, ConfigError
from validation.data import synthetic_bars
from validation.engine import BacktestEngine
from validation.runner import run_backtest


def _config(**kwargs) -> BacktestConfig:
    base = dict(
        asset="COMEX",
        timeframe="1d",
        run_id="test_engine",
        warmup_bars=30,
        min_bars=60,
        use_stops=False,
        horizon_bars=2,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    base.update(kwargs)
    return BacktestConfig(**base)


class TestBacktestEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars(180, seed=11)

    def test_reproducible_decisions(self):
        """同一数据 + 同一参数：决策序列哈希必须完全一致。"""
        first = run_backtest(_config(), bars=self.bars, persist=False)
        second = run_backtest(_config(), bars=self.bars, persist=False)
        self.assertGreater(len(first["decisions"]), 0)
        self.assertEqual(first["decisions_hash"], second["decisions_hash"])
        self.assertTrue(first["reproducible"])

    def test_no_lookahead_entry_next_bar(self):
        """第 i 根收盘决策，必须在第 i+1 根开盘成交，且价格可核对。"""
        run = BacktestEngine(_config()).run(self.bars)
        self.assertGreater(len(run.trades), 0)
        for trade in run.trades[:30]:
            entry_row = self.bars.iloc[trade.entry_index]
            decide_row = self.bars.iloc[trade.decision_index]
            self.assertEqual(trade.entry_index, trade.decision_index + 1)
            self.assertEqual(
                trade.entry_datetime,
                pd.Timestamp(entry_row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.assertEqual(
                trade.entry_price, round(float(entry_row["Open"]), 6)
            )
            self.assertNotEqual(
                trade.entry_datetime,
                pd.Timestamp(decide_row["Datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
            )

    def test_exit_within_horizon(self):
        """关闭止损后，持仓不超过配置的 horizon_bars，且按持有期平仓。"""
        run = BacktestEngine(_config(horizon_bars=2, use_stops=False)).run(self.bars)
        self.assertGreater(len(run.trades), 0)
        for trade in run.trades:
            self.assertEqual(trade.exit_reason, "horizon")
            self.assertLessEqual(trade.holding_bars, 2)
            self.assertLessEqual(trade.exit_index, trade.entry_index + 1)

    def test_stop_loss_exit_reason(self):
        """启用止损时，出场原因只能来自 stop/target/horizon 三种。"""
        run = BacktestEngine(
            _config(use_stops=True, stop_atr_mult=0.5, risk_reward=1.0, horizon_bars=5)
        ).run(self.bars)
        reasons = {t.exit_reason for t in run.trades}
        self.assertTrue(reasons.issubset({"stop", "target", "horizon"}))
        self.assertIn("stop", reasons)

    def test_incremental_reuses_cached_decisions(self):
        """增量回测：相同 run_id 且指纹一致时应全部复用，不重复计算。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(run_id="inc", incremental=False)
            first = run_backtest(cfg, bars=self.bars, runs_dir=tmp, persist=True)
            self.assertEqual(first["stats"]["reused_decisions"], 0)

            cfg2 = _config(run_id="inc", incremental=True)
            second = run_backtest(cfg2, bars=self.bars, runs_dir=tmp, persist=True)
            self.assertEqual(second["stats"]["computed_decisions"], 0)
            self.assertEqual(second["stats"]["reused_decisions"], second["stats"]["decisions"])
            self.assertEqual(first["decisions_hash"], second["decisions_hash"])

    def test_changed_params_invalidate_cache(self):
        """参数变化后不得复用旧决策（指纹失效）。"""
        with tempfile.TemporaryDirectory() as tmp:
            run_backtest(_config(run_id="inv"), bars=self.bars, runs_dir=tmp, persist=True)
            changed = run_backtest(
                _config(run_id="inv", incremental=True, weight_indicator=0.9, weight_trend=0.05,
                        weight_pattern=0.05),
                bars=self.bars,
                runs_dir=tmp,
                persist=True,
            )
            self.assertEqual(changed["stats"]["reused_decisions"], 0)
            self.assertGreater(changed["stats"]["computed_decisions"], 0)

    def test_recorded_strategy(self):
        """已记录决策回放：按时间戳精确对齐，决策数等于记录覆盖的 bar 数。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.csv"
            rows = {
                "Datetime": self.bars["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "decision": ["LONG" if i % 2 == 0 else "SHORT" for i in range(len(self.bars))],
                "confidence": [0.8] * len(self.bars),
            }
            pd.DataFrame(rows).to_csv(path, index=False)
            cfg = _config(run_id="recorded", strategy="recorded", recorded_path=str(path))
            run = run_backtest(cfg, bars=self.bars, persist=False)
            self.assertEqual(run["stats"]["decisions"], run["stats"]["trades"])
            self.assertGreater(run["stats"]["decisions"], 0)

    def test_insufficient_data_raises(self):
        tiny = synthetic_bars(20, seed=3)
        with self.assertRaises(ValueError):
            BacktestEngine(_config()).run(tiny)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ConfigError):
            BacktestConfig(run_id="bad id!")
        with self.assertRaises(ConfigError):
            BacktestConfig(timeframe="2h")
        with self.assertRaises(ConfigError):
            BacktestConfig(horizon_bars=0)
        with self.assertRaises(ConfigError):
            BacktestConfig(warmup_bars=10)
        with self.assertRaises(ConfigError):
            BacktestConfig(strategy="recorded")


if __name__ == "__main__":
    unittest.main()
