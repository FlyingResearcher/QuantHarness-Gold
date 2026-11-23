"""指标计算测试：用可手算的样本验证准确率、收益、风险、时效与基准指标。"""

import os
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation.config import BacktestConfig
from validation.engine import BacktestEngine, BacktestRun, Decision, Trade
from validation.metrics import accuracy_metrics, compute_metrics, performance_metrics, risk_metrics


def make_trade(i, net, gross=None, direction="LONG", correct=None, hold=1, reason="horizon",
               bench=None, conf=0.8, mfe=0.0, mfe_bars=0) -> Trade:
    return Trade(
        decision_index=i,
        entry_index=i + 1,
        exit_index=i + hold,
        entry_datetime="2025-01-01 00:00:00",
        exit_datetime="2025-01-02 00:00:00",
        direction=direction,
        entry_price=100.0,
        exit_price=100.0 + (net if direction == "LONG" else -net),
        stop_price=None,
        target_price=None,
        exit_reason=reason,
        holding_bars=hold,
        gross_return_pct=net if gross is None else gross,
        net_return_pct=net,
        mfe_pct=mfe,
        mae_pct=min(0.0, net),
        time_to_mfe_bars=mfe_bars,
        correct=(None if net == 0 else net > 0) if correct is None else correct,
        confidence=conf,
        benchmark_return_pct=net if bench is None else bench,
    )


def make_run(trades, config=None) -> BacktestRun:
    cfg = config or BacktestConfig(run_id="metrics_test")
    series = BacktestEngine._build_series(trades)
    return BacktestRun(
        run_id=cfg.run_id,
        config=cfg,
        data_meta={},
        decisions=[],
        trades=trades,
        series=series,
        stats={"eligible_decisions": len(trades), "decisions": len(trades)},
        decisions_hash="x",
        decision_cache={},
        data_issues=[],
        timings={},
        strategy_name="rule",
    )


class TestMetrics(unittest.TestCase):
    def test_accuracy_hand_computed(self):
        trades = [
            make_trade(0, 1.0, correct=True),
            make_trade(1, 2.0, correct=True),
            make_trade(2, -1.0, direction="SHORT", correct=False),
        ]
        acc = accuracy_metrics(trades)
        self.assertEqual(acc["evaluated_trades"], 3)
        self.assertAlmostEqual(acc["accuracy"], 2 / 3, places=6)
        self.assertEqual(acc["confusion"]["tp"], 2)
        self.assertEqual(acc["confusion"]["fn"], 1)

    def test_tie_excluded_from_accuracy(self):
        trades = [make_trade(0, 0.0, correct=None), make_trade(1, 1.0, correct=True)]
        acc = accuracy_metrics(trades)
        self.assertEqual(acc["evaluated_trades"], 1)
        self.assertEqual(acc["ties"], 1)
        self.assertAlmostEqual(acc["accuracy"], 1.0)

    def test_profit_factor_and_expectancy(self):
        trades = [make_trade(0, 2.0), make_trade(1, -1.0)]
        perf = performance_metrics(make_run(trades))
        self.assertAlmostEqual(perf["profit_factor"], 2.0, places=6)
        self.assertAlmostEqual(perf["avg_trade_return_pct"], 0.5, places=6)
        self.assertAlmostEqual(perf["total_return_pct"], (1.02 * 0.99 - 1) * 100, places=4)
        self.assertAlmostEqual(perf["win_rate"], 0.5)

    def test_max_drawdown_hand_computed(self):
        trades = [make_trade(0, 10.0), make_trade(1, -20.0)]
        risk = risk_metrics(make_run(trades))
        # 净值 1.0 → 1.1 → 0.88，回撤 = 0.88/1.1 - 1 = -20%
        self.assertAlmostEqual(risk["max_drawdown_pct"], -20.0, places=6)

    def test_benchmark_beta_and_alpha(self):
        trades = [
            make_trade(0, 2.0, bench=2.0),
            make_trade(1, -1.0, bench=-1.0),
            make_trade(2, 3.0, bench=3.0),
        ]
        perf = performance_metrics(make_run(trades))
        self.assertAlmostEqual(perf["beta"], 1.0, places=6)
        self.assertAlmostEqual(perf["alpha_pct"], 0.0, places=6)
        self.assertAlmostEqual(perf["excess_return_pct"], 0.0, places=6)

    def test_var_cvar(self):
        trades = [make_trade(i, r) for i, r in enumerate([-5.0, -3.0, -1.0, 1.0, 2.0])]
        risk = risk_metrics(make_run(trades))
        self.assertAlmostEqual(risk["var_95_pct"], -4.6, places=6)
        self.assertAlmostEqual(risk["cvar_95_pct"], -5.0, places=6)

    def test_empty_trades_no_nan(self):
        metrics = compute_metrics(make_run([]))
        self.assertIsNone(metrics["accuracy"]["accuracy"])
        self.assertEqual(metrics["performance"]["total_return_pct"], 0.0)
        self.assertEqual(metrics["risk"]["max_consecutive_losses"], 0)
        self.assertGreaterEqual(metrics["composite"]["score"], 0.0)

    def test_composite_score_range(self):
        trades = [make_trade(i, 0.5) for i in range(20)]
        metrics = compute_metrics(make_run(trades))
        self.assertGreaterEqual(metrics["composite"]["score"], 0.0)
        self.assertLessEqual(metrics["composite"]["score"], 100.0)

    def test_timeliness_latency(self):
        cfg = BacktestConfig(run_id="latency")
        run = make_run([make_trade(0, 1.0)])
        run.decisions = [
            Decision(
                index=0, datetime="2025-01-01 00:00:00", direction="LONG", confidence=0.8,
                score=0.4, risk_reward_ratio=1.5, sub_signals=None, signal_parts={},
                signals={}, latency_ms=ms, strategy="rule", window_digest="d",
            )
            for ms in (10.0, 20.0, 30.0)
        ]
        from validation.metrics import timeliness_metrics

        t = timeliness_metrics(run)
        self.assertAlmostEqual(t["avg_decision_latency_ms"], 20.0)
        self.assertAlmostEqual(t["max_decision_latency_ms"], 30.0)
        self.assertEqual(t["execution_lag_bars"], 1)


if __name__ == "__main__":
    unittest.main()
