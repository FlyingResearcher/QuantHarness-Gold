"""异常决策标记测试：逐类触发并校验归类、计数与严重度。"""

import os
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from validation.anomalies import detect_anomalies
from validation.config import BacktestConfig
from validation.engine import BacktestEngine, BacktestRun, Decision, Trade


def make_decision(i, direction="LONG", confidence=0.8, error=None, sub=None, score=0.4):
    return Decision(
        index=i,
        datetime=f"2025-01-{i + 1:02d} 00:00:00",
        direction=direction,
        confidence=confidence,
        score=score,
        risk_reward_ratio=1.5,
        sub_signals=sub,
        signal_parts={},
        signals={"atr_14": 1.0},
        latency_ms=1.0,
        strategy="rule",
        window_digest=f"d{i}",
        error=error,
    )


def make_trade(i, net, gross=None, direction="LONG", correct=True, reason="horizon", hold=1):
    return Trade(
        decision_index=i,
        entry_index=i + 1,
        exit_index=i + hold,
        entry_datetime="2025-01-01 00:00:00",
        exit_datetime="2025-01-02 00:00:00",
        direction=direction,
        entry_price=100.0,
        exit_price=101.0,
        stop_price=None,
        target_price=None,
        exit_reason=reason,
        holding_bars=hold,
        gross_return_pct=net if gross is None else gross,
        net_return_pct=net,
        mfe_pct=0.0,
        mae_pct=0.0,
        time_to_mfe_bars=0,
        correct=correct,
        confidence=0.8,
        benchmark_return_pct=0.0,
    )


def make_run(decisions, trades, config=None, issues=None) -> BacktestRun:
    cfg = config or BacktestConfig(run_id="anomaly_test", max_repeat_signals=3)
    return BacktestRun(
        run_id=cfg.run_id,
        config=cfg,
        data_meta={},
        decisions=decisions,
        trades=trades,
        series=BacktestEngine._build_series(trades),
        stats={"eligible_decisions": len(decisions), "decisions": len(decisions)},
        decisions_hash="x",
        decision_cache={},
        data_issues=issues or [],
        timings={},
        strategy_name="rule",
    )


class TestAnomalies(unittest.TestCase):
    def test_wrong_direction_and_risk_breach(self):
        cfg = BacktestConfig(run_id="a1", error_return_threshold_pct=0.1, max_trade_loss_pct=2.0)
        dec = make_decision(0, confidence=0.9)
        trade = make_trade(0, net=-3.0, gross=-2.5, correct=False, reason="stop")
        result = detect_anomalies(make_run([dec], [trade], cfg))
        codes = result["by_category"]
        self.assertIn("WRONG_DIRECTION", codes)
        self.assertIn("RISK_BREACH", codes)
        self.assertEqual(result["by_severity"]["high"], 2)  # 方向错误 + 风控超限
        self.assertIn("LATE_ENTRY", codes)

    def test_marginal_error(self):
        cfg = BacktestConfig(run_id="a2", error_return_threshold_pct=0.5)
        dec = make_decision(0)
        trade = make_trade(0, net=-0.2, gross=-0.2, correct=False)
        result = detect_anomalies(make_run([dec], [trade], cfg))
        self.assertIn("MARGINAL_ERROR", result["by_category"])
        self.assertNotIn("WRONG_DIRECTION", result["by_category"])

    def test_signal_conflict_and_low_confidence(self):
        cfg = BacktestConfig(run_id="a3", min_confidence=0.35)
        dec = make_decision(
            0, direction="LONG", confidence=0.10,
            sub={"indicator": -0.6, "trend": -0.5, "pattern": 0.1},
        )
        result = detect_anomalies(make_run([dec], [make_trade(0, 1.0)], cfg))
        self.assertIn("SIGNAL_CONFLICT", result["by_category"])
        self.assertIn("LOW_CONFIDENCE", result["by_category"])

    def test_late_entry(self):
        cfg = BacktestConfig(run_id="a4", late_entry_bars=1)
        dec = make_decision(0)
        trade = make_trade(0, net=-1.0, reason="stop", hold=1)
        result = detect_anomalies(make_run([dec], [trade], cfg))
        self.assertIn("LATE_ENTRY", result["by_category"])

    def test_duplicate_signal_flagged_once_per_run(self):
        cfg = BacktestConfig(run_id="a5", max_repeat_signals=3)
        decisions = [make_decision(i, direction="LONG") for i in range(6)]
        trades = [make_trade(i, 0.5) for i in range(6)]
        result = detect_anomalies(make_run(decisions, trades, cfg))
        self.assertEqual(result["by_category"]["DUPLICATE_SIGNAL"]["count"], 1)

    def test_data_anomaly(self):
        dec = make_decision(0)
        issues = [{"index": 0, "datetime": dec.datetime, "issue_bars": 2, "current_bar_issue": True}]
        result = detect_anomalies(make_run([dec], [make_trade(0, 1.0)], issues=issues))
        self.assertIn("DATA_ANOMALY", result["by_category"])
        self.assertEqual(result["data_count"], 1)

    def test_engine_error(self):
        dec = make_decision(0, direction=None, error="RuntimeError: boom")
        result = detect_anomalies(make_run([dec], []))
        self.assertIn("ENGINE_ERROR", result["by_category"])
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["total_flags"], 1)

    def test_no_anomaly_clean_run(self):
        cfg = BacktestConfig(run_id="a6", max_repeat_signals=10, min_confidence=0.0,
                             error_return_threshold_pct=99.0)
        dec = make_decision(0)
        result = detect_anomalies(make_run([dec], [make_trade(0, 1.0)], cfg))
        self.assertEqual(result["total_flags"], 0)
        self.assertEqual(result["flagged_decisions"], 0)


if __name__ == "__main__":
    unittest.main()
