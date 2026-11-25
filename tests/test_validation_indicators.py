"""回测侧的共享指标接入：口径一致、阈值重标、指纹升版、准确率钉死。

回测以前自己算一套指标（快速 %K、无周期后缀的键）。接入共享模块后，
它与报告、Agent 读同一批数字，Stochastic 阈值也随慢速 %K 重标。
"""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import indicators
import validation.indicators as backtest_indicators
from validation.config import ENGINE_VERSION, BacktestConfig
from validation.data import synthetic_bars
from validation.runner import run_backtest


def _config(**kwargs) -> BacktestConfig:
    base = dict(
        asset="COMEX",
        timeframe="1d",
        run_id="seam",
        warmup_bars=30,
        min_bars=60,
        use_stops=False,
        horizon_bars=2,
        fee_pct=0.0,
        slippage_pct=0.0,
    )
    base.update(kwargs)
    return BacktestConfig(**base)


# 固定夹具：400 根合成行情（seed=7），回测与校准都基于它。
_FIXTURE_BARS = 400
_FIXTURE_SEED = 7
_WARMUP = 30

# 迁移前（快速 %K × 85/15）在该夹具上的 Stochastic 极端区触发率。
# 阈值下调到 83/17 就是为了把这个比例拿回来。
_PRE_CHANGE_FIRE_RATE = 0.43089430894308944
# 重标后的实测触发率。83/17 比原频率高 1.9pp；这个夹具上 83/16 恰好等于原频率，
# 但那不是本票选定的阈值对（见 ADR-0001），所以这里的容差是 2.5pp，不是 0。
_RECALIBRATED_FIRE_RATE = 0.44986449864498645
# 容差 2.5pp：把「大致回到原水平」和「完全等价」区分开。
_FIRE_RATE_TOLERANCE = 0.025


def _engine(bars):
    return backtest_indicators.RuleSignalEngine(bars, _config())


class TestBacktestReadsTheSharedModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars(_FIXTURE_BARS, seed=_FIXTURE_SEED)
        cls.engine = _engine(cls.bars)
        cls.shared = indicators.compute_series(cls.bars)

    def test_every_shared_indicator_series_is_the_shared_one(self):
        """回测序列里每个共享键都与共享模块逐 bar 相同（在同样的取整口径下）。"""
        keys = [k for k in indicators.INDICATOR_KEYS if k in self.engine.series]
        self.assertIn("stoch_k", keys)
        self.assertIn("atr_14", keys)
        for i in range(_WARMUP, len(self.bars) - 1):
            row = self.engine._row(i)
            for key in keys:
                expected = backtest_indicators._safe(self.shared[key].iloc[i])
                with self.subTest(bar=i, key=key):
                    self.assertEqual(row[key], expected)

    def test_backtest_still_has_its_own_channel_and_volume_series(self):
        """共享模块不负责的量能/通道序列仍由回测自己提供。"""
        for key in ("open", "vol_ratio", "roll_high", "roll_low", "trend_slope_pct"):
            with self.subTest(key=key):
                self.assertIn(key, self.engine.series)

    def test_delegation_is_real(self):
        """把共享计算替换成哨兵值，回测读到的必须跟着变。"""
        original = indicators.compute_series

        def stamped(df):
            out = original(df)
            return {k: (v + 12.34 if k == "stoch_k" else v) for k, v in out.items()}

        with patch.object(backtest_indicators.shared_indicators, "compute_series", stamped):
            engine = _engine(self.bars)
        expected = backtest_indicators._safe(self.shared["stoch_k"].iloc[100] + 12.34)
        self.assertEqual(engine._row(100)["stoch_k"], expected)


class TestSlowPercentK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synthetic_bars(_FIXTURE_BARS, seed=_FIXTURE_SEED)
        cls.engine = _engine(cls.bars)

    def test_backtest_percent_k_is_the_smoothed_value(self):
        """回测的 %K 等于共享模块的慢速 %K，而不是原始的快速 %K。"""
        slow = indicators.compute_series(self.bars)["stoch_k"]
        row = self.engine._row(100)
        self.assertEqual(row["stoch_k"], backtest_indicators._safe(slow.iloc[100]))
        fast = indicators.talib.STOCH(
            self.bars["High"].to_numpy(dtype=float),
            self.bars["Low"].to_numpy(dtype=float),
            self.bars["Close"].to_numpy(dtype=float),
            fastk_period=14,
            slowk_period=1,
            slowd_period=1,
        )[0]
        self.assertNotEqual(row["stoch_k"], backtest_indicators._safe(fast[100]))

    def test_stochastic_thresholds_are_named_and_recalibrated(self):
        """83/17 以具名常量出现，且确实被用于打分——不是只写在注释里。"""
        self.assertEqual(backtest_indicators.STOCH_OVERBOUGHT, 83.0)
        self.assertEqual(backtest_indicators.STOCH_OVERSOLD, 17.0)

        # 构造一个 %K 落在 83 与 85 之间、%K≈%D 的读数：
        # 新阈值下应扣分（超买），旧阈值下则不应扣分。
        row = {"stoch_k": 84.0, "stoch_d": 84.0}
        score, parts = backtest_indicators._indicator_signal(row)
        self.assertLess(parts["stoch"], 0.0, "慢速 %K 在 84 应被判定为超买")

        row = {"stoch_k": 16.0, "stoch_d": 16.0}
        _, parts = backtest_indicators._indicator_signal(row)
        self.assertGreater(parts["stoch"], 0.0, "慢速 %K 在 16 应被判定为超卖")

    def test_stochastic_fires_at_roughly_its_previous_rate(self):
        """重标后 Stochastic 的极端区触发率回到迁移前的水平。

        慢速 %K 平滑掉了快速 %K 的尖峰，极端区停留更短。83/17 把触发率从
        85/15 下的 40.11% 拉回到 44.99%，与快速 %K 的 43.09% 只差 1.9pp。
        """
        rate = self._extreme_rate(backtest_indicators.STOCH_OVERBOUGHT,
                                  backtest_indicators.STOCH_OVERSOLD)
        self.assertAlmostEqual(rate, _RECALIBRATED_FIRE_RATE, places=10)
        self.assertAlmostEqual(rate, _PRE_CHANGE_FIRE_RATE, delta=_FIRE_RATE_TOLERANCE)

    def _extreme_rate(self, overbought: float, oversold: float) -> float:
        k = self.engine.series["stoch_k"]
        values = np.array(
            [float(k.iloc[i]) for i in range(_WARMUP, len(self.bars) - 1)], dtype=float
        )
        values = values[np.isfinite(values)]
        return float(((values > overbought) | (values < oversold)).mean())

    def test_old_thresholds_would_have_under_fired(self):
        """保留 85/15 会让触发率明显下降——这正是重标的理由。"""
        stale = self._extreme_rate(85.0, 15.0)
        self.assertLess(stale, _PRE_CHANGE_FIRE_RATE - 0.005)
        self.assertLess(stale, self._extreme_rate(
            backtest_indicators.STOCH_OVERBOUGHT, backtest_indicators.STOCH_OVERSOLD
        ))


class TestSnapshotVocabulary(unittest.TestCase):
    def test_snapshot_keys_carry_their_period(self):
        bars = synthetic_bars(180, seed=11)
        out = _engine(bars).evaluate(100)
        for key in ("rsi_14", "roc_10", "willr_14", "atr_14"):
            with self.subTest(key=key):
                self.assertIn(key, out["signals"])
        for stale in ("rsi", "roc", "willr", "atr"):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, out["signals"])

    def test_snapshot_values_are_the_shared_series(self):
        """快照里每个值都取自同名共享序列——改一个键的取值来源会立刻暴露。"""
        bars = synthetic_bars(180, seed=11)
        engine = _engine(bars)
        shared = indicators.compute_series(bars)
        out = engine.evaluate(120)
        for key in ("rsi_14", "roc_10", "willr_14", "atr_14", "macd_hist", "stoch_k"):
            with self.subTest(key=key):
                self.assertEqual(
                    out["signals"][key],
                    backtest_indicators._safe(shared[key].iloc[120]),
                )

    def test_stops_read_atr_from_the_stored_decision(self):
        """引擎从持久化决策里读 atr_14：止损价必须由它算出来。"""
        bars = synthetic_bars(180, seed=11)
        cfg = _config(use_stops=True, stop_atr_mult=2.0, risk_reward=1.5)
        run = run_backtest(cfg, bars=bars, persist=False)
        stopped = [t for t in run["decisions"] if t["signals"].get("atr_14")]
        self.assertGreater(len(stopped), 0)
        decision = stopped[0]
        entry = float(bars.iloc[decision["index"] + 1]["Open"])
        atr = float(decision["signals"]["atr_14"])
        expected_stop = (
            entry - 2.0 * atr if decision["direction"] == "LONG" else entry + 2.0 * atr
        )
        trade = next(t for t in run["trades"] if t["decision_index"] == decision["index"])
        self.assertAlmostEqual(trade["stop_price"], round(expected_stop, 6), places=4)


class TestEngineVersionBump(unittest.TestCase):
    def test_version_moved_past_the_old_one(self):
        major, minor, _ = (int(p) for p in ENGINE_VERSION.split("."))
        self.assertEqual((major, minor), (1, 1))

    def test_decision_fingerprint_changes_with_the_version(self):
        """版本参与决策指纹；升版必须让增量缓存失效。"""
        cfg = _config()
        current = cfg.decision_fingerprint()
        with patch("validation.config.ENGINE_VERSION", "1.0.0"):
            previous = cfg.decision_fingerprint()
        self.assertNotEqual(current, previous)

    def test_a_cache_written_under_the_old_version_is_recomputed(self):
        """端到端：1.0.0 留下的 manifest 在 1.1.0 下必须全部重算。

        只断言指纹不同是不够的——真正要保护的是「增量回测不会把旧口径的
        决策拿来复用」这条路径。
        """
        import json
        import tempfile
        from pathlib import Path

        bars = synthetic_bars(180, seed=11)
        cfg = _config(run_id="stale", incremental=True)
        with tempfile.TemporaryDirectory() as tmp:
            run = run_backtest(cfg, bars=bars, runs_dir=tmp, persist=True)
            manifest_path = Path(tmp) / "stale" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["engine_version"], ENGINE_VERSION)

            # 把 manifest 改写成旧版本，模拟上一版留下的缓存目录。
            manifest["engine_version"] = "1.0.0"
            manifest["decision_fingerprint"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            rerun = run_backtest(cfg, bars=bars, runs_dir=tmp, persist=True)
            self.assertEqual(rerun["stats"]["reused_decisions"], 0)
            self.assertEqual(rerun["stats"]["computed_decisions"], run["stats"]["decisions"])


class TestBacktestSeam(unittest.TestCase):
    """固定夹具上的回测结果钉死，迁移前后都落在这个范围内。"""

    def test_accuracy_and_trade_count_are_pinned(self):
        bars = synthetic_bars(_FIXTURE_BARS, seed=_FIXTURE_SEED)
        run = run_backtest(_config(), bars=bars, persist=False)
        accuracy = run["metrics"]["accuracy"]["accuracy"]
        self.assertEqual(run["stats"]["trades"], 369)
        self.assertAlmostEqual(accuracy, 0.607046, places=6)

    def test_horizon_one_accuracy_is_pinned(self):
        bars = synthetic_bars(_FIXTURE_BARS, seed=_FIXTURE_SEED)
        run = run_backtest(_config(horizon_bars=1), bars=bars, persist=False)
        self.assertEqual(run["stats"]["trades"], 369)
        self.assertAlmostEqual(
            run["metrics"]["accuracy"]["accuracy"], 0.563686, places=6
        )

    def test_result_is_reproducible_under_the_shared_module(self):
        bars = synthetic_bars(_FIXTURE_BARS, seed=_FIXTURE_SEED)
        first = run_backtest(_config(), bars=bars, persist=False)
        second = run_backtest(_config(), bars=bars, persist=False)
        self.assertEqual(first["decisions_hash"], second["decisions_hash"])


if __name__ == "__main__":
    unittest.main()