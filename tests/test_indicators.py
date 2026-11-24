"""共享指标模块测试：口径一致性、%K 约定锁定、其余指标数值钉死。

这些断言全部落在模块的公开接口上（compute_series / latest_values /
tool_payload），不触碰内部实现。
"""

import os
import subprocess
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

import indicators


# 一段显式构造的锯齿行情：可手工核对，用于锁定 %K 口径。
# 60 根——MACD 需要 26+9-1=34 根才出值，短于此则 MACD 完全未被覆盖。
CLOSES = [
    100, 102, 101, 104, 103, 106, 105, 108, 107, 110, 109, 112, 111, 114, 113,
    116, 115, 118, 117, 120, 119, 122, 121, 124, 123, 126, 125, 128, 127, 130,
    129, 132, 131, 134, 133, 136, 135, 138, 137, 140, 139, 142, 141, 144, 143,
    146, 145, 148, 147, 150, 149, 152, 151, 154, 153, 156, 155, 158, 157, 160,
]
HIGHS = [c + 1.5 for c in CLOSES]
LOWS = [c - 1.5 for c in CLOSES]
VOLUMES = [1000000 + i * 1000 for i in range(len(CLOSES))]

# 由独立参考实现（显式循环，不经 TA-Lib / pandas）算出的 %K / %D 末值。
# 慢速 %K = SMA3(快速 %K)；%D = SMA3(%K)。
EXPECTED_SLOW_K = 89.23611111111113
EXPECTED_D = 88.42592592592597
# 快速 %K（未平滑）——与慢速 %K 不同，正是本次要钉死的区别。
EXPECTED_FAST_K = 91.66666666666666

# TA-Lib 计算的指标（RSI / MACD / ROC / Williams %R）的末值。
EXPECTED_RSI_14 = 76.38181456186884
EXPECTED_MACD = 7.044587285831369
EXPECTED_MACD_SIGNAL = 7.004043777098295
EXPECTED_MACD_HIST = 0.040543508733073885
EXPECTED_ROC_10 = 6.666666666666665
EXPECTED_WILLR_14 = -8.333333333333332

# pandas 计算的指标（布林带 / ATR / 均线）的末值：合并前后必须一致。
EXPECTED_BB_UP = 161.68281792655455
EXPECTED_BB_MID = 149.5
EXPECTED_BB_LO = 137.31718207344545
EXPECTED_BB_PCT = 93.09347830403605
EXPECTED_ATR_14 = 3.7676915347047233
EXPECTED_MA5 = 157.2
EXPECTED_MA20 = 149.5
EXPECTED_MA60 = 129.51666666666668


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=len(CLOSES), freq="1D"),
            "Open": CLOSES,
            "High": HIGHS,
            "Low": LOWS,
            "Close": CLOSES,
            "Volume": VOLUMES,
        }
    )


class TestIndicatorSeries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = indicators.compute_series(_frame())

    def _last(self, key: str) -> float:
        """序列最后一个有效值。"""
        values = self.series[key].dropna()
        return float(values.iloc[-1])

    def test_series_exposes_the_canonical_keys(self):
        """序列投影必须暴露全部二十个指标键。"""
        self.assertEqual(set(self.series.keys()), set(indicators.INDICATOR_KEYS))
        self.assertEqual(len(self.series), 20)

    def test_key_set_is_twenty_even_without_volume(self):
        """键集恒为二十个：缺少成交量列时也必须补齐，调用方不必先判断。"""
        series = indicators.compute_series(_frame().drop(columns=["Volume"]))
        self.assertEqual(set(series.keys()), set(indicators.INDICATOR_KEYS))
        self.assertIn("volume", series)

    def test_stoch_k_is_the_smoothed_percent_k(self):
        """%K 必须是慢速 %K（SMA3 后的值），不是未平滑的快速 %K。"""
        actual = self._last("stoch_k")
        self.assertAlmostEqual(actual, EXPECTED_SLOW_K, places=8)
        # 明确排除快速 %K：这个值不同，正是要防回归的那一步。
        self.assertNotAlmostEqual(actual, EXPECTED_FAST_K, places=6)

    def test_stoch_d_is_the_average_of_percent_k(self):
        """%D 必须是 %K 的 3 周期均值。"""
        self.assertAlmostEqual(self._last("stoch_d"), EXPECTED_D, places=8)

    def test_talib_indicators_have_their_pinned_values(self):
        """TA-Lib 计算的指标数值必须钉死——它们不随实现改动而漂移。"""
        self.assertAlmostEqual(self._last("rsi_14"), EXPECTED_RSI_14, places=8)
        self.assertAlmostEqual(self._last("macd"), EXPECTED_MACD, places=8)
        self.assertAlmostEqual(self._last("macd_signal"), EXPECTED_MACD_SIGNAL, places=8)
        self.assertAlmostEqual(self._last("macd_hist"), EXPECTED_MACD_HIST, places=8)
        self.assertAlmostEqual(self._last("roc_10"), EXPECTED_ROC_10, places=8)
        self.assertAlmostEqual(self._last("willr_14"), EXPECTED_WILLR_14, places=8)

    def test_macd_is_actually_exercised(self):
        """MACD 需要 34 根才出值——夹具必须足够长，否则 MACD 形同虚设。"""
        self.assertFalse(self.series["macd"].dropna().empty)
        self.assertFalse(self.series["macd_signal"].dropna().empty)
        self.assertFalse(self.series["macd_hist"].dropna().empty)

    def test_bollinger_values_are_unchanged(self):
        """布林带数值必须与合并前一致（pandas 样本标准差口径）。"""
        self.assertAlmostEqual(self._last("bb_up"), EXPECTED_BB_UP, places=9)
        self.assertAlmostEqual(self._last("bb_mid"), EXPECTED_BB_MID, places=9)
        self.assertAlmostEqual(self._last("bb_lo"), EXPECTED_BB_LO, places=9)
        self.assertAlmostEqual(self._last("bb_pct"), EXPECTED_BB_PCT, places=9)

    def test_atr_and_moving_averages_are_unchanged(self):
        """ATR 与均线数值必须与合并前一致。"""
        self.assertAlmostEqual(self._last("atr_14"), EXPECTED_ATR_14, places=9)
        self.assertAlmostEqual(self._last("ma5"), EXPECTED_MA5, places=9)
        self.assertAlmostEqual(self._last("ma20"), EXPECTED_MA20, places=9)
        self.assertAlmostEqual(self._last("ma60"), EXPECTED_MA60, places=9)

    def test_atr_has_values_from_the_first_bar(self):
        """ATR 不设 min_periods——与合并前一致，头部即有值。"""
        self.assertFalse(self.series["atr_14"].iloc[0:3].isna().any())


class TestIndicatorProjections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = indicators.compute_series(_frame())

    def test_all_projections_report_the_same_percent_k(self):
        """三种投影必须给出同一个 %K / %D——这是本次合并的核心保证。"""
        latest = indicators.latest_values(self.series)
        payload = indicators.tool_payload(self.series)

        series_k = float(self.series["stoch_k"].dropna().iloc[-1])
        series_d = float(self.series["stoch_d"].dropna().iloc[-1])

        self.assertAlmostEqual(latest["stoch_k"], series_k, places=8)
        self.assertAlmostEqual(latest["stoch_d"], series_d, places=8)
        self.assertAlmostEqual(payload["stoch_k"][-1], series_k, places=2)
        self.assertAlmostEqual(payload["stoch_d"][-1], series_d, places=2)

    def test_latest_values_exposes_the_report_keys(self):
        """最新值投影必须提供报告解读所需的全部键。"""
        latest = indicators.latest_values(self.series)
        expected = set(indicators.INDICATOR_KEYS) | {"macd_hist_prev"}
        self.assertEqual(set(latest.keys()), expected)

    def test_latest_values_high_and_low_span_the_whole_range(self):
        """high / low 取整段区间的极值，与既有报告口径一致。"""
        latest = indicators.latest_values(self.series)
        self.assertAlmostEqual(latest["high"], max(HIGHS), places=9)
        self.assertAlmostEqual(latest["low"], min(LOWS), places=9)

    def test_missing_volume_is_reported_as_none_not_zero(self):
        """缺少成交量时必须报 None——补零会伪造一个不存在的指标。"""
        series = indicators.compute_series(_frame().drop(columns=["Volume"]))
        self.assertIsNone(indicators.latest_values(series)["volume"])

    def test_tool_payload_keeps_the_agent_tool_shape(self):
        """工具载荷必须沿用 Agent 既有工具的键名与窗口长度。"""
        payload = indicators.tool_payload(self.series)
        self.assertEqual(
            set(payload.keys()),
            {"rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc", "willr"},
        )
        for key in ("rsi", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc", "willr"):
            self.assertLessEqual(len(payload[key]), indicators.TOOL_WINDOW)
        # macd 在既有工具里返回完整序列，这里保持一致。
        self.assertEqual(len(payload["macd"]), len(self.series["macd"]))


class TestIndicatorModuleStandalone(unittest.TestCase):
    def test_imports_without_the_agent_stack(self):
        """模块必须能单独导入：不牵入 Agent 栈或绘图库。"""
        script = (
            "import sys, indicators;"
            "heavy = [m for m in ('langchain_core', 'langchain', 'langgraph',"
            " 'matplotlib', 'mplfinance') if m in sys.modules];"
            "print(','.join(heavy))"
        )
        repo_root = os.path.join(os.path.dirname(__file__), "..")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
