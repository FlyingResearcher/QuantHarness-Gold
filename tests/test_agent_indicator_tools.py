"""Agent 指标工具测试：工具载荷与共享模块同源，且数值不变。

Agent 的工具包装器以前自己用 TA-Lib 算指标，是同一批指标的第三份实现。
迁移到共享 `indicators` 模块后，Agent 与报告读的是同一批数字——这是本文件
要锁住的东西。
"""

import os
import sys
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import indicators
from graph_util import TechnicalTools


# 45 根——Agent 实际收到的窗口长度（llm_window_bars=45）。
_FIXTURE_BARS = 45


def _kline_data(seed: int = 11, bars: int = _FIXTURE_BARS) -> dict:
    """确定性行情夹具，与 Agent 实际收到的输入同形。"""
    rng = np.random.default_rng(seed)
    close = (100 + np.cumsum(rng.normal(0, 1.2, bars))).astype(float)
    return {
        "Open": close.tolist(),
        "High": (close + 1.4).tolist(),
        "Low": (close - 1.4).tolist(),
        "Close": close.tolist(),
        "Volume": [1000000.0] * bars,
    }


def _fixture(bars: int = _FIXTURE_BARS, seed: int = 11) -> pd.DataFrame:
    """夹具的 DataFrame 形式。"""
    return pd.DataFrame(_kline_data(seed, bars)).astype(float)


def _frame(kline_data: dict) -> pd.DataFrame:
    return pd.DataFrame(kline_data).astype(float)


# 迁移前 Agent 工具产出的末值——迁移后必须逐一相同。
EXPECTED_RSI = 42.37
EXPECTED_MACD = -0.99
EXPECTED_MACD_SIGNAL = -1.22
EXPECTED_MACD_HIST = 0.24
EXPECTED_STOCH_K = 58.43
EXPECTED_STOCH_D = 64.47
EXPECTED_ROC = 0.02
EXPECTED_WILLR = -57.68


class TestAgentIndicatorTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kline_data = _kline_data()
        cls.toolkit = TechnicalTools()

    def _payload(self) -> dict:
        """Agent 五个工具当前的返回内容，合并成一个 dict。"""
        payload = {}
        payload.update(self.toolkit.compute_rsi.invoke(
            {"kline_data": self.kline_data, "period": 14}))
        payload.update(self.toolkit.compute_macd.invoke(
            {"kline_data": self.kline_data}))
        payload.update(self.toolkit.compute_stoch.invoke(
            {"kline_data": self.kline_data}))
        payload.update(self.toolkit.compute_roc.invoke(
            {"kline_data": self.kline_data, "period": 10}))
        payload.update(self.toolkit.compute_willr.invoke(
            {"kline_data": self.kline_data, "period": 14}))
        return payload

    def test_tool_values_are_unchanged(self):
        """Agent 工具的读数必须与迁移前逐一相同。"""
        payload = self._payload()
        self.assertAlmostEqual(payload["rsi"][-1], EXPECTED_RSI, places=6)
        self.assertAlmostEqual(payload["macd"][-1], EXPECTED_MACD, places=6)
        self.assertAlmostEqual(payload["macd_signal"][-1], EXPECTED_MACD_SIGNAL, places=6)
        self.assertAlmostEqual(payload["macd_hist"][-1], EXPECTED_MACD_HIST, places=6)
        self.assertAlmostEqual(payload["stoch_k"][-1], EXPECTED_STOCH_K, places=6)
        self.assertAlmostEqual(payload["stoch_d"][-1], EXPECTED_STOCH_D, places=6)
        self.assertAlmostEqual(payload["roc"][-1], EXPECTED_ROC, places=6)
        self.assertAlmostEqual(payload["willr"][-1], EXPECTED_WILLR, places=6)

    def test_tool_shape_is_unchanged(self):
        """工具返回的形状不能变——LLM 看到的内容必须一模一样。"""
        payload = self._payload()
        self.assertEqual(set(payload.keys()), {
            "rsi", "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d", "roc", "willr",
        })
        # 窗口长度：除 macd 外都是近期 28 根；macd 返回完整序列。
        for key in ("rsi", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "roc", "willr"):
            self.assertEqual(len(payload[key]), indicators.TOOL_WINDOW)
        self.assertEqual(len(payload["macd"]), _FIXTURE_BARS)

    def test_tools_match_the_shared_module_payload(self):
        """工具必须与共享模块的 tool_payload 投影逐字节相同。"""
        payload = self._payload()
        shared = indicators.tool_payload(_frame(self.kline_data))
        self.assertEqual(payload, shared)

    def test_tools_read_from_the_shared_module(self):
        """工具必须真的读共享模块，而不是自己另算一套。"""
        original = indicators.tool_payload
        sentinel = 5.0
        try:
            def fake_tool_payload(df, **kwargs):
                payload = original(df, **kwargs)
                payload["stoch_k"] = [value + sentinel for value in payload["stoch_k"]]
                return payload

            indicators.tool_payload = fake_tool_payload
            payload = self.toolkit.compute_stoch.invoke(
                {"kline_data": self.kline_data})
        finally:
            indicators.tool_payload = original

        baseline = self.toolkit.compute_stoch.invoke(
            {"kline_data": self.kline_data})
        self.assertAlmostEqual(
            payload["stoch_k"][-1], baseline["stoch_k"][-1] + sentinel, places=6
        )

    def test_rsi_tool_still_honours_a_custom_period(self):
        """LLM 可以传 period 参数，工具必须照样尊重它。"""
        default = self.toolkit.compute_rsi.invoke(
            {"kline_data": self.kline_data, "period": 14})["rsi"]
        custom = self.toolkit.compute_rsi.invoke(
            {"kline_data": self.kline_data, "period": 7})["rsi"]
        self.assertNotEqual(default[-1], custom[-1])


class TestFallbackReportAgreesWithReportTable(unittest.TestCase):
    """Agent 的兜底报告与报告表格必须给出同一个 %K / %D。"""

    def test_fallback_report_matches_the_report_table(self):
        """兜底报告的 %K / %D 必须与报告表格一致——两者现在同源。"""
        import re

        from indicator_agent import local_indicator_report
        import report_util

        frame = _fixture(120, seed=3)
        kline_data = report_util.build_kline_dict(frame)

        toolkit = TechnicalTools()
        fallback = local_indicator_report(kline_data, [
            toolkit.compute_macd,
            toolkit.compute_rsi,
            toolkit.compute_roc,
            toolkit.compute_stoch,
            toolkit.compute_willr,
        ])
        match = re.search(r"Stochastic %K/%D: ([-\d.]+)/([-\d.]+)", fallback)
        self.assertIsNotNone(match, "兜底报告里没有 Stochastic 行")

        table = report_util.compute_indicators(kline_data)
        self.assertAlmostEqual(float(match.group(1)), table["stoch_k"], places=2)
        self.assertAlmostEqual(float(match.group(2)), table["stoch_d"], places=2)


if __name__ == "__main__":
    unittest.main()
