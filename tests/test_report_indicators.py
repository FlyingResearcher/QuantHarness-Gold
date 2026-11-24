"""报告指标一致性测试：报告表格与 Agent 散文的 %K / %D 必须是同一个数。

这个测试锁住的是一次真实缺陷：合并前，报告的指标表用快速 %K，
而 Agent 的工具用慢速 %K，同一份报告里两处数字对不上。
"""

import os
import re
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from graph_util import TechnicalTools
from indicator_agent import local_indicator_report
from report_util import build_kline_dict, generate_markdown_report


def _fixture(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """确定性行情夹具。"""
    rng = np.random.default_rng(seed)
    close = (100 + np.cumsum(rng.normal(0, 1, n))).astype(float)
    return pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=n, freq="1D"),
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(n, 1e6),
        }
    )


def _agent_indicator_report(kline_data: dict) -> str:
    """Agent 的确定性兜底报告——不需要 LLM。"""
    toolkit = TechnicalTools()
    tools = [
        toolkit.compute_macd,
        toolkit.compute_rsi,
        toolkit.compute_roc,
        toolkit.compute_stoch,
        toolkit.compute_willr,
    ]
    return local_indicator_report(kline_data, tools)


def _generate_report(kline_data: dict, indicator_report: str) -> str:
    """用固定夹具生成一份完整报告。"""
    frame = pd.DataFrame(kline_data)
    return generate_markdown_report(
        args=SimpleNamespace(asset="COMEX", timeframe="1d", source=None),
        info={"name": "COMEX"},
        used_source="eastmoney",
        attempts=[{"source": "eastmoney", "label": "EastMoney", "rows": len(frame)}],
        kline_data=kline_data,
        final_state={
            "indicator_report": indicator_report,
            "pattern_report": "模式报告",
            "trend_report": "趋势报告",
            "final_trade_decision": '{"decision": "LONG", "risk_reward_ratio": 2.0}',
            "analysis_provenance": {},
        },
        pre_dt=1.0,
        analysis_dt=2.0,
        post_dt=3.0,
        report_dir=None,
        stamp=None,
        models={},
    )


def _table_stochastic(markdown: str) -> tuple:
    """从报告的指标表里取出 %K / %D。"""
    match = re.search(r"\| Stochastic\(14\) \| %K=([-\d.]+)<br>%D=([-\d.]+) \|", markdown)
    assert match is not None, "报告里没有 Stochastic 表格行"
    return float(match.group(1)), float(match.group(2))


def _prose_stochastic(indicator_report: str) -> tuple:
    """从 Agent 散文里取出 %K / %D。"""
    match = re.search(r"Stochastic %K/%D: ([-\d.]+)/([-\d.]+)", indicator_report)
    assert match is not None, "Agent 报告里没有 Stochastic 行"
    return float(match.group(1)), float(match.group(2))


class TestReportIndicatorConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = _fixture()
        cls.kline_data = build_kline_dict(cls.frame)
        cls.agent_report = _agent_indicator_report(cls.kline_data)
        cls.markdown = _generate_report(cls.kline_data, cls.agent_report)

    def test_table_and_agent_prose_agree_on_percent_k(self):
        """报告表格的 %K / %D 必须与 Agent 散文里的完全一致。

        合并前这是失败的：表格用快速 %K，Agent 用慢速 %K。
        """
        table_k, table_d = _table_stochastic(self.markdown)
        prose_k, prose_d = _prose_stochastic(self.agent_report)
        self.assertAlmostEqual(table_k, prose_k, places=2)
        self.assertAlmostEqual(table_d, prose_d, places=2)

    def test_report_states_the_slow_percent_k_convention(self):
        """报告的速查表必须描述慢速 %K 口径，不能与代码相矛盾。"""
        self.assertIn("慢速 %K", self.markdown)
        # 旧描述把快速 %K 直接叫做 %K，再把它的 SMA3 叫做 %D——正是错位的根源。
        self.assertNotIn(
            "(C − L14) / (H14 − L14) × 100；%D = SMA3(%K)", self.markdown
        )

    def test_overbought_thresholds_are_unchanged(self):
        """超买超卖阈值保持 80/20 不变。"""
        self.assertIn("超买 >80 / 超卖 <20", self.markdown)

    def test_report_does_not_claim_a_crossover_it_cannot_detect(self):
        """报告不能声称检测到了 %K/%D 交叉——它没有这个能力。"""
        self.assertNotIn("金叉形成", self.markdown)
        self.assertNotIn("死叉形成", self.markdown)

    def test_bollinger_and_atr_rows_are_present(self):
        """布林带与 ATR 的读数仍然出现在报告里。"""
        self.assertIn("| Bollinger(20,2) |", self.markdown)
        self.assertIn("| ATR(14) |", self.markdown)


class TestReportReadsTheSharedModule(unittest.TestCase):
    """报告必须真的读共享模块，而不是自己另算一套。"""

    def test_report_values_come_from_the_shared_module(self):
        """替换共享模块的计算结果，报告的读数必须随之改变。

        这验证的是"报告读共享模块"这件事本身——只断言两个数字相等，
        无法区分"读同一份"与"各自独立算出同样的值"。
        """
        import indicators
        import report_util

        frame = _fixture()
        kline_data = build_kline_dict(frame)

        original = indicators.compute_series
        sentinel = 12.34
        try:
            def fake_compute_series(df):
                series = {key: original(df)[key] for key in original(df)}
                series["stoch_k"] = series["stoch_k"] + sentinel
                return series

            indicators.compute_series = fake_compute_series
            values = report_util.compute_indicators(kline_data)
        finally:
            indicators.compute_series = original

        baseline = report_util.compute_indicators(kline_data)
        self.assertAlmostEqual(
            values["stoch_k"], baseline["stoch_k"] + sentinel, places=8
        )


if __name__ == "__main__":
    unittest.main()
