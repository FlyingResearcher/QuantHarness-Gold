"""回测侧的 TA-Lib 依赖守卫：共享指标模块必须拿到真的 TA-Lib。

历史问题：某个测试模块曾在**导入期**给 TA-Lib 装上 mock，于是回测自己的测试
拿到的是替身，指标读数全是 `MagicMock`。这里把两件事钉住：

  1. 回测用的指标模块（`validation.indicators`）解析出来的 `talib` 是真库，
     且与共享模块持有的是同一个模块对象；
  2. 没有任何测试模块把 TA-Lib 重新写进 mock 清单——直接扫源码，而不是导入
     模块后读常量：这样既能抓住 `MOCK_MODULES` 列表，也能抓住在
     `MockedDependencies([...])` 里顺手写进去的内联写法，还不会在扫描时
     触发其它测试模块的模块级装配（例如需要 `.env` 的那一个）。
"""

import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, NonCallableMock, patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import indicators
import validation.indicators

TESTS_DIR = Path(__file__).resolve().parent
TALIB = "talib"


def _is_real_talib(module) -> bool:
    """真库：是模块对象（不是 mock），且带字符串版本号。"""
    if isinstance(module, NonCallableMock) or not isinstance(module, types.ModuleType):
        return False
    return isinstance(getattr(module, "__version__", None), str)


class TestBacktestResolvesTheRealTalib(unittest.TestCase):
    """AC：回测侧的指标模块解析到真库，而不是替身。"""

    def test_backtest_indicators_module_holds_the_real_talib(self):
        resolved = validation.indicators.shared_indicators.talib
        self.assertTrue(_is_real_talib(resolved), f"TA-Lib 被替换了：{resolved!r}")
        self.assertIs(resolved, indicators.talib)
        self.assertTrue(
            resolved.__file__.replace("\\", "/").endswith("talib/__init__.py"),
            f"TA-Lib 来自非预期位置：{resolved.__file__}",
        )

    def test_backtest_actually_computes_with_that_library(self):
        """回测算出来的 Stochastic 来自真库：末值有限，且与共享模块一致。"""
        from validation.data import synthetic_bars

        bars = synthetic_bars(180, seed=11)
        shared = indicators.compute_series(bars)
        engine = validation.indicators.RuleSignalEngine(
            bars, validation.indicators.BacktestConfig(warmup_bars=30, min_bars=60)
        )
        self.assertEqual(
            engine._row(100)["stoch_k"],
            validation.indicators._safe(shared["stoch_k"].iloc[100]),
        )
        slowk = validation.indicators.shared_indicators.talib.STOCH(
            bars["High"].to_numpy(dtype=float),
            bars["Low"].to_numpy(dtype=float),
            bars["Close"].to_numpy(dtype=float),
        )[0]
        self.assertIsInstance(slowk, np.ndarray)
        self.assertTrue(np.isfinite(slowk[-1]))


def _talib_mock_sites(path: Path) -> list:
    """扫一个测试文件，返回把 talib 写进 mock 清单的位置（行号 + 写法）。

    识别两种写法：
      * ``MOCK_MODULES = [..., "talib", ...]``
      * ``MockedDependencies(["talib", ...])``
    只看代码结构，不执行文件——所以模块级装配（读 .env 之类）不会被打扰。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites = []

    def mentions_talib(node) -> bool:
        return any(
            isinstance(elt, ast.Constant) and elt.value == TALIB for elt in node.elts
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "MOCK_MODULES" in names and isinstance(node.value, (ast.List, ast.Tuple)):
                if mentions_talib(node.value):
                    sites.append((node.lineno, "MOCK_MODULES"))
        elif isinstance(node, ast.Call):
            func = node.func
            func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if func_name == "MockedDependencies" and node.args:
                first = node.args[0]
                if isinstance(first, (ast.List, ast.Tuple, ast.Set)) and mentions_talib(first):
                    sites.append((node.lineno, "MockedDependencies([...])"))
    return sites


class TestNoTestModuleMocksTalib(unittest.TestCase):
    """回归守卫：TA-Lib 不得重新进入任何测试文件的 mock 清单。"""

    def test_no_test_file_lists_talib_as_a_mock(self):
        offenders = {}
        for path in sorted(TESTS_DIR.glob("*.py")):
            sites = _talib_mock_sites(path)
            if sites:
                offenders[path.name] = sites
        self.assertEqual(offenders, {}, f"这些测试文件又把 TA-Lib 标成了 mock：{offenders}")

    def test_the_scanner_catches_both_forms(self):
        """自检：扫描器对两种写法都有效，不是恒真断言。"""
        sample = (
            'MOCK_MODULES = ["langchain_qwq", "talib"]\n'
            'with MockedDependencies(["talib"]):\n'
            "    pass\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.py"
            path.write_text(sample, encoding="utf-8")
            forms = {form for _, form in _talib_mock_sites(path)}
        self.assertEqual(forms, {"MOCK_MODULES", "MockedDependencies([...])"})


class TestDetectorRejectsAMock(unittest.TestCase):
    """自检：检测器真能把替身认出来。"""

    def test_detector_rejects_a_mocked_talib(self):
        with patch.dict(sys.modules, {"talib": MagicMock()}):
            self.assertFalse(_is_real_talib(sys.modules["talib"]))
            self.assertTrue(_is_real_talib(indicators.talib))


if __name__ == "__main__":
    unittest.main()
