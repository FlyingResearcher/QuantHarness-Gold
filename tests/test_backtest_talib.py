"""回测侧的 TA-Lib 依赖守卫：共享指标模块必须拿到真的 TA-Lib。

历史问题：某个测试模块曾在**导入期**给 TA-Lib 装上 mock，于是回测自己的测试
拿到的是替身，指标读数全是 `MagicMock`。这里把两件事钉住：

  1. 共享指标模块（回测将复用它）持有的 `talib` 是真库，而不是替身；
  2. 没有任何测试模块重新把 TA-Lib 列进 mock 清单——因为 mock 会在
     `tearDownModule` 复原，只靠运行时断言抓不到「有人把 talib 加回
     mock 列表」这种回归。
"""

import importlib
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, NonCallableMock, patch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import indicators


def _is_real_talib(module) -> bool:
    """真库：是模块对象（不是 mock），且带字符串版本号。"""
    if isinstance(module, NonCallableMock) or not isinstance(module, types.ModuleType):
        return False
    return isinstance(getattr(module, "__version__", None), str)


class TestBacktestTalibIsReal(unittest.TestCase):
    def test_shared_module_resolves_the_real_talib(self):
        """共享模块里的 talib 就是 sys.modules 里的那个真库。"""
        talib = indicators.talib
        self.assertTrue(_is_real_talib(talib), f"TA-Lib 被替换了：{talib!r}")
        self.assertIs(talib, sys.modules.get("talib"))
        self.assertTrue(
            talib.__file__.replace("\\", "/").endswith("talib/__init__.py"),
            f"TA-Lib 来自非预期位置：{talib.__file__}",
        )

    def test_backtest_side_import_keeps_talib_real(self):
        """回测包导入后，内存里的 TA-Lib 仍是同一个真库。

        回测尚未直接调用 TA-Lib（那是下一张票的事），但它一旦改走共享模块，
        拿到的必须是这个真库——这里先把「导入不会替换它」钉住。
        """
        importlib.import_module("validation.indicators")
        self.assertIs(sys.modules.get("talib"), indicators.talib)
        self.assertTrue(_is_real_talib(sys.modules["talib"]))

    def test_real_talib_actually_computes(self):
        """真库返回 ndarray；替身只会返回 MagicMock。"""
        close = np.linspace(100.0, 160.0, 60)
        high, low = close + 1.5, close - 1.5
        slowk, slowd = indicators.talib.STOCH(high, low, close)
        self.assertIsInstance(slowk, np.ndarray)
        self.assertIsInstance(slowd, np.ndarray)
        self.assertTrue(np.isfinite(slowk[-1]))
        self.assertTrue(np.isfinite(slowd[-1]))


def _test_modules() -> dict:
    """导入 tests 包下的每个模块，返回 {模块名: 模块对象}。"""
    import pkgutil

    import tests as tests_package

    loaded = {}
    for info in pkgutil.iter_modules(tests_package.__path__):
        name = f"{tests_package.__name__}.{info.name}"
        loaded[name] = importlib.import_module(name)
    return loaded


class TestNoTestModuleMocksTalib(unittest.TestCase):
    """回归守卫：TA-Lib 不得重新进入任何测试模块的 mock 清单。

    逐个模块检查而不是写死名单——新加的测试模块若又把 TA-Lib 标成 mock，
    这里会立刻变红，而不是等到某次顺序巧合下的指标读数失真才发现。
    """

    def test_importing_every_test_module_keeps_talib_real(self):
        """导入全部测试模块不会把 TA-Lib 换成替身。

        老缺陷的形态正是「某个测试模块在导入期装 mock」。导入本身才是要盯住
        的动作，所以在导入全部模块之后再验一次真库。
        """
        _test_modules()
        self.assertTrue(_is_real_talib(sys.modules.get("talib")))
        self.assertIs(sys.modules.get("talib"), indicators.talib)

    def test_mock_lists_do_not_contain_talib(self):
        modules = _test_modules()
        self.assertIn("tests.test_minimax_provider", modules)
        for name, module in modules.items():
            with self.subTest(module=name):
                self.assertNotIn("talib", getattr(module, "MOCK_MODULES", ()))


class TestDetectorRejectsAMock(unittest.TestCase):
    """自检：检测器真能把替身认出来，不是一个恒真的断言。"""

    def test_detector_rejects_a_mocked_talib(self):
        with patch.dict(sys.modules, {"talib": MagicMock()}):
            self.assertFalse(_is_real_talib(sys.modules["talib"]))
            self.assertTrue(_is_real_talib(indicators.talib))


if __name__ == "__main__":
    unittest.main()