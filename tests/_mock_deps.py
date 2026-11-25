"""测试用的依赖替身：让重依赖只在需要它的模块测试期间存在。

为什么需要它：某些测试要导入 `trading_graph` / `web_interface`，而它们的依赖
里有 TA-Lib（C 库）与 langchain（pydantic v1/v2 冲突）。把这些模块 mock 掉能
让测试跑起来，但如果在**导入期**装 mock，就会污染其后导入的所有模块——例如让
`graph_util` 的 `@tool` 装饰器变成 MagicMock，测试结果因此依赖文件名的排序。

这里用 `patch.dict` 把 mock 的生命周期限制在单个测试模块内，并在结束时丢弃
期间被污染的项目模块。
"""

import sys
import types
from unittest.mock import MagicMock, patch

# 预先导入共享指标模块：它要真的 TA-Lib，不能落进任何 mock 窗口。
# 在 mock 窗口内首次导入会把 `indicators.talib` 绑成替身，之后所有指标读数
# 都会变成 MagicMock——而且不同测试模块会拿到不同的模块对象。
import indicators  # noqa: F401  (导入即生效)


class FakeStateGraph:
    """langgraph 的替身：只需能满足 `trading_graph` 的导入与调用形状。"""

    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *args, **kwargs):
        pass

    def add_edge(self, *args, **kwargs):
        pass

    def compile(self):
        return MagicMock()


# 会被 mock 的项目模块——它们在 mock 期间被导入，结束后必须丢弃。
PROJECT_MODULES_TO_PURGE = (
    "trading_graph",
    "web_interface",
    "graph_setup",
    "static_util",
    "graph_util",
)


class MockedDependencies:
    """在 `with` 期间把重依赖换成替身，退出时恢复现场。

    用法：

        with MockedDependencies(["talib", "langchain_qwq"]):
            from trading_graph import TradingGraph
    """

    def __init__(self, module_names, purge_project_modules: bool = True):
        self._module_names = list(module_names)
        self._purge = purge_project_modules
        self._patcher = None

    def _build(self) -> dict:
        mocks = {name: MagicMock() for name in self._module_names}
        if "langchain_core.language_models" in mocks:
            mocks["langchain_core.language_models"].BaseChatModel = object
        if "langgraph.graph" in mocks:
            mocks["langgraph.graph"].END = "__end__"
            mocks["langgraph.graph"].START = "__start__"
            mocks["langgraph.graph"].StateGraph = FakeStateGraph
        if "langchain_core.messages" in mocks:
            messages_module = types.ModuleType("langchain_core.messages")
            messages_module.AIMessage = MagicMock
            messages_module.BaseMessage = MagicMock
            messages_module.HumanMessage = MagicMock
            messages_module.SystemMessage = MagicMock
            messages_module.ToolMessage = MagicMock
            mocks["langchain_core.messages"] = messages_module
        return mocks

    def __enter__(self) -> "MockedDependencies":
        self._patcher = patch.dict(sys.modules, self._build())
        self._patcher.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._patcher is not None:
            self._patcher.stop()
            self._patcher = None
        if self._purge:
            for name in PROJECT_MODULES_TO_PURGE:
                sys.modules.pop(name, None)
