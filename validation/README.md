# 决策正确性验证模块（Backtest / Decision Validation）

面向 QuantHarness-Gold 现有决策系统的**独立验证模块**：用历史数据全量回放决策逻辑，
从准确率、收益、风险、响应时效四个维度量化评估，自动标记异常决策，并输出可复现的
HTML 报表。模块与线上决策流程低耦合，可独立运行、增量回测，不修改 `trading_graph.py` /
`decision_agent.py` 的任何逻辑。

---

## 1. 快速开始

### 1.1 命令行

```bash
# 在线取数（gold_data 多源自动降级）
python -m validation.cli run --asset COMEX --timeframe 1d --run-id comex_1d

# 离线：本地 CSV（列名 Datetime/Open/High/Low/Close[/Volume]）
python -m validation.cli run --csv record.csv --run-id demo_rule --fee 0.01 --slippage 0.01

# 离线：确定性合成行情（无需网络 / 无需 API Key）
python -m validation.cli run --synthetic 400 --run-id demo_synthetic

# 增量回测（复用同 run_id 已计算决策，只算新增部分）
python -m validation.cli run --synthetic 400 --run-id demo_synthetic --incremental

# 可复现性校验（同数据同参数连跑两次，比对决策哈希）
python -m validation.cli check --synthetic 400 --run-id reproducibility

# 查看历史 / 单次结果
python -m validation.cli list
python -m validation.cli show demo_synthetic
```

### 1.2 Python 调用

```python
from validation import BacktestConfig, run_backtest

payload = run_backtest(
    BacktestConfig(asset="COMEX", timeframe="1d", run_id="comex_1d", fee_pct=0.01),
)
print(payload["metrics"]["accuracy"]["accuracy"])
print(payload["files"] if "files" in payload else payload["run_id"])
```

### 1.3 可视化界面

```bash
# 独立启动（端口 5001）
python -m validation.cli serve --port 5001
# 打开 http://127.0.0.1:5001/validation/
```

也可以挂载到现有 Web 界面：`web_interface.py` 已用 `try/except` 注册 Blueprint，
启动 `python web_interface.py` 后访问 `http://127.0.0.1:5000/validation/` 即可；
即使挂载失败也只打印提示，不影响主流程。

---

## 2. 模块结构

```
validation/
├── config.py       # BacktestConfig：全部参数 + 校验 + 决策指纹
├── data.py         # 在线 / CSV / 合成数据统一加载、数据指纹、窗口指纹
├── indicators.py   # 纯 pandas/numpy 指标与规则信号（不依赖 TA-Lib）
├── strategies.py   # 可插拔策略：rule / recorded / llm
├── engine.py       # 逐 bar 回放、成交模拟、增量缓存、决策哈希
├── metrics.py      # 准确率 / 收益 / 风险 / 时效 / 基准 / 综合评分
├── anomalies.py    # 异常决策自动标记与归类
├── report.py       # 自包含 HTML / Markdown 报表（内联 SVG，无 CDN）
├── runner.py       # 编排、持久化、增量、可复现校验
├── cli.py          # 命令行入口
├── web_app.py      # Flask Blueprint + 独立应用
└── templates/validation.html
```

运行时产物：`validation_runs/<run_id>/`（已加入 `.gitignore`）

```
validation_runs/<run_id>/
├── manifest.json    # 配置、数据指纹、决策哈希、耗时、可复现标记
├── decisions.jsonl  # 每个决策一行（含缓存 key），供增量回放
├── result.json      # 指标、异常、曲线、成交明细（不含决策体）
├── report.html      # 自包含可视化报表
└── report.md        # Markdown 摘要
```

---

## 3. 回放与成交模型

| 环节 | 规则 |
|---|---|
| 决策时点 | 第 i 根 K 线**收盘后**，只使用 `[0, i]` 的数据 |
| 成交时点 | 第 i+1 根 K 线**开盘价**（`execution_lag_bars = 1`） |
| 成本 | 单边手续费 + 单边滑点；入场与出场分别计提 |
| 止损/止盈 | 止损 = ATR(14) × `stop_atr_mult`；止盈 = 风险距离 × `risk_reward`（或 `target_atr_mult`） |
| 同 bar 冲突 | 同时触发止损与止盈时按**止损**处理（悲观撮合） |
| 超时平仓 | 达到 `horizon_bars` 仍未触发止损/止盈，按收盘价平仓 |
| 正确性判定 | 按**毛收益方向**判定（不受费用干扰），零收益记为平局并从准确率分母剔除 |

> 规则策略复刻决策智能体在提示词中声明的策略：技术指标 45% + 趋势通道 35% + 形态 20%，
> 只输出 LONG / SHORT（与线上一致，不允许 HOLD）；多空僵持时以趋势通道方向兜底。
> 这是**确定性代理**，用于可复现的回归验证；如需评估真实 LLM 决策，使用 `recorded` 或 `llm` 策略。

---

## 4. 参数说明（BacktestConfig）

| 参数 | 默认 | 说明 |
|---|---|---|
| `asset` / `timeframe` | COMEX / 1d | 与 gold_data 同一口径；周期支持 1m~1mo |
| `source` | None | eastmoney / ths / akshare；None 用默认源并自动降级 |
| `start` / `end` | None | 回测区间（含端点） |
| `strategy` | rule | rule / recorded / llm |
| `horizon_bars` | 1 | 预测与最长持有根数 |
| `warmup_bars` | 30 | 最小历史窗口（≥26，MACD 需要） |
| `fee_pct` / `slippage_pct` | 0 / 0 | 单边成本（%） |
| `use_stops` | True | 是否启用 ATR 止损/止盈 |
| `stop_atr_mult` | 2.0 | 止损 ATR 倍数 |
| `target_atr_mult` | 0 | 0 表示由 `risk_reward` 推导 |
| `risk_reward` | 1.5 | 目标风险收益比 |
| `benchmark` | buy_hold | buy_hold（同区间满仓做多）/ flat（空仓） |
| `error_return_threshold_pct` | 0.10 | 方向错误的「实质性」阈值 |
| `min_confidence` | 0.35 | 低置信度阈值 |
| `max_trade_loss_pct` | 2.0 | 单笔风控上限 |
| `max_repeat_signals` | 3 | 连续同向决策容忍次数 |
| `late_entry_bars` | 1 | 入场后多少根内止损视为迟滞 |
| `run_id` / `incremental` | default / False | 结果命名与增量复用 |
| `recorded_path` | None | `strategy=recorded` 时的决策文件（CSV/JSON/JSONL） |

`recorded` 文件要求：`Datetime` 列 + `decision`（或 `direction`/`signal`）列，
方向支持 LONG/SHORT/BUY/SELL/做多/做空；可选 `confidence`、`risk_reward_ratio`。

---

## 5. 指标体系

### 5.1 决策准确率
- 方向准确率、平衡准确率、多/空准确率、精确率、召回率、F1
- 混淆矩阵（TP/FP/FN/TN）
- 置信度校准：正确/错误决策的平均置信度、置信度落差、ECE（5 分箱）

### 5.2 收益表现
- 累计净收益、年化收益（近似）、平均每笔收益、胜率、盈亏比、盈利因子、期望收益
- Sharpe、Sortino、Calmar、最佳/最差单笔

### 5.3 风险指标
- 最大回撤及持续笔数、波动率（逐笔/年化）、下行标准差
- VaR(95%)、CVaR(95%)、单笔最大亏损、最大连续亏损、持仓暴露比例

### 5.4 响应时效性
- 平均 / P50 / P95 / 最大决策耗时（毫秒）
- 执行滞后（固定 1 根）、平均持仓、到达最大浮盈所需根数及其占持仓比例
- 迟滞入场笔数与比例、决策覆盖率、增量缓存复用率

### 5.5 基准对比
- 基准累计收益、超额收益、Beta、Alpha、跟踪误差、信息比率
- 逐笔收益与「同区间满仓做多」的区间收益对比，成本口径一致

### 5.6 综合评分（0-100）

```
0.40 × 准确率
+ 0.25 × clip(50 + 超额收益, 0, 100)
+ 0.20 × clip(100 - 4 × 最大回撤%, 0, 100)
+ 0.15 × clip(100 - 60 × 迟滞率 - min(40, P95耗时/50), 0, 100)
```

> 评分仅用于面板横向比较，口径固定、可复算，不构成投资建议。

---

## 6. 异常决策自动标记

| 代码 | 归类 | 严重度 | 触发条件 |
|---|---|---|---|
| `WRONG_DIRECTION` | 错误 | high | 方向做反且毛亏 ≥ `error_return_threshold_pct` |
| `MARGINAL_ERROR` | 错误 | low | 方向做反但幅度小于阈值 |
| `RISK_BREACH` | 错误 | high | 单笔净亏损 > `max_trade_loss_pct` |
| `ENGINE_ERROR` | 错误 | high | 决策计算抛错（引擎已隔离，回测继续） |
| `SIGNAL_CONFLICT` | 偏差 | medium | 决策方向与 ≥2 个分项信号（指标/趋势/形态）相反 |
| `LATE_ENTRY` | 偏差 | medium | 入场后 `late_entry_bars` 根内即触发止损 |
| `LOW_CONFIDENCE` | 偏差 | low | 置信度 < `min_confidence` |
| `DUPLICATE_SIGNAL` | 偏差 | low | 连续同向超过 `max_repeat_signals`（每段只记一次） |
| `DATA_ANOMALY` | 数据 | medium | 决策窗口内存在缺口/缺失/零振幅/重复时间戳 |

报表按严重度输出 Top 样本，并给出分类计数、错误/偏差/数据三类汇总。

---

## 7. 增量回测与可复现

- **决策缓存键** = `时间戳 | 历史窗口指纹`。窗口指纹由最近 `warmup_bars` 根 K 线内容哈希得到，
  数据被供应商修订后缓存自动失效。
- **决策指纹** = 影响决策的参数哈希（策略、权重、窗口等）；费用/滑点/止损变化不影响缓存复用，
  但会重新计算绩效。
- **决策序列哈希**（`decisions_hash`）由每条决策的方向、置信度、评分、风险收益比生成，
  同数据同参数重跑必须一致；CLI `check` 会连跑两次比对。
- `manifest.json` 记录引擎版本、数据指纹、参数指纹、决策哈希与耗时，便于归档与审计。
- `llm` 策略是非确定性的，`reproducible` 会被标记为 `false`，不纳入可复现断言。

---

## 8. 与主系统的耦合关系

- **不修改** `trading_graph.py` / `decision_agent.py` / `agent_state.py`；
- `rule` / `recorded` 策略不导入 LangGraph、不需要 API Key、不产生外部副作用；
- `llm` 策略在 `prepare()` 时才延迟导入 `TradingGraph`，与线上使用同一套 `.env` 配置；
- Web 集成仅一行 `register_blueprint`，且被 `try/except` 包裹；验证页面不参与线上决策请求链路；
- 数据层通过 `gold_data` 的公开函数取数，未修改其行为。

---

## 9. 测试

```bash
# 全部验证模块测试（33 个用例，覆盖正常/边界/异常）
python -m unittest discover -s tests -p "test_validation_*.py" -v
```

| 文件 | 覆盖点 |
|---|---|
| `tests/test_validation_engine.py` | 可复现、无未来函数、持有期、止损出场、增量复用、参数变更失效、已记录策略、数据不足 |
| `tests/test_validation_metrics.py` | 手算准确率/盈亏比/最大回撤/VaR-CVaR/Beta/空集边界/时效 |
| `tests/test_validation_anomalies.py` | 九类异常逐一触发、计数与严重度、干净样本零标记 |
| `tests/test_validation_report_store.py` | 产物完整性、HTML 转义、空 payload、JSON 可序列化、Web 端点与 400 错误 |

> 测试全部使用确定性合成数据，不访问网络、不调用 LLM。

---

## 10. 已知限制与扩展方向

1. 规则策略是线上 LLM 决策的**确定性代理**，用于回归验证；评估真实决策请导出决策记录后用
   `recorded` 回放，或使用 `llm` 策略（慢、需 Key、非确定性）。
2. 年化指标按「平均持仓根数」折算，样本量较小时波动较大，仅作横向参考。
3. 成交模拟未包含盘口深度、部分成交与保证金/杠杆，适用于策略逻辑层面的正确性评估。
4. 新增策略只需继承 `DecisionStrategy` 并实现 `prepare()` / `decide()`，回测引擎无需改动。
