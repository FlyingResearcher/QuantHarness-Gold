# QuantHarness-Gold

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, accessed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles are used verbatim: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Code map

| Module | Role |
|---|---|
| `indicators.py` | Shared Indicator module: TA-Lib for RSI/MACD/Stochastic/ROC/Williams %R; pandas for Bollinger/ATR/MA. Projections: `compute_series` (backtest) / `latest_values` (Report) / one `*_tool_payload` per agent tool. See `docs/adr/0001-stochastic-percent-k-convention.md` |
| `gold_data.py` | K-line fetching across `eastmoney`/`ths`/`akshare`, with failover and OHLCV validation |
| `report_util.py` | Report assembly: indicator values, interpretations, Markdown output |
| `graph_util.py` | `TechnicalTools` (LLM-facing tool wrappers), trendline/pattern detection |
| `trading_graph.py` | LangGraph decision graph and provider/LLM plumbing |
| `validation/` | Backtest engine, metrics, strategies. CLI: `python -m validation.cli` |
| `web_interface.py` | Flask UI. Largest file (1356 lines) |
| `static_util.py` | mplfinance chart rendering |

> **In progress:** the consolidation onto `indicators.py` is not complete. The Report
> (`report_util.py`) and the agent's tool wrappers (`graph_util.py`) now read the
> shared module; the backtest (`validation/indicators.py`) still computes
> Indicators itself and is being migrated in ticket #7. Track progress on issue #1.

## Commands

```bash
# Interpreter: use the venv directly. Bare `python` hits the MS Store stub.
.venv/Scripts/python.exe run_gold_analysis.py COMEX 1d akshare --bars 120
.venv/Scripts/python.exe -m validation.cli run --synthetic 400 --run-id demo
.venv/Scripts/python.exe -m unittest discover -s tests -t .   # 99 tests, ~5s
```

## Conventions

- Indicator keys carry their period: `rsi_14`, `roc_10`, `willr_14`, `atr_14`.
- No `uv`, no `pytest`. Tests are stdlib `unittest`.

