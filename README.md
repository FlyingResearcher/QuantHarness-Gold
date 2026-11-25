# QuantHarness Gold

**Price-Driven Multi-Agent LLM Analysis for Gold Markets**

[Overview](#overview) | [Features](#key-features) | [Quick Start](#quick-start) | [Configuration](#configuration) | [Usage](#usage) | [Troubleshooting](#troubleshooting-and-faq) | [Citation](#research-and-citation)

## Overview

QuantHarness Gold is a research-oriented, multi-agent LLM system for technical analysis of gold markets. It retrieves recent OHLCV data, generates price charts and technical indicators, coordinates specialized analysis agents with LangGraph, and produces a final **LONG** or **SHORT** decision with supporting reports.

This repository is a gold-focused implementation of the [QuantHarness](https://github.com/Y-Research-SBU/QuantHarness) research architecture. The current application supports **COMEX gold futures** and **SHFE gold futures**; it is not the general stock, cryptocurrency, and index application described in earlier versions of this document.

The system does **not** connect to a broker, manage a portfolio, or execute real orders. Market data comes from public sources and should not be treated as a guaranteed real-time or trading-grade feed.

## Key Features

- **Gold-specific market coverage**: Analyze COMEX gold in USD per troy ounce or SHFE main continuous gold futures in CNY per gram.
- **Multiple public data sources**: Use Eastmoney, Tonghuashun (THS), or AKShare, with automatic source fallback and standardized OHLCV output.
- **Technical indicator analysis**: Calculate RSI, MACD, ROC, Stochastic Oscillator, and Williams %R through agent tools. Shared reports also include moving averages, Bollinger Bands, and ATR.
- **Chart pattern recognition**: Generate a recent K-line chart and use a vision-capable LLM to describe relevant price patterns.
- **Trend analysis**: Fit support and resistance lines, annotate the chart, and summarize direction, channel structure, and consolidation.
- **Final decision synthesis**: Combine indicator, pattern, and trend findings into JSON-formatted text containing `forecast_horizon`, `decision`, `justification`, and `risk_reward_ratio`.
- **Execution provenance**: Record whether each analysis stage used an LLM, a vision model, or a local programmatic fallback through `analysis_provenance`.
- **CLI, Web, and Python interfaces**: Run timestamped Markdown reports from the terminal, use the Flask Web UI, or invoke the graph directly.

## Analysis Pipeline

```text
Public gold market data
        |
        v
Standardized OHLCV + integrity checks
        |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v
 Indicator Agent       Pattern Agent         Trend Agent
 RSI/MACD/etc.         K-line chart           Support/resistance
        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                      Decision Agent
                             |
                             v
             LONG/SHORT decision + analysis reports
```

The Decision Agent synthesizes the three analysis reports. There is no independent Risk Agent in the current pipeline, and the system does not place trades.

### Model Slots

QuantHarness Gold uses two configurable LLM slots:

| Slot | Current responsibilities |
| --- | --- |
| `graph_llm` | Indicator tool calling and reporting, Pattern/Trend visual analysis, and the final decision |
| `agent_llm` | Auxiliary Pattern/Trend tool calls when pre-generated chart images are not supplied |

Both the CLI and Web interface pre-generate pattern and trend images. In those standard workflows, `agent_llm` may only be needed by fallback paths, but a model and API key must still resolve for both slots during configuration validation.

For the complete workflow, the `graph_llm` model should support **tool calling and image input**. Some analysis stages can fall back to local programmatic methods with a text-only model, but the final decision still requires an LLM. Check `analysis_provenance` to confirm how each stage actually ran.

## Requirements

- **Python 3.11** is recommended.
- **TA-Lib**, including its native runtime, is required. Installing only a partial Python wrapper is not sufficient.
- Network access to at least one supported market data source and your selected LLM endpoint.
- A valid LLM API key. A vision- and tool-capable model is recommended for `graph_llm`.
- Python packages listed in [requirements.txt](requirements.txt), including Flask, LangChain, LangGraph, pandas, NumPy, SciPy, matplotlib, mplfinance, Pillow, requests, OpenAI/Anthropic SDKs, and `akshare>=1.20.46`.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ChaoPeng13/QuantHarness-Gold.git
cd QuantHarness-Gold
```

### 2. Create an Environment and Install Dependencies

Conda is the simplest option because `conda-forge` can provide TA-Lib and its native dependency together:

```bash
conda create -n quantharness-gold python=3.11 -y
conda activate quantharness-gold
conda install -c conda-forge ta-lib -y
python -m pip install -r requirements.txt
```

Alternatively, use a virtual environment. Install the TA-Lib native library first by following the [official TA-Lib Python installation guide](https://github.com/TA-Lib/ta-lib-python#installation), then run:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the core imports before continuing:

```bash
python -c "import talib, flask, langgraph, akshare; print('Core dependencies imported successfully')"
```

### 3. Configure the LLM

Copy the example configuration and edit `.env`:

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

At minimum, each model slot must resolve a provider, model, and API key. A single shared configuration looks like this:

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-vision-and-tool-capable-model
LLM_API_KEY=your-api-key
LLM_TEMPERATURE=0.1
```

Replace the model placeholder with a model available to your account. See [Configuration](#configuration) for provider-specific settings, split-model examples, and Bob API configuration.

### 4. Run an Analysis

Run the default COMEX daily analysis:

```bash
python run_gold_analysis.py
```

Or start the Web interface:

```bash
python web_interface.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Both commands require working market-data access and valid LLM configuration for a full analysis.

## Configuration

Configuration is loaded by [env_config.py](env_config.py). Values are resolved in this order:

1. System environment variables
2. The project-root `.env` file
3. Built-in defaults, where available

The `.env` file is git-ignored. Restart the process after editing it, and check for stale system environment variables if your changes appear to have no effect.

### Shared and Per-Slot Variables

| Variable | Description |
| --- | --- |
| `LLM_PROVIDER` | Shared provider: `openai`, `anthropic`, `qwen`, `minimax`, `minimax_cn`, `gemini`, `deepseek`, or `bob-api` |
| `LLM_BASE_URL` | Shared endpoint URL, especially for OpenAI-compatible providers |
| `LLM_MODEL` | Shared model name; required unless both slots define their own model |
| `LLM_API_KEY` | Shared API key; required unless both slots define their own key |
| `LLM_TEMPERATURE` | Shared sampling temperature; defaults to `0.1` |
| `LLM_AGENT_PROVIDER` | Optional provider override for `agent_llm` |
| `LLM_AGENT_BASE_URL` | Optional endpoint override for `agent_llm` |
| `LLM_AGENT_MODEL` | Optional model override for `agent_llm` |
| `LLM_AGENT_API_KEY` | Optional key override for `agent_llm` |
| `LLM_AGENT_TEMPERATURE` | Optional temperature override for `agent_llm` |
| `LLM_GRAPH_PROVIDER` | Optional provider override for `graph_llm` |
| `LLM_GRAPH_BASE_URL` | Optional endpoint override for `graph_llm` |
| `LLM_GRAPH_MODEL` | Optional model override for `graph_llm` |
| `LLM_GRAPH_API_KEY` | Optional key override for `graph_llm` |
| `LLM_GRAPH_TEMPERATURE` | Optional temperature override for `graph_llm` |
| `GOLD_DATA_SOURCE` | Preferred market data source: `eastmoney`, `ths`, or `akshare`; defaults to `eastmoney` |

Unset slot-specific values inherit the corresponding shared `LLM_*` value. Provider endpoints and model capabilities depend on your service and account.

### Provider Endpoints

| Provider | Typical base URL or integration |
| --- | --- |
| `openai` | `https://api.openai.com/v1` |
| `anthropic` | Anthropic SDK integration |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `minimax` | `https://api.minimax.io/v1` |
| `minimax_cn` | `https://api.minimaxi.com/v1` |
| `gemini` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `deepseek` | `https://api.deepseek.com/v1`; the repository examples use text models, so configure a separate vision model for full chart analysis |
| `bob-api` | `https://api.bob-api.com/`; the client normalizes the OpenAI-compatible `/v1` endpoint |

### Bob API Example

Bob API can be configured with its dedicated variables instead of repeating shared slot settings:

```ini
LLM_PROVIDER=bob-api
BOB_API_KEY=your-bob-api-key
BOB_API_BASE_URL=https://api.bob-api.com/
BOB_API_MODEL=deepseek-v4.1-flash
LLM_TEMPERATURE=0.1
```

The built-in Bob API defaults are `https://api.bob-api.com/` and `deepseek-v4.1-flash`; `BOB_API_KEY` is still required. The default model name is not a guarantee of tool-calling or vision support. Select a model that provides the capabilities required by your workflow, or override `LLM_GRAPH_*` with a suitable multimodal model.

### Using Different Models for the Two Slots

Slot-specific values take precedence over shared values. For example:

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=your-agent-model
LLM_API_KEY=your-openai-key

LLM_GRAPH_PROVIDER=qwen
LLM_GRAPH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_GRAPH_MODEL=your-vision-capable-qwen-model
LLM_GRAPH_API_KEY=your-dashscope-key
LLM_GRAPH_TEMPERATURE=0.1
```

### Web UI Configuration Note

Changing the provider, endpoint, model, or API key in the Web UI updates the running process configuration and environment. It **does not rewrite `.env`**. Add persistent settings to `.env` yourself and restart the application.

## Gold Assets and Data Sources

### Supported Assets

| Primary symbol | Aliases | Market and units |
| --- | --- | --- |
| `COMEX` | `XAUUSD`, `AU`, `GC` | COMEX gold futures, USD per troy ounce |
| `SHFE_AU` | `AUM` | SHFE main continuous gold futures, CNY per gram |

`XAUUSD` is an alias for the COMEX-anchored series in this repository. It is **not** an independent spot-gold data stream.

### Source and Timeframe Matrix

| Source | COMEX / XAUUSD | SHFE_AU | 4-hour handling |
| --- | --- | --- | --- |
| `eastmoney` | `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1mo` | Same intervals as COMEX | Resampled from `1h` |
| `ths` | Not supported | `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`, `1mo` | Native `4h` data |
| `akshare` | `1d` only | `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` | Resampled from `1h` |

AKShare does not provide weekly or monthly gold data in this integration. The Web interface currently exposes `1m`, `15m`, `1h`, `4h`, `1d`, `1w`, and `1mo`; actual availability still depends on the selected asset and source.

All loaders return a standardized DataFrame with `Datetime`, `Open`, `High`, `Low`, `Close`, and `Volume`. The automatic loader validates OHLC relationships and can fall back to another compatible source. Public endpoints may change, throttle requests, or return incomplete history.

## Usage

### Command-Line Reports

```bash
# Default: COMEX, daily interval, automatic source selection
python run_gold_analysis.py

# COMEX hourly data with Eastmoney preferred
python run_gold_analysis.py COMEX 1h eastmoney

# SHFE 4-hour data from THS, which provides native 4h bars
python run_gold_analysis.py SHFE_AU 4h ths

# COMEX daily data from AKShare, requesting more history
python run_gold_analysis.py COMEX 1d akshare --bars 120

# Run without writing a Markdown report
python run_gold_analysis.py SHFE_AU 1d eastmoney --no-report

# Choose a different report directory
python run_gold_analysis.py COMEX 1d --report-dir reports/custom
```

| Argument | Description |
| --- | --- |
| `asset` | `COMEX` or `SHFE_AU`, including supported aliases; defaults to `COMEX` |
| `timeframe` | A supported interval; defaults to `1d` |
| `source` | Optional preferred source: `eastmoney`, `ths`, or `akshare` |
| `--bars` | Minimum history requested from the loader; defaults to `60`. This is passed as `min_bars` and is not a strict cap on returned rows |
| `--report` / `--no-report` | Enable or disable Markdown report generation; enabled by default |
| `--report-dir` | Report output directory; defaults to `reports/` |

The CLI prints the final decision and indicator, pattern, and trend reports. By default, it also writes a timestamped Markdown report and chart assets under `reports/`, including model details, timings, source attempts, and execution provenance.

Use `python run_gold_analysis.py --help` for the current command-line signature.

### Web Interface

```bash
python web_interface.py
```

1. Open [http://127.0.0.1:5000](http://127.0.0.1:5000).
2. Select a gold asset, timeframe, and compatible data source.
3. Confirm the model provider, endpoint, model names, and API keys.
4. Start the analysis and review the indicator, pattern, trend, and final decision sections.

The Web workflow analyzes the latest **45 bars** returned by the selected source. The Flask server currently binds to `127.0.0.1:5000` and is intended for local development rather than exposed production deployment.

### Python API

The following example fetches data, generates the required chart images, and invokes the same graph used by the CLI:

```python
import gold_data
import static_util
from report_util import build_kline_dict
from trading_graph import TradingGraph


df, used_source, attempts = gold_data.fetch_gold_kline_auto(
    "COMEX",
    "1h",
    min_bars=60,
    preferred="eastmoney",
)
if df is None or df.empty:
    raise RuntimeError(f"No gold data returned; attempts={attempts}")

kline_data = build_kline_dict(df)
info = gold_data.asset_info("COMEX", source=used_source)

initial_state = {
    "kline_data": kline_data,
    "analysis_results": None,
    "messages": [],
    "time_frame": "1h",
    "stock_name": info["name"],
    "pattern_image": static_util.generate_kline_image(kline_data)["pattern_image"],
    "trend_image": static_util.generate_trend_image(kline_data)["trend_image"],
}

final_state = TradingGraph().graph.invoke(initial_state)

print(final_state.get("indicator_report"))
print(final_state.get("pattern_report"))
print(final_state.get("trend_report"))
print(final_state.get("final_trade_decision"))
print(final_state.get("analysis_provenance"))
```

Important return fields include:

| Field | Content |
| --- | --- |
| `indicator_report` | Technical indicator analysis |
| `pattern_report` | Chart pattern analysis |
| `trend_report` | Trend and support/resistance analysis |
| `final_trade_decision` | JSON-formatted decision text, rather than a Python dictionary by default |
| `analysis_provenance` | Per-stage execution method and model information |
| `pattern_image`, `trend_image` | Generated chart image data |

## Data Verification

Use the verification utility to exercise source/asset/timeframe combinations and inspect OHLC integrity:

```bash
python verify_gold_data.py --source eastmoney
python verify_gold_data.py --source ths
python verify_gold_data.py --source akshare
python verify_gold_data.py --source all
```

The utility checks supported combinations, cross-source behavior, 4-hour handling, and writes verification charts under `verification/`. These checks require network access.

Review the individual results, not only the final summary. A `NO DATA` or `SKIP` entry means that a particular combination was not successfully exercised; an overall `PASS` alone does not prove that every source and timeframe returned data.

## Project Structure

```text
.
├── run_gold_analysis.py   # CLI entry point and Markdown report generation
├── web_interface.py       # Flask Web application
├── trading_graph.py       # LLM construction and graph invocation interface
├── graph_setup.py         # LangGraph workflow setup
├── agent_state.py         # Shared graph state
├── indicator_agent.py     # Technical indicator agent
├── pattern_agent.py       # Chart pattern agent
├── trend_agent.py         # Trend analysis agent
├── decision_agent.py      # Final decision synthesis
├── gold_data.py           # Gold asset aliases, loaders, fallback, and validation
├── indicators.py          # Shared TA-Lib and pandas indicator calculations
├── static_util.py         # K-line and trend chart generation
├── graph_util.py          # Chart, trend-line, and agent tool helpers
├── report_util.py         # Report formatting, provenance, and data conversion
├── env_config.py          # .env/environment configuration loader
├── verify_gold_data.py    # Data-source verification utility
├── templates/             # Flask templates
├── assets/                # Documentation and UI images
├── reports/               # Generated CLI reports
├── verification/          # Generated data-verification outputs
└── requirements.txt       # Python dependencies
```

## Troubleshooting and FAQ

### TA-Lib fails to install or import

Use the Conda installation sequence in [Quick Start](#quick-start), or install the native library using the [official TA-Lib instructions](https://github.com/TA-Lib/ta-lib-python#installation) before installing `requirements.txt`. Then verify:

```bash
python -c "import talib; print(talib.__version__)"
```

Make sure the Python interpreter running the application is the same interpreter where the dependency was installed.

### The application reports a missing model or API key

Set `LLM_MODEL` and `LLM_API_KEY`, or provide `LLM_AGENT_MODEL` / `LLM_GRAPH_MODEL` and their corresponding keys. Bob API can instead resolve its model and key from `BOB_API_MODEL` and `BOB_API_KEY` when the relevant provider is `bob-api`.

Remember that system environment variables override `.env`, and configuration changes require a process restart.

### The LLM returns 401, 403, 429, quota, or timeout errors

- Confirm that the API key belongs to the selected provider and endpoint.
- Check that the model name is available to your account.
- For `429` or quota errors, inspect account balance, rate limits, and provider status before retrying.
- For connection or timeout errors, verify network access, proxy settings, and the base URL. Bob API accepts `https://api.bob-api.com/` and normalizes its OpenAI-compatible `/v1` path.
- Do not paste real API keys into logs, reports, screenshots, issues, or commits.

### Pattern or trend analysis does not use image input

Use a `graph_llm` model that supports both image input and tool calling. Text-only models may cause some stages to use local programmatic fallback methods. Inspect `analysis_provenance` and the generated report to see which method was used; the final decision still requires an LLM.

### A data request returns no rows or an unsupported timeframe

Check the [source and timeframe matrix](#source-and-timeframe-matrix). For example, THS does not support COMEX in this integration, and AKShare supports COMEX daily data only. Try another compatible source or omit the CLI source argument to use the configured preference and automatic fallback.

Public data sources can also be temporarily unavailable. Review the CLI's source-attempt output and individual `NO DATA` / `SKIP` results from `verify_gold_data.py`.

### Indicators are missing or contain NaN values

Longer-period indicators need enough valid history. In particular, MACD and the 60-period moving average can remain unavailable for short windows. Request more history in the CLI, for example:

```bash
python run_gold_analysis.py COMEX 1d eastmoney --bars 120
```

The actual history returned still depends on the source. The Web workflow uses 45 bars, so long-period values may be unavailable there.

### Chinese chart labels show missing-glyph warnings

Install a font with Chinese glyph coverage, such as Microsoft YaHei or Noto Sans CJK, make it available to matplotlib, and restart the process. Font availability is operating-system dependent.

### Port 5000 is already in use

Stop the process currently using port `5000`, then restart `web_interface.py`. The Web entry point currently uses the fixed address `127.0.0.1:5000`; it does not expose a CLI port option.

### Web UI model settings disappear after restart

This is expected. Web UI changes apply to the current process and are not persisted to `.env`. Store permanent settings in `.env` or system environment variables.

### Does `XAUUSD` retrieve an independent spot feed?

No. `XAUUSD` currently resolves to the same COMEX-anchored data as `COMEX`.

### Can the current application analyze stocks or cryptocurrencies?

The maintained data and analysis workflow in this repository is gold-specific. Use `COMEX` or `SHFE_AU` and their documented aliases.

### Does QuantHarness Gold place real trades?

No. It produces research outputs only and has no broker integration or order-execution path.

## Security and Operational Notes

- Keep `.env` and API credentials out of version control. Rotate any key that may have been exposed.
- Treat generated decisions and reports as untrusted research output, not executable trading instructions.
- Public market-data endpoints can change without notice and may have rate limits, delays, or missing history.
- The included Flask server is a local development interface, not a hardened production deployment.

## Research and Citation

The original QuantHarness paper describes the research architecture behind this project. Refer to the paper for the original methodology and experimental results; this README describes the current gold-focused repository implementation.

- Paper: [QuantHarness: Price-Driven Multi-Agent LLMs for High-Frequency Trading](https://arxiv.org/abs/2509.09995)
- Project website: [Y-Research-SBU.github.io/QuantHarness](https://Y-Research-SBU.github.io/QuantHarness)

```bibtex
@article{xiong2025quantharness,
  title={QuantHarness: Price-Driven Multi-Agent LLMs for High-Frequency Trading},
  author={Fei Xiong and Xiang Zhang and Aosong Feng and Siqi Sun and Chenyu You},
  journal={arXiv preprint arXiv:2509.09995},
  year={2025}
}
```

## Contributing

1. Fork the repository and create a focused feature branch.
2. Keep changes consistent with the current gold data and agent workflow.
3. Run the relevant existing checks and update documentation when behavior changes.
4. Do not commit `.env`, API keys, caches, or generated private reports.
5. Open a pull request describing the change and how it was validated.

## Acknowledgements

### Special Thanks to the QuantHarness Team

This gold-focused project is deeply indebted to the original **QuantHarness** research team — Fei Xiong, Xiang Zhang, Aosong Feng, Siqi Sun, and Chenyu You — whose pioneering work demonstrated that multi-agent LLM architectures could produce coherent, technically grounded market analysis. Their paper *"QuantHarness: Price-Driven Multi-Agent LLMs for High-Frequency Trading"* (arXiv:2509.09995) introduced the core insight that chart images, technical indicators, and structured decision synthesis could be orchestrated through LangGraph into a compelling research system.

We gratefully stand on their shoulders. This repository adapts and extends the QuantHarness architecture for gold markets, adding multi-source OHLCV collection (Eastmoney / THS / AKShare), a dual-slot LLM configuration model, execution provenance tracking, and a gold-specific data normalization layer. The original vision — letting multiple specialized LLM agents collaborate on real market data rather than running as monolithic prompts — remains the heartbeat of this codebase.

## Disclaimer

This software is for educational and research purposes only. It does not provide financial advice, guaranteed market forecasts, or real trading execution. Public data and LLM outputs may be delayed, incomplete, or incorrect. Always perform independent research and consult a qualified financial professional before making investment decisions.
