# QuantHarness-Gold

Domain vocabulary for the multi-agent gold-trading analysis system: what the K-line data is, what the agents compute from it, and what the report claims about it.

## Language

**K-line (bar)**:
One candle of price data for an asset at a timeframe: open, high, low, close, and volume. The unit of analysis everywhere in this system.
_Avoid_: Candle, tick, row

**Asset**:
A tradable instrument, identified by a friendly key such as `COMEX` (gold futures) or `SHFE_AU` (Shanghai gold futures).
_Avoid_: Symbol, ticker, instrument

**Source**:
A vendor feed an asset's K-lines can be fetched from — `eastmoney`, `ths`, or `akshare`. Each declares which assets and timeframes it can serve.
_Avoid_: Provider, feed, datasource

**Failover**:
Trying the preferred source first and falling back to other capable sources when it returns no rows, so a vendor outage does not end the run.
_Avoid_: Retry, fallback chain

**Decision**:
The agent graph's final trade instruction for a window of K-lines: direction, risk/reward ratio, forecast horizon, and justification.
_Avoid_: Signal, trade, recommendation

**Indicator**:
A derived series computed from K-lines — RSI, MACD, Stochastic, ROC, Williams %R, moving averages, Bollinger bands, ATR.
_Avoid_: Metric, stat, formula

**%K / %D (Stochastic)**:
The Stochastic oscillator's two lines. Under this project's canonical convention, **%K is the smoothed (slow) %K** — the 3-period simple moving average of the raw fast %K — and **%D is the 3-period moving average of %K**. See ADR-0001.
_Avoid_: Fast %K, raw %K (these mean the unsmoothed oscillator, which this project does not report as %K)

**Slow %K / Fast %K**:
The unsmoothed 14-period Stochastic oscillator is *fast %K*; its 3-period moving average is *slow %K*. Trading platforms typically display *slow %K* and label it "%K".
_Avoid_: %K, %D (reserved for the canonical convention)

**Report**:
The generated Markdown analysis of one run: market snapshot, charts, indicator values and interpretations, the three agent reports, the final decision, and provenance.
_Avoid_: Summary, output, document

**Provenance**:
The record of how each analysis stage was produced — which model, or which deterministic fallback — so a report's numbers can be traced to their source.
_Avoid_: Metadata, lineage, audit trail

**Backtest run**:
A replay of historical K-lines through a decision strategy, producing trades and metrics, recorded as a manifest plus per-decision JSONL.
_Avoid_: Simulation, replay, evaluation

**Strategy**:
A pluggable source of decisions for a backtest run: `rule` (deterministic signals), `recorded` (decisions replayed from a file), or `llm` (the live decision graph).
_Avoid_: Mode, plugin, adapter

**Reproducible**:
A backtest run whose decisions are byte-identical to a previous run with the same inputs — asserted by comparing the decisions hash and the decision fingerprint.
_Avoid_: Deterministic (that describes the property; this describes the achieved guarantee)
