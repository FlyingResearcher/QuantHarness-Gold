# Stochastic %K is the smoothed (slow) %K

> **Status: accepted, implemented.** The shared module `indicators.py` implements
> this convention (ticket #2); the Report reads from it (ticket #3), the agent's tool
> wrappers read from it (ticket #4), and the backtest reads from it (ticket #7).

The project reports Stochastic **%K as the smoothed value** — the 3-period simple moving average of the raw 14-period oscillator — and **%D as the 3-period moving average of %K**. This matches TA-Lib's `STOCH(fastk_period=14, slowk_period=3, slowd_period=3)` output and the convention mainstream trading platforms display.

## Context

Three implementations of the same indicators existed with two different %K conventions. `report_util.compute_indicators()` and `validation/indicators.py` computed the **fast** %K (unsmoothed) and labelled its 3-period average "%D". `graph_util.TechnicalTools.compute_stoch()` used TA-Lib's default and therefore reported the **slow** %K as "%K". All three have since been migrated onto the shared module.

The result was a live correctness defect, not a cosmetic one. A single generated report contained both conventions: the indicator table printed `%K=18.89 %D=14.68` while the agent's prose in the same file said `%K 14.68 versus %D 10.49` — the report's "%D" was numerically the agent's "%K". The report's own reference table documented "%D = SMA3(%K)", which described the fast convention. Because Stochastic is the one indicator the report's overbought/oversold thresholds key off, the mislabelling changed how every generated report classified overbought and oversold conditions.

## Decision

%K is the slow %K (TA-Lib's default `STOCH(14, 3, 3)` output). All indicator computation lives in one module, `indicators.py`, which uses TA-Lib for the five indicators it owns (RSI, MACD, Stochastic, ROC, Williams %R) and pandas for Bollinger, ATR, and moving averages.

Two consequences are deliberate:

- **TA-Lib's `BBANDS` uses population stdev (ddof=0) while the repo's pandas code used sample stdev (ddof=1)**, so Bollinger stays in pandas. Moving it would shift `%B` by up to 1.9 for no benefit.
- **The validation backtest's Stochastic thresholds are recalibrated from 85/15 to 83/17.** Smoothing shortens the time %K spends in the extremes, so keeping 85/15 would have cut the fraction of bars where the Stochastic vote fires. On the 400-bar synthetic fixture (seed 7) the extreme-zone rate was 43.09% (159/369) under fast %K × 85/15, 40.11% (148/369) under slow %K × 85/15, and 44.99% (166/369) under slow %K × 83/17. **83/17 was the pair specified for this migration, and it overshoots the old rate by 1.9pp.** Note for anyone revisiting this: on this one fixture 83/16 and 83/15 both land on 159/369 = 43.09% exactly, and 84/16 sits 1.6pp low. Those alternatives have not been checked across other fixtures, so switching to one of them is a calibration change that needs its own measurement — not an obvious correction of a mistake.

## Consequences

The backtest's rule-strategy accuracy on the 400-bar synthetic fixture (seed 7, horizon 1) moved from 0.569106 to 0.563686 once the backtest was migrated — the residual difference is the smoothed oscillator's slightly different timing, not a loss of signal. At horizon 2 the accuracy is 0.607046 both before and after; the decision sequence changes at only 4 of 369 bars.

**RSI and MACD values also change.** TA-Lib seeds its exponential averages with a simple moving average, while the repo's hand-rolled `ewm(adjust=False)` starts from the first value. Migrating RSI and MACD to TA-Lib therefore shifts the report's readings — RSI by up to ~2.75 on a short fixture. This is part of the decision: one backend for each indicator means one number per indicator. The shared module's tests pin the new values so the shift is deliberate and visible rather than incidental.

Reports generated before this decision remain committed in `reports/` and carry the fast-%K labels. They are left as historical records of the runs that actually happened; the canonical convention applies to every report generated afterwards.

`validation/` no longer avoids TA-Lib. Its "pure pandas/numpy, no TA-Lib" guarantee is withdrawn — test mocks of TA-Lib were removed in ticket #5 and the backtest's own indicator maths was replaced by the shared module in ticket #7 — because the alternative was preserving a second implementation, which is the drift this consolidation exists to prevent.

The backtest's engine version moved from 1.0.0 to 1.1.0 alongside the migration. It participates in the decision fingerprint, so the bump invalidates every cached decision: an incremental run cannot mix decisions computed under fast %K with decisions computed under slow %K.
