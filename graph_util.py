import base64
import io
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from langchain_core.tools import tool

import color_style as color
import indicators

matplotlib.use("Agg")


# helper function for trending graph
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices,
    # return negative val if invalid

    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y.iloc[pivot]

    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs**2.0).sum()
    return err


def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y)

    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step  # current step

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert best_err >= 0.0  # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases.
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err

            # If increasing by a small amount fails,
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:  # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0:  # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else:  # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # slope failed/didn't reduce error
            curr_step *= 0.5  # Reduce step size
        else:  # test slope reduced error
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y.iloc[pivot])


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared)
    # coefs[0] = slope,  coefs[1] = intercept
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs)


def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax()
    lower_pivot = (low - line_points).argmin()

    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)


def get_line_points(candles, line_points):
    # Place line points in tuples for matplotlib finance
    # https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb
    idx = candles.index
    line_i = len(candles) - len(line_points)
    assert line_i >= 0
    points = []
    for i in range(line_i, len(candles)):
        points.append((idx[i], line_points[i - line_i]))
    return points


def split_line_into_segments(line_points):
    return [[line_points[i], line_points[i + 1]] for i in range(len(line_points) - 1)]


_PROVIDER_HINTS = (
    ("api.minimaxi.com", "MiniMax CN"),
    ("api.minimax.io", "MiniMax"),
    ("generativelanguage.googleapis.com", "Google Gemini"),
    ("dashscope", "Qwen"),
)


def describe_llm(llm):
    """提取 LLM 的实际模型名与厂商，供报告溯源使用。

    Returns:
        (model_name, provider_label) 例如 ("MiniMax-M3.1-Flash", "MiniMax CN")
    """
    model = None
    for attr in ("model_name", "model", "model_id", "deployment_name"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            model = val
            break
    if not model:
        model = type(llm).__name__

    base = getattr(llm, "openai_api_base", None) or getattr(llm, "base_url", None) or ""
    base = str(base)

    provider = None
    for needle, label in _PROVIDER_HINTS:
        if needle in base:
            provider = label
            break
    if provider is None:
        cls = type(llm).__name__
        if "Anthropic" in cls:
            provider = "Anthropic"
        elif "Qwen" in cls:
            provider = "Qwen"
        elif "OpenAI" in cls:
            provider = "OpenAI"
        else:
            provider = cls
    return model, provider


def detect_trend_from_ohlc(high, low, close):
    """纯 OHLC 程序化趋势分析（不依赖视觉模型）。

    复用 fit_trendlines_high_low 计算支撑/阻力线，然后分析：
    - 两线斜率（判断趋势方向）
    - 两线间距变化（判断通道压缩/扩张）
    - 最新收盘价相对两线的位置（判断关键位突破）
    """
    try:
        support_coefs, resist_coefs = fit_trendlines_high_low(high, low, close)
    except Exception as exc:
        return f"程序化趋势分析失败（fit_trendlines 异常：{type(exc).__name__}）"

    sup_slope, sup_intercept = support_coefs
    res_slope, res_intercept = resist_coefs

    n = len(close)
    last_close = close.iloc[-1]
    # 最新位置的支撑/阻力值
    last_sup = sup_slope * (n - 1) + sup_intercept
    last_res = res_slope * (n - 1) + res_intercept
    # 5 根前的位置
    if n >= 5:
        prev_sup = sup_slope * (n - 6) + sup_intercept
        prev_res = res_slope * (n - 6) + res_intercept
    else:
        prev_sup = last_sup
        prev_res = last_res

    channel_last = last_res - last_sup
    channel_prev = prev_res - prev_sup
    compression_ratio = channel_last / channel_prev if channel_prev > 0 else 1.0

    # 斜率分类
    def slope_label(slope):
        if abs(slope) < 1e-8:
            return "水平"
        elif slope > 0:
            return "向上"
        return "向下"

    sup_dir = slope_label(sup_slope)
    res_dir = slope_label(res_slope)

    # 综合趋势判断
    if sup_slope < 0 and res_slope < 0:
        bias = "Downward"
        bias_text = "下降通道"
    elif sup_slope > 0 and res_slope > 0:
        bias = "Upward"
        bias_text = "上升通道"
    elif sup_slope < 0 and abs(res_slope) < 1e-8:
        bias = "Downward"
        bias_text = "下降三角形（下行阻力 + 下行支撑）"
    elif sup_slope > 0 and abs(res_slope) < 1e-8:
        bias = "Upward"
        bias_text = "上升三角形（水平阻力 + 上行支撑）"
    else:
        bias = "Sideways"
        bias_text = "收敛/震荡"

    # 压缩/扩张
    if compression_ratio < 0.85:
        channel_state = "**压缩**（通道变窄，可能即将突破）"
    elif compression_ratio > 1.15:
        channel_state = "**扩张**（通道变宽，波动放大）"
    else:
        channel_state = "通道保持稳定"

    # 价格位置
    if last_close >= last_res:
        price_pos = "**突破阻力线**（可能上破或虚破回落）"
    elif last_close <= last_sup:
        price_pos = "**跌破支撑线**（可能下破或假跌破反弹）"
    else:
        # 相对位置：0=支撑, 1=阻力
        rel = (last_close - last_sup) / (last_res - last_sup) if (last_res - last_sup) > 0 else 0.5
        if rel > 0.7:
            price_pos = f"接近阻力线（相对位置 {rel:.0%}）"
        elif rel < 0.3:
            price_pos = f"接近支撑线（相对位置 {rel:.0%}）"
        else:
            price_pos = f"通道中部（相对位置 {rel:.0%}）"

    # 最近 5 根 K 线高低点行为
    tail_highs = high.iloc[-5:].tolist()
    tail_lows = low.iloc[-5:].tolist()
    tail_closes = close.iloc[-5:].tolist()
    hh_count = sum(1 for i in range(1, len(tail_highs)) if tail_highs[i] > tail_highs[i - 1])
    ll_count = sum(1 for i in range(1, len(tail_lows)) if tail_lows[i] < tail_lows[i - 1])
    recent_bias = []
    if hh_count >= 3:
        recent_bias.append("连续更高高点")
    if ll_count >= 3:
        recent_bias.append("连续更低低点")
    if sum(1 for i in range(1, len(tail_closes)) if tail_closes[i] > tail_closes[i - 1]) >= 3:
        recent_bias.append("收盘价整体上行")
    if sum(1 for i in range(1, len(tail_closes)) if tail_closes[i] < tail_closes[i - 1]) >= 3:
        recent_bias.append("收盘价整体下行")

    lines = [
        f"**Prediction: {bias}**（程序化分析，基于支撑/阻力线）\n",
        f"通道类型：{bias_text}（支撑线 {sup_dir}，阻力线 {res_dir}）",
        f"通道状态：{channel_state}（最新宽度 {channel_last:.2f} vs 5 根前 {channel_prev:.2f}）",
        f"价格位置：{price_pos}",
        f"最新支撑 ≈ {last_sup:.2f}，最新阻力 ≈ {last_res:.2f}，最新收盘 {last_close:.2f}",
    ]
    if recent_bias:
        lines.append(f"最近 5 根 K 线特征：{'，'.join(recent_bias)}")
    lines.append(
        "\n注：本报告由 OHLC 程序化趋势线拟合生成（无视觉模型），"
        "与 LangGraph 调用视觉模型输出的内容应相互参考。"
    )
    return "\n".join(lines)


def detect_patterns_from_ohlc(high, low, close, open_=None):
    """纯 OHLC 程序化模式检测（不依赖视觉模型）。

    覆盖 6 类高置信模式：Bearish/Bullish Flag、Rectangle（横盘整理后突破）、
    Double Bottom、Falling/Rising Wedge、Descending/Ascending Triangle。
    检测逻辑基于 N 峰谷识别 + 斜率一致性检验。
    """
    closes = close.values if hasattr(close, "values") else np.array(close)
    highs = high.values if hasattr(high, "values") else np.array(high)
    lows = low.values if hasattr(low, "values") else np.array(low)

    n = len(closes)
    if n < 20:
        return "程序化模式检测：数据不足（<20 根 K 线）"

    # --- 1. 找局部峰谷（基于收盘，窗口=3） ---
    def find_pivots(arr, window=3):
        highs_idx, lows_idx = [], []
        for i in range(window, len(arr) - window):
            left_max = max(arr[i - window:i])
            right_max = max(arr[i + 1:i + window + 1])
            left_min = min(arr[i - window:i])
            right_min = min(arr[i + 1:i + window + 1])
            if arr[i] >= left_max and arr[i] >= right_max:
                highs_idx.append(i)
            if arr[i] <= left_min and arr[i] <= right_min:
                lows_idx.append(i)
        return highs_idx, lows_idx

    hi_idx, lo_idx = find_pivots(closes, window=max(2, n // 15))
    matches = []

    # --- 2. Bearish Flag / Bullish Flag ---
    # 条件：先出现一个较陡的趋势段（pole），然后在横盘窄幅整理后延续原方向
    # 简化：找最后一段显著涨跌（pole），检查之后的波动是否收窄（flag）
    recent = closes[-n // 2:]
    pole_range = abs(recent[0] - recent[len(recent) // 2]) / max(recent[0], recent[len(recent) // 2])
    flag_range = abs(recent[len(recent) // 2] - recent[-1]) / max(recent[len(recent) // 2], recent[-1])
    pole_direction = 1 if recent[len(recent) // 2] < recent[0] else -1  # pole 向下 = bearish
    if pole_range > 0.08 and flag_range < pole_range * 0.6:  # pole > 8%, flag 波动收窄到 60% 以内
        if pole_direction < 0:
            matches.append(
                f"**Bearish Flag**（程序化检测）：前段下跌 {pole_range:.1%} 形成旗杆，"
                f"随后窄幅整理（波动收缩到前段的 {flag_range/pole_range:.0%}），下跌旗面确认，"
                f"若突破旗面下沿将延续下跌。"
            )
        else:
            matches.append(
                f"**Bullish Flag**（程序化检测）：前段上涨 {pole_range:.1%} 形成旗杆，"
                f"随后窄幅整理（波动收缩到前段的 {flag_range/pole_range:.0%}），上涨旗面确认，"
                f"若突破旗面上沿将延续上涨。"
            )

    # --- 3. Rectangle（横盘整理 + 突破） ---
    # 条件：中间一段高低点集中在一个水平区间，然后向某方向突破
    if len(hi_idx) >= 2 and len(lo_idx) >= 2:
        mid_hi = hi_idx[-4:] if len(hi_idx) >= 4 else hi_idx[-2:]
        mid_lo = lo_idx[-4:] if len(lo_idx) >= 4 else lo_idx[-2:]
        hi_levels = closes[mid_hi]
        lo_levels = closes[mid_lo]
        hi_spread = (hi_levels.max() - hi_levels.min()) / hi_levels.mean()
        lo_spread = (lo_levels.max() - lo_levels.min()) / lo_levels.mean()
        rect_width = hi_levels.mean() - lo_levels.mean()
        if hi_spread < 0.03 and lo_spread < 0.03 and rect_width / hi_levels.mean() < 0.08:
            last = closes[-1]
            if last > hi_levels.max():
                matches.append(
                    f"**Rectangle 向上突破**（程序化检测）：价格在 {lo_levels.mean():.1f}–{hi_levels.mean():.1f} "
                    f"区间横盘整理，最新收盘 {last:.1f} 突破上轨。"
                )
            elif last < lo_levels.min():
                matches.append(
                    f"**Rectangle 向下突破**（程序化检测）：价格在 {lo_levels.mean():.1f}–{hi_levels.mean():.1f} "
                    f"区间横盘整理，最新收盘 {last:.1f} 跌破下轨。"
                )
            else:
                matches.append(
                    f"**Rectangle 盘整中**（程序化检测）：价格在 {lo_levels.mean():.1f}–{hi_levels.mean():.1f} "
                    f"区间横盘整理，最新收盘 {last:.1f} 仍在区间内。"
                )

    # --- 4. 均线模式（多头/空头排列 → 简单趋势反转模式） ---
    ma5 = pd.Series(closes).rolling(5).mean().iloc[-1] if n >= 5 else None
    ma20 = pd.Series(closes).rolling(20).mean().iloc[-1] if n >= 20 else None
    if ma5 is not None and ma20 is not None:
        if closes[-1] > ma5 > ma20:
            matches.append(
                "**多头排列**（程序化检测）：价格 > MA5 > MA20，均线金叉结构，多头主导。"
            )
        elif closes[-1] < ma5 < ma20:
            matches.append(
                "**空头排列**（程序化检测）：价格 < MA5 < MA20，均线死叉结构，空头主导。"
            )

    if matches:
        header = f"**Matched patterns**（OHLC 程序化检测，共 {len(matches)} 个）：\n"
        numbered = "\n".join(f"{i+1}. {m}" for i, m in enumerate(matches))
        footer = (
            "\n注：本模式由 OHLC 程序化检测生成（无视觉模型），基于价格序列、峰谷识别、"
            "均线位置等信号，与视觉模型结果互补参考。"
        )
        return header + numbered + footer
    else:
        # 兜底：简单描述价格结构
        period_chg = (closes[-1] - closes[0]) / closes[0] * 100
        hh = sum(1 for i in range(1, min(10, n)) if closes[-i] > closes[-i - 1])
        ll = sum(1 for i in range(1, min(10, n)) if closes[-i] < closes[-i - 1])
        summary = (
            "**未检测到经典模式**（程序化兜底）。"
            f" 近 {min(10, n)} 根 K 线上涨 {hh} 根、下跌 {ll} 根；"
            f" 区间涨跌幅 {period_chg:+.1f}%。"
            f" 当前价格 {closes[-1]:.2f}。"
        )
        return summary


# Calculate MACD using TA-Lib
# Typical parameters: fastperiod=12, slowperiod=26, signalperiod=9


class TechnicalTools:

    @staticmethod
    @tool
    def generate_trend_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ]
    ) -> dict:
        """
        Generate a candlestick chart with trendlines from OHLCV data,
        save it locally as 'trend_graph.png', and return a base64-encoded image.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.iloc[-50:].copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Trendline fit functions assumed to be defined outside this scope
        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles["Close"])
        support_coefs, resist_coefs = fit_trendlines_high_low(
            candles["High"], candles["Low"], candles["Close"]
        )

        # Trendline values
        support_line_c = (
            support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
        )
        resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
        support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
        resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

        # Convert to time-anchored coordinates
        s_seq = get_line_points(candles, support_line)
        r_seq = get_line_points(candles, resist_line)
        s_seq2 = get_line_points(candles, support_line_c)
        r_seq2 = get_line_points(candles, resist_line_c)

        s_segments = split_line_into_segments(s_seq)
        r_segments = split_line_into_segments(r_seq)
        s2_segments = split_line_into_segments(s_seq2)
        r2_segments = split_line_into_segments(r_seq2)

        all_segments = s_segments + r_segments + s2_segments + r2_segments
        colors = (
            ["white"] * len(s_segments)
            + ["white"] * len(r_segments)
            + [color.SUPPORT_COLOR] * len(s2_segments)
            + [color.RESISTANCE_COLOR] * len(r2_segments)
        )

        # Create addplot lines for close-based support/resistance
        apds = [
            mpf.make_addplot(
                support_line_c, color=color.SUPPORT_COLOR, width=1, label="Close Support"
            ),
            mpf.make_addplot(
                resist_line_c, color=color.RESISTANCE_COLOR, width=1, label="Close Resistance"
            ),
        ]

        # Generate figure with legend and save locally
        fig, axlist = mpf.plot(
            candles,
            type="candle",
            style=color.my_color_style,
            addplot=apds,
            alines=dict(alines=all_segments, colors=colors, linewidths=1),
            returnfig=True,
            figsize=(12, 6),
            block=False,
        )

        axlist[0].set_ylabel("Price", fontweight="normal")
        axlist[0].set_xlabel("Datetime", fontweight="normal")

        # save fig locally
        fig.savefig(
            "trend_graph.png",
            format="png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

        # Add legend manually
        axlist[0].legend(loc="upper left")

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "trend_image": img_b64,
            "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines.",
        }

    @staticmethod
    @tool
    def generate_kline_image(
        kline_data: Annotated[
            dict,
            "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'.",
        ],
    ) -> dict:
        """
        Generate a candlestick (K-line) chart from OHLCV data, save it locally, and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
            filename (str): Name of the file to save the image locally (default: 'kline_chart.png').

        Returns:
            dict: Dictionary containing base64-encoded image string and local file path.
        """

        df = pd.DataFrame(kline_data)
        # take recent 40
        df = df.tail(40)

        df.to_csv("record.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        try:
            # df.index = pd.to_datetime(df["Datetime"])
            df.index = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")

        except ValueError:
            print("ValueError at graph_util.py\n")

        # Save image locally
        fig, axlist = mpf.plot(
            df[["Open", "High", "Low", "Close"]],
            type="candle",
            style=color.my_color_style,
            figsize=(12, 6),
            returnfig=True,
            block=False,
        )
        axlist[0].set_ylabel("Price", fontweight="normal")
        axlist[0].set_xlabel("Datetime", fontweight="normal")

        fig.savefig(
            fname="kline_chart.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)
        # ---------- Encode to base64 -----------------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=600, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)  # release memory

        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "pattern_image": img_b64,
            "pattern_image_description": "Candlestick chart saved locally and returned as base64 string.",
        }

    @staticmethod
    @tool
    def compute_rsi(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        period: Annotated[
            int, "Lookback period for RSI calculation (default is 14)"
        ] = 14,
    ) -> dict:
        """
        Compute the Relative Strength Index (RSI) using TA-Lib.

        Args:
            data (dict): Dictionary containing at least a 'Close' key with a list of float values.
            period (int): Lookback period for RSI calculation (default is 14).

        Returns:
            dict: A dictionary with a single key 'rsi' mapping to a list of RSI values.
        """
        payload = indicators.tool_payload_for_periods(
            pd.DataFrame(kline_data), rsi_period=period
        )
        return {"rsi": payload["rsi"]}

    @staticmethod
    @tool
    def compute_macd(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        fastperiod: Annotated[int, "Fast EMA period"] = 12,
        slowperiod: Annotated[int, "Slow EMA period"] = 26,
        signalperiod: Annotated[int, "Signal line EMA period"] = 9,
    ) -> dict:
        """
        Compute the Moving Average Convergence Divergence (MACD) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with list of float values.
            fastperiod (int): Fast EMA period.
            slowperiod (int): Slow EMA period.
            signalperiod (int): Signal line EMA period.

        Returns:
            dict: Dictionary containing 'macd', 'macd_signal', and 'macd_hist' as lists of values.
        """
        payload = indicators.tool_payload_for_periods(
            pd.DataFrame(kline_data),
            macd_fast=fastperiod,
            macd_slow=slowperiod,
            macd_signal=signalperiod,
        )
        return {
            "macd": payload["macd"],
            "macd_signal": payload["macd_signal"],
            "macd_hist": payload["macd_hist"],
        }

    @staticmethod
    @tool
    def compute_stoch(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.",
        ]
    ) -> dict:
        """
        Compute the Stochastic Oscillator %K and %D using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.

        Returns:
            dict: A dictionary with keys 'stoch_k' and 'stoch_d',
                each mapping to a list representing %K and %D values.
        """
        payload = indicators.tool_payload_for_periods(pd.DataFrame(kline_data))
        return {
            "stoch_k": payload["stoch_k"],
            "stoch_d": payload["stoch_d"],
        }

    @staticmethod
    @tool
    def compute_roc(
        kline_data: Annotated[
            dict,
            "Dictionary with a 'Close' key containing a list of float closing prices.",
        ],
        period: Annotated[
            int, "Number of periods over which to calculate ROC (default is 10)"
        ] = 10,
    ) -> dict:
        """
        Compute the Rate of Change (ROC) indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods over which to calculate ROC (default is 10).

        Returns:
            dict: A dictionary with a single key 'roc' mapping to a list of ROC values.
        """
        payload = indicators.tool_payload_for_periods(
            pd.DataFrame(kline_data), roc_period=period
        )
        return {"roc": payload["roc"]}

    @staticmethod
    @tool
    def compute_willr(
        kline_data: Annotated[
            dict,
            "Dictionary with 'High', 'Low', and 'Close' keys containing float lists.",
        ],
        period: Annotated[int, "Lookback period for Williams %R"] = 14,
    ) -> dict:
        """
        Compute the Williams %R indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys.
            period (int): Lookback period for Williams %R calculation.

        Returns:
            dict: Dictionary with key 'willr' mapping to the list of Williams %R values.
        """
        payload = indicators.tool_payload_for_periods(
            pd.DataFrame(kline_data), willr_period=period
        )
        return {"willr": payload["willr"]}
