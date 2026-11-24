import mplfinance as mpf

# ---------------------------------------------------------------------------
# 金价涨跌配色规范（全项目唯一来源）：红涨绿跌
#   up   = 收盘价 >= 开盘价（上涨）→ 红色
#   down = 收盘价 <  开盘价（下跌）→ 绿色
# 所有 K 线图、成交量柱、报告文案、决策徽章都必须复用下面两个常量。
# ---------------------------------------------------------------------------
UP_COLOR = "#A02128"    # 上涨：红
DOWN_COLOR = "#006340"  # 下跌：绿

# 趋势线配色：支撑用蓝色、阻力用橙色。阻力线不能再用红色，
# 否则会与「红色 = 上涨」的语义冲突，也会和红色阳线混在一起难以分辨。
SUPPORT_COLOR = "#1F4FD8"
RESISTANCE_COLOR = "#FF8C00"

font = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.weight": "normal",
    "font.size": 15,
}

my_color_style = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(
        up=UP_COLOR,  # color for bullish (rising) candles
        down=DOWN_COLOR,  # color for bearish (falling) candles
        edge="none",  # use candle fill color for edge
        wick="black",  # color of the wicks
        volume="inherit",  # volume bars follow the up/down rule as well
    ),
    gridstyle="-",
    facecolor="white",  # background color
    rc=font,
)
