# Poker Prize Calculator (Streamlit+) 2025-05
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

# ---------- 核心演算法 ---------- #
def build_group_lengths(total_paid: int, beta: float) -> List[int]:
    """初步依 beta 生成 [2, 3, 4, ...] 等長度序列 (頭 9 名獨立)"""
    if total_paid <= 9:
        return []
    remain, length, groups = total_paid - 9, 2, []
    while remain > 0:
        groups.append(length)
        remain -= length
        length = max(1, round(length * beta))
    if remain < 0:
        groups[-1] += remain
    return groups


def enforce_non_decreasing(lengths: List[int]) -> List[int]:
    """若最後一組比前一組少，合併直到遞增"""
    i = len(lengths) - 1
    while i > 0:
        if lengths[i] < lengths[i - 1]:
            lengths[i - 1] += lengths[i]
            lengths.pop(i)
        i -= 1
    return lengths


def payjump_ok(prizes: List[int]) -> bool:
    jumps = [a - b for a, b in zip(prizes[:-1], prizes[1:])]
    return all(j1 > j2 for j1, j2 in zip(jumps[:-1], jumps[1:]))


def calc_prize_distribution(
    total_players: int,
    itm_percent: float,
    pool: int,
    round_digits: int,
    alpha: float,
    beta: float,
    mincash: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """回傳 (prizes_full, group_ranges)；若無可行組合會 raise ValueError"""
    # 1. ITM 人數
    paid = math.ceil(total_players * itm_percent / 100)
    paid = max(1, min(paid, total_players))

    # 2. 初始權重
    ranks = np.arange(1, paid + 1)
    weights = ranks ** (-alpha)
    raw = pool * weights / weights.sum()

    # 3. 分組長度 (頭 9 名獨立)
    group_lengths = build_group_lengths(paid, beta)
    group_lengths = enforce_non_decreasing(group_lengths)

    # 4. 以中位數取組獎金
    prizes_group: List[int] = []
    prizes_group.extend(raw[: min(9, paid)].round(2).tolist())
    idx = 9
    for length in group_lengths:
        if idx >= paid:
            break
        seg = raw[idx : idx + length]
        prizes_group.append(float(seg[len(seg) // 2]))
        idx += length

    # 5. round-floor 並加上 mincash 下限
    unit = 10 ** (-round_digits)
    prizes_group = [max(mincash, math.floor(p / unit) * unit) for p in prizes_group]

    # 6. 調整金額使總和回到 pool
    total_after = sum(
        p * (1 if i < 9 else group_lengths[i - 9]) for i, p in enumerate(prizes_group)
    )
    diff_pool = pool - total_after
    prizes_group[0] += diff_pool  # 先把差額放到第一名 (可能是正也可能負)

    # 7. 嚴格遞增 payjump 修正
    delta = unit
    extra_added = 0
    for i in range(len(prizes_group) - 1, 0, -1):
        jump = prizes_group[i - 1] - prizes_group[i]
        if i == len(prizes_group) - 1:
            max_jump = jump
            continue
        if jump <= max_jump:
            need = max_jump + delta - jump
            prizes_group[i - 1] += need
            extra_added += need
            max_jump = jump + need
        else:
            max_jump = jump
    prizes_group[0] -= extra_added  # 回收多加的錢

    # 8. 重新檢查 mincash、payjump
    if prizes_group[-1] < mincash or prizes_group[0] <= 0 or not payjump_ok(prizes_group):
        raise ValueError

    # 9. 展開到每位玩家
    prizes_full: List[int] = []
    prizes_full.extend(prizes_group[: min(9, paid)])
    gi = 0
    for length in group_lengths:
        if len(prizes_full) >= paid:
            break
        prizes_full.extend([prizes_group[9 + gi]] * length)
        gi += 1
    prizes_full = prizes_full[:paid]

    # 10. 生成 group_ranges 方便表格顯示
    group_ranges: List[Tuple[int, int]] = []
    start = 1
    for i, p in enumerate(prizes_full):
        if i == len(prizes_full) - 1 or prizes_full[i + 1] != p:
            group_ranges.append((start, i + 1))
            start = i + 2
    return prizes_full, group_ranges


def find_suggestion(
    which: str,
    cur_alpha: float,
    cur_beta: float,
    total_players: int,
    itm_percent: float,
    pool: int,
    round_digits: int,
    mincash: int,
) -> float | None:
    """嘗試找第一個可行 Alpha 或 Beta 建議值"""
    if which == "alpha":
        for offset in np.linspace(0, 0.5, 501):
            for a in (round(cur_alpha + offset, 3), round(cur_alpha - offset, 3)):
                if 0.5 <= a <= 1:
                    try:
                        calc_prize_distribution(
                            total_players,
                            itm_percent,
                            pool,
                            round_digits,
                            a,
                            cur_beta,
                            mincash,
                        )
                        return a
                    except ValueError:
                        continue
    else:
        for offset in np.linspace(0, 1, 101):
            for b in (round(cur_beta + offset, 1), round(cur_beta - offset, 1)):
                if 1 <= b <= 2:
                    try:
                        calc_prize_distribution(
                            total_players,
                            itm_percent,
                            pool,
                            round_digits,
                            cur_alpha,
                            b,
                            mincash,
                        )
                        return b
                    except ValueError:
                        continue
    return None


# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="Poker Prize Calculator", layout="wide")
st.title("Poker Prize Calculator")

col_input, col_output = st.columns([1, 2], gap="medium")

with col_input:
    st.header("參數設定")

    total_players = st.number_input("總參賽人數", min_value=2, step=1, value=524)
    itm_percent = st.number_input(
        "獎勵圈百分比 (%)",
        min_value=0.5,
        max_value=100.0,
        step=0.5,
        value=12.5,
        format="%.1f",
    )
    pool = st.number_input(
        "總獎池",
        min_value=0,
        max_value=2_000_000_000,
        step=1000,
        value=182_000,
        format="%d",
    )
    mincash = st.number_input(
        "最低分組獎金 mincash",
        min_value=0,
        max_value=1_000_000,
        step=100,
        value=900,
        format="%d",
    )

    round_digits = st.number_input(
        "Round Digits (-1 到 -10)",
        min_value=-10,
        max_value=-1,
        step=1,
        value=-2,
        format="%d",
    )

    alpha = st.number_input(
        "Alpha (0.5 到 1.0)",
        min_value=0.5,
        max_value=1.0,
        step=0.001,
        value=0.831,
        format="%.3f",
    )
    beta = st.number_input(
        "Beta (1.0 到 2.0)",
        min_value=1.0,
        max_value=2.0,
        step=0.1,
        value=1.5,
        format="%.1f",
    )

with col_output:
    st.header("結果")

    try:
        prizes, group_ranges = calc_prize_distribution(
            total_players,
            itm_percent,
            pool,
            round_digits,
            alpha,
            beta,
            mincash,
        )

        st.success("計算成功")
        st.subheader(f"ITM 人數：{len(prizes)}")

        table_rows = [
            {"Rank": f"{s}" if s == e else f"{s}-{e}", "Prize": f"{prizes[s-1]:,}"}
            for s, e in group_ranges
        ]
        st.table(pd.DataFrame(table_rows))

        fmt = FuncFormatter(
            lambda x, pos: f"{x/1e9:.1f}B"
            if x >= 1e9
            else f"{x/1e6:.1f}M"
            if x >= 1e6
            else f"{x/1e3:.0f}K"
            if x >= 1e3
            else f"{int(x)}"
        )
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(range(1, len(prizes) + 1), prizes, linewidth=1.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Prize")
        ax.set_title("Prize distribution")
        ax.yaxis.set_major_formatter(fmt)
        st.pyplot(fig, use_container_width=True)

    except ValueError:
        st.error("無法計算 - 請調整 Alpha / Beta / Round Digits 或 mincash")

        sugg_a = find_suggestion(
            "alpha",
            alpha,
            beta,
            total_players,
            itm_percent,
            pool,
            round_digits,
            mincash,
        )
        sugg_b = find_suggestion(
            "beta",
            alpha,
            beta,
            total_players,
            itm_percent,
            pool,
            round_digits,
            mincash,
        )
        st.info(
            (f"Alpha 建議值：{sugg_a:.3f}" if sugg_a else "Alpha 無可行建議")
            + " | "
            + (f"Beta 建議值：{sugg_b:.1f}" if sugg_b else "Beta 無可行建議")
        )
