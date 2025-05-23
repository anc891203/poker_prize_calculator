# Poker Prize Calculator (Streamlit 版；可手動輸入 RoundDigits / Alpha / Beta)
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

# ------------------- 核心演算法 ------------------- #
def build_group_lengths(total_paid: int, beta: float) -> List[int]:
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
) -> Tuple[List[int], List[Tuple[int, int]]]:
    paid = math.ceil(total_players * itm_percent / 100)
    paid = max(1, min(paid, total_players))

    ranks = np.arange(1, paid + 1)
    weights = ranks ** (-alpha)
    raw = pool * weights / weights.sum()

    group_lengths = build_group_lengths(paid, beta)

    prizes_group: List[int] = []
    prizes_group.extend(raw[: min(9, paid)].round(2).tolist())
    idx = 9
    for length in group_lengths:
        if idx >= paid:
            break
        seg = raw[idx : idx + length]
        prizes_group.append(float(seg[len(seg) // 2]))
        idx += length

    unit = 10 ** (-round_digits)
    prizes_group = [math.floor(p / unit) * unit for p in prizes_group]
    diff_pool = pool - sum(
        p * (1 if i < 9 else group_lengths[i - 9]) for i, p in enumerate(prizes_group)
    )
    prizes_group[0] += diff_pool

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

    prizes_group[0] -= extra_added
    if not payjump_ok(prizes_group):
        raise ValueError

    prizes_full: List[int] = []
    prizes_full.extend(prizes_group[: min(9, paid)])
    gi = 0
    for length in group_lengths:
        if len(prizes_full) >= paid:
            break
        prizes_full.extend([prizes_group[9 + gi]] * length)
        gi += 1
    prizes_full = prizes_full[:paid]

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
) -> float | None:
    if which == "alpha":
        for offset in np.linspace(0, 0.5, 501):
            for a in (round(cur_alpha + offset, 3), round(cur_alpha - offset, 3)):
                if 0.5 <= a <= 1:
                    try:
                        calc_prize_distribution(
                            total_players, itm_percent, pool, round_digits, a, cur_beta
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
                            total_players, itm_percent, pool, round_digits, cur_alpha, b
                        )
                        return b
                    except ValueError:
                        continue
    return None


# ------------------- Streamlit UI ------------------- #
st.set_page_config(page_title="Poker Prize Calculator", layout="wide")

st.title("Poker Prize Calculator")

# 兩欄：左輸入、右輸出
col_input, col_output = st.columns([1, 2], gap="medium")

with col_input:
    st.header("參數設定")

    total_players = st.number_input("總參賽人數", min_value=2, step=1, value=524)
    itm_percent = st.number_input(
        "獎勵圈百分比 (%)", min_value=0.5, max_value=100.0, step=0.5, value=12.5, format="%.1f"
    )
    pool = st.number_input(
        "總獎池", min_value=0, max_value=2_000_000_000, step=1000, value=182_000, format="%d"
    )

    round_digits = st.number_input(
        "Round Digits (−1 ~ −10)", min_value=-10, max_value=-1, step=1, value=-2, format="%d"
    )

    alpha = st.number_input(
        "Alpha (0.5 ~ 1.0)", min_value=0.5, max_value=1.0, step=0.001, value=0.825, format="%.3f"
    )
    beta = st.number_input(
        "Beta (1.0 ~ 2.0)", min_value=1.0, max_value=2.0, step=0.1, value=1.5, format="%.1f"
    )

with col_output:
    st.header("結果")

    try:
        prizes, group_ranges = calc_prize_distribution(
            total_players, itm_percent, pool, round_digits, alpha, beta
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
        st.error("無法計算 - 請調整 Alpha / Beta 或 Round Digits")

        sugg_a = find_suggestion(
            "alpha", alpha, beta, total_players, itm_percent, pool, round_digits
        )
        sugg_b = find_suggestion(
            "beta", alpha, beta, total_players, itm_percent, pool, round_digits
        )
        st.info(
            (f"Alpha 建議值：{sugg_a:.3f}" if sugg_a else "Alpha 無可行建議")
            + " | "
            + (f"Beta 建議值：{sugg_b:.1f}" if sugg_b else "Beta 無可行建議")
        )
