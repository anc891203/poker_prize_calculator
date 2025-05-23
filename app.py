# Poker Prize Calculator – Streamlit edition (English, fixed-mincash)
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st

# ---------- helper functions ---------- #
def build_group_lengths(total_paid: int, beta: float) -> List[int]:
    """Initial group lengths after the top 9 single places."""
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
    """Merge from the end until every later group has >= players than the previous."""
    i = len(lengths) - 1
    while i > 0:
        if lengths[i] < lengths[i - 1]:
            lengths[i - 1] += lengths[i]
            lengths.pop(i)
        i -= 1
    return lengths


def payjump_ok(prizes: List[int]) -> bool:
    """True if pay-jumps are strictly decreasing (i.e. prize gaps strictly increasing)."""
    jumps = [a - b for a, b in zip(prizes[:-1], prizes[1:])]
    return all(j1 > j2 for j1, j2 in zip(jumps[:-1], jumps[1:]))


# ---------- core algorithm ---------- #
def calc_prize_distribution(
    total_players: int,
    itm_percent: float,
    pool: int,
    round_digits: int,
    alpha: float,
    beta: float,
    mincash: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return per-player prizes and compact group ranges; raise ValueError if impossible."""
    # 1. number of paid places
    paid = math.ceil(total_players * itm_percent / 100)
    paid = max(1, min(paid, total_players))

    # 2. raw weights
    ranks = np.arange(1, paid + 1)
    weights = ranks ** (-alpha)
    raw = pool * weights / weights.sum()

    # 3. group lengths
    group_lengths = enforce_non_decreasing(build_group_lengths(paid, beta))

    # 4. median prize per group
    group_prizes: List[int] = []
    group_prizes.extend(raw[: min(9, paid)].round(2).tolist())
    idx = 9
    for length in group_lengths:
        if idx >= paid:
            break
        segment = raw[idx : idx + length]
        group_prizes.append(float(segment[len(segment) // 2]))
        idx += length

    # 5. round-floor to given digits
    unit = 10 ** (-round_digits)
    group_prizes = [math.floor(p / unit) * unit for p in group_prizes]

    # 6. force last group to exact mincash
    if paid > 9:
        group_prizes[-1] = mincash

    # 7. adjust total to match pool
    total_after = sum(
        p * (1 if i < 9 else group_lengths[i - 9]) for i, p in enumerate(group_prizes)
    )
    diff = pool - total_after
    group_prizes[0] += diff  # push difference into 1st prize for now

    # 8. enforce strictly increasing pay-jumps
    delta = unit
    extra_added = 0
    for i in range(len(group_prizes) - 1, 0, -1):
        jump = group_prizes[i - 1] - group_prizes[i]
        if i == len(group_prizes) - 1:
            max_jump = jump
            continue
        if jump <= max_jump:
            need = max_jump + delta - jump
            group_prizes[i - 1] += need
            extra_added += need
            max_jump = jump + need
        else:
            max_jump = jump
    group_prizes[0] -= extra_added  # recover the over-allocation

    # 9. final sanity checks
    if (
        group_prizes[-1] != mincash
        or group_prizes[0] <= 0
        or not payjump_ok(group_prizes)
    ):
        raise ValueError

    # 10. expand to every player
    prizes_full: List[int] = []
    prizes_full.extend(group_prizes[: min(9, paid)])
    gi = 0
    for length in group_lengths:
        if len(prizes_full) >= paid:
            break
        prizes_full.extend([group_prizes[9 + gi]] * length)
        gi += 1
    prizes_full = prizes_full[:paid]

    # 11. compact range list for table
    ranges: List[Tuple[int, int]] = []
    start = 1
    for i, p in enumerate(prizes_full):
        if i == len(prizes_full) - 1 or prizes_full[i + 1] != p:
            ranges.append((start, i + 1))
            start = i + 2
    return prizes_full, ranges


def suggest_value(
    which: str,
    alpha: float,
    beta: float,
    total_players: int,
    itm_percent: float,
    pool: int,
    round_digits: int,
    mincash: int,
) -> float | None:
    if which == "alpha":
        for offset in np.linspace(0, 0.5, 501):
            for candidate in (round(alpha + offset, 3), round(alpha - offset, 3)):
                if 0.5 <= candidate <= 1:
                    try:
                        calc_prize_distribution(
                            total_players,
                            itm_percent,
                            pool,
                            round_digits,
                            candidate,
                            beta,
                            mincash,
                        )
                        return candidate
                    except ValueError:
                        continue
    else:  # beta
        for offset in np.linspace(0, 1, 101):
            for candidate in (round(beta + offset, 1), round(beta - offset, 1)):
                if 1 <= candidate <= 2:
                    try:
                        calc_prize_distribution(
                            total_players,
                            itm_percent,
                            pool,
                            round_digits,
                            alpha,
                            candidate,
                            mincash,
                        )
                        return candidate
                    except ValueError:
                        continue
    return None


# ---------- Streamlit UI ---------- #
st.set_page_config(page_title="Poker Prize Calculator", layout="wide")
st.title("Poker Prize Calculator")

col_left, col_right = st.columns([1, 2], gap="medium")

with col_left:
    st.header("Parameters")
    total_players = st.number_input("Total entrants", min_value=2, step=1, value=524)
    itm_percent = st.number_input(
        "Payout percentage (%)",
        min_value=0.5,
        max_value=100.0,
        step=0.5,
        value=12.5,
        format="%.1f",
    )
    pool = st.number_input(
        "Prize pool",
        min_value=0,
        max_value=2_000_000_000,
        step=1_000,
        value=182_000,
        format="%d",
    )
    mincash = st.number_input(
        "Fixed min-cash (last group)",
        min_value=0,
        max_value=1_000_000,
        step=100,
        value=1_000,
        format="%d",
    )
    round_digits = st.number_input(
        "Round digits (-1 to -10)",
        min_value=-10,
        max_value=-1,
        step=1,
        value=-2,
        format="%d",
    )
    alpha = st.number_input(
        "Alpha (0.5 – 1.0)",
        min_value=0.5,
        max_value=1.0,
        step=0.001,
        value=0.879,
        format="%.3f",
    )
    beta = st.number_input(
        "Beta (1.0 - )",
        min_value=1.0,
        max_value=10.0,
        step=0.1,
        value=1.6,
        format="%.1f",
    )

with col_right:
    st.header("Results")
    try:
        prizes, ranges = calc_prize_distribution(
            total_players,
            itm_percent,
            pool,
            round_digits,
            alpha,
            beta,
            mincash,
        )

        st.success("Computation successful")
        st.subheader(f"ITM players: {len(prizes)}")

        df = pd.DataFrame(
            {
                "Rank": [f"{s}" if s == e else f"{s}-{e}" for s, e in ranges],
                "Prize": [f"{prizes[s-1]:,}" for s, _ in ranges],
            }
        )
        st.table(df)

        fmt = FuncFormatter(
            lambda x, pos: f"{x/1e9:.1f} B"
            if x >= 1e9
            else f"{x/1e6:.1f} M"
            if x >= 1e6
            else f"{x/1e3:.0f} K"
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
        st.error("Infeasible parameters – adjust Alpha, Beta, round digits, or min-cash.")

        sugg_a = suggest_value(
            "alpha",
            alpha,
            beta,
            total_players,
            itm_percent,
            pool,
            round_digits,
            mincash,
        )
        sugg_b = suggest_value(
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
            (f"Suggested α: {sugg_a:.3f}" if sugg_a else "No α suggestion")
            + " | "
            + (f"Suggested β: {sugg_b:.1f}" if sugg_b else "No β suggestion")
        )
