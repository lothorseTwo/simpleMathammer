import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
import re

# =========================================================================
# DICE PARSING
# =========================================================================

def parse_attack_input(val: str) -> dict[int, float] | None:
    """
    Parses an attack input string into a probability distribution.
    Supports:
        - Plain integer:        "10"
        - Single die:           "d6", "d3", "d2"
        - Multiple dice:        "2d6", "3d3"
        - Die plus modifier:    "d6+2", "2d6+3"
    Returns a dict {num_attacks: probability} or None if invalid.
    """
    val = val.strip().lower()

    # Plain integer
    if re.fullmatch(r'\d+', val):
        n = int(val)
        if n < 1:
            return None
        return {n: 1.0}

    # Dice pattern: optional multiplier, die type, optional modifier
    match = re.fullmatch(r'(\d*)d(\d+)(?:\+(\d+))?', val)
    if not match:
        return None

    num_dice   = int(match.group(1)) if match.group(1) else 1
    die_size   = int(match.group(2))
    modifier   = int(match.group(3)) if match.group(3) else 0

    if die_size not in (2, 3, 6):
        return None
    if num_dice < 1:
        return None

    # Build distribution for a single die
    single_die = {face: 1.0 / die_size for face in range(1, die_size + 1)}

    # Convolve for multiple dice
    combined = {0: 1.0}
    for _ in range(num_dice):
        new_combined = defaultdict(float)
        for v1, p1 in combined.items():
            for v2, p2 in single_die.items():
                new_combined[v1 + v2] += p1 * p2
        combined = dict(new_combined)

    # Apply modifier
    if modifier:
        combined = {k + modifier: v for k, v in combined.items()}

    # Validate all outcomes are >= 1
    if min(combined.keys()) < 1:
        return None

    return combined


# =========================================================================
# PROBABILITY CALCULATION
# =========================================================================

def calculate_hit_probs(
    to_hit: int,
    crit_hit_threshold: int,
    hit_rerolls: int | bool,
) -> tuple[float, float]:
    """
    Calculates p_crit_hit and p_normal_hit correctly accounting for
    rerolls that may include successful faces.

    Uses set arithmetic to precisely determine which faces are kept,
    which are rerolled, and what the rerolled dice contribute.
    """
    faces_crit   = set(range(crit_hit_threshold, 7))
    faces_hit    = set(range(to_hit, crit_hit_threshold))
    faces_reroll = set(range(1, hit_rerolls + 1)) if hit_rerolls is not False else set()

    p_crit_raw = len(faces_crit) / 6
    p_hit_raw  = len(faces_hit)  / 6

    p_crit_kept    = len(faces_crit - faces_reroll) / 6
    p_hit_kept     = len(faces_hit  - faces_reroll) / 6
    p_total_reroll = len(faces_reroll)              / 6

    p_crit_hit   = p_crit_kept + p_total_reroll * p_crit_raw
    p_normal_hit = p_hit_kept  + p_total_reroll * p_hit_raw

    return p_crit_hit, p_normal_hit


def calculate_wound_probs(
    to_wound: int,
    crit_wound_threshold: int,
    wound_rerolls: int | bool,
) -> tuple[float, float]:
    """
    Calculates p_crit_wound and p_normal_wound correctly accounting for
    rerolls that may include successful faces.
    """
    faces_crit_w   = set(range(crit_wound_threshold, 7))
    faces_hit_w    = set(range(to_wound, crit_wound_threshold))
    faces_reroll_w = set(range(1, wound_rerolls + 1)) if wound_rerolls is not False else set()

    p_crit_wound_raw   = len(faces_crit_w) / 6
    p_normal_wound_raw = len(faces_hit_w)  / 6

    p_crit_wound_kept   = len(faces_crit_w - faces_reroll_w) / 6
    p_normal_wound_kept = len(faces_hit_w  - faces_reroll_w) / 6
    p_total_wreroll     = len(faces_reroll_w)                / 6

    p_crit_wound   = p_crit_wound_kept   + p_total_wreroll * p_crit_wound_raw
    p_normal_wound = p_normal_wound_kept + p_total_wreroll * p_normal_wound_raw

    return p_crit_wound, p_normal_wound


def calculate_attack_distribution(
    attack_input: str,
    to_hit: int,
    crit_hits: int | bool = False,
    hit_rerolls: int | bool = False,
    sussy: bool = False,
    lethal: bool = False,
    to_wound: int = 4,
    crit_wounds: int | bool = False,
    wound_rerolls: int | bool = False,
    devvy: bool = False,
    to_save: int | bool = False,
) -> dict:

    # --- Parse attack input ---
    attack_dist = parse_attack_input(attack_input)
    if attack_dist is None:
        raise ValueError("Invalid attack input. Please enter a number or a dice value e.g. 10, d6, 2d6, d6+3")

    # --- Input Validation ---
    if not (2 <= to_hit <= 6):
        raise ValueError("to_hit must be between 2 and 6")
    if crit_hits is not False and not (2 <= crit_hits <= 6):
        raise ValueError("crit_hits must be False or an integer between 2 and 6")
    if hit_rerolls is not False and not (1 <= hit_rerolls <= 5):
        raise ValueError("hit_rerolls must be False or an integer between 1 and 5")
    if not (2 <= to_wound <= 6):
        raise ValueError("to_wound must be between 2 and 6")
    if crit_wounds is not False and not (2 <= crit_wounds <= 6):
        raise ValueError("crit_wounds must be False or an integer between 2 and 6")
    if wound_rerolls is not False and not (1 <= wound_rerolls <= 5):
        raise ValueError("wound_rerolls must be False or an integer between 1 and 5")
    if to_save is not False and not (2 <= to_save <= 6):
        raise ValueError("to_save must be false or between 2 and 6")

    # --- Hit probabilities ---
    crit_hit_threshold = crit_hits if crit_hits is not False else 6
    p_crit_hit, p_normal_hit = calculate_hit_probs(to_hit, crit_hit_threshold, hit_rerolls)

    # --- Wound probabilities ---
    crit_wound_threshold = crit_wounds if crit_wounds is not False else 6
    p_crit_wound, p_normal_wound = calculate_wound_probs(to_wound, crit_wound_threshold, wound_rerolls)

    # --- Save probability ---
    p_fail_save = 1.0 if to_save is False else (to_save - 1) / 6

    # --- Single attack wound distribution ---
    def single_attack_wound_dist() -> dict[int, float]:
        dist   = defaultdict(float)
        p_miss = 1 - p_crit_hit - p_normal_hit
        dist[0] += p_miss

        def wound_roll_unsaved(p_crit_w, p_norm_w, p_fail_s, lethal_auto=False) -> dict[int, float]:
            d = defaultdict(float)
            if lethal_auto:
                d[1] += p_fail_s
                d[0] += (1 - p_fail_s)
            else:
                if devvy:
                    d[1] += p_crit_w
                else:
                    d[1] += p_crit_w * p_fail_s
                    d[0] += p_crit_w * (1 - p_fail_s)
                d[1] += p_norm_w * p_fail_s
                d[0] += p_norm_w * (1 - p_fail_s)
                d[0] += 1 - p_crit_w - p_norm_w
            return d

        def convolve_dists(d1: dict, d2: dict) -> dict[int, float]:
            result = defaultdict(float)
            for w1, p1 in d1.items():
                for w2, p2 in d2.items():
                    result[w1 + w2] += p1 * p2
            return result

        normal_hit_wound_dist = wound_roll_unsaved(p_crit_wound, p_normal_wound, p_fail_save)
        for w, p in normal_hit_wound_dist.items():
            dist[w] += p_normal_hit * p

        if lethal:
            lethal_wound_dist = wound_roll_unsaved(p_crit_wound, p_normal_wound, p_fail_save, lethal_auto=True)
            if sussy:
                combined = convolve_dists(lethal_wound_dist, normal_hit_wound_dist)
                for w, p in combined.items():
                    dist[w] += p_crit_hit * p
            else:
                for w, p in lethal_wound_dist.items():
                    dist[w] += p_crit_hit * p
        else:
            crit_as_wound_dist = wound_roll_unsaved(p_crit_wound, p_normal_wound, p_fail_save)
            if sussy:
                combined = convolve_dists(crit_as_wound_dist, normal_hit_wound_dist)
                for w, p in combined.items():
                    dist[w] += p_crit_hit * p
            else:
                for w, p in crit_as_wound_dist.items():
                    dist[w] += p_crit_hit * p

        return dist

    single_dist = single_attack_wound_dist()

    # --- Convolve across attack distribution ---
    full_dist = defaultdict(float)

    for num_attacks, attack_prob in attack_dist.items():
        attack_wound_dist = defaultdict(float)
        attack_wound_dist[0] = 1.0

        for _ in range(num_attacks):
            new_dist = defaultdict(float)
            for w1, p1 in attack_wound_dist.items():
                for w2, p2 in single_dist.items():
                    new_dist[w1 + w2] += p1 * p2
            attack_wound_dist = new_dist

        for w, p in attack_wound_dist.items():
            full_dist[w] += p * attack_prob

    # --- Summary statistics ---
    wounds   = np.array(sorted(full_dist.keys()))
    probs    = np.array([full_dist[w] for w in wounds])

    mean     = float(np.sum(wounds * probs))
    variance = float(np.sum((wounds - mean) ** 2 * probs))
    std_dev  = float(np.sqrt(variance))

    cdf    = np.cumsum(probs)
    q1     = int(wounds[np.searchsorted(cdf, 0.25)])
    median = int(wounds[np.searchsorted(cdf, 0.50)])
    q3     = int(wounds[np.searchsorted(cdf, 0.75)])
    mode   = int(wounds[np.argmax(probs)])

    mask   = probs >= 0.01
    wounds = wounds[mask]
    probs  = probs[mask]

    return {
        "wounds"        : wounds,
        "probs"         : probs,
        "mean"          : mean,
        "median"        : median,
        "mode"          : mode,
        "std_dev"       : std_dev,
        "q1"            : q1,
        "q3"            : q3,
        "attack_input"  : attack_input,
        "to_hit"        : to_hit,
        "to_wound"      : to_wound,
        "to_save"       : to_save,
        "sussy"         : sussy,
        "lethal"        : lethal,
        "devvy"         : devvy,
        "hit_rerolls"   : hit_rerolls,
        "wound_rerolls" : wound_rerolls,
        "crit_hits"     : crit_hits,
        "crit_wounds"   : crit_wounds,
    }


# =========================================================================
# PLOTTING
# =========================================================================

def plot_distribution(results: dict) -> plt.Figure:
    wounds  = results["wounds"]
    probs   = results["probs"]
    mean    = results["mean"]
    median  = results["median"]
    mode    = results["mode"]
    std_dev = results["std_dev"]
    q1      = results["q1"]
    q3      = results["q3"]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    bar_colors = ["#FF6359" if w == mode else "#41605E" for w in wounds]
    bars = ax.bar(wounds, probs * 100, color=bar_colors, edgecolor="#2e2e3e", linewidth=0.5, zorder=3)

    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

    cumulative_or_better = np.array([np.sum(probs[i:]) for i in range(len(wounds))])

    for bar, prob, cum_prob in zip(bars, probs, cumulative_or_better):
        bar_x      = bar.get_x() + bar.get_width() / 2
        bar_height = bar.get_height()
        ax.text(bar_x, bar_height + 0.3,       f"{prob * 100:.1f}%",       ha="center", va="bottom", color="#A0AFAF", fontsize=7.5)
        ax.text(bar_x, bar_height + 0.3 + 1.5, f"≥{cum_prob * 100:.1f}%", ha="center", va="bottom", color="#FF6359", fontsize=7.5)

    ax.axvspan(q1 - 0.5, q3 + 0.5, alpha=0.15, color="#A0AFAF", label="IQR (Q1–Q3)", zorder=2)
    ax.axvline(mean,   color="#FF6359", linestyle="--", linewidth=1.5, label=f"Mean: {mean:.2f}", zorder=4)
    ax.axvline(median, color="#A0AFAF", linestyle=":",  linewidth=1.5, label=f"Median: {median}", zorder=4)

    ax.set_xlabel("Unsaved Wounds", color="#A0AFAF", fontsize=12)
    ax.set_ylabel("Probability (%)", color="#A0AFAF", fontsize=12)

    # --- Build subtitle ---
    save_label = "None" if results["to_save"] is False else f"{results['to_save']}+"

    hit_mods = [f"Hit: {results['to_hit']}+"]
    if results["crit_hits"] is not False and results["crit_hits"] != 6:
        hit_mods.append(f"crit {results['crit_hits']}+")
    if results["lethal"]: hit_mods.append("lethal")
    if results["sussy"]:  hit_mods.append("sustained")
    if results["hit_rerolls"] is not False:
        rr = results["hit_rerolls"]
        hit_mods.append(f"re-rolling {'1s' if rr == 1 else f'1-{rr}'}")

    wound_mods = [f"Wound: {results['to_wound']}+"]
    if results["crit_wounds"] is not False and results["crit_wounds"] != 6:
        wound_mods.append(f"crit {results['crit_wounds']}+")
    if results["devvy"]: wound_mods.append("devastating")
    if results["wound_rerolls"] is not False:
        rr = results["wound_rerolls"]
        wound_mods.append(f"re-rolling {'1s' if rr == 1 else f'1-{rr}'}")

    ax.set_title(
        f"Warhammer 40K — Unsaved Wound Distribution\n"
        f"{results['attack_input']} Attacks | {', '.join(hit_mods)} | {', '.join(wound_mods)} | Save: {save_label}",
        color="white", fontsize=13, pad=15
    )

    ax.tick_params(colors="#A0AFAF")
    ax.set_xticks(wounds)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3e3e4e")

    ax.yaxis.grid(True, color="#3e3e4e", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    stats_text = (
        f"  Mean:    {mean:.2f}\n"
        f"  Median:  {median}\n"
        f"  Mode:    {mode}\n"
        f"  Std Dev: {std_dev:.2f}\n"
        f"  Q1:      {q1}\n"
        f"  Q3:      {q3}"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10, verticalalignment="top", horizontalalignment="right",
        color="#A0AFAF",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2e2e3e", edgecolor="#3e3e4e", alpha=0.9)
    )
    ax.legend(loc="upper left", facecolor="#2e2e3e", edgecolor="#3e3e4e", labelcolor="#A0AFAF")

    return fig


# =========================================================================
# STREAMLIT UI
# =========================================================================

def parse_dropdown(val: str) -> int:
    return int(''.join(filter(str.isdigit, val)))

st.set_page_config(page_title="MatHammer", layout="wide")
st.title("Warhammer 40k Attack Sequence Statistics")

# --- Initialise session state ---
if "plots" not in st.session_state:
    st.session_state.plots = []

plus_options   = ["2+", "3+", "4+", "5+", "6+"]
reroll_options = ["None", "1s", "2s and under", "3s and under", "4s and under", "5s and under"]

# --- Sidebar controls ---
with st.sidebar:
    st.header("Attack Parameters")

    attack_input = st.text_input(
        "Num Attacks",
        value="10",
        help="Enter a number (e.g. 10), a die (e.g. d6), multiple dice (e.g. 2d6), or dice with modifier (e.g. 2d6+3)"
    )
    if parse_attack_input(attack_input) is None:
        st.error("Invalid input. Please enter a number (e.g. 10), or a dice value (e.g. d6, 2d6, d6+3). Supported dice: d2, d3, d6.")

    st.subheader("Hit Roll")
    to_hit      = st.selectbox("To Hit",        plus_options,   index=1)
    crit_hits   = st.selectbox("Crit Hits",      plus_options,   index=4)
    hit_rerolls = st.selectbox("Hit Rerolls",    reroll_options, index=0)
    sussy       = st.selectbox("Sustained Hits", ["No", "Yes"],  index=0)
    lethal      = st.selectbox("Lethal Hits",    ["No", "Yes"],  index=0)

    st.subheader("Wound Roll")
    to_wound    = st.selectbox("To Wound",       plus_options,   index=2)
    crit_wounds = st.selectbox("Crit Wounds",    plus_options,   index=4)
    wound_rr    = st.selectbox("Wound Rerolls",  reroll_options, index=0)
    devvy       = st.selectbox("Dev. Wounds",    ["No", "Yes"],  index=0)

    st.subheader("Save Roll")
    to_save     = st.selectbox("To Save", ["No Save"] + plus_options, index=0)

    col1, col2  = st.columns(2)
    with col1:
        calculate = st.button("Calculate",   use_container_width=True)
    with col2:
        clear     = st.button("Clear Plots", use_container_width=True)

# --- Clear plots ---
if clear:
    st.session_state.plots = []

# --- Calculate and store new plot ---
if calculate:
    try:
        results = calculate_attack_distribution(
            attack_input  = attack_input,
            to_hit        = parse_dropdown(to_hit),
            crit_hits     = parse_dropdown(crit_hits),
            hit_rerolls   = False if hit_rerolls == "None" else parse_dropdown(hit_rerolls),
            sussy         = sussy == "Yes",
            lethal        = lethal == "Yes",
            to_wound      = parse_dropdown(to_wound),
            crit_wounds   = parse_dropdown(crit_wounds),
            wound_rerolls = False if wound_rr == "None" else parse_dropdown(wound_rr),
            devvy         = devvy == "Yes",
            to_save       = False if to_save == "No Save" else parse_dropdown(to_save),
        )
        fig = plot_distribution(results)
        st.session_state.plots.insert(0, fig)

    except Exception as e:
        st.error(f"Error: {e}")

# --- Render all stored plots ---
if st.session_state.plots:
    for fig in st.session_state.plots:
        st.pyplot(fig)
else:
    st.info("Configure your attack parameters in the sidebar and press Calculate.")
