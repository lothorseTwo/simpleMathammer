import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict

# =========================================================================
# PROBABILITY CALCULATION
# =========================================================================

def calculate_attack_distribution(
    num_attacks: int,
    to_hit: int,
    crit_hits: int | bool = False,
    hit_rerolls: int | bool = False,
    sussy: bool = False,
    lethal: bool = False,
    to_wound: int = 4,
    crit_wounds: int | bool = False,
    wound_rerolls: int | bool = False,
    devvy: bool = False,
    to_save: int = 6,
) -> dict:

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
    if not (2 <= to_save <= 6):
        raise ValueError("to_save must be between 2 and 6")

    crit_hit_threshold = crit_hits if crit_hits is not False else 6
    p_crit_raw         = (7 - crit_hit_threshold) / 6
    p_hit_raw          = max(0, (crit_hit_threshold - to_hit)) / 6

    if hit_rerolls is not False:
        p_reroll     = hit_rerolls / 6
        p_crit_hit   = p_crit_raw + p_reroll * p_crit_raw
        p_normal_hit = p_hit_raw  + p_reroll * p_hit_raw
    else:
        p_crit_hit   = p_crit_raw
        p_normal_hit = p_hit_raw

    crit_wound_threshold = crit_wounds if crit_wounds is not False else 6
    p_crit_wound_raw     = (7 - crit_wound_threshold) / 6
    p_normal_wound_raw   = max(0, (crit_wound_threshold - to_wound)) / 6

    if wound_rerolls is not False:
        p_wreroll      = wound_rerolls / 6
        p_crit_wound   = p_crit_wound_raw  + p_wreroll * p_crit_wound_raw
        p_normal_wound = p_normal_wound_raw + p_wreroll * p_normal_wound_raw
    else:
        p_crit_wound   = p_crit_wound_raw
        p_normal_wound = p_normal_wound_raw

    p_fail_save = (to_save - 1) / 6

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

    single_dist  = single_attack_wound_dist()
    full_dist    = defaultdict(float)
    full_dist[0] = 1.0

    for _ in range(num_attacks):
        new_dist = defaultdict(float)
        for w1, p1 in full_dist.items():
            for w2, p2 in single_dist.items():
                new_dist[w1 + w2] += p1 * p2
        full_dist = new_dist

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
        "wounds"      : wounds,
        "probs"       : probs,
        "mean"        : mean,
        "median"      : median,
        "mode"        : mode,
        "std_dev"     : std_dev,
        "q1"          : q1,
        "q3"          : q3,
        "num_attacks" : num_attacks,
        "to_hit"      : to_hit,
        "to_wound"    : to_wound,
        "to_save"     : to_save,
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
    ax.set_title(
        f"Warhammer 40K — Unsaved Wound Distribution\n"
        f"{results['num_attacks']} Attacks | Hit {results['to_hit']}+ | Wound {results['to_wound']}+ | Save {results['to_save']}+",
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
    return int(val.replace("s and under", "").replace("s", "").replace("+", ""))

st.set_page_config(page_title="MatHammer", layout="wide")
st.title("Warhammer 40k Attack Sequence Statistics")

plus_options   = ["2+", "3+", "4+", "5+", "6+"]
reroll_options = ["None", "1s", "2s and under", "3s and under", "4s and under", "5s and under"]

# --- Sidebar controls ---
with st.sidebar:
    st.header("Attack Parameters")

    num_attacks = st.number_input("Num Attacks", min_value=1, max_value=1000, value=10, step=1)

    st.subheader("Hit Roll")
    to_hit       = st.selectbox("To Hit",         plus_options,   index=1)
    crit_hits    = st.selectbox("Crit Hits",       plus_options,   index=4)
    hit_rerolls  = st.selectbox("Hit Rerolls",     reroll_options, index=0)
    sussy        = st.selectbox("Sustained Hits",  ["No", "Yes"],  index=0)
    lethal       = st.selectbox("Lethal Hits",     ["No", "Yes"],  index=0)

    st.subheader("Wound Roll")
    to_wound     = st.selectbox("To Wound",        plus_options,   index=2)
    crit_wounds  = st.selectbox("Crit Wounds",     plus_options,   index=4)
    wound_rr     = st.selectbox("Wound Rerolls",   reroll_options, index=0)
    devvy        = st.selectbox("Dev. Wounds",     ["No", "Yes"],  index=0)

    st.subheader("Save Roll")
    to_save      = st.selectbox("To Save",         plus_options,   index=4)

    calculate    = st.button("Calculate", use_container_width=True)

# --- Main area ---
if calculate:
    try:
        results = calculate_attack_distribution(
            num_attacks   = int(num_attacks),
            to_hit        = parse_dropdown(to_hit),
            crit_hits     = parse_dropdown(crit_hits),
            hit_rerolls   = False if hit_rerolls == "None" else parse_dropdown(hit_rerolls),
            sussy         = sussy == "Yes",
            lethal        = lethal == "Yes",
            to_wound      = parse_dropdown(to_wound),
            crit_wounds   = parse_dropdown(crit_wounds),
            wound_rerolls = False if wound_rr == "None" else parse_dropdown(wound_rr),
            devvy         = devvy == "Yes",
            to_save       = parse_dropdown(to_save),
        )
        fig = plot_distribution(results)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Configure your attack parameters in the sidebar and press Calculate.")
