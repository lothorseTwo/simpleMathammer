import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import tkinter as tk
from tkinter import ttk

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
    p_miss_raw         = (to_hit - 1) / 6

    if hit_rerolls is not False:
        p_reroll     = hit_rerolls / 6
        p_crit_hit   = p_crit_raw   + p_reroll * p_crit_raw
        p_normal_hit = p_hit_raw    + p_reroll * p_hit_raw
    else:
        p_crit_hit   = p_crit_raw
        p_normal_hit = p_hit_raw

    crit_wound_threshold = crit_wounds if crit_wounds is not False else 6
    p_crit_wound_raw     = (7 - crit_wound_threshold) / 6
    p_normal_wound_raw   = max(0, (crit_wound_threshold - to_wound)) / 6

    if wound_rerolls is not False:
        p_wreroll      = wound_rerolls / 6
        p_crit_wound   = p_crit_wound_raw   + p_wreroll * p_crit_wound_raw
        p_normal_wound = p_normal_wound_raw  + p_wreroll * p_normal_wound_raw
    else:
        p_crit_wound   = p_crit_wound_raw
        p_normal_wound = p_normal_wound_raw

    p_fail_save = (to_save - 1) / 6

    def single_attack_wound_dist() -> dict[int, float]:
        dist = defaultdict(float)
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
                p_miss_w = 1 - p_crit_w - p_norm_w
                d[0] += p_miss_w
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

def plot_distribution(results: dict, ax, canvas):
    ax.clear()
    ax.set_facecolor("#1e1e2e")

    wounds  = results["wounds"]
    probs   = results["probs"]
    mean    = results["mean"]
    median  = results["median"]
    mode    = results["mode"]
    std_dev = results["std_dev"]
    q1      = results["q1"]
    q3      = results["q3"]

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

    canvas.draw()


# =========================================================================
# UI
# =========================================================================

def build_ui():
    root = tk.Tk()
    root.title("Warhammer 40K Attack Calculator")
    root.configure(bg="#1e1e2e")

    # --- Styling ---
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel",    background="#1e1e2e", foreground="#A0AFAF", font=("Helvetica", 10))
    style.configure("TFrame",    background="#1e1e2e")
    style.configure("TCombobox", fieldbackground="#ffffff", background="#ffffff", foreground="#000000", font=("Helvetica", 10))
    style.configure("TButton",   background="#41605E", foreground="white", font=("Helvetica", 11, "bold"), padding=6)
    style.configure("TEntry",    fieldbackground="#ffffff", foreground="#000000", font=("Helvetica", 10))
    style.map("TButton", background=[("active", "#FF6359")])

    # --- Layout frames ---
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(side=tk.LEFT, fill=tk.Y)

    chart_frame = ttk.Frame(root)
    chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # --- Matplotlib canvas ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e2e")
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Helper to create a labelled dropdown ---
    def make_dropdown(parent, label, options, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        var = tk.StringVar(value=default)
        cb  = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=10)
        cb.grid(row=row, column=1, padx=8, pady=4)
        return var

    # --- Helper to parse "None" / bool / int from dropdown ---
    def parse_value(val: str) -> int | bool:
        if val == "None": return False
        if val == "Yes":  return True
        if val == "No":   return False
        return int(val.replace("+", ""))

    # --- Number of attacks field (numbers only) ---
    ttk.Label(control_frame, text="Num Attacks").grid(row=0, column=0, sticky=tk.W, pady=4)
    vcmd = root.register(lambda P: P.isdigit() or P == "")
    num_attacks_var = tk.StringVar(value="10")
    ttk.Entry(
        control_frame, textvariable=num_attacks_var,
        validate="key", validatecommand=(vcmd, "%P"), width=12
    ).grid(row=0, column=1, padx=8, pady=4)

    # --- Dropdowns ---
    to_hit_var      = make_dropdown(control_frame, "To Hit",         ["2+","3+","4+","5+","6+"],              "3+",   row=1)
    crit_hits_var   = make_dropdown(control_frame, "Crit Hits",      ["2+","3+","4+","5+","6+"],              "6+",   row=2)
    hit_rerolls_var = make_dropdown(control_frame, "Hit Rerolls",    ["None","1+","2+","3+","4+","5+"],       "None", row=3)
    sussy_var       = make_dropdown(control_frame, "Sustained Hits", ["No","Yes"],                            "No",   row=4)
    lethal_var      = make_dropdown(control_frame, "Lethal Hits",    ["No","Yes"],                            "No",   row=5)
    to_wound_var    = make_dropdown(control_frame, "To Wound",       ["2+","3+","4+","5+","6+"],              "4+",   row=6)
    crit_wounds_var = make_dropdown(control_frame, "Crit Wounds",    ["2+","3+","4+","5+","6+"],              "6+",   row=7)
    wound_rr_var    = make_dropdown(control_frame, "Wound Rerolls",  ["None","1+","2+","3+","4+","5+"],       "None", row=8)
    devvy_var       = make_dropdown(control_frame, "Dev. Wounds",    ["No","Yes"],                            "No",   row=9)
    to_save_var     = make_dropdown(control_frame, "To Save",        ["2+","3+","4+","5+","6+"],              "6+",   row=10)

    # --- Error label ---
    error_var = tk.StringVar()
    ttk.Label(control_frame, textvariable=error_var, foreground="#FF6359").grid(row=12, column=0, columnspan=2, pady=4)

    # --- Calculate button callback ---
    def on_calculate():
        error_var.set("")
        try:
            num_attacks = int(num_attacks_var.get())
            if num_attacks < 1:
                raise ValueError("Number of attacks must be at least 1")

            results = calculate_attack_distribution(
                num_attacks   = num_attacks,
                to_hit        = parse_value(to_hit_var.get()),
                crit_hits     = parse_value(crit_hits_var.get()),
                hit_rerolls   = parse_value(hit_rerolls_var.get()),
                sussy         = parse_value(sussy_var.get()),
                lethal        = parse_value(lethal_var.get()),
                to_wound      = parse_value(to_wound_var.get()),
                crit_wounds   = parse_value(crit_wounds_var.get()),
                wound_rerolls = parse_value(wound_rr_var.get()),
                devvy         = parse_value(devvy_var.get()),
                to_save       = parse_value(to_save_var.get()),
            )
            plot_distribution(results, ax, canvas)

        except Exception as e:
            error_var.set(f"Error: {e}")

    ttk.Button(control_frame, text="Calculate", command=on_calculate).grid(
        row=11, column=0, columnspan=2, pady=12, sticky=tk.EW
    )

    root.mainloop()


if __name__ == "__main__":
    build_ui()
