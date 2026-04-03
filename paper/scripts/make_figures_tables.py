import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
mpl.rcParams['ps.fonttype']=42
mpl.rcParams['font.family']='DejaVu Sans'
#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "paper_data.json").read_text())
FIG = ROOT / "figures"
TAB = ROOT / "tables"

name_map = {"planted": "S1 planted", "mini_ioi": "S2 mini-IOI", "gpt2_ioi": "S3 GPT-2 small IOI"}

def draw_box(ax, xy, w, h, text, fc="#f7f7f7", ec="#333333", lw=1.2, fontsize=10):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.025",
                           linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return patch

def make_method_figure():
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    draw_box(ax, (0.02, 0.58), 0.23, 0.30,
             "Candidate abstraction\n$A=(H,V_{\\mathrm{int}},b,\\{S_v\\},\\{\\tau_v\\},m,h)$\n"
             "$V_{\\mathrm{int}}=\\{N_1,N_2,R\\}$\natomic residual-stream sites only",
             fc="#edf4ff", ec="#2b5fab", fontsize=10)
    draw_box(ax, (0.31, 0.66), 0.23, 0.22,
             "Structural bits\n$L_{\\mathrm{struct}}(A)$\nhigh-level model + budget +\nsite groups + family/hyper +\nparameter code",
             fc="#fff3e8", ec="#b76e00", fontsize=9.5)
    draw_box(ax, (0.31, 0.34), 0.23, 0.22,
             "Residual bits\n$L_{\\mathrm{res}}(A;D_{\\mathrm{test}})$\ncandidate-specific patching\n+ symbolic prediction\n+ validation-set $\\epsilon_A$",
             fc="#f5eefe", ec="#7a4fb3", fontsize=9.5)
    draw_box(ax, (0.60, 0.58), 0.17, 0.30,
             "Matched nulls\nrandom_site\nshuffled_pair\nuntrained_model\n(exact budget/family matching)",
             fc="#eef9f0", ec="#2f8f46", fontsize=10)
    draw_box(ax, (0.81, 0.58), 0.16, 0.30,
             "Balanced null frontier\n2-bit structural bins\nvalid if each family has $\\geq 5$\nno extrapolation",
             fc="#eef9f0", ec="#2f8f46", fontsize=10)
    draw_box(ax, (0.58, 0.12), 0.39, 0.24,
             "Support rule\n1. candidate is in the best-bits class\n($\\leq$ best $+0.01$ bits/example)\n"
             "2. grouped-bootstrap LCB of $g_{\\mathrm{test}}$ is $>0$\n3. $g_{\\mathrm{shift}}>0$ without refitting",
             fc="#fff8d9", ec="#9b870c", fontsize=10)

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.5, color="#444")
    for a, b, cs in [((0.25,0.73),(0.31,0.77),None),
                     ((0.25,0.63),(0.31,0.45),None),
                     ((0.54,0.77),(0.60,0.73),None),
                     ((0.77,0.73),(0.81,0.73),None),
                     ((0.425,0.34),(0.71,0.36),"arc3,rad=-0.2"),
                     ((0.89,0.58),(0.86,0.36),None)]:
        kw = dict(arrow_kw)
        if cs is not None:
            kw["connectionstyle"] = cs
        ax.add_patch(FancyArrowPatch(a, b, **kw))
    ax.text(0.02, 0.97, "Locked control-calibrated acceptance pipeline",
            fontsize=13, fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.03,
            "Output is a supported abstraction class, not necessarily a unique interpretation. "
            "In the full-locked reruns, no setting certifies a supported class.",
            fontsize=9.5, ha="left", va="bottom")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "method_overview.pdf", bbox_inches="tight")
    fig.savefig(FIG / "method_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def make_summary_df():
    rows = []
    for s in DATA["settings"]:
        sid = s["setting_id"]
        cp = s["candidate_pool_scope"]
        pr = s["primary"]
        oracle = s.get("oracle")
        # values already stored in final summary
        best_bits = None  # not needed in table generation here
        rows.append({
            "sid": sid,
            "label": name_map[sid],
            "cands": cp["recorded_candidate_records"],
            "unev": cp["recorded_unevaluable_candidate_cells"],
            "valid_test": pr["test_valid_bins"],
            "valid_shift": pr["shift_valid_bins"],
            "best_eligible": pr["best_candidate_frontier_eligible"],
            "fd_bits": pr["best_frontier_defined_candidate_test_bits_per_example"],
            "fd_gtest": pr["best_frontier_defined_candidate_g_test"],
            "fd_gshift": pr["best_frontier_defined_candidate_g_shift"],
            "fd_within_best": pr["best_frontier_defined_candidate_within_best_bits"],
            "n_supported": pr["n_supported"],
            "calib_changes": pr["control_calibration_changed_decision"],
            "best_changed": s["best_candidate_changed"],
            "support_changed": s["support_changed"],
            "oracle_gtest": None if oracle is None else oracle["g_test"],
            "oracle_gshift": None if oracle is None else oracle["g_shift"],
        })
    return pd.DataFrame(rows)

def make_summary_panels():
    # These exact values come from the full-locked final package.
    labels = ["S1\nplanted", "S2\nmini-IOI", "S3\nGPT-2\nsmall IOI"]
    recorded = np.array([120, 116, 92])
    unev = np.array([0, 4, 28])
    delta = np.array([2.007500926773174, 0.06604010419670203, 0.19812031259016294])
    gtest = np.array([0.0, -0.1548902260915881, -2.3398500028846243])
    gshift = np.array([0.0, -15.582107852061, -37.015904391262076])

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.1), gridspec_kw={"width_ratios":[1.1,1.0,1.35]})
    x = np.arange(3)

    ax = axes[0]
    ax.bar(x, recorded, width=0.6, label="recorded", edgecolor="black", linewidth=0.5)
    ax.bar(x, unev, bottom=recorded, width=0.6, label="unevaluable", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_ylim(0, 132); ax.set_ylabel("candidate cells", fontsize=9)
    ax.set_title("Full-locked candidate-pool coverage", fontsize=10.2)
    for i, (r, u) in enumerate(zip(recorded, unev)):
        ax.text(i, r+u+2, f"{int(r+u)}/120", ha="center", va="bottom", fontsize=8.3)
        if u > 0:
            ax.text(i, r+u/2, f"{int(u)}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.7, loc="upper right")
    ax.tick_params(axis="y", labelsize=8.5)

    ax = axes[1]
    ax.axhline(0.01, color="black", linestyle="--", linewidth=1, label="best + 0.01 gate")
    ax.bar(x, delta, width=0.6, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_ylabel(r"$\Delta$ bits/example", fontsize=9)
    ax.set_title("Closest frontier-defined candidate", fontsize=10.2)
    for i, v in enumerate(delta):
        ax.text(i, v + (0.03 if v < 1 else 0.05), f"{v:.3f}", ha="center", va="bottom", fontsize=8.0)
    ax.set_ylim(0, 2.55)
    ax.legend(frameon=False, fontsize=7.3, loc="upper left")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.text(0.5, 0.79, "All global best candidates\nare frontier-ineligible",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7"))

    ax = axes[2]
    w = 0.32
    ax.axhline(0, color="black", linewidth=1)
    ax.bar(x-w/2, gtest, width=w, label=r"$g_{\mathrm{test}}$", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, gshift, width=w, label=r"$g_{\mathrm{shift}}$", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.7)
    ax.set_title("Best frontier-defined null gaps", fontsize=10.2)
    ax.set_ylabel("gap (frontier - candidate)", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(-50, 6)
    ax.tick_params(axis="y", labelsize=8.5)
    for i, v in enumerate(gtest):
        ax.text(i-w/2, v + (0.8 if v >= 0 else -2.0), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.8)
    for i, v in enumerate(gshift):
        ax.text(i+w/2, v + (0.8 if v >= 0 else -2.0), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.8)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIG / "summary_panels.pdf", bbox_inches="tight")
    fig.savefig(FIG / "summary_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_method_figure()
    make_summary_panels()
