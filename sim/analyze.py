#!/usr/bin/env python3
"""
Analysis + figure generation for the incentive-allocation study.

AI-assisted implementation (large language model). Figures are AI-generated data
visualizations, marked as such in their captions. This script computes descriptive
statistics and paired contrasts and renders them; it draws no conclusions.

Reads ../results/full.csv (produced by `sim.py full`).
Writes ../results/*.csv tables and ../results/figs/*.pdf|png figures.

Requires numpy, scipy, pandas, matplotlib (figure/stat tooling only; the data of
record in full.csv was produced by the stdlib-only sim.py).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RES, "figs")
os.makedirs(FIGS, exist_ok=True)

POLICY_LABEL = {
    "P0": "Random", "P1": "Monetary-only", "P2": "Static taxonomy\n(theory baseline)",
    "P3": "Learner\n(retention reward)", "P4": "Learner + audit",
    "P5": "Oracle\n(true satisfaction)",
}
POLICY_SHORT = {"P0": "Random", "P1": "Mon-only", "P2": "Static", "P3": "Learner",
                "P4": "Learner+audit", "P5": "Oracle"}
CATS = ["M", "R", "F", "D", "S"]
CAT_NAME = {"M": "Monetary", "R": "Recognition", "F": "Flexibility",
            "D": "Development", "S": "Social"}

# Muted, color-blind-safe, print-honest palette (Okabe-Ito derived).
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#8256A8", grey="#7A7A7A", teal="#56B4E9", ink="#222222")
POLICY_C = {"P0": C["grey"], "P1": C["orange"], "P2": C["green"],
            "P3": C["red"], "P4": C["blue"], "P5": C["purple"]}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return 0.0
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def load():
    df = pd.read_csv(os.path.join(RES, "full.csv"))
    return df


# ------------------------------ tables ------------------------------

def summary_table(df):
    """Per-policy descriptive stats at each handcuff level (P4 at p=0.10)."""
    rows = []
    for h in sorted(df.handcuff.unique()):
        for pol in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            if pol == "P4":
                sub = df[(df.handcuff == h) & (df.policy == pol) & (df.p_audit_cfg == 0.10)]
            else:
                sub = df[(df.handcuff == h) & (df.policy == pol) & (df.p_audit_cfg == 0.15)]
            if sub.empty:
                continue
            rows.append(dict(
                handcuff=h, policy=pol, n=len(sub),
                attr=sub.attr_rate.mean(), attr_ci=ci95(sub.attr_rate),
                sat=sub.mean_s.mean(), sat_ci=ci95(sub.mean_s),
                gap=sub.arch_gap.mean(), gap_ci=ci95(sub.arch_gap),
                share_M=sub.share_M.mean(),
            ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "table_summary.csv"), index=False)
    return out


def paired_contrast(df, pol_a, pol_b, h, metric, pa_a=0.15, pa_b=0.15):
    """Paired-by-seed contrast pol_a - pol_b at handcuff h. Returns dict.

    Primary evidence = mean_diff with Monte-Carlo SE and 95% CI, plus per-seed
    SIGN-STABILITY. cohen_dz is reported but is WITHIN-MODEL SEPARATION (it
    inflates with employees/seed), NOT a behavioral effect size — do not headline
    it (see VERIFICATION_REPORT.md).
    """
    a = df[(df.policy == pol_a) & (df.handcuff == h) & (df.p_audit_cfg == pa_a)].set_index("seed")[metric]
    b = df[(df.policy == pol_b) & (df.handcuff == h) & (df.p_audit_cfg == pa_b)].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    t, p = stats.ttest_rel(a.loc[common], b.loc[common])
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else 0.0
    mcse = sd / np.sqrt(n) if n else float("nan")
    sign_pos = float((d > 0).mean())
    return dict(pol_a=pol_a, pol_b=pol_b, handcuff=h, metric=metric, n=n,
                mean_diff=d.mean(), mcse=mcse, ci=ci95(d), t=t, p=p,
                sign_stability=max(sign_pos, 1 - sign_pos), cohen_dz_within_model=dz)


def contrasts_table(df):
    rows = []
    for h in sorted(df.handcuff.unique()):
        for metric in ["attr_rate", "mean_s", "arch_gap"]:
            rows.append(paired_contrast(df, "P3", "P2", h, metric))  # Goodhart core
            rows.append(paired_contrast(df, "P4", "P3", h, metric, pa_a=0.10))  # audit effect
            rows.append(paired_contrast(df, "P5", "P3", h, metric))  # oracle ceiling
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "table_contrasts.csv"), index=False)
    return out


def audit_frontier_table(df):
    rows = []
    for h in sorted(df.handcuff.unique()):
        base3 = df[(df.policy == "P3") & (df.handcuff == h) & (df.p_audit_cfg == 0.15)].set_index("seed").mean_s
        orc = df[(df.policy == "P5") & (df.handcuff == h) & (df.p_audit_cfg == 0.15)].set_index("seed").mean_s
        for pa in sorted(df[df.policy == "P4"].p_audit_cfg.unique()):
            sub = df[(df.policy == "P4") & (df.handcuff == h) & (df.p_audit_cfg == pa)].set_index("seed").mean_s
            common = sub.index.intersection(base3.index)
            recov_denom = (orc.loc[common].mean() - base3.loc[common].mean())
            recov = ((sub.loc[common].mean() - base3.loc[common].mean()) / recov_denom
                     if abs(recov_denom) > 1e-9 else float("nan"))
            rows.append(dict(handcuff=h, p_audit=pa, sat=sub.mean(), sat_ci=ci95(sub),
                             recovery_frac=recov))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "table_audit_frontier.csv"), index=False)
    return out


# ------------------------------ figures ------------------------------

def fig_goodhart(summ):
    """Fig 2: the money figure — observed KPI vs hidden KPI, main config h=1.0."""
    s = summ[summ.handcuff == 1.0].set_index("policy")
    order = ["P0", "P1", "P2", "P3", "P4", "P5"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
    # panel a: observed retention (1 - attrition), higher = better
    ret = 1 - s.loc[order, "attr"]
    ret_ci = s.loc[order, "attr_ci"]
    ax[0].bar(range(len(order)), ret, yerr=ret_ci, capsize=3,
              color=[POLICY_C[p] for p in order], edgecolor="white", linewidth=0.6)
    ax[0].set_ylim(0.90, 0.95)
    ax[0].set_ylabel("Observed retention / quarter")
    ax[0].set_title("(a) What the organization SEES", fontsize=9, loc="left")
    for i, (v, c) in enumerate(zip(ret, ret_ci)):
        ax[0].text(i, v + c + 0.0015, f"{v:.3f}", ha="center", fontsize=6.5)
    # panel b: latent satisfaction, higher = better
    sat = s.loc[order, "sat"]; sat_ci = s.loc[order, "sat_ci"]
    ax[1].bar(range(len(order)), sat, yerr=sat_ci, capsize=3,
              color=[POLICY_C[p] for p in order], edgecolor="white", linewidth=0.6)
    ax[1].set_ylim(0.44, 0.60)
    ax[1].set_ylabel("Latent satisfaction (unobserved)")
    ax[1].set_title("(b) What it CANNOT see", fontsize=9, loc="left")
    ax[1].axhline(s.loc["P2", "sat"], ls="--", lw=0.9, color=C["green"], alpha=0.7)
    for i, (v, c) in enumerate(zip(sat, sat_ci)):
        ax[1].text(i, v + c + 0.004, f"{v:.3f}", ha="center", fontsize=6.5)
    for a in ax:
        a.set_xticks(range(len(order)))
        a.set_xticklabels([POLICY_SHORT[p] for p in order], rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_goodhart.pdf"))
    fig.savefig(os.path.join(FIGS, "fig2_goodhart.png"))
    plt.close(fig)


def fig_handcuff_sweep(df, summ):
    """Fig 3: Goodhart gap grows with the handcuff (unmediated-retention) channel."""
    hs = sorted(df.handcuff.unique())
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for pol in ["P2", "P3", "P5"]:
        m = [summ[(summ.handcuff == h) & (summ.policy == pol)].sat.values[0] for h in hs]
        e = [summ[(summ.handcuff == h) & (summ.policy == pol)].sat_ci.values[0] for h in hs]
        ax[0].errorbar(hs, m, yerr=e, marker="o", ms=4, capsize=2.5,
                       color=POLICY_C[pol], label=POLICY_SHORT[pol])
    ax[0].set_xlabel("Handcuff strength $h$ (retention not via satisfaction)")
    ax[0].set_ylabel("Latent satisfaction")
    ax[0].set_title("(a) Hidden welfare vs. handcuff channel", fontsize=9, loc="left")
    ax[0].legend(fontsize=7)
    # panel b: paired Goodhart gap P2 - P3 in satisfaction
    gap, gci = [], []
    for h in hs:
        c = paired_contrast(df, "P2", "P3", h, "mean_s")
        gap.append(c["mean_diff"]); gci.append(c["ci"])
    ax[1].bar(range(len(hs)), gap, yerr=gci, capsize=3, color=C["red"],
              edgecolor="white", linewidth=0.6)
    ax[1].set_xticks(range(len(hs))); ax[1].set_xticklabels(hs)
    ax[1].set_xlabel("Handcuff strength $h$")
    ax[1].set_ylabel("Satisfaction penalty of learning\n(Static − Learner, paired)")
    ax[1].set_title("(b) Goodhart penalty (paired by seed)", fontsize=9, loc="left")
    ax[1].axhline(0, color=C["ink"], lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_handcuff.pdf"))
    fig.savefig(os.path.join(FIGS, "fig3_handcuff.png"))
    plt.close(fig)


def fig_audit_frontier(front):
    """Fig 4: does auditing satisfaction fix it? Recovery vs audit frequency."""
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    hs = sorted(front.handcuff.unique())
    cmap = {0.0: C["teal"], 0.5: C["blue"], 1.0: C["purple"], 1.5: C["red"]}
    for h in hs:
        sub = front[front.handcuff == h].sort_values("p_audit")
        ax[0].errorbar(sub.p_audit, sub.sat, yerr=sub.sat_ci, marker="o", ms=4,
                       capsize=2.5, color=cmap.get(h, C["grey"]), label=f"$h$={h}")
    ax[0].set_xlabel("Audit frequency $p$ (fraction surveyed / quarter)")
    ax[0].set_ylabel("Latent satisfaction (P4)")
    ax[0].set_title("(a) Audit dose–response", fontsize=9, loc="left")
    ax[0].legend(fontsize=7, title="handcuff", title_fontsize=7)
    for h in hs:
        sub = front[front.handcuff == h].sort_values("p_audit")
        ax[1].plot(sub.p_audit, sub.recovery_frac, marker="s", ms=4,
                   color=cmap.get(h, C["grey"]), label=f"$h$={h}")
    ax[1].axhline(1.0, ls="--", lw=0.8, color=C["ink"], alpha=0.6)
    ax[1].axhline(0.0, ls=":", lw=0.8, color=C["ink"], alpha=0.6)
    ax[1].set_xlabel("Audit frequency $p$")
    ax[1].set_ylabel("Recovery fraction toward oracle")
    ax[1].set_title("(b) Fraction of Goodhart gap closed", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_audit.pdf"))
    fig.savefig(os.path.join(FIGS, "fig4_audit.png"))
    plt.close(fig)


def fig_allocation(df):
    """Fig 5: what each policy actually hands out (category mix), h=1.0."""
    order = ["P2", "P3", "P5"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bottom = np.zeros(len(order))
    catC = {"M": C["red"], "R": C["orange"], "F": C["blue"], "D": C["green"], "S": C["purple"]}
    for cat in CATS:
        vals = [df[(df.policy == p) & (df.handcuff == 1.0) & (df.p_audit_cfg == 0.15)][f"share_{cat}"].mean()
                for p in order]
        ax.bar(range(len(order)), vals, bottom=bottom, label=CAT_NAME[cat],
               color=catC[cat], edgecolor="white", linewidth=0.6)
        bottom += np.array(vals)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([POLICY_SHORT[p] for p in order])
    ax.set_ylabel("Share of incentives allocated")
    ax.set_title("Allocation mix by policy ($h$=1.0)", fontsize=9, loc="left")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_allocation.pdf"))
    fig.savefig(os.path.join(FIGS, "fig5_allocation.png"))
    plt.close(fig)


def main():
    df = load()
    summ = summary_table(df)
    con = contrasts_table(df)
    front = audit_frontier_table(df)
    fig_goodhart(summ)
    fig_handcuff_sweep(df, summ)
    fig_audit_frontier(front)
    fig_allocation(df)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n================ DESCRIPTIVE SUMMARY (per policy) ================")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n================ PAIRED CONTRASTS (Goodhart / audit / oracle) ================")
    show = con[con.handcuff == 1.0]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n================ AUDIT FRONTIER ================")
    print(front.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFigures written to results/figs/: fig2_goodhart, fig3_handcuff, fig4_audit, fig5_allocation (pdf+png)")


if __name__ == "__main__":
    main()
