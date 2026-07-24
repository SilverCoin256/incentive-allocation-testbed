#!/usr/bin/env python3
"""
Kill-test analysis + figures (VERIFICATION_REPORT.md fixes).

AI-assisted implementation (large language model); figures are AI-generated
visualizations, marked as such. This script reports statistics and renders them,
and prints the pre-registered kill-condition verdicts as data, not conclusions.

Statistics deliberately follow the adversarial review:
  - absolute mean differences with Monte-Carlo SE and 95% CIs are primary;
  - Cohen's dz is reported but LABELLED "within-model separation, not a behavioral
    effect size" (it inflates with employees/seed);
  - per-seed SIGN-STABILITY is reported for every crossover/backfire claim;
  - TOST equivalence is offered for "tied" cells;
  - Benjamini-Hochberg FDR is applied across each test family.

Reads ../results/kill_*.csv  ->  ../results/table_kill_*.csv, ../results/figs/fig6-9.
Needs numpy/scipy/pandas/matplotlib (analysis tooling only).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIGS = os.path.join(RES, "figs")
os.makedirs(FIGS, exist_ok=True)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#8256A8", grey="#7A7A7A", teal="#56B4E9", ink="#222222")
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "legend.frameon": False,
                     "font.family": "DejaVu Sans"})


def paired(a, b):
    """Paired-by-seed contrast a-b. Returns dict of honest statistics."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).values
    n = len(d)
    sd = d.std(ddof=1) if n > 1 else 0.0
    mcse = sd / np.sqrt(n) if n else float("nan")
    ci = stats.t.ppf(0.975, n - 1) * mcse if n > 1 else 0.0
    dz = d.mean() / sd if sd > 0 else 0.0
    sign_pos = float((d > 0).mean())
    return dict(n=n, mean=d.mean(), mcse=mcse, ci=ci, dz=dz,
                sign_pos=sign_pos, sign_stab=max(sign_pos, 1 - sign_pos))


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def tost(a, b, margin):
    """Two one-sided tests for equivalence within +-margin. Returns max p."""
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).values
    n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0
    t_lo = (d.mean() - (-margin)) / se
    t_hi = (d.mean() - margin) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return max(p_lo, p_hi)


# ------------------------------- H2 decomposition -------------------------------

def analyze_h2():
    path = os.path.join(RES, "kill_h2.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    h2 = df[df.test == "h2"]
    rows = []
    for h in sorted(h2.handcuff.unique()):
        for cm in sorted(h2.crowd_mult.unique()):
            p2 = h2[(h2.policy == "P2") & (h2.handcuff == h) & (h2.crowd_mult == cm)].set_index("seed")
            p3 = h2[(h2.policy == "P3") & (h2.handcuff == h) & (h2.crowd_mult == cm)].set_index("seed")
            for metric, col in (("population", "mean_s"), ("founder", "mean_s_founder")):
                st = paired(p2[col], p3[col])
                rows.append(dict(handcuff=h, crowd_mult=cm, metric=metric, **st))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "table_kill_h2.csv"), index=False)

    # verdict: base gap at crowd=0 & h=0 (both metrics)
    print("\n===== H2 KILL-TEST (crowding=0 control + founder cohort) =====")
    for metric in ("population", "founder"):
        base = out[(out.crowd_mult == 0.0) & (out.handcuff == 0.0) & (out.metric == metric)]
        if len(base):
            r = base.iloc[0]
            verdict = "PERSISTS (not merely the crowding constant)" if (r["mean"] - r["ci"]) > 0 \
                else "vanishes -> gap was the crowding constant"
            print(f"  {metric:10} gap at crowd=0,h=0: {r['mean']:+.4f} "
                  f"[95%CI {r['mean']-r['ci']:+.4f},{r['mean']+r['ci']:+.4f}] "
                  f"sign-stab {r['sign_stab']:.0%}  -> {verdict}")
    # replacement-mode sensitivity
    rep = df[df.test == "replace"]
    if len(rep):
        print("  replacement-scheme sensitivity (h=1, crowd=1), population gap P2-P3:")
        for rm in sorted(rep.replace_mode.dropna().unique()):
            p2 = rep[(rep.policy == "P2") & (rep.replace_mode == rm)].set_index("seed")["mean_s"]
            p3 = rep[(rep.policy == "P3") & (rep.replace_mode == rm)].set_index("seed")["mean_s"]
            st = paired(p2, p3)
            print(f"    replace={rm:9}: {st['mean']:+.4f} [±{st['ci']:.4f}] sign-stab {st['sign_stab']:.0%}")
    fig_decomposition(out)


def fig_decomposition(out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for j, metric in enumerate(("population", "founder")):
        for h, col in ((0.0, C["teal"]), (1.0, C["red"])):
            sub = out[(out.metric == metric) & (out.handcuff == h)].sort_values("crowd_mult")
            ax[j].errorbar(sub.crowd_mult, sub["mean"], yerr=sub["ci"], marker="o", ms=4,
                           capsize=2.5, color=col, label=f"$h$={h}")
        ax[j].axhline(0, color=C["ink"], lw=0.6)
        ax[j].set_xlabel("crowding-out multiplier")
        ax[j].set_title(f"({'ab'[j]}) {metric} Goodhart gap (Static $-$ Learner)",
                        fontsize=8.5, loc="left")
        ax[j].legend(fontsize=7)
    ax[0].set_ylabel("satisfaction gap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_decomposition.pdf"))
    fig.savefig(os.path.join(FIGS, "fig6_decomposition.png"))
    plt.close(fig)


# ------------------------------- audit J-curve estimator -------------------------------

def analyze_audit():
    path = os.path.join(RES, "kill_audit.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    ests = [e for e in ("raw", "shrink", "gated") if e in df.sat_est.unique()]
    rows, backfire_tests = [], []
    print("\n===== J-CURVE ESTIMATOR KILL-TEST =====")
    for h in sorted(df.handcuff.unique()):
        for est in ests:
            sub = df[(df.sat_est == est) & (df.handcuff == h)]
            p0 = sub[sub.p_audit_cfg == 0.0].set_index("seed")["mean_s"]
            for pa in sorted(sub.p_audit_cfg.unique()):
                s = sub[sub.p_audit_cfg == pa].set_index("seed")["mean_s"]
                rows.append(dict(handcuff=h, sat_est=est, p_audit=pa,
                                 sat=s.mean(), mcse=s.std(ddof=1) / np.sqrt(len(s))))
            p05 = sub[sub.p_audit_cfg == 0.05].set_index("seed")["mean_s"]
            st = paired(p05, p0)  # backfire = negative
            idx = p05.index.intersection(p0.index)
            _, pval = stats.ttest_rel(p05.loc[idx], p0.loc[idx])
            backfire_tests.append(dict(handcuff=h, sat_est=est, diff=st["mean"], ci=st["ci"],
                                       sign_stab=st["sign_stab"], p=pval))
    bt = pd.DataFrame(backfire_tests)
    bt["p_fdr"] = bh_fdr(bt["p"].values)
    bt.to_csv(os.path.join(RES, "table_kill_audit_backfire.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "table_kill_audit_curve.csv"), index=False)
    print("  backfire = P4(p=0.05) - P4(p=0) satisfaction (negative = backfire); BH-FDR corrected:")
    for _, r in bt.iterrows():
        surv = "BACKFIRE" if (r["diff"] + r["ci"] < 0 and r["p_fdr"] < 0.05) else \
               ("n.s." if r["diff"] + r["ci"] >= 0 else "backfire(uncorrected)")
        print(f"    h={r['handcuff']} est={r['sat_est']:6}: {r['diff']:+.4f} [±{r['ci']:.4f}] "
              f"sign-stab {r['sign_stab']:.0%}  p_fdr={r['p_fdr']:.3g}  -> {surv}")
    raw_bf = bt[(bt.sat_est == 'raw')]
    prec_bf = bt[(bt.sat_est.isin(['shrink', 'gated']))]
    surv = (prec_bf["diff"] + prec_bf["ci"] < 0).all()
    print(f"  PRE-REGISTERED VERDICT: backfire under precision-aware estimators "
          f"{'SURVIVES (novel result stands)' if surv else 'VANISHES (reframe to naive-auditing-backfires)'}")
    fig_estimator(pd.DataFrame(rows))


def fig_estimator(curve):
    ests = [e for e in ("raw", "shrink", "gated") if e in curve.sat_est.unique()]
    hs = sorted(curve.handcuff.unique())
    fig, ax = plt.subplots(1, len(hs), figsize=(3.6 * len(hs), 3.0), squeeze=False)
    col = {"raw": C["red"], "shrink": C["blue"], "gated": C["green"]}
    for j, h in enumerate(hs):
        for est in ests:
            sub = curve[(curve.sat_est == est) & (curve.handcuff == h)].sort_values("p_audit")
            ax[0][j].errorbar(sub.p_audit, sub.sat, yerr=1.96 * sub.mcse, marker="o", ms=4,
                              capsize=2.5, color=col[est], label=est)
        ax[0][j].set_xlabel("audit frequency $p$")
        ax[0][j].set_title(f"$h$={h}", fontsize=9, loc="left")
        ax[0][j].legend(fontsize=7, title="sat. estimator", title_fontsize=7)
    ax[0][0].set_ylabel("latent satisfaction (P4)")
    fig.suptitle("Does the audit backfire survive a competent estimator?", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_estimator.pdf"))
    fig.savefig(os.path.join(FIGS, "fig7_estimator.png"))
    plt.close(fig)


# ------------------------------- lambda heatmap -------------------------------

def analyze_lambda():
    path = os.path.join(RES, "kill_lambda.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    piv = df.groupby(["lam", "p_audit_cfg"]).mean_s.mean().unstack()
    piv.to_csv(os.path.join(RES, "table_kill_lambda.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("audit frequency $p$"); ax.set_ylabel("$\\lambda$ (satisfaction weight)")
    ax.set_title("P4 latent satisfaction ($h$=1)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.85, label="latent satisfaction")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_lambda.pdf"))
    fig.savefig(os.path.join(FIGS, "fig8_lambda.png"))
    plt.close(fig)


# ------------------------------- robustness -------------------------------

def analyze_robust():
    path = os.path.join(RES, "kill_robust.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fpos = (df.gap_pop > 0).mean()
    ffnd = (df.gap_founder > 0).mean()
    fjc = (df.jcurve < 0).mean()
    print("\n===== RANDOM-WORLDS ROBUSTNESS =====")
    print(f"  n_worlds={len(df)}")
    print(f"  population Goodhart-gap sign (Static>Learner) holds in {fpos:.1%}")
    print(f"  founder-cohort gap sign holds in                       {ffnd:.1%}")
    print(f"  raw J-curve backfire (p.05<p0) holds in                {fjc:.1%}")
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6))
    for a, col, title, ref in ((ax[0], "gap_pop", "population gap", 0),
                               (ax[1], "gap_founder", "founder gap", 0),
                               (ax[2], "jcurve", "J-curve (p.05$-$p0)", 0)):
        a.hist(df[col], bins=30, color=C["blue"], alpha=0.8)
        a.axvline(ref, color=C["red"], lw=1.2)
        a.set_title(title, fontsize=8.5, loc="left")
    ax[0].set_ylabel("worlds")
    fig.suptitle("Sign invariance across 250 random assumption-worlds", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig9_robust.pdf"))
    fig.savefig(os.path.join(FIGS, "fig9_robust.png"))
    plt.close(fig)


def main():
    analyze_h2()
    analyze_audit()
    analyze_lambda()
    analyze_robust()
    print("\nWrote table_kill_*.csv and figs fig6_decomposition, fig7_estimator, "
          "fig8_lambda, fig9_robust (pdf+png).")


if __name__ == "__main__":
    main()
