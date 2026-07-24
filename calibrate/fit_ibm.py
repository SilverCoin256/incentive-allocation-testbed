#!/usr/bin/env python3
"""
Calibrate the response-model parameters to the IBM HR Analytics benchmark.

AI-assisted implementation (large language model); interpretation of the fit is
the author's. This turns the three most-attacked "invented" numbers into
benchmark-fit quantities:
  * HAZ_A1 (satisfaction -> attrition slope): logistic regression coefficient.
  * HAZ_A0 (baseline log-odds of leaving): the fitted intercept.
  * handcuff channel existence/sign: the StockOptionLevel coefficient, i.e. an
    off-satisfaction retention lever (vesting/options) present in the real data.
  * archetype prevalence: real KMeans cluster shares, not equal 25%.

Note (also stated in the paper): the IBM HR Analytics set is itself a
*synthetic benchmark* built by IBM data scientists, and JobSatisfaction is a
cross-sectional 1-4 self-report, not a longitudinal latent. So this calibrates
the hazard RELATIONSHIP and the archetype MIX to a standard external benchmark;
it does NOT calibrate the satisfaction DYNAMICS (gains/crowding/habituation),
which remain illustrative. It converts "where did 3.2 come from?" into "a
logistic fit to the IBM HR benchmark, coefficient = <value> [95% CI]."

Reads the IBM CSV (path arg or auto-detect); writes calibrated_params.json and
diagnostic figures. Needs pandas, numpy, statsmodels, scikit-learn, matplotlib.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = HERE
FIGS = os.path.join(HERE, "..", "results", "figs")
os.makedirs(FIGS, exist_ok=True)

# Portable lookup: env var, then the repo's calibrate/ dir, then a home-relative
# default. Set HR_CSV=/path/to/csv, or drop the file in calibrate/, or pass the
# path as argv[1]. The IBM HR Analytics CSV is NOT redistributed here (Kaggle
# license) — download "IBM HR Analytics Employee Attrition & Performance".
CANDIDATES = [
    os.environ.get("HR_CSV", ""),
    os.path.join(HERE, "WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    os.path.join(HERE, "ibm_hr_attrition.csv"),
    os.path.expanduser("~/Documents/RL Research/WA_Fn-UseC_-HR-Employee-Attrition.csv"),
]


def find_csv():
    for p in ([sys.argv[1]] if len(sys.argv) > 1 else []) + CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("IBM HR CSV not found. Pass path as arg. "
                            "Kaggle: 'IBM HR Analytics Employee Attrition & Performance'.")


def main():
    path = find_csv()
    df = pd.read_csv(path)
    df["y"] = (df["Attrition"] == "Yes").astype(int)
    # map ordinal JobSatisfaction {1,2,3,4} -> s in [0,1] to match the sim scale.
    df["s"] = (df["JobSatisfaction"] - 1) / 3.0
    df["overtime"] = (df["OverTime"] == "Yes").astype(int)
    df["log_income"] = np.log(df["MonthlyIncome"])
    n = len(df)
    base_attr = df["y"].mean()

    out = {"dataset": os.path.basename(path), "n_employees": int(n),
           "overall_attrition_rate": round(float(base_attr), 4),
           "note": "IBM HR Analytics is a synthetic benchmark; JobSatisfaction is "
                   "cross-sectional 1-4. Calibrates hazard RELATIONSHIP + archetype "
                   "MIX only; dynamics remain illustrative."}

    # ---- hazard: logit(P leave) = a + b_s * s (+ controls) ----
    def fit(cols):
        X = sm.add_constant(df[cols])
        m = sm.Logit(df["y"], X).fit(disp=0)
        return m

    m_uni = fit(["s"])
    m_ctrl = fit(["s", "overtime", "log_income", "YearsAtCompany",
                  "JobInvolvement", "WorkLifeBalance", "StockOptionLevel"])
    b_s_uni = m_uni.params["s"]
    b_s_ctrl = m_ctrl.params["s"]
    ci_uni = m_uni.conf_int().loc["s"].tolist()
    ci_ctrl = m_ctrl.conf_int().loc["s"].tolist()

    # sim hazard is sigmoid(A0 - A1*s - h*cuff): A1 = -(coef on s), A0 = intercept.
    out["hazard_fit"] = {
        "HAZ_A1_univariate": round(float(-b_s_uni), 4),
        "HAZ_A1_univariate_ci": [round(-ci_uni[1], 4), round(-ci_uni[0], 4)],
        "HAZ_A1_controlled": round(float(-b_s_ctrl), 4),
        "HAZ_A1_controlled_ci": [round(-ci_ctrl[1], 4), round(-ci_ctrl[0], 4)],
        "HAZ_A0_univariate": round(float(m_uni.params["const"]), 4),
        "HAZ_A0_controlled_at_means": round(float(
            m_ctrl.params["const"]
            + sum(m_ctrl.params[c] * df[c].mean() for c in
                  ["overtime", "log_income", "YearsAtCompany", "JobInvolvement",
                   "WorkLifeBalance", "StockOptionLevel"])), 4),
        "interpretation": "HAZ_A1 = reduction in log-odds of leaving per unit s "
                          "(s in [0,1]). Sim default was 3.2 (illustrative).",
    }

    # ---- handcuff channel: does an off-satisfaction retention lever exist? ----
    # StockOptionLevel = vesting/options proxy. Negative coef (controlling for s)
    # => retention that does NOT run through satisfaction: the handcuff, in data.
    out["handcuff_evidence"] = {
        "StockOptionLevel_coef_controlled": round(float(m_ctrl.params["StockOptionLevel"]), 4),
        "StockOptionLevel_ci": [round(x, 4) for x in m_ctrl.conf_int().loc["StockOptionLevel"].tolist()],
        "reading": "negative => higher stock options reduce attrition controlling "
                   "for satisfaction, i.e. an empirical off-satisfaction retention "
                   "(handcuff) channel exists in the benchmark.",
    }

    # ---- archetypes: cluster, get real prevalence, map to taxonomy ----
    feats = ["MonthlyIncome", "YearsSinceLastPromotion", "overtime", "WorkLifeBalance",
             "JobInvolvement", "TrainingTimesLastYear", "Age", "TotalWorkingYears",
             "NumCompaniesWorked", "PercentSalaryHike"]
    Z = StandardScaler().fit_transform(df[feats])
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(Z)
    df["cluster"] = km.labels_
    cent = pd.DataFrame(km.cluster_centers_, columns=feats)  # z-scored centroids
    # map each cluster to the archetype whose signature feature it most expresses
    sig = {"A0_LowComp": ("MonthlyIncome", -1), "A1_LowVelocity": ("YearsSinceLastPromotion", +1),
           "A2_Burnout": ("overtime", +1), "A3_Ambition": ("JobInvolvement", +1)}
    assign, used = {}, set()
    # greedy: assign strongest signature match first
    scores = []
    for arch, (f, sgn) in sig.items():
        for c in range(4):
            scores.append((sgn * cent.loc[c, f], arch, c))
    for val, arch, c in sorted(scores, reverse=True):
        if arch in assign or c in used:
            continue
        assign[arch] = c
        used.add(c)
    prevalence = {arch: round(float((df["cluster"] == c).mean()), 4) for arch, c in assign.items()}
    out["archetype_prevalence"] = prevalence
    out["archetype_cluster_map"] = {a: int(c) for a, c in assign.items()}
    out["archetype_attrition"] = {arch: round(float(df.loc[df.cluster == c, "y"].mean()), 4)
                                  for arch, c in assign.items()}

    with open(os.path.join(OUT, "calibrated_params.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    # ---- diagnostic figure ----
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    js = sorted(df.JobSatisfaction.unique())
    emp = [df.loc[df.JobSatisfaction == j, "y"].mean() for j in js]
    ax[0].bar([str(j) for j in js], emp, color="#0072B2", alpha=0.8,
              label="empirical attrition")
    ss = np.linspace(0, 1, 50)
    fit_p = 1 / (1 + np.exp(-(m_uni.params["const"] + b_s_uni * ss)))
    ax[0].plot(ss * 3, fit_p, color="#D55E00", lw=2, label="logistic fit")
    ax[0].set_xlabel("JobSatisfaction (1–4)"); ax[0].set_ylabel("P(attrition)")
    ax[0].set_title("(a) hazard calibration", fontsize=9, loc="left"); ax[0].legend(fontsize=7)
    archs = list(prevalence.keys())
    ax[1].bar(range(4), [prevalence[a] for a in archs], color="#009E73", alpha=0.85)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([a.split("_")[0] for a in archs],
                                                      rotation=25, ha="right", fontsize=7)
    ax[1].axhline(0.25, ls="--", color="#7A7A7A", lw=1, label="equal-share assumption")
    ax[1].set_ylabel("prevalence"); ax[1].set_title("(b) archetype mix (IBM)", fontsize=9, loc="left")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig10_calibration.pdf"))
    fig.savefig(os.path.join(FIGS, "fig10_calibration.png"), dpi=300)
    print("wrote calibrated_params.json and fig10_calibration")


if __name__ == "__main__":
    main()
