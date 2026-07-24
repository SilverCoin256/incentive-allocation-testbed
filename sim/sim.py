#!/usr/bin/env python3
"""
Incentive allocation under partial observability — simulation engine.

AI-assisted implementation (large language model). The author directed the
research design and owns the interpretation; this file is the simulation engine.
Design decisions (archetypes, gains, costs, hazard model, budget) are modelling
choices documented in the paper's Method section.

IMPORTANT (scale + provenance honesty; see VERIFICATION_REPORT.md): latent
satisfaction s is a UNITLESS illustrative variable in [0,1]. Every magnitude
below (gains, crowding, hazard coefficients) is a NON-CALIBRATED modelling
choice, not an estimate from any cited paper. Cited works (Deci et al. 1999;
Holmström–Milgrom 1991; Frey–Jegen 2001) motivate the SIGN and FUNCTIONAL FORM
of a mechanism only — never a numeric value. Findings are therefore reported as
structural (sign / direction / invariance across a swept region), never as
calibrated effect magnitudes; dz and raw p-values are demoted to within-model
separation, not behavioral effect sizes.

Modes:
  python3 sim.py validate   # correctness battery (V1-V4); must pass before any use
  python3 sim.py pilot      # engineering smoke test — NOT paper data
  python3 sim.py full       # main grid — the primary data
  python3 sim.py kill       # pre-registered kill-tests (crowding=0, estimator, lambda,
                            #   founder-cohort, random-worlds robustness) — see
                            #   VERIFICATION_REPORT.md

Stdlib only. Deterministic per seed.
"""
import csv
import hashlib
import math
import os
import random
import statistics
import sys

# ----------------------------- parameters (modelling choices) -----------------------------

CATS = ["M", "R", "F", "D", "S"]          # Monetary, Recognition, Flexibility, Development, Social
NONE_ACT = "NONE"
ACTIONS = CATS + [NONE_ACT]

COST = {"M": 3.0, "R": 1.0, "F": 2.0, "D": 2.0, "S": 1.0, NONE_ACT: 0.0}

ARCHETYPES = ["A0_LowComp", "A1_LowVelocity", "A2_Burnout", "A3_Ambition"]

# Satisfaction gain per category, per archetype (applied as gain*(1-s)).
GAIN = {
    "A0_LowComp":     {"M": 0.12, "R": 0.06, "F": 0.03, "D": 0.05, "S": 0.03},
    "A1_LowVelocity": {"M": 0.04, "R": 0.05, "F": 0.03, "D": 0.12, "S": 0.06},
    "A2_Burnout":     {"M": 0.03, "R": 0.03, "F": 0.14, "D": 0.04, "S": 0.05},
    "A3_Ambition":    {"M": 0.05, "R": 0.12, "F": 0.03, "D": 0.08, "S": 0.03},
}
BEST_CAT = {k: max(v, key=v.get) for k, v in GAIN.items()}  # analytic optimum for V1

# Monetary crowding-out of intrinsic satisfaction, applied with a one-quarter
# lag. SIGN/FORM motivated by the crowding-out literature (Deci et al. 1999;
# Frey–Jegen 2001); MAGNITUDES are illustrative, non-calibrated, and are the
# variable swept in the crowding kill-test (crowd_mult), so no result depends
# on the specific value 0.06.
CROWDING = {"A0_LowComp": 0.0, "A1_LowVelocity": 0.0, "A2_Burnout": 0.02, "A3_Ambition": 0.06}

DRIFT = 0.02            # untreated satisfaction decay per quarter
HABIT_WINDOW = 3        # quarters
HABIT_PENALTY = 0.30    # per repeat within window
SURVEY_NOISE = 0.10

# Attrition hazard: P(leave) = sigmoid(A0H - A1H*s - h*handcuff)
HAZ_A0 = -0.8
HAZ_A1 = 3.2
REPLACE_S = (0.45, 0.65)

BUDGET_PER_N = 1.5      # cost units per employee per quarter

# Bandit
EPS0, EPS_DECAY, Q_INIT, LAMBDA = 0.15, 0.95, 1.0, 1.0
SAT_PRIOR = 0.5     # neutral prior for the satisfaction estimate (no data yet)
SHRINK_K = 20       # shrinkage strength: precision weight = n_sat / (n_sat + K)
GATE_N = 8          # confidence-gated: ignore a cell's Q_sat until audited n >= this

DEFAULTS = dict(N=400, T=12, handcuff=1.0, p_audit=0.15)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------- core simulation -----------------------------

class Bandit:
    """Per-archetype epsilon-greedy value table over ACTIONS.

    Cohort-batch semantics: values are exact running means of quarterly
    cohort-level rewards per (archetype, action) cell — i.e., the allocator
    learns from aggregated retention/survey statistics per quarter, the way an
    HR analytics function actually observes them, not from per-person credit
    assignment (which pilot v1 showed cannot converge at organizational
    horizons: binary per-employee rewards leave ~0.01 Q-gaps under noise).
    """

    def __init__(self, rng, lam=0.0, sat_est="raw"):
        # Two separately-estimated KPIs per cell, combined only at decision
        # time (value = Q_stay + lam * effective_Q_sat). Summing them inside one
        # reward (pilot v2) made cells with a survey observation incommensurable
        # with cells without one — the learner chased survey PRESENCE (a
        # rich-get-richer artifact), not satisfaction.
        #
        # sat_est selects how the satisfaction term enters the value (this is
        # the J-curve artifact kill-test, VERIFICATION_REPORT.md):
        #   "raw"    : unshrunk running mean at full weight (the naive default
        #              the adversarial review flagged as noise-injecting).
        #   "shrink" : precision-weighted toward the neutral prior; the sat
        #              term's weight = lam * n/(n+SHRINK_K), so a low-n (sparse
        #              audit) cell contributes ~0 until data accrues.
        #   "gated"  : the sat term is used only once audited n >= GATE_N.
        self.q_stay = {k: {a: Q_INIT for a in ACTIONS} for k in ARCHETYPES}
        self.n_stay = {k: {a: 0 for a in ACTIONS} for k in ARCHETYPES}
        self.q_sat = {k: {a: SAT_PRIOR for a in ACTIONS} for k in ARCHETYPES}
        self.n_sat = {k: {a: 0 for a in ACTIONS} for k in ARCHETYPES}
        self.eps = EPS0
        self.lam = lam
        self.sat_est = sat_est
        self.rng = rng

    def _sat_term(self, arch, act):
        n = self.n_sat[arch][act]
        q = self.q_sat[arch][act]
        if self.sat_est == "raw":
            return q
        if self.sat_est == "shrink":
            w = n / (n + SHRINK_K)               # precision weight in [0,1)
            return w * q + (1 - w) * SAT_PRIOR
        if self.sat_est == "gated":
            return q if n >= GATE_N else SAT_PRIOR
        raise ValueError(self.sat_est)

    def value(self, arch, act):
        return self.q_stay[arch][act] + self.lam * self._sat_term(arch, act)

    def pick(self, arch, budget_left):
        affordable = [a for a in ACTIONS if COST[a] <= budget_left]
        if self.rng.random() < self.eps:
            return self.rng.choice(affordable)
        return max(affordable, key=lambda a: self.value(arch, a))

    @staticmethod
    def _mean_update(qtab, ntab, arch, act, obs):
        n0 = ntab[arch][act]
        qtab[arch][act] = (qtab[arch][act] * n0 + sum(obs)) / (n0 + len(obs)) \
            if n0 > 0 else statistics.fmean(obs)
        ntab[arch][act] = n0 + len(obs)

    def update_stay(self, arch, act, obs):
        if obs:
            self._mean_update(self.q_stay, self.n_stay, arch, act, obs)

    def update_sat(self, arch, act, obs):
        # store the raw running mean + count; the estimator transform is applied
        # at decision time in _sat_term (so shrink/gate see the true audited n).
        if obs:
            self._mean_update(self.q_sat, self.n_sat, arch, act, obs)

    def decay(self):
        self.eps *= EPS_DECAY


def new_employee(arch, rng, founder=False):
    return dict(arch=arch, s=rng.uniform(*REPLACE_S), hist=[], pending=0.0,
                last_m=0, founder=founder)


def run(policy, seed, N=None, T=None, handcuff=None, p_audit=None,
        attrition_on=True, reward_mode=None, crowd_mult=1.0, lam=None,
        sat_est="raw", replace_mode="fixed", gain=None, crowding=None,
        haz_a1=None, haz_a0=None, prevalence=None):
    """One simulated organization. Returns metrics dict.

    policy in {P0,P1,P2,P3,P4,P5}. reward_mode overrides (for validation).
    crowd_mult scales the crowding-out vector (0 = the H2 kill-test control).
    lam overrides LAMBDA for P4/P5; sat_est in {raw,shrink,gated} selects the
    satisfaction estimator (J-curve kill-test). replace_mode in
    {fixed,none,survivor} sets how a leaver's slot is refilled (metric-confound
    sensitivity). gain/crowding/haz_a1 override the world parameters (random-
    worlds robustness sweep).
    """
    p = dict(DEFAULTS)
    for k, v in dict(N=N, T=T, handcuff=handcuff, p_audit=p_audit).items():
        if v is not None:
            p[k] = v
    GAINw = gain if gain is not None else GAIN
    CROWDw = crowding if crowding is not None else CROWDING
    A1 = haz_a1 if haz_a1 is not None else HAZ_A1
    A0 = haz_a0 if haz_a0 is not None else HAZ_A0
    rng = random.Random(seed)
    if prevalence is not None:
        # build the initial population at the calibrated archetype mix
        # (deterministic: integer counts by share, then shuffle).
        counts = {a: int(round(prevalence.get(a, 0.0) * p["N"])) for a in ARCHETYPES}
        while sum(counts.values()) < p["N"]:
            counts[max(ARCHETYPES, key=lambda a: prevalence.get(a, 0))] += 1
        while sum(counts.values()) > p["N"]:
            counts[max(counts, key=counts.get)] -= 1
        archs = [a for a in ARCHETYPES for _ in range(counts[a])]
        rng.shuffle(archs)
        emps = [new_employee(archs[i], rng, founder=True) for i in range(p["N"])]
    else:
        emps = [new_employee(ARCHETYPES[i % 4], rng, founder=True) for i in range(p["N"])]
    lam_eff = 0.0 if (policy == "P3" or reward_mode == "delta_s") \
        else (lam if lam is not None else LAMBDA)
    bandit = Bandit(rng, lam=lam_eff, sat_est=sat_est) \
        if policy in ("P3", "P4", "P5") else None
    static_map = {"A0_LowComp": "M", "A1_LowVelocity": "D",
                  "A2_Burnout": "F", "A3_Ambition": "R"}

    leavers = 0
    s_accum, s_by_arch = [], {k: [] for k in ARCHETYPES}
    founder_s = []            # satisfaction of never-left t=0 employees (harm-vs-trap)
    tenure_s = {}             # tenure-bucket -> [s]  (Kitagawa-Oaxaca decomposition)
    act_counts = {a: 0 for a in ACTIONS}

    for t in range(p["T"]):
        budget = BUDGET_PER_N * p["N"]
        order = list(range(p["N"]))
        rng.shuffle(order)
        assigned = {}
        for i in order:
            e = emps[i]
            if e.get("departed"):
                continue
            if policy == "P0":
                aff = [a for a in ACTIONS if COST[a] <= budget]
                act = rng.choice(aff)
            elif policy == "P1":
                act = "M" if COST["M"] <= budget else NONE_ACT
            elif policy == "P2":
                pref = static_map[e["arch"]]
                act = pref if COST[pref] <= budget else NONE_ACT
            else:
                act = bandit.pick(e["arch"], budget)
            budget -= COST[act]
            assert budget >= -1e-9, "budget exceeded"  # V3 guard
            assigned[i] = act
            act_counts[act] += 1

        # satisfaction dynamics
        for i, e in enumerate(emps):
            if e.get("departed"):
                continue
            act = assigned[i]
            ds = -DRIFT
            ds -= e["pending"]           # lagged crowding penalty
            e["pending"] = 0.0
            if act in CATS:
                repeats = e["hist"][-HABIT_WINDOW:].count(act)
                g = GAINw[e["arch"]][act] * max(0.0, 1.0 - HABIT_PENALTY * repeats)
                ds += g * (1.0 - e["s"])
                if act == "M":
                    e["pending"] = CROWDw[e["arch"]] * crowd_mult
            e["s"] = min(1.0, max(0.0, e["s"] + ds))
            e["hist"].append(act)
            e["last_m"] = 1 if act == "M" else (1 if (e["last_m"] and len(e["hist"]) >= 2 and e["hist"][-2] == "M") else 0)

        # surveys (partial observability)
        surveyed = {}
        for i, e in enumerate(emps):
            if e.get("departed"):
                continue
            if rng.random() < p["p_audit"]:
                surveyed[i] = min(1.0, max(0.0, e["s"] + rng.uniform(-SURVEY_NOISE, SURVEY_NOISE)))

        # attrition + cohort-level bandit reward (batched per archetype x action)
        stay_obs = {k: {a: [] for a in ACTIONS} for k in ARCHETYPES}
        surv_obs = {k: {a: [] for a in ACTIONS} for k in ARCHETYPES}
        gain_obs = {k: {a: [] for a in ACTIONS} for k in ARCHETYPES}
        alive_s = [e["s"] for e in emps if not e.get("departed")]  # for survivor-draw
        for i, e in enumerate(emps):
            if e.get("departed"):
                continue
            act = assigned[i]
            cuff = 1 if ("M" in e["hist"][-2:]) else 0
            p_leave = sigmoid(A0 - A1 * e["s"] - p["handcuff"] * cuff) if attrition_on else 0.0
            left = rng.random() < p_leave
            stay_obs[e["arch"]][act].append(0.0 if left else 1.0)
            if policy == "P4" and i in surveyed:
                surv_obs[e["arch"]][act].append(surveyed[i])
            elif policy == "P5":
                surv_obs[e["arch"]][act].append(e["s"])  # oracle sees true s
            gain_obs[e["arch"]][act].append(
                10.0 * (GAINw[e["arch"]][act] if act in CATS else 0.0))
            if left:
                leavers += 1
                if replace_mode == "none":
                    e["departed"] = True                 # slot goes empty
                elif replace_mode == "survivor":
                    ns = new_employee(e["arch"], rng)
                    ns["s"] = rng.choice(alive_s) if alive_s else ns["s"]
                    emps[i] = ns
                else:                                     # "fixed"
                    emps[i] = new_employee(e["arch"], rng)

        if bandit is not None:
            for k in ARCHETYPES:
                for a in ACTIONS:
                    if not stay_obs[k][a]:
                        continue
                    if reward_mode == "delta_s":
                        bandit.update_stay(k, a, [statistics.fmean(gain_obs[k][a])])
                        continue
                    bandit.update_stay(k, a, [statistics.fmean(stay_obs[k][a])])
                    if surv_obs[k][a]:
                        # pass the full per-employee surveyed list so audited n
                        # reflects true audit VOLUME (needed for shrink/gate).
                        bandit.update_sat(k, a, surv_obs[k][a])
            bandit.decay()
        for e in emps:
            if e.get("departed"):
                continue
            s_accum.append(e["s"])
            s_by_arch[e["arch"]].append(e["s"])
            if e.get("founder"):
                founder_s.append(e["s"])            # never-left original cohort
            ten = len(e["hist"])                    # quarters experienced
            tenure_s.setdefault(min(ten, p["T"]), []).append(e["s"])

    arch_means = {k: statistics.fmean(v) for k, v in s_by_arch.items() if v}
    total_assign = sum(act_counts.values()) or 1
    m = dict(policy=policy, seed=seed,
             attr_rate=leavers / (p["N"] * p["T"]),
             mean_s=statistics.fmean(s_accum) if s_accum else float("nan"),
             mean_s_founder=statistics.fmean(founder_s) if founder_s else float("nan"),
             founder_frac=len(founder_s) / (p["N"] * p["T"]),
             arch_gap=(max(arch_means.values()) - min(arch_means.values()))
                      if arch_means else float("nan"))
    for a in ACTIONS:
        m[f"share_{a}"] = act_counts[a] / total_assign
    m["_bandit"] = bandit
    m["_arch_means"] = arch_means
    m["_tenure_s"] = {k: statistics.fmean(v) for k, v in tenure_s.items()}
    return m


# ----------------------------- validation battery -----------------------------

def validate():
    ok = True
    # V1: learning machinery recovers the known-best category per archetype
    # (aligned reward, no attrition, no crowding/handcuff interference).
    res = run("P5", seed=7, N=200, T=40, handcuff=0.0, p_audit=0.0,
              attrition_on=False, reward_mode="delta_s")
    b = res["_bandit"]
    for k in ARCHETYPES:
        q = dict(b.q_stay[k]); q.pop(NONE_ACT)
        learned = max(q, key=q.get)
        good = learned == BEST_CAT[k]
        ok &= good
        print(f"V1 {k}: learned={learned} expected={BEST_CAT[k]} {'PASS' if good else 'FAIL'}")
    # V1b: mechanism reachability — at high handcuff strength the retention-only
    # learner must be ABLE to discover the monetary channel (precondition for
    # any Goodhart hypothesis test; this is machinery, not a result).
    res = run("P3", seed=11, N=1000, T=16, handcuff=1.5, p_audit=0.0)
    b = res["_bandit"]
    hits = 0
    for k in ARCHETYPES:
        q = dict(b.q_stay[k])
        learned = max(q, key=q.get)
        hit = learned == "M"
        hits += hit
        print(f"V1b {k}: retention-only learner argmax={learned} (expect M at h=1.5) {'PASS' if hit else 'FAIL'}")
    ok &= hits >= 3
    # V2: attrition monotone in satisfaction
    lo = statistics.fmean(sigmoid(HAZ_A0 - HAZ_A1 * 0.2) for _ in range(1))
    hi = statistics.fmean(sigmoid(HAZ_A0 - HAZ_A1 * 0.8) for _ in range(1))
    good = lo > hi
    ok &= good
    print(f"V2 hazard(s=0.2)={lo:.3f} > hazard(s=0.8)={hi:.3f}: {'PASS' if good else 'FAIL'}")
    # V3: budget assertions are active in every run mode (exercised implicitly);
    # run one config of each policy to trip any violation.
    for pol in ("P0", "P1", "P2", "P3", "P4", "P5"):
        run(pol, seed=1, N=100, T=8)
    print("V3 budget guard exercised across all policies: PASS")
    # V4: determinism — identical seed, identical metrics
    a = run("P3", seed=42, N=150, T=10)
    bb = run("P3", seed=42, N=150, T=10)
    key = lambda r: (round(r["attr_rate"], 12), round(r["mean_s"], 12), round(r["arch_gap"], 12))
    good = key(a) == key(bb)
    ok &= good
    print(f"V4 determinism: {'PASS' if good else 'FAIL'}")
    print("ALL VALIDATIONS PASSED" if ok else "VALIDATION FAILURE — DO NOT USE RESULTS")
    return 0 if ok else 1


# ----------------------------- experiment drivers -----------------------------

def ci95(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    t = 2.093 if n <= 25 else 1.96
    return t * statistics.stdev(xs) / math.sqrt(n)


def summarize(rows, label):
    print(f"\n=== {label} ===")
    pols = sorted({r["policy"] for r in rows})
    hdr = f"{'policy':6} {'attrition/qtr':>14} {'latent satisfaction':>21} {'archetype gap':>15} {'share M':>8}"
    print(hdr)
    for pol in pols:
        rs = [r for r in rows if r["policy"] == pol]
        a = [r["attr_rate"] for r in rs]
        s = [r["mean_s"] for r in rs]
        g = [r["arch_gap"] for r in rs]
        mM = statistics.fmean(r["share_M"] for r in rs)
        print(f"{pol:6} {statistics.fmean(a):.4f} ±{ci95(a):.4f}   "
              f"{statistics.fmean(s):.4f} ±{ci95(s):.4f}     "
              f"{statistics.fmean(g):.4f} ±{ci95(g):.4f}   {mM:.3f}")


def write_csv(rows, path):
    # union of all non-underscore keys across rows (rows may have differing
    # shapes, e.g. h2 vs replace-mode arms), preserving first-seen order.
    cols = []
    for r in rows:
        for c in r:
            if not c.startswith("_") and c not in cols:
                cols.append(c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {path} ({len(rows)} rows)")


def pilot():
    rows = []
    for pol in ("P0", "P1", "P2", "P3", "P4", "P5"):
        for seed in range(20):
            rows.append(run(pol, seed, N=200, T=12, handcuff=1.0, p_audit=0.15))
    summarize(rows, "PILOT (engineering smoke test — NOT paper data) N=200 T=12 seeds=20 h=1.0 p=0.15")
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "pilot.csv"))


def full():
    rows = []
    seeds = range(200)
    for h in (0.0, 0.5, 1.0, 1.5):
        for pol in ("P0", "P1", "P2", "P3", "P5"):
            for seed in seeds:
                r = run(pol, seed, handcuff=h, p_audit=0.15)
                r["handcuff"], r["p_audit_cfg"] = h, 0.15
                rows.append(r)
        for pa in (0.0, 0.05, 0.1, 0.2, 0.4):
            for seed in seeds:
                r = run("P4", seed, handcuff=h, p_audit=pa)
                r["handcuff"], r["p_audit_cfg"] = h, pa
                rows.append(r)
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "full.csv"))
    summarize([r for r in rows if r["handcuff"] == 1.0 and r["p_audit_cfg"] in (0.15, 0.1)],
              "FULL @ h=1.0 (main configuration)")


# ----------------------------- pre-registered kill-tests -----------------------------
# Each answers a specific fatal objection from VERIFICATION_REPORT.md, with the
# kill condition stated BEFORE the run. All CPU/stdlib, deterministic per seed.

def _tag(r, **kw):
    r.update(kw)
    return r


def kill_h2(seeds=200):
    """H2 tautology test + harm-vs-trap. Sweep crowd_mult (crowding=0 control +
    dose-response) x handcuff, policies P2/P3, reporting BOTH the population
    mean_s and the founder-cohort mean_s. Kill condition (pre-registered):
    if at crowd_mult=0 AND h=0 the population gap P2-P3 is ~0 -> the Goodhart
    gap is generated by crowding-out (a set constant), not discovered. If the
    FOUNDER gap vanishes while the population gap persists -> the effect is
    survivorship composition (trapping), not per-person harm."""
    rows = []
    for cm in (0.0, 1.0 / 3, 1.0, 2.0):
        for h in (0.0, 1.0):
            for pol in ("P2", "P3"):
                for s in range(seeds):
                    rows.append(_tag(run(pol, s, handcuff=h, crowd_mult=cm),
                                     test="h2", crowd_mult=round(cm, 3), handcuff=h))
    # replacement-scheme sensitivity (metric-confound), crowd=1, h=1
    for rm in ("none", "survivor"):
        for pol in ("P2", "P3"):
            for s in range(seeds):
                rows.append(_tag(run(pol, s, handcuff=1.0, replace_mode=rm),
                                 test="replace", replace_mode=rm, crowd_mult=1.0, handcuff=1.0))
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "kill_h2.csv"))


def kill_audit(seeds=200):
    """J-curve artifact test. P4 audit-fraction sweep under three satisfaction
    estimators (raw / shrink / gated), with P3 and P5 baselines for the recovery
    denominator. Kill condition (pre-registered): if the p=0.05 < p=0 backfire
    present under 'raw' DISAPPEARS under a precision-aware estimator (shrink or
    gated) at both h, the J-curve is an unshrunk-estimator artifact and the
    'auditing backfires' claim must be reframed to 'naive level-based auditing
    backfires.'"""
    rows = []
    for h in (0.0, 1.0):
        for pol in ("P3", "P5"):
            for s in range(seeds):
                rows.append(_tag(run(pol, s, handcuff=h), test="audit",
                                 sat_est="-", handcuff=h, p_audit_cfg=0.15))
        for est in ("raw", "shrink", "gated"):
            for pa in (0.0, 0.05, 0.1, 0.2, 0.4):
                for s in range(seeds):
                    rows.append(_tag(run("P4", s, handcuff=h, p_audit=pa, sat_est=est),
                                     test="audit", sat_est=est, handcuff=h, p_audit_cfg=pa))
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "kill_audit.csv"))


def kill_lambda(seeds=150):
    """Fairness of lambda=1. Full lambda x audit-fraction grid (raw estimator,
    h=1). Shows whether the backfire is specific to the untuned lambda=1."""
    rows = []
    for lam in (0.0, 0.5, 1.0, 2.0, 4.0):
        for pa in (0.0, 0.05, 0.1, 0.2, 0.4):
            for s in range(seeds):
                rows.append(_tag(run("P4", s, handcuff=1.0, p_audit=pa, lam=lam, sat_est="raw"),
                                 test="lambda", lam=lam, p_audit_cfg=pa, handcuff=1.0))
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "kill_lambda.csv"))


def _sample_world(wr):
    g = {k: {a: max(0.005, GAIN[k][a] * wr.uniform(0.6, 1.4)) for a in CATS}
         for k in ARCHETYPES}
    c3 = wr.uniform(0.0, 0.15)
    crowd = {"A0_LowComp": 0.0, "A1_LowVelocity": 0.0,
             "A2_Burnout": c3 * wr.uniform(0.0, 0.6), "A3_Ambition": c3}
    return g, crowd, wr.uniform(1.5, 5.0)


def kill_robust(n_worlds=250, seeds=5):
    """Random-worlds robustness. Draw GAIN, CROWDING, hazard-slope over plausible
    ranges; report the FRACTION of the assumption space where (i) the population
    Goodhart gap sign (P2>P3 satisfaction), (ii) the founder-gap sign, and
    (iii) the raw J-curve backfire (P4 p=0.05 < p=0) hold. Converts 'my numbers
    give this' into 'the sign is invariant across a wide region.'"""
    rows = []
    wr = random.Random(90210)
    for w in range(n_worlds):
        g, c, a1 = _sample_world(wr)
        def mean(pol, **kw):
            xs = [run(pol, s, handcuff=1.0, gain=g, crowding=c, haz_a1=a1, **kw)
                  for s in range(seeds)]
            return (statistics.fmean(x["mean_s"] for x in xs),
                    statistics.fmean(x["mean_s_founder"] for x in xs))
        p2, p2f = mean("P2")
        p3, p3f = mean("P3")
        j0, _ = mean("P4", p_audit=0.0, sat_est="raw")
        j5, _ = mean("P4", p_audit=0.05, sat_est="raw")
        rows.append(dict(world=w, gap_pop=p2 - p3, gap_founder=p2f - p3f,
                         jcurve=j5 - j0, haz_a1=round(a1, 3),
                         crowd3=round(c["A3_Ambition"], 4)))
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "kill_robust.csv"))
    fpos = statistics.fmean(1.0 if r["gap_pop"] > 0 else 0.0 for r in rows)
    ffnd = statistics.fmean(1.0 if r["gap_founder"] > 0 else 0.0 for r in rows)
    fjc = statistics.fmean(1.0 if r["jcurve"] < 0 else 0.0 for r in rows)
    print(f"ROBUSTNESS over {n_worlds} random worlds:")
    print(f"  population Goodhart gap sign (P2>P3) holds in {fpos:.1%}")
    print(f"  founder-cohort gap sign holds in           {ffnd:.1%}")
    print(f"  raw J-curve backfire (p.05<p0) holds in    {fjc:.1%}")


def calibrated(seeds=200):
    """Re-run the load-bearing contrasts with parameters fit to the IBM HR
    benchmark (calibrate/calibrated_params.json): the satisfaction->attrition
    slope HAZ_A1, and the real archetype prevalence. Tests whether R1 (Goodhart
    gap), R2 (founder gap), and R3 (shrinkage removes the audit backfire) survive
    calibration. Time unit is abstract; we calibrate the protective SLOPE and the
    archetype MIX (the two most-attacked 'invented' choices), and run the fitted
    intercept as a sensitivity."""
    import json
    cfgpath = os.path.join(os.path.dirname(__file__), "..", "calibrate", "calibrated_params.json")
    if not os.path.exists(cfgpath):
        print("run calibrate/fit_ibm.py first"); return
    cfg = json.load(open(cfgpath))
    prev = cfg["archetype_prevalence"]
    a1_uni = abs(cfg["hazard_fit"]["HAZ_A1_univariate"])
    a1_ctrl = abs(cfg["hazard_fit"]["HAZ_A1_controlled"])
    a0_fit = cfg["hazard_fit"]["HAZ_A0_univariate"]
    settings = [
        ("illustrative_default", dict()),                                   # HAZ_A1=3.2, equal mix
        ("calib_slope_uni",  dict(haz_a1=a1_uni, prevalence=prev)),         # fitted slope + real mix
        ("calib_slope_ctrl", dict(haz_a1=a1_ctrl, prevalence=prev)),
        ("calib_full",       dict(haz_a1=a1_uni, haz_a0=a0_fit, prevalence=prev)),  # + fitted intercept
    ]
    rows = []
    for name, kw in settings:
        for pol in ("P2", "P3", "P5"):
            for s in range(seeds):
                rows.append(_tag(run(pol, s, handcuff=1.0, **kw),
                                 setting=name, sat_est="-", p_audit_cfg=0.15))
        for est in ("raw", "shrink"):
            for pa in (0.0, 0.05):
                for s in range(seeds):
                    rows.append(_tag(run("P4", s, handcuff=1.0, p_audit=pa, sat_est=est, **kw),
                                     setting=name, sat_est=est, p_audit_cfg=pa))
    write_csv(rows, os.path.join(os.path.dirname(__file__), "..", "results", "calibrated.csv"))
    # inline survival summary
    def mean(name, pol, est="-", pa=0.15):
        xs = [r["mean_s"] for r in rows if r["setting"] == name and r["policy"] == pol
              and r.get("sat_est", "-") == est and abs(float(r["p_audit_cfg"]) - pa) < 1e-9]
        return statistics.fmean(xs) if xs else float("nan")
    def meanf(name, pol):
        xs = [r["mean_s_founder"] for r in rows if r["setting"] == name and r["policy"] == pol
              and r.get("sat_est", "-") == "-"]
        return statistics.fmean(xs) if xs else float("nan")
    print("\n===== IBM-CALIBRATION SURVIVAL (h=1.0, %d seeds) =====" % seeds)
    print(f"{'setting':20} {'R1 gap(P2-P3)':>14} {'R2 founder gap':>15} "
          f"{'R3 raw backfire':>16} {'R3 shrink':>10}")
    for name, _ in settings:
        r1 = mean(name, "P2") - mean(name, "P3")
        r2 = meanf(name, "P2") - meanf(name, "P3")
        raw_bf = mean(name, "P4", "raw", 0.05) - mean(name, "P4", "raw", 0.0)
        shr_bf = mean(name, "P4", "shrink", 0.05) - mean(name, "P4", "shrink", 0.0)
        print(f"{name:20} {r1:>+14.4f} {r2:>+15.4f} {raw_bf:>+16.4f} {shr_bf:>+10.4f}")
    print("R1>0 & R2>0 => Goodhart/founder gap survives; raw<0 & shrink~0 => audit finding survives.")


def kill():
    print(">>> kill_h2 (crowding=0 control + founder cohort)");    kill_h2()
    print(">>> kill_audit (estimator variants)");                   kill_audit()
    print(">>> kill_lambda (lambda sweep)");                        kill_lambda()
    print(">>> kill_robust (random worlds)");                       kill_robust()
    print(">>> kill-tests complete; analyze with analyze_kill.py")


if __name__ == "__main__":
    with open(__file__, "rb") as f:
        print(f"sim.py sha256={hashlib.sha256(f.read()).hexdigest()[:16]}")
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        sys.exit(validate())
    elif mode == "pilot":
        pilot()
    elif mode == "full":
        full()
    elif mode == "kill":
        kill()
    elif mode == "calibrated":
        calibrated()
    else:
        print("usage: sim.py [validate|pilot|full|kill|calibrated]")
