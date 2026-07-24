# Results Digest — descriptive readout
A compact map of every headline number to the CSV it comes from. Source: `results/full.csv` + `results/kill_*.csv` (200 seeds/config). See `paper/` for the full analysis and `VERIFICATION_REPORT.md` for the falsification tests.

> **CORRECTION BANNER (read before trusting Facts 1–5 below):** Facts 1–5 describe the *naive* main grid and include a raw-estimator audit "J-curve." The pre-registered kill-tests (Facts 6–9) show that J-curve is **largely an artifact of the naive unshrunk estimator** — a shrinkage estimator removes it (Fact 8). So the descriptive "sparse auditing backfires" pattern in Fact 4 does **not** support a general claim that auditing backfires. What is robust: the Goodhart/equity gap (Facts 2, 6, 7) and the *naive-estimator* backfire + its fix (Fact 8).

## Design summary of the run
6 policies × 200 seeds × 4 handcuff strengths h∈{0,0.5,1,1.5}; P4 additionally swept over audit rate p∈{0,.05,.1,.2,.4}. N=400 employees, T=12 quarters. Metrics per run: attrition/quarter (observed), mean latent satisfaction (hidden), archetype satisfaction gap (max−min across the 4 archetypes), category allocation shares.

## Fact 1 — On the OBSERVED metric, the learner's rank vs the static baseline flips with h
Paired-by-seed contrast P3 (learner, retention reward) − P2 (static theory baseline), attrition/quarter (lower = better retention):
- h=0.0: +0.0061 [95% CI ±0.00074], t=16.2, dz=+1.14  → learner attrites MORE (worse)
- h=0.5: +0.0031 [±0.00075], dz=+0.57                 → learner worse
- h=1.0: −0.0005 [±0.00073], t=−1.37, p=0.17          → indistinguishable
- h=1.5: −0.0059 [±0.00071], t=−16.3, dz=−1.15         → learner attrites LESS (better)

*Open question:* what does a rank-flip on the observed metric as the handcuff channel strengthens mean for the H1 claim?

## Fact 2 — On the HIDDEN metric, the learner is below the static baseline at every h
Paired contrast P3 − P2, mean latent satisfaction (higher = better), and archetype gap (lower = more equitable):
- satisfaction: h=0.0 −0.0248 [±0.0012], h=0.5 −0.0280, h=1.0 −0.0368, h=1.5 −0.0450. All t ≤ −41, p ≤ 2×10⁻¹⁰⁰, Cohen dz −2.95 → −3.89.
- archetype gap: h=0.0 +0.0239, h=0.5 +0.0294, h=1.0 +0.0447, h=1.5 +0.0604. All t ≥ 17, dz +1.24 → +2.16.
Both magnitudes increase monotonically with h.

*Open question:* is this the pre-registered H2 signature (Goodhart)? Does K1 fire or not? (K1 = "if this gap does not appear CI-separated across the h grid, the central claim is dead.")

## Fact 3 — Perfect observation (oracle) does not reach the static baseline
Mean latent satisfaction: P5 (oracle, true s in reward) vs P2 (static) at h=1.0: 0.5466 [±0.0017] vs 0.5662 [±0.0003]. P5 − P3 satisfaction is positive at all h (+0.012 → +0.021, all p<10⁻²⁶), i.e., the oracle recovers part of the P3→P2 gap but P5 < P2 everywhere.

*Open question:* if even perfect satisfaction observation underperforms the theory baseline, what does that imply about whether the problem is *observability* vs *objective design*?

## Fact 4 — Audit dose–response is NON-MONOTONE (the "J-curve")
P4 mean latent satisfaction as audit rate p rises, and recovery fraction toward the oracle = (P4−P3)/(P5−P3), at h=1.0:
- p=0.00: 0.5276, recovery −0.10
- p=0.05: 0.5196, recovery −0.57   ← sparse auditing is WORSE than no auditing
- p=0.10: 0.5238, recovery −0.33
- p=0.20: 0.5309, recovery +0.08
- p=0.40: 0.5356, recovery +0.36
The same dip-then-rise shape holds at h=0, 0.5, 1.5 (see `results/table_audit_frontier.csv`). Best observed recovery across the grid is ≈+0.51 (h=1.5, p=0.4); no cell reaches 1.0.

*Open question:* the pre-registered H3 ("most recovery arrives by p≈0.1–0.2") is not what the data shows. Does K2 apply, does the finding become a negative/nuanced result about audit design, or does it reframe? This is the single most important interpretive decision in the paper.

## Fact 5 — What each policy actually allocates (h=1.0, category shares)
- P2 static: M .19 / R .21 / F .27 / D .24 / S .09 (spread across archetype-matched categories)
- P3 learner: M .25 / R .17 / F .23 / D .21 / S .05 (monetary share up, social down)
- Monetary share of the learner rises with h (0.15 → 0.30 from h=0 to 1.5; `table_summary.csv`).

*Open question:* does the allocation mix support a mechanism story for Facts 1–2?

## KILL-TEST FACTS (from `kill_*.csv` via `analyze_kill.py`; the load-bearing ones)

### Fact 6 — the Goodhart gap is NOT merely the crowding constant (H2 tautology test)
Paired P2−P3 latent satisfaction at **crowding-multiplier = 0 and handcuff = 0** (all amplifiers off): population +0.0151 [95% CI ±0.0010], founder-cohort +0.0141 [±0.0011], both 98% seed-sign-stable. The gap grows monotonically with the crowding multiplier and with h (fig6). The base channel present at crowd=0 is exploration/estimator cost; H2 is a Holmström–Milgrom instantiation, not a new mechanism.

### Fact 7 — the gap is genuine per-person, not survivorship composition (harm-vs-trap)
The founder-cohort gap (never-left originals only) persists (Fact 6). Under alternative replacement schemes at h=1: leavers not replaced +0.0237 [±0.0017]; replacement drawn from survivors +0.0489 [±0.0019]; both ≥98% sign-stable. The sign is invariant to how leavers are handled. *Open question: does this rule out the "trapping, not harming" confound?*

### Fact 8 — the audit backfire does NOT survive a precision-aware estimator (the decisive kill-test)
Backfire = P4(p=0.05) − P4(p=0) latent satisfaction (negative = backfire), BH-FDR corrected across the 6-cell family:
- raw estimator: h=0 −0.0085 [±0.0019] (p_fdr=1×10⁻¹⁴), h=1 −0.0076 [±0.0026] (p_fdr=7×10⁻⁸) → **backfire**
- **shrink estimator: h=0 +0.0003 [±0.0020] (n.s.), h=1 +0.0006 [±0.0024] (n.s.) → backfire GONE**
- gated estimator: h=0 −0.0048 (backfire), h=1 −0.0017 (n.s.)
Pre-registered verdict fired: **the backfire is largely a naive-estimator artifact; precision weighting removes it** (fig7). The supported claim is "naive audit rewards backfire; precision-aware estimation fixes it," not "auditing backfires."

### Fact 9 — robustness across 250 random assumption-worlds
Drawing gains, crowding, and hazard slope over plausible ranges: population Goodhart-gap sign holds in **100%**, founder-gap sign in **100%**, raw J-curve backfire in only **62%** (fig9). *Open question: which findings can be called "robust across the assumption space" and which cannot?*

### Fact 10 — IBM HR benchmark calibration (`calibrate/fit_ibm.py`, `sim.py calibrated`)
Calibrating the two most load-bearing choices to the IBM HR Analytics set ($n{=}1470$): the satisfaction$\to$attrition hazard slope is fit at **0.75 (univariate) to 0.95 (controlled)**, CI [0.38, 1.36] — vs. the illustrative default 3.2 (i.e., my invented value was ~3–4× too strong). Archetype mix is skewed (LowComp 39% vs. equal-share 25%). An off-satisfaction retention channel is empirically present: **StockOptionLevel coefficient −0.51 [−0.72, −0.31]** controlling for satisfaction (grounds the handcuff). Re-running the key contrasts with the fitted slope + real mix:
- **Welfare gap (P2−P3) SURVIVES:** +0.033 (calibrated) vs +0.037 (illustrative).
- **Equity gap (learner more unequal) SURVIVES, if anything larger:** +0.047 (calibrated) vs +0.045.
- **Founder gap SURVIVES:** +0.036–0.038.
- **Audit backfire does NOT survive:** raw-estimator backfire −0.0076 (illustrative) → **+0.003 (calibrated)** — it needed the strong illustrative coupling. Combined with Fact 8 (shrinkage removes it), the audit backfire is doubly fragile.

The headline is welfare+equity erosion (robust, calibrated, a distributional Goodhart/HM sharpening); the audit finding is a negative/regime-specific result, not a headline claim.

## Numbers you can quote verbatim (all traced to results/*.csv)
Every figure and every number above is regenerated by `analyze.py` from `full.csv`.
Do **not** put any number in the paper that you have not personally regenerated and
logged. If you change any G2 parameter, re-run `full` and this digest is void until
re-generated.
