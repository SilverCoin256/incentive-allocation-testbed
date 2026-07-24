# Future Work / Extensions
All CPU-only and reproducible from this repo. Ordered by how much each would strengthen the findings.

## Calibrate the dynamics, not just the hazard slope + mix
The current calibration (`calibrate/fit_ibm.py`) fits the satisfaction→attrition hazard slope and the archetype prevalence to the IBM HR Analytics benchmark. The satisfaction *dynamics* (per-category gains, crowding-out, habituation) remain illustrative because that benchmark is cross-sectional. A **longitudinal turnover-and-engagement panel** would let one fit the gains and the crowding-out magnitude directly, converting the last illustrative magnitudes into estimated ones. Candidate open/secondary sources to anchor parameters: aggregate satisfaction indices, the US BLS JOLTS quits series, and published engagement–turnover correlations.

## Scientific extensions (compute-only)
1. **Calibrated-parameter robustness.** Re-run the full grid with gains/crowding fit to real data instead of theory defaults; report whether the welfare/equity erosion and the audit null survive fitted dynamics (extends `sim.py calibrated`).
2. **Smarter-learner ablation.** Add a contextual learner and a satisfaction-aware but sparsely-sampled learner, to test whether the erosion is fundamental or an artifact of the cohort-bandit class (pre-empts the "strawman learner" objection).
3. **Fixed-cohort / survivor-conditioned welfare.** The founder-cohort metric already separates harm from trapping; add a Kitagawa–Oaxaca tenure decomposition and a quit-adjusted (outside-option) welfare accounting for a fully composition-robust measure.
4. **Mechanism decomposition.** Toggle crowding-out vs. handcuff vs. habituation one at a time to attribute the erosion to each channel, and separate the h=0 (crowding/exploration) channel from the h>0 (handcuff) channel.
5. **Audit-design sweep.** Extend the estimator study (`sim.py kill`) with inverse-variance weighting, IPW for survey propensity, and active (uncertainty-targeted) auditing, to map which audit designs help and in which regimes.

## Replication
The testbed is open and CPU-only; independent replication on a different archetype distribution (e.g. an organization's aggregated, de-identified employee mix) would test external validity. See `sim.py` for the entry points and the reproduction commands in `README.md`.
