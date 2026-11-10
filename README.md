# Incentive Allocation under Partial Observability

This simulation tests what happens when an allocator learns from employee retention and has no direct measure of satisfaction. It runs on a CPU with the Python standard library.

## Simulation results

The main experiment used 200 seeds. See the [results digest](RESULTS_DIGEST.md) for the comparisons.
- A learner rewarded only on observed **retention** matches a theory-based static
  allocator on retention while degrading unobserved **latent satisfaction** and, more
  sharply, **cross-archetype equity** (~4× the inequality of the static baseline).
- This erosion is robust: it survives a zero-crowding control, a never-left "founder"
  cohort (ruling out survivorship composition), 250 randomized parameter worlds, and
  **calibration of the attrition-hazard slope and archetype mix to the IBM HR Analytics
  benchmark** — in which an off-satisfaction "handcuff" retention channel is itself
  empirically present (stock-option coefficient −0.51 controlling for satisfaction).
- The equity result describes the distribution of outcomes in a model of multitask principal–agent theory.
- Adding audited satisfaction to the reward appeared to make outcomes worse only with a naive estimator and strong coupling. A precision-aware estimator or weaker coupling removed the effect. The [verification report](VERIFICATION_REPORT.md) gives the conditions and results.

> **AI-use disclosure.** The simulation code, analysis/calibration code, and figures were
> AI-assisted. The research questions, experimental design, execution, interpretation,
> and paper text are the author's own. See the paper's Acknowledgments.

## What's here
```
sim/sim.py             simulation engine — stdlib only. validate | pilot | full | kill | calibrated
sim/analyze.py         descriptive stats + figures 2–5 (numpy/scipy/pandas/matplotlib)
sim/analyze_kill.py    kill-test stats (Monte-Carlo SE, sign-stability, TOST, BH-FDR) + figs 6–9
sim/fig1_schematic.py  Figure 1 conceptual schematic
calibrate/fit_ibm.py   fit hazard slope + archetype mix to the IBM HR benchmark → fig10
results/*.csv          run grids + summary/contrast/kill/calibration tables
results/figs/          figures 1–10 (pdf + png)
paper/main.tex         full paper (compiles to main.pdf)
paper/main_short.tex   short / extended-abstract version (main_short.pdf)
VERIFICATION_REPORT.md adversarial review + pre-registered kill-tests + citation checks
RESULTS_DIGEST.md      descriptive readout of every headline number
ROADMAP.md             calibration + validation extensions
```

## Reproduce from a clean machine
```bash
cd sim
python3 sim.py validate      # prints ALL VALIDATIONS PASSED
python3 sim.py full          # ~18 min single core → ../results/full.csv (8000 rows)
python3 sim.py kill          # kill-tests → ../results/kill_*.csv
python3 sim.py calibrated    # calibration survival → ../results/calibrated.csv

# IBM calibration (place the Kaggle "IBM HR Analytics Employee Attrition &
# Performance" CSV in calibrate/ or set HR_CSV=/path/to.csv — not redistributed here):
python3 ../calibrate/fit_ibm.py

# analysis tooling (figures/tables):
python3 -m pip install -r ../requirements.txt
python3 analyze.py && python3 analyze_kill.py && python3 fig1_schematic.py
```
Every run is seeded; `sim.py full` regenerates `results/full.csv` bit-for-bit on the same
Python, and each script prints its own `sha256` so results tie to an exact code version.

## Method notes
Four employee archetypes, five incentive categories, and their affinity mapping;
theory-anchored behavioral mechanisms (intrinsic-motivation crowding-out; a
hygiene/motivator distinction; a satisfaction-independent "handcuff" retention channel).
All magnitudes are illustrative/non-calibrated except the hazard slope and archetype mix,
which are fit to the IBM HR benchmark; central findings are reported across a
pre-registered sensitivity grid and randomized-world sweep so they do not depend on any
single constant. Full design and falsification tests: `paper/` and `VERIFICATION_REPORT.md`.

## Citing
See `CITATION.cff`.

## License
Code: MIT (see `LICENSE`). Paper text and figures: author retains rights.
