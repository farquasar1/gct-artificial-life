# Generative Constraint Theory: Interventional Closure and Viable Compositional Reach

Code and data for the article of the same title by Armando Vieira (Institute of Computer Science, University of Tartu; armando.vieira@ut.ee; ORCID 0000-0002-4447-2291), submitted to *Artificial Life* (MIT Press).

The repository contains the two deterministic artificial chemistries used in the paper, every script needed to regenerate its numbers and figures, and the outputs as they were reported.

## Requirements

Python 3.13 (tested with 3.13.5), plus numpy 2.5.4, scipy 1.18.1 and matplotlib 3.11.2 (pinned in `requirements.txt`). Nothing else is used and no network access is needed.

## Quick start

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    PYTHON=.venv/bin/python bash reproduce.sh

On a normal laptop the full run takes about 1.5 minutes (85 s measured on one CPU). It overwrites `results/` and `figures/`. With the pinned versions the regenerated CSV and JSON files are byte-identical to the committed ones.

## Scripts

| Script | What it does | Writes |
|---|---|---|
| `run_all.py` | Experiment 1. A three-catalyst loop with one reusable intermediate, run just above and below the spectral threshold (k = 0.22 and k = 0.18, with delta = 0.2). It applies a 90% washout, then a novel-substrate assay from t = 150. It also covers the intermediate-rescue control, 400 parameter draws and a 2,100-run rate sweep. | `results/summary.json`, `parameters.json`, `replicates.csv`, `sweep.csv`, `trajectories.csv`; figures 1 to 3 |
| `validate.py` | Independent checks of the nominal experiment, including the interventional estimate of M against the analytic matrix. | `results/validation.json` |
| `topology_experiment.py` | Experiment 2. Two rewired six-catalyst topologies with matched initial stocks, resources and total flux. It runs 300 paired seeds each and four fixed constructions (2,400 outcomes), then compares seed-held-out prediction against an aggregate-throughput baseline. The fixed analysis plan is in `topology_protocol.md`, and its hash is checked. | `results/topology/*.csv`, `summary.json`, `parameters.json`; figure 4 |
| `topology_sensitivity.py` | A post-hoc comparator, labelled as such, that uses a topology-by-product label baseline. It runs no new simulations. | `results/topology/sensitivity.json` |
| `round3_checks.py` | Recalculates the recovery and early-assay diagnostics independently. It also integrates 24 full S/P assays to check the integrated-hazard yield formula. | `results/round3_checks.json` |

## Paper figures and tables

| In paper | File | Produced by |
|---|---|---|
| Fig. 2: nominal recovery after washout | `figures/Vieira.fig01.*` | `run_all.py` |
| Fig. 3: rate sweep (recovery, viability, yield) | `figures/Vieira.fig02.*` | `run_all.py` |
| Fig. 1: intermediate rescue as a positive control | `figures/Vieira.fig03.*` | `run_all.py` |
| Fig. 4: construction identity under rewiring and held-out prediction | `figures/Vieira.fig04.*` | `topology_experiment.py` |
| Construction-success table (Experiment 2) | `results/topology/summary.json` | `topology_experiment.py` |
| Rescue 200/200, viability 56/200; early-assay difference 0.247 (95% CI 0.240 to 0.255) | `results/summary.json`, `results/round3_checks.json` | `run_all.py`, `round3_checks.py` |
| Log loss 0.3595 (baseline) vs 0.0404 (with producer membership); topology-label baseline 0.0456 | `results/topology/summary.json`, `sensitivity.json` | `topology_experiment.py`, `topology_sensitivity.py` |

File numbers are the generated file names. They differ from the figure order in the paper, which puts the rescue figure first.

## Citation

See `CITATION.cff`. Please cite the article:

Vieira, A. (2026). Generative Constraint Theory: Interventional Closure and Viable Compositional Reach. Manuscript submitted to *Artificial Life*.

## License

MIT. See `LICENSE`.
