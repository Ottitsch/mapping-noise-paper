# Measuring Atom Mapping Error Growth in Reaction Barrier Models

Code and results for the paper: how the barrier error of a Chemprop CGR model grows when a controlled share of heavy atom maps is wrong, in training maps, test maps or both, and where RXNMapper's errors fall on that curve.

- `research/src/` data preparation, corruption operator (`lib/amn.py`), training driver (`lib/train.py`), job queue, analysis and figures
- `research/results/` every model's test predictions and metrics, the analysis tables and the figures

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r research/requirements.txt
cd research
```

## Reproduce

```
bash src/data_prep/download.sh && python3 -I src/data_prep/prep.py   # public datasets, splits
python3 -I src/repro/make_e2sn2_folds.py && bash src/repro/run.sh     # published clean points
bash src/rxnmapper_rate/run.sh                                       # RXNMapper per atom error rate
bash src/analysis/run_operator.sh                                    # realised corruption fractions
bash src/grid/run.sh                                                 # all Chemprop models, CPU, many hours
bash src/baseline_mfp/run.sh                                         # Morgan fingerprint baseline
python3 -I src/analysis/operator_counts.py && python3 -I src/analysis/profile_and_subset.py
python3 -I src/analysis/analyse.py && python3 -I src/analysis/results_md.py
python3 -I src/figures/make_figures.py
```

`results/` already holds every output, so the analysis and figure steps run without training.

## License

MIT, see `LICENSE`.
