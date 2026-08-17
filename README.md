# Prime Pattern Lab

This package tests whether prime-pair data contain a scale-stable residual after the known Hardy–Littlewood modular correction is removed.

It deliberately separates four layers:

1. exact prime generation by segmented sieve;
2. Hardy–Littlewood singular-series predictions;
3. a wheel-conditioned random null model;
4. chronological holdout tests for interpretable and shallow machine-learning models.

The analysis includes ridge regression, a shallow boosted tree, a sparse second-order feature library, PCA across offsets and a deliberately low-resolution spectral scan on the `log(x)` axis.

The principal quantity is

```text
R_h(block) = sum Lambda(n)Lambda(n+h) / (S(h) * usable_block_length) - 1
```

where `S(h)` is the prime-pair singular series. Prime powers are included in `Lambda`.

## Run

From this directory:

```bash
python -m unittest discover -s tests -v
python run_experiment.py --quick
python run_experiment.py --start 1000000 --limit 100000000 --max-gap 10000
```

The full command uses blockwise FFT autocorrelation. It never loops over all integers separately for every offset.

## Important controls

- Blocks are ordered by magnitude; late blocks are never mixed into training.
- A model must beat the `residual = 0` baseline on later scales.
- Wheel-random sequences expose patterns caused solely by small-prime exclusions.
- Raw prime labels are not used as a generic classification task.
- Correlation p-values are diagnostic and are not corrected discovery claims.

## Output

The `results` directory contains:

- `REPORT.md` — automatically generated findings;
- `pair_metrics.csv.gz` — measurement for every block and even offset;
- `offset_summary.csv` — aggregate residuals and factorization features;
- `block_summary.csv` and `block_residual_summary.csv`;
- `analysis_summary.json` and `run_metadata.json`;
- a normalized prime-gap histogram;
- diagnostic PNG plots.

## Interpretation

A finite-range fit is not evidence of a new prime law. A candidate becomes interesting only if it transfers to larger ranges, unseen offsets, different wheel sizes and mathematically distinct null models. Even then it is a conjecture until translated into a proof.
