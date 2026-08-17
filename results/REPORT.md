# Prime Pattern Lab — Initial Run Report

## Scope

- Numerical range: `1,000,000` to `100,000,000`
- Logarithmic blocks: `12`
- Tested even offsets: `2` through `10,000`
- Primary statistic: weighted Hardy–Littlewood pair residual
- Validation: chronological holdout plus a wheel-conditioned random null

## Main numerical results

- Aggregate raw-count observed/predicted ratio: `0.99977362`
- Aggregate weighted observed/predicted ratio: `0.99997339`
- Median absolute per-offset weighted residual: `0.000439`
- Largest absolute per-offset weighted residual: `0.002562` at `h=8612`
- Final-block weighted residual RMSE across offsets: `0.002531`
- Final-block wheel-null RMSE across offsets: `0.015355`

## Result

No transferable residual law was found in the tested model class. The aggregate weighted prediction error was `-0.00002661`, and every fitted model performed worse on later numerical scales than simply predicting a zero residual. The correct conclusion from this run is that the Hardy–Littlewood singular series explains the tested pair frequencies extremely well through `10^8`; it is not evidence that the conjecture has been proved.

## Out-of-scale prediction

The models were trained only on earlier blocks and evaluated on later blocks.

| Model | Holdout RMSE | Holdout MAE |
|---|---:|---:|
| No-extra-pattern baseline (residual = 0) | 0.002106 | 0.001619 |
| Interpretable ridge model | 0.002133 | 0.001643 |
| Shallow boosted-tree model | 0.002121 | 0.001632 |
| Sparse second-order feature model | 0.002110 | 0.001622 |

An AI model is useful only if it beats the zero-residual baseline on later numerical scales. Training fit is deliberately not reported as evidence.

## Log-scale spectral scan

The scan used only `12` block samples. Its strongest prime/null power ratio was `1.2964` at approximately `1.000` cycles across the observed log-range. With so few scale samples, this can reject a large common oscillation but cannot resolve or identify a zeta-zero-like spectrum.

## Arithmetic-feature check

The strongest tested rank correlation between an offset feature and the aggregate weighted residual was `omega`, with Spearman `rho=-0.02505` and uncorrected `p=0.0766`. Because several features were tested and offsets are arithmetically dependent, this is diagnostic rather than discovery evidence.

## Interpretation rules

1. Ratios near one support the known singular-series baseline; they do not prove Hardy–Littlewood.
2. A model that loses to residual = 0 has found no transferable extra law.
3. The wheel null is a finite modular model, not a complete model of primes.
4. Heatmap bands aligned with factors of offsets are presumed modular until they survive larger wheels.
5. This range is far too small to assess the Riemann hypothesis or prove any fixed prime-pair conjecture.

## Files

- `pair_metrics.csv.gz`: every block/offset measurement
- `offset_summary.csv`: aggregate statistics and arithmetic features
- `block_summary.csv`: sieve and gap summary by block
- `block_residual_summary.csv`: residual error by scale
- `analysis_summary.json`: model, correlation and PCA results
- `pca_offset_loadings.csv`: dominant cross-offset modes
- `spectral_summary.csv`: low-resolution log-scale frequency scan
- `gap_distribution.csv`: normalized gap histogram
- PNG files: diagnostic plots
