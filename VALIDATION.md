# Validation record

## Automated checks

The following command completed with all eight tests passing:

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- ordinary and segmented sieve correctness;
- exact handling of prime powers in the von Mangoldt function;
- complete, non-overlapping geometric block construction;
- FFT autocorrelation against a direct dot-product implementation;
- prime-pair counts against direct Boolean enumeration;
- known Hardy–Littlewood relative factors for gaps 2, 4, 6, 10, 14 and 30;
- the local obstruction for odd pair offsets.

## Full-run checks

- Range: `[1,000,000, 100,000,000)`
- Prime count: `5,682,957`, equal to `pi(100,000,000) - pi(1,000,000)` because neither endpoint changes the count.
- Pair result rows: `60,000` (`12` blocks times `5,000` offsets).
- All numeric values were finite; no missing values or negative counts occurred.
- Direct enumeration independently reproduced raw and von-Mangoldt-weighted pair counts for gaps `2`, `6`, `30`, `210`, `2310` and `10000` in the first, eighth and final blocks.
- Runtime for the complete sieve, three autocorrelations per block, analysis and plotting: `33.83 seconds` on the recorded runtime.

## Scope limit

These checks establish implementation consistency. They do not prove the Hardy–Littlewood conjecture or turn finite numerical agreement into an asymptotic theorem.
