# 6N-shortinterval-variance

**Part XXIX — Twin Counts in Short Intervals: Singular-Series Rigidity and the Density-Gradient Variance**

Ruqing Chen · GUT Geoservice Inc., Montreal · June 2026

Companion code and data for Part XXIX of *Arithmetic Geodynamics on the 6N Skeleton*.
This is the Montgomery–Soundararajan short-interval variance question, transposed to the twin
constellation on the 6N skeleton. **Everything here is a measured sieve result — no fitted
parameters, no fabricated numbers.**

## The question

Primes > 3 live at `6N ± 1`. A **twin centre** is an `N` with both `6N-1` and `6N+1` prime. We sieve
every centre to `6N ≤ 6×10⁸` (so `N ≤ 10⁸`), giving **2,166,300 twin centres** at global rate
`p₀ = 0.0216630`. We partition `[1, N_max]` into disjoint windows of `H` consecutive centres, count
twin centres per window, and study the **Fano factor** `F(H) = Var/mean` versus `H`. A Poisson process
gives `F ≡ 1`; the deviation is the signal.

## The result, in one paragraph

`F(H)` is **sub-Poisson** for small windows (dips to `F ≈ 0.92` near `H ≈ 50`), crosses `F = 1` at
`H* ≈ 410`, and rises steeply **super-Poisson** afterward (`F ≈ 37` at `H = 10⁵`). Two cleanly
separable mechanisms explain the whole curve:

1. **Local arithmetic — the Hardy–Littlewood 4-tuple.** The pair correlation of twin centres at
   separation `k` is governed by the singular series of the offset 4-tuple `{−1, 1, 6k−1, 6k+1}`:

   ```
   C(k) = G · Π_{p|k}   (p−2)/(p−4)
            · Π_{p|3k−1} (p−3)/(p−4)
            · Π_{p|3k+1} (p−3)/(p−4) ,   p > 3,
   G = Π_{p>3} (1 − 4/(p−2)²) = 0.39688.
   ```

   Verified against the data to a **median 1.7%** across `k ≤ 3000`; every arithmetic spike is
   reproduced. The integrated local correlation is **negative** (`Σ(C_HL−1) = −5.5`): the four-fold
   killer constraint makes twin centres locally *more regular* than Poisson. This rigidity *is* the
   sub-Poisson dip.

2. **Density gradient — the Montgomery–Soundararajan term.** The super-Poisson rise is not local: it
   is the smooth deterministic decline of twin density across `[1, X]`. Density-detrending splits the
   variance into `F_local` (stays sub-Poisson, tracks the singular-series prediction to ~1%) and
   `F_trend ∝ H` (`Var_trend ∝ H²`), which reaches **92%** of the total variance at `H = 10⁵`. The
   crossover `H* ≈ 410` is where the global density gradient overtakes the local rigidity.

**What is left open (honestly):** the genuine Montgomery–Soundararajan lower-order term — the
`H·log(X/H)` fluctuation *at fixed density* and the asymptotic Gaussianity — sits below resolution at
reachable `H` and `X = 6×10⁸`, dominated by the rigidity and the deterministic trend. We bound where
it must live; we do not claim to measure it. No claim is made about the infinitude of twins.

## Reproducing

```bash
pip install -r requirements.txt
cd code
python3 var_compute.py     # sieve to 6N=6e8; Fano ladder, C(k), HL prediction, variance identity
                           #   -> data/pair_correlation.csv, fano_ladder.csv,
                           #      variance_identity.csv, summary.csv          (~55 s, ~1.5 GB RAM)
python3 var_detrend.py     # density-detrended decomposition F_total / F_local / F_trend
                           #   -> data/fano_decomposition.csv                  (~15 s)
python3 var_robust.py      # robustness of F_local to the smoothing scale W (console output)
python3 makefigs29.py      # reads the CSVs -> figures/p29_fig1.pdf, p29_fig2.pdf
```

`explore_var.py` is a faster exploratory pass at `6N ≤ 2×10⁸` (writes `data/explore_fano.csv`).

Scripts re-run the sieve each time (≈10 s) and are self-contained; paths resolve relative to the
script (outputs land in `../data` and `../figures`). Requires ~1.5 GB RAM for the `10⁸`-centre
cumulative-sum arrays. Single-threaded.

## Files

```
code/    var_compute.py   var_detrend.py   var_robust.py   makefigs29.py   explore_var.py
data/    pair_correlation.csv   k, C_measured, C_HL_prediction          (k = 1..3000)
         fano_ladder.csv        H, n_windows, mean, var, F_total
         fano_decomposition.csv H, F_total, F_local_detrended, F_trend
         variance_identity.csv  H, F_measured, F_from_C_measured, F_from_C_HL
         summary.csv            parameter, value  (p0, G, crossover H*, the correlation sums, ...)
figures/ p29_fig1.pdf  p29_fig2.pdf
paper/   paper29.tex   paper29.pdf
```

All data files are plain CSV — openable in any text editor or spreadsheet.

## Citation

See `CITATION.cff`. The paper is archived on Zenodo (DOI in the citation file once minted).

## License

MIT (see `LICENSE`).
