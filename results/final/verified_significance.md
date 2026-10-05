| metric            |   budget |    skc | best_baseline         |   baseline |    diff |    ci_lo |   ci_hi |      p |   n_traj |
|:------------------|---------:|-------:|:----------------------|-----------:|--------:|---------:|--------:|-------:|---------:|
| State correct     |     4000 | 48.7   | Random selection      |    43.8    |  4.94   |  4.46    |  5.44   | 0.0002 |     1050 |
| State correct     |     8000 | 62.8   | Random selection      |    56.8    |  5.94   |  5.33    |  6.55   | 0.0002 |     1015 |
| State correct     |    16000 | 76.1   | Random selection      |    71.9    |  4.14   |  3.46    |  4.84   | 0.0002 |      906 |
| State correct     |    32000 | 87.9   | Random selection      |    84.5    |  3.41   |  2.63    |  4.24   | 0.0002 |      582 |
| State stale       |     4000 |  0.207 | Recency window        |     0.233  | -0.0253 | -0.1     |  0.0355 | 0.496  |     1050 |
| State stale       |     8000 |  0.253 | Recency window        |     0.211  |  0.0421 | -0.0193  |  0.0972 | 0.166  |     1015 |
| State stale       |    16000 |  0.201 | Recency window        |     0.175  |  0.0264 | -0.0366  |  0.0842 | 0.38   |      906 |
| State stale       |    32000 |  0.137 | Recency window        |     0.0838 |  0.0531 |  0.00421 |  0.0997 | 0.0324 |      582 |
| Use (long range)  |     4000 | 64.8   | BM25 (chunks)         |    69      | -4.13   | -5.43    | -2.76   | 0.0002 |      967 |
| Use (long range)  |     8000 | 79.1   | BM25 (chunks)         |    83.8    | -4.67   | -5.81    | -3.5    | 0.0002 |      907 |
| Use (long range)  |    16000 | 90.3   | BM25 (chunks)         |    93.7    | -3.42   | -4.42    | -2.39   | 0.0002 |      779 |
| Use (long range)  |    32000 | 96.6   | BM25 (chunks)         |    98.1    | -1.5    | -2.18    | -0.847  | 0.0002 |      473 |
| Copy (long range) |     4000 | 55.2   | BM25 (chunks)         |    59      | -3.8    | -6.21    | -1.39   | 0.0024 |      761 |
| Copy (long range) |     8000 | 74.2   | Truncate observations |    78.8    | -4.69   | -7.58    | -1.74   | 0.0004 |      719 |
| Copy (long range) |    16000 | 89.3   | BM25 (chunks)         |    90.5    | -1.13   | -3.05    |  0.8    | 0.256  |      607 |
| Copy (long range) |    32000 | 96.6   | BM25 (chunks)         |    97.3    | -0.663  | -2.27    |  0.994  | 0.454  |      322 |