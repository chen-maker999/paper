| metric            |   budget |     skc | best_baseline         |   baseline |    diff |   ci_lo |   ci_hi |      p |   n_traj |
|:------------------|---------:|--------:|:----------------------|-----------:|--------:|--------:|--------:|-------:|---------:|
| State correct     |     4000 | 50.5    | Random selection      |    43.8    |  6.67   |  6.18   |  7.17   | 0.0002 |     1050 |
| State correct     |     8000 | 66.2    | Random selection      |    56.8    |  9.34   |  8.75   |  9.94   | 0.0002 |     1015 |
| State correct     |    16000 | 82      | Random selection      |    71.9    | 10.1    |  9.54   | 10.7    | 0.0002 |      906 |
| State correct     |    32000 | 94.8    | Random selection      |    84.5    | 10.2    |  9.63   | 10.9    | 0.0002 |      582 |
| State stale       |     4000 |  0.217  | Recency window        |     0.233  | -0.0156 | -0.0902 |  0.0451 | 0.688  |     1050 |
| State stale       |     8000 |  0.239  | Recency window        |     0.211  |  0.0282 | -0.0362 |  0.0871 | 0.367  |     1015 |
| State stale       |    16000 |  0.154  | Recency window        |     0.175  | -0.0208 | -0.0838 |  0.0332 | 0.532  |      906 |
| State stale       |    32000 |  0.0663 | Recency window        |     0.0838 | -0.0176 | -0.0603 |  0.0223 | 0.385  |      582 |
| Use (long range)  |     4000 | 66.7    | BM25 (chunks)         |    69      | -2.29   | -3.6    | -0.931  | 0.0004 |      967 |
| Use (long range)  |     8000 | 82.2    | BM25 (chunks)         |    83.8    | -1.65   | -2.76   | -0.485  | 0.0048 |      907 |
| Use (long range)  |    16000 | 93.4    | BM25 (chunks)         |    93.7    | -0.282  | -1.18   |  0.658  | 0.531  |      779 |
| Use (long range)  |    32000 | 98.9    | BM25 (chunks)         |    98.1    |  0.76   |  0.116  |  1.51   | 0.0228 |      473 |
| Copy (long range) |     4000 | 57.2    | BM25 (chunks)         |    59      | -1.88   | -4.26   |  0.561  | 0.126  |      761 |
| Copy (long range) |     8000 | 78.2    | Truncate observations |    78.8    | -0.627  | -3.47   |  2.29   | 0.695  |      719 |
| Copy (long range) |    16000 | 92.9    | BM25 (chunks)         |    90.5    |  2.48   |  0.675  |  4.24   | 0.006  |      607 |
| Copy (long range) |    32000 | 98.2    | BM25 (chunks)         |    97.3    |  0.916  | -0.625  |  2.38   | 0.228  |      322 |