| metric            |   budget |    skc | best_baseline    |   baseline |    diff |   ci_lo |    ci_hi |      p |   n_traj |
|:------------------|---------:|-------:|:-----------------|-----------:|--------:|--------:|---------:|-------:|---------:|
| State correct     |     4000 | 67.1   | Random selection |    57.8    |  9.31   |  8.84   |  9.82    | 0.0002 |     1382 |
| State correct     |     8000 | 78.9   | Random selection |    67.3    | 11.6    | 10.9    | 12.2     | 0.0002 |     1162 |
| State correct     |    16000 | 89.1   | Random selection |    75.7    | 13.4    | 12.7    | 14.1     | 0.0002 |      846 |
| State correct     |    32000 | 94.8   | Random selection |    82.5    | 12.4    | 11.6    | 13.1     | 0.0002 |      519 |
| State stale       |     4000 |  0.24  | Recency window   |     0.318  | -0.0779 | -0.16   | -0.00104 | 0.0464 |     1382 |
| State stale       |     8000 |  0.176 | Recency window   |     0.24   | -0.064  | -0.151  |  0.0148  | 0.107  |     1162 |
| State stale       |    16000 |  0.104 | Recency window   |     0.223  | -0.119  | -0.2    | -0.0418  | 0.0002 |      846 |
| State stale       |    32000 |  0.055 | Recency window   |     0.0952 | -0.0402 | -0.0888 |  0.00483 | 0.0836 |      519 |
| Use (long range)  |     4000 | 81.5   | BM25 (chunks)    |    80.5    |  0.992  | -0.581  |  2.54    | 0.209  |      948 |
| Use (long range)  |     8000 | 90.3   | BM25 (chunks)    |    89.4    |  0.836  | -0.648  |  2.35    | 0.277  |      709 |
| Use (long range)  |    16000 | 95.9   | BM25 (chunks)    |    95.4    |  0.499  | -0.738  |  1.69    | 0.435  |      480 |
| Use (long range)  |    32000 | 99     | BM25 (chunks)    |    99      |  0.0229 | -0.987  |  0.952   | 0.95   |      229 |
| Copy (long range) |     4000 | 70.5   | BM25 (chunks)    |    75.7    | -5.17   | -7.77   | -2.44    | 0.0002 |      772 |
| Copy (long range) |     8000 | 84.6   | BM25 (chunks)    |    87.7    | -3.12   | -5.56   | -0.725   | 0.0108 |      611 |
| Copy (long range) |    16000 | 94.2   | BM25 (chunks)    |    95.3    | -1.11   | -3.09   |  0.837   | 0.268  |      416 |
| Copy (long range) |    32000 | 97.2   | BM25 (chunks)    |    99      | -1.77   | -3.89   | -0.0232  | 0.0476 |      195 |