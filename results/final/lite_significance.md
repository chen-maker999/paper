| metric            |   budget |     skc | best_baseline    |   baseline |    diff |    ci_lo |   ci_hi |      p |   n_traj |
|:------------------|---------:|--------:|:-----------------|-----------:|--------:|---------:|--------:|-------:|---------:|
| State correct     |     4000 | 65      | Random selection |    57.8    |  7.19   |   6.7    |  7.73   | 0.0002 |     1382 |
| State correct     |     8000 | 74.8    | Random selection |    67.3    |  7.45   |   6.74   |  8.16   | 0.0002 |     1162 |
| State correct     |    16000 | 85      | Random selection |    75.7    |  9.24   |   8.42   | 10.1    | 0.0002 |      846 |
| State correct     |    32000 | 91.8    | Random selection |    82.5    |  9.35   |   8.53   | 10.1    | 0.0002 |      519 |
| State stale       |     4000 |  0.292  | Recency window   |     0.318  | -0.0258 |  -0.111  |  0.0559 | 0.564  |     1382 |
| State stale       |     8000 |  0.223  | Recency window   |     0.24   | -0.017  |  -0.11   |  0.0691 | 0.72   |     1162 |
| State stale       |    16000 |  0.166  | Recency window   |     0.223  | -0.0569 |  -0.144  |  0.0259 | 0.186  |      846 |
| State stale       |    32000 |  0.0662 | Recency window   |     0.0952 | -0.029  |  -0.0808 |  0.019  | 0.256  |      519 |
| Use (long range)  |     4000 | 79.5    | BM25 (chunks)    |    80.5    | -0.996  |  -2.59   |  0.579  | 0.208  |      948 |
| Use (long range)  |     8000 | 88.3    | BM25 (chunks)    |    89.4    | -1.11   |  -2.62   |  0.425  | 0.154  |      709 |
| Use (long range)  |    16000 | 94.6    | BM25 (chunks)    |    95.4    | -0.854  |  -2.13   |  0.416  | 0.194  |      480 |
| Use (long range)  |    32000 | 97.4    | BM25 (chunks)    |    99      | -1.55   |  -3.44   |  0.0117 | 0.0524 |      229 |
| Copy (long range) |     4000 | 67.6    | BM25 (chunks)    |    75.7    | -8.12   | -10.7    | -5.3    | 0.0002 |      772 |
| Copy (long range) |     8000 | 81.3    | BM25 (chunks)    |    87.7    | -6.44   |  -8.94   | -3.93   | 0.0002 |      611 |
| Copy (long range) |    16000 | 90      | BM25 (chunks)    |    95.3    | -5.36   |  -7.69   | -3.04   | 0.0002 |      416 |
| Copy (long range) |    32000 | 96.2    | BM25 (chunks)    |    99      | -2.72   |  -5.06   | -0.758  | 0.0068 |      195 |