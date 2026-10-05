| reader            | metric     | versus               |    diff |   ci_lo |   ci_hi |      p |   n |
|:------------------|:-----------|:---------------------|--------:|--------:|--------:|-------:|----:|
| Qwen3.8-Flash     | qa_correct | Recency window       |  36.9   |   29.5  |   44.2  | 0.0002 | 217 |
| Qwen3.8-Flash     | qa_correct | Observation masking  |  45.2   |   37.3  |   53    | 0.0002 | 217 |
| Qwen3.8-Flash     | qa_correct | BM25 (turns)         |  45.2   |   38.2  |   52.1  | 0.0002 | 217 |
| Qwen3.8-Flash     | qa_correct | LLM summary          |  74.7   |   68.7  |   80.2  | 0.0002 | 217 |
| Qwen3.8-Flash     | qa_correct | LLM summary + recent |  39.2   |   31.3  |   47    | 0.0002 | 217 |
| Qwen3.8-Flash     | path_hit   | Recency window       |  -1.15  |   -9.2  |    6.9  | 0.884  |  87 |
| Qwen3.8-Flash     | path_hit   | Observation masking  |  -1.15  |  -10.3  |    8.05 | 0.903  |  87 |
| Qwen3.8-Flash     | path_hit   | BM25 (turns)         |  -2.3   |  -11.5  |    6.9  | 0.726  |  87 |
| Qwen3.8-Flash     | path_hit   | LLM summary          |   9.2   |    0    |   18.4  | 0.0648 |  87 |
| Qwen3.8-Flash     | path_hit   | LLM summary + recent |   0     |   -8.05 |    8.05 | 1      |  87 |
| Qwen3.8-Flash     | ident_f1   | Recency window       |  -2.18  |   -6.41 |    2.12 | 0.317  | 239 |
| Qwen3.8-Flash     | ident_f1   | Observation masking  |  -2.23  |   -6.61 |    2.15 | 0.306  | 239 |
| Qwen3.8-Flash     | ident_f1   | BM25 (turns)         |  -0.378 |   -5.32 |    4.65 | 0.882  | 239 |
| Qwen3.8-Flash     | ident_f1   | LLM summary          |  15.8   |   11    |   20.9  | 0.0002 | 239 |
| Qwen3.8-Flash     | ident_f1   | LLM summary + recent |   0.613 |   -3.75 |    5.21 | 0.793  | 239 |
| Qwen3.7-Flash     | qa_correct | Recency window       |  25.3   |   18.9  |   31.8  | 0.0002 | 217 |
| Qwen3.7-Flash     | qa_correct | Observation masking  |  30.9   |   24    |   37.8  | 0.0002 | 217 |
| Qwen3.7-Flash     | qa_correct | BM25 (turns)         |  29.5   |   23    |   35.9  | 0.0002 | 217 |
| Qwen3.7-Flash     | qa_correct | LLM summary          |  46.1   |   39.2  |   53    | 0.0002 | 217 |
| Qwen3.7-Flash     | qa_correct | LLM summary + recent |  32.3   |   25.3  |   39.2  | 0.0002 | 217 |
| Qwen3.7-Flash     | path_hit   | Recency window       |   3.45  |   -1.15 |    9.2  | 0.236  |  87 |
| Qwen3.7-Flash     | path_hit   | Observation masking  |   1.15  |   -3.45 |    5.75 | 0.808  |  87 |
| Qwen3.7-Flash     | path_hit   | BM25 (turns)         |   2.3   |   -5.75 |   10.3  | 0.666  |  87 |
| Qwen3.7-Flash     | path_hit   | LLM summary          |   5.75  |   -1.15 |   12.6  | 0.12   |  87 |
| Qwen3.7-Flash     | path_hit   | LLM summary + recent |   1.15  |   -5.75 |    8.05 | 0.871  |  87 |
| Qwen3.7-Flash     | ident_f1   | Recency window       |  -2.52  |   -6.74 |    1.54 | 0.231  | 239 |
| Qwen3.7-Flash     | ident_f1   | Observation masking  |  -5.32  |   -9.26 |   -1.46 | 0.0076 | 239 |
| Qwen3.7-Flash     | ident_f1   | BM25 (turns)         |   1.44  |   -2.95 |    5.94 | 0.51   | 239 |
| Qwen3.7-Flash     | ident_f1   | LLM summary          |   9.47  |    4.67 |   14.3  | 0.0004 | 239 |
| Qwen3.7-Flash     | ident_f1   | LLM summary + recent |  -2.96  |   -7.15 |    1.34 | 0.178  | 239 |
| DeepSeek-V4-Flash | qa_correct | Recency window       |  32.7   |   25.3  |   40.1  | 0.0002 | 217 |
| DeepSeek-V4-Flash | qa_correct | Observation masking  |  35     |   27.6  |   42.4  | 0.0002 | 217 |
| DeepSeek-V4-Flash | qa_correct | BM25 (turns)         |  35.5   |   28.6  |   42.4  | 0.0002 | 217 |
| DeepSeek-V4-Flash | qa_correct | LLM summary          |  53.5   |   47    |   59.9  | 0.0002 | 217 |
| DeepSeek-V4-Flash | qa_correct | LLM summary + recent |  34.6   |   27.6  |   41.5  | 0.0002 | 217 |
| DeepSeek-V4-Flash | path_hit   | Recency window       |  -6.9   |  -14.9  |    1.15 | 0.1    |  87 |
| DeepSeek-V4-Flash | path_hit   | Observation masking  |  -5.75  |  -14.9  |    2.3  | 0.23   |  87 |
| DeepSeek-V4-Flash | path_hit   | BM25 (turns)         |  -8.05  |  -16.1  |    0    | 0.0564 |  87 |
| DeepSeek-V4-Flash | path_hit   | LLM summary          |   0     |  -11.5  |   11.5  | 1      |  87 |
| DeepSeek-V4-Flash | path_hit   | LLM summary + recent | -10.3   |  -19.5  |   -1.15 | 0.0328 |  87 |
| DeepSeek-V4-Flash | ident_f1   | Recency window       |  -3.31  |   -8.09 |    1.55 | 0.182  | 239 |
| DeepSeek-V4-Flash | ident_f1   | Observation masking  |  -2.53  |   -7.85 |    2.69 | 0.343  | 239 |
| DeepSeek-V4-Flash | ident_f1   | BM25 (turns)         |  -1.62  |   -6.19 |    2.94 | 0.49   | 239 |
| DeepSeek-V4-Flash | ident_f1   | LLM summary          |   9.38  |    4.26 |   14.7  | 0.0008 | 239 |
| DeepSeek-V4-Flash | ident_f1   | LLM summary + recent |  -3.66  |   -8.82 |    1.29 | 0.14   | 239 |