| reader            | method              |   qa_correct |   qa_stale |   qa_unknown |   qa_wrong |   path_hit |   ident_f1 |   ctx_tokens |   n_qa |
|:------------------|:--------------------|-------------:|-----------:|-------------:|-----------:|-----------:|-----------:|-------------:|-------:|
| Qwen3.8-Flash     | Recency window      |         40.1 |        0.9 |         44.7 |       14.3 |       42.5 |       47.9 |       7524.5 |    217 |
| Qwen3.8-Flash     | Observation masking |         31.8 |        0.0 |         50.2 |       18.0 |       42.5 |       47.9 |       7325.7 |    217 |
| Qwen3.8-Flash     | BM25 (turns)        |         31.8 |        4.1 |         53.5 |       10.6 |       43.7 |       46.1 |       8056.3 |    217 |
| Qwen3.8-Flash     | LLM summary         |          2.3 |        0.0 |         94.0 |        3.7 |       32.2 |       29.9 |       2364.1 |    217 |
| Qwen3.8-Flash     | SKC (ours)          |         77.0 |        0.9 |         10.1 |       12.0 |       41.4 |       45.7 |       8284.9 |    217 |
| Qwen3.7-Flash     | Recency window      |         24.9 |        0.5 |         57.1 |       17.5 |       29.9 |       34.2 |       7524.5 |    217 |
| Qwen3.7-Flash     | Observation masking |         19.4 |        1.4 |         60.4 |       18.9 |       32.2 |       37.0 |       7325.7 |    217 |
| Qwen3.7-Flash     | BM25 (turns)        |         20.7 |        3.7 |         59.0 |       16.6 |       31.0 |       30.2 |       8056.3 |    217 |
| Qwen3.7-Flash     | LLM summary         |          4.1 |        0.0 |         87.1 |        8.8 |       27.6 |       22.2 |       2518.2 |    217 |
| Qwen3.7-Flash     | SKC (ours)          |         50.2 |        2.3 |         23.0 |       24.4 |       33.3 |       31.6 |       8284.9 |    217 |
| DeepSeek-V4-Flash | Recency window      |         27.2 |        1.4 |         53.9 |       17.5 |       37.9 |       40.5 |       7524.5 |    217 |
| DeepSeek-V4-Flash | Observation masking |         24.9 |        0.9 |         60.8 |       13.4 |       36.8 |       39.7 |       7325.7 |    217 |
| DeepSeek-V4-Flash | BM25 (turns)        |         24.4 |        3.2 |         58.1 |       14.3 |       39.1 |       38.8 |       8056.3 |    217 |
| DeepSeek-V4-Flash | LLM summary         |          6.5 |        0.0 |         82.9 |       10.6 |       31.0 |       27.8 |       2115.5 |    217 |
| DeepSeek-V4-Flash | SKC (ours)          |         59.9 |        1.4 |         20.3 |       18.4 |       31.0 |       37.2 |       8284.9 |    217 |