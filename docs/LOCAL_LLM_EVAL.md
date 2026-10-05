# 用本地模型做 LLM 在环评测（操作说明）

本实验用一个真实 LLM 来“读”各压缩方法输出的上下文，检验压缩后的信息能否真正被模型用上：

| 任务 | 做法 | 指标 |
|---|---|---|
| A. 状态问答 | 问：“命令 `X` 最近一次运行的输出最后一行是什么？”（所选命令都在 ≥3 步之前运行过） | 正确 / **过期**（答成更早一次的输出）/ 错误 / UNKNOWN |
| B. 下一步动作预测 | 让模型写出 Agent 的下一个工具调用，与真实动作比对 | 文件路径命中率、标识符 F1 |

对比方法：Recency window、Observation masking、BM25、**SKC（本文）**，以及 **LLM 摘要基线**（同一个模型把较早的历史总结成不超过预算一半的摘要，再拼上最近两轮）。预算为 8k token。

---

## 1. 硬件与模型建议

- 上下文长度：**至少 16k**（8k 的压缩上下文加上提示词）；LLM 摘要基线建议 **32k**。
- 推荐模型（建议至少跑两个，结论更稳）：
  - `Qwen/Qwen3-14B`（关闭思考模式，加 `--no-think`），或 `Qwen/Qwen2.5-14B-Instruct`；
  - `Qwen/Qwen2.5-Coder-7B-Instruct` / `meta-llama/Llama-3.1-8B-Instruct`（显存较小时）。
- 显存参考：14B 的 bf16 约需 40GB 以上；24GB 显卡可用 7–8B，或 14B 的 AWQ/GPTQ 量化版。
- 规模与耗时：默认每个 LLM 抽 60 个决策点，共 300 个样本，约 3,300 次请求。在单卡 vLLM 上大约 1–3 小时。时间允许时可用 `--per-model 100`。

## 1.1 针对 AMD RX 9070 XT（16GB）+ 32GB 内存的推荐设置

- 后端：Windows 上用 **LM Studio**（开启 Developer → Local Server，默认地址 `http://localhost:1234/v1`），或 llama.cpp 的 **Vulkan** 版 `llama-server`；Linux 上用 llama.cpp（Vulkan 或 ROCm ≥ 6.4）。
- 模型（GGUF）：`Qwen3-8B` Q6_K（主模型），`Llama-3.1-8B-Instruct` Q6_K（不同家族）；可选 `Qwen3-14B` Q4_K_M。
- 服务端参数：上下文 **32768**、并行槽位 **1**、开启 Flash Attention、KV 缓存 **q8_0**。llama.cpp 示例：
  ```bash
  llama-server -m Qwen3-8B-Q6_K.gguf -c 32768 -np 1 -fa on -ctk q8_0 -ctv q8_0 -ngl 99 --port 8080
  ```
- 评测参数：`--per-model 40 --concurrency 1 --max-context 32768`。8B 模型每个约 4–5 小时，14B 约 8 小时，可以挂一夜。
- Windows PowerShell 中设置环境变量用：`$env:SKC_TOKENIZER = "$PWD\data\tokenizer.json"`。

## 2. 准备代码与数据

```bash
git clone https://github.com/chen-maker999/paper.git
cd paper
git checkout claude/agent-context-compaction
pip install -r requirements.txt openai

# 真实 BPE tokenizer（所有预算都以它计数）
python experiments/get_tokenizer.py data/tokenizer.json
export SKC_TOKENIZER=$PWD/data/tokenizer.json

# 下载 SWE-agent 公开轨迹（SWE-bench Lite，5 个 LLM，约 6GB）并解析
python experiments/download_trajs.py data/trajs
python experiments/corpus.py data/trajs data/cache
```

SKC 的效用模型已放在仓库的 `results/utility/` 下，无需重训。如需重训：
`python experiments/train_utility.py --cache data/cache --out results/utility`，CPU 约 20 分钟。

## 3. 启动本地模型服务（任选其一）

只要服务端兼容 OpenAI 的 `/v1/chat/completions` 接口即可。

**vLLM（推荐，吞吐最高）**
```bash
vllm serve Qwen/Qwen3-14B --max-model-len 32768 --port 8000
# 对应参数：--base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B
```

**Ollama**（注意：默认上下文只有 2k–4k，必须调大，否则输入会被静默截断）
```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
ollama pull qwen3:14b
# 对应参数：--base-url http://localhost:11434/v1 --model qwen3:14b
```

**llama.cpp**
```bash
llama-server -m qwen3-14b-q4_k_m.gguf -c 32768 --port 8080
# 对应参数：--base-url http://localhost:8080/v1 --model qwen3-14b
```

## 4. 运行评测

```bash
export SKC_TOKENIZER=$PWD/data/tokenizer.json
OUT=results/llm_qwen3-14b                       # 每个模型用不同目录
ARGS="--backend local --base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B --no-think --max-context 32768 --concurrency 16"

python experiments/llm_eval.py prepare   --cache data/cache --out $OUT --per-model 60
python experiments/llm_eval.py summarise --out $OUT $ARGS     # LLM 摘要基线
python experiments/llm_eval.py submit    --out $OUT $ARGS     # 问答 + 下一步预测
python experiments/llm_eval.py collect   --out $OUT           # 评分，输出汇总表
```

- 每一步的结果都会写入缓存，中断后重新执行同一条命令会从断点继续。
- `--no-think` 只对 Qwen3 这类带思考开关的聊天模板有效；其他模型去掉即可。模型若仍输出 `<think>…</think>`，脚本会自动去掉。
- `--concurrency` 按服务端承受能力调整：vLLM 可设 16–64，Ollama 和 llama.cpp 建议 2–4。

## 4.1 使用云端 API（以阿里云百炼为例）

脚本的 `local` 后端适用于任何兼容 OpenAI 接口的服务，包括云端 API。密钥只从环境变量读取，不要写进代码。

```bash
export SKC_TOKENIZER=$PWD/data/tokenizer.json        # 并已设置 DASHSCOPE_API_KEY
OUT=results/llm_qwen3.8-flash
ARGS="--backend local --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
      --api-key-env DASHSCOPE_API_KEY --model qwen3.8-flash --no-think --max-context 32768 --concurrency 8"
python experiments/llm_eval.py prepare   --cache data/cache --out $OUT --per-model 60
python experiments/llm_eval.py summarise --out $OUT $ARGS
python experiments/llm_eval.py submit    --out $OUT $ARGS
python experiments/llm_eval.py collect   --out $OUT
```

`--no-think` 会关闭 qwen 的思考模式（百炼用 `enable_thinking: false`），既省钱，也避免回答被思考 token 截断。
第二个模型用 `--model glm-5.3`，输出目录改为 `results/llm_glm-5.3`。glm-5.3 的思考模式无法关闭（传 `enable_thinking: false` 会报错），因此去掉 `--no-think`，改加 `--think-tokens 8192`：思考 token 也计入 `max_tokens`，不留余量时问答（上限 256）会在思考阶段就被截断。百炼对该模型不支持 `thinking_budget`。思考内容在单独的 `reasoning_content` 字段里，不会混进答案。

## 4.2 只给新版本方法补跑（不重跑已有方法）

已有方法的答案都缓存在 `answers.json` 里。下面的命令只会给新加入的方法（默认 `skc2`，即最终版 SKC）发送请求：

```bash
python experiments/llm_eval.py add     --cache data/cache --out $OUT        # 生成新方法的上下文
python experiments/llm_eval.py submit  --out $OUT $ARGS                     # 只请求新方法
python experiments/llm_eval.py collect --out $OUT
```

## 5. 把结果交回

把 `$OUT/` 目录下的这几个文件提交到仓库（或直接发给我）：

- `llm_eval_summary.csv`：各方法汇总；
- `llm_eval_rows.csv`：逐样本评分，用来算置信区间；
- `answers.json`、`summaries.json`：原始输出，便于复核；
- `samples.jsonl`：样本与上下文，体积较大，可选。

同时注明所用的模型名、量化方式和服务端（vLLM/Ollama/llama.cpp）。我会据此做统计检验，并写进论文。
