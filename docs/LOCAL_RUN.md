# 在本地电脑上运行大规模评测

评测全部是 CPU 计算，不需要显卡。速度大致与 CPU 核心数成正比：14 核的机器用 `--workers 12`，约为 4 核云端的 2.5–3 倍。

## 1. 环境与数据（一次性，约 30–60 分钟，需约 15GB 磁盘）

```bash
git clone https://github.com/chen-maker999/paper.git && cd paper
git checkout claude/agent-context-compaction
pip install -r requirements.txt scikit-learn tokenizers

python experiments/get_tokenizer.py data/tokenizer.json
export SKC_TOKENIZER=$PWD/data/tokenizer.json          # Windows PowerShell: $env:SKC_TOKENIZER="$PWD\data\tokenizer.json"

python experiments/download_trajs.py data/trajs              # SWE-agent / SWE-bench Lite，约 6GB
python experiments/download_trajs.py data/trajs_v --verified # OpenHands 等 / SWE-bench Verified，约 4GB
python experiments/corpus.py data/trajs   data/cache
python experiments/corpus.py data/trajs_v data/cache verified
```

效用模型已在仓库的 `results/utility/` 中，无需重新训练。

## 2. 运行评测（每一步都支持断点续跑，中断后重新执行同一条命令即可）

```bash
W=12   # 并行进程数，建议设为 CPU 核心数减 2
python experiments/run_eval.py --suite main     --exclude-dev --cache data/cache --out results --budgets 4000,8000,16000,32000 --workers $W
python experiments/run_eval.py --suite main     --corpora verified --per-model 150 --cache data/cache --out results --budgets 4000,8000,16000,32000 --workers $W
python experiments/run_eval.py --suite ablation --exclude-dev --per-model 100 --budgets 4000,8000 --cache data/cache --out results --tag v2 --workers $W
python experiments/stats.py data/cache results
python experiments/run_synthetic.py
```

Windows 上建议在 WSL2（Ubuntu）里运行。原生 Windows 理论上也可以，但多进程的启动方式不同，速度会慢一些。
