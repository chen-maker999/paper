# 预算约束下的攻击路径阻断（CMTI）实验框架

问题定义、复杂度分析和算法设计见：

- `docs/stage1_literature_review.md`：文献调研
- `docs/stage2_problem_formulation.md`：问题形式化、IP 模型、复杂度、算法 A–D

本仓库的代码只处理**中性的带权有向图**（节点、边、删除成本、是否可删、入口权重、目标价值），
与数据来源无关。把你的数据转换成下面的格式后即可运行全部实验。

## 环境

Python 3.10+，只需要 CPU。

```bash
pip install -r requirements.txt
python -m pytest -q          # 约 1 分钟，235 个与暴力枚举对照的正确性测试
```

PuLP 自带 CBC 求解器；LP 松弛使用 scipy 的 HiGHS；最大流使用 scipy 的 Dinic（要求整数成本）。

## 数据格式（graphio）

一个实例 = 一个目录，目录名约定为 `<组名>_s<种子>`（如 `vul_1k_s3`），同组不同种子用于计算均值与标准差。

`edges.csv`

```
src,dst,cost,deletable
n1,n7,3,1
n7,n9,,0
```

- `src -> dst` 表示“控制 src 后可以控制 dst”；
- `deletable` 为 1/0；`cost` 为正整数（`deletable=0` 时留空）；
- 同一对 `(src,dst)` 出现多行会合并成一条边：成本相加，任一行不可删则整条不可删。

`nodes.csv`

```
node,entry_weight,target_value
n1,1.0,0
n9,0,10
```

- 只需列出入口（`entry_weight>0`）和目标（`target_value>0`）；
- 一个节点不能既是入口又是目标。

`meta.json`（可选）：任意键值，比如数据来源、生成参数，会原样保存。

转换脚本需要你自己编写：从数据源生成上面两个 CSV，成本规则按阶段 2 文档表 1。
写完后，建议先用 `adinterdict.graphio.load_instance` 读回来检查节点数、边数和入口/目标数。

### AD 导出读取与合成数据

仓库提供离线 AD 图工具：`adinterdict.adio.load_ad_export` 读取已经导出的
JSON 图数据，`adinterdict.ad_generator.generate_ad_graph` 生成用于算法验证的
抽象用户、组、计算机和 Tier-0 目标。它们都只生成/读取中性的图实例，不连接域，
也不执行目录修改。

```bash
python -m experiments.ad_generator --users 1000 --groups 80 \
  --computers 300 --targets 5 --seeds 0-4 --out data/ad_smoke
```

读取单个规范化 JSON 或包含多个 JSON 文件的目录：

```python
from adinterdict.adio import load_ad_export

inst = load_ad_export("exports/graph.json")
```

规范化 JSON 使用 `nodes`、`edges`、`entries`、`targets` 四个数组；节点字段为
`id`、`type`，边字段为 `source`、`target`、`type`。读取器也接受常见的
BloodHound 风格 `data`、`Properties`、`ObjectIdentifier` 字段，并把边类型映射到
现有成本规则。入口和目标建议通过 `entry_weight`、`target_value` 显式标注，或在
调用 `load_ad_export` 时通过 `entry_weight=`、`target_value=` 传入。

## 目录结构

```
adinterdict/
  model.py         成本表、Instance 定义、build_graph
  indexed.py       整数下标实例；R(F)、可达对比例、平均最短路长度等指标
  graphio.py       数据格式读写
  flow.py          最小割（scipy Dinic），自动排除无法切断的源点
  dominators.py    支配树（Cooper–Harvey–Kennedy），一次算出全部单边收益（引理 2）
  reduction.py     精确约简 R1–R4（D1）
  lp.py            (IP-T) 的 LP 松弛：下界 + D 的 LP 引导补救
  algorithms/
    greedy.py      A  贪心删边
    mincut.py      B1 全局最小割 / B2 目标割 + 贪心补充
    ip.py          C  整数规划 (IP-T)，PuLP + CBC
    hubcut.py      D  约简 + 枢纽割贪心 + LP 补救 + 局部搜索
experiments/
  synthetic.py     通用合成图（仅用于冒烟测试和定理 2 的构造）
  run.py           批量实验，逐行写入原始结果 CSV
  aggregate.py     均值 ± 标准差汇总，计算与最优解/下界的差距
  plot.py          论文插图（PNG + PDF）
tests/             正确性测试
```

## 运行实验

```bash
# 1. 准备数据：每个实例一个目录，放在 data/ 下（data/ 不进 git）
#    冒烟测试可以先用通用合成图：
python -m experiments.synthetic layered --n 1000 --seeds 0-4 --out data/smoke
python -m experiments.synthetic clique  --n 40 --p 0.2 --seeds 0-4 --out data/smoke

# 2. 小规模：包含 IP 最优解 C 和 LP 下界
python -m experiments.run --data "data/<组名>_s*" --algs A,B2,C,D --lp-bound \
       --ip-time-limit 300 --out results/raw_small.csv

# 3. 大规模：不跑 C，用 LP 下界报告差距
python -m experiments.run --data "data/<大图组名>_s*" --algs A,B2,D --lp-bound \
       --out results/raw_large.csv

# 4. 消融实验
python -m experiments.run --data "data/<组名>_s*" \
       --algs D,D-noLP,D-nohub,D-noLS,D-ratio,D-noreduce --out results/raw_ablation.csv

# 5. 汇总与作图
python -m experiments.aggregate results/raw_*.csv --out results/summary
python -m experiments.plot results/summary.csv --out figures
```

- 预算取 λ（切断全部可切断入口—目标对的最小割成本，观察 1）的比例，默认为 `0.05,0.1,0.25,0.5,0.75`，可用 `--fracs` 修改；
- 实验中断后加 `--resume` 可以从断点继续；
- `results/summary.md` 是结果表（R/R0、与最优解的差距、与下界的差距、运行时间，均为均值 ± 标准差）。

### 指标

| 指标 | 含义 |
|---|---|
| `risk_ratio` | 剩余风险比例 $R(F)/R(\varnothing)$，越低越好 |
| `gap_opt` | $(P^\star-P)/P^\star$，$P=R_0-R$，$P^\star$ 取 C 的最优解（仅统计状态为 optimal 的实例） |
| `gap_lb` | 用 LP 下界代替 $P^\star$ 得到的差距，是真实差距的**上界** |
| `reach_pair_frac` | 仍然连通的（入口, 目标）对的比例（不加权） |
| `mean_dist` | 仍连通对的平均最短路跳数 |
| `time` | 求解时间（秒）；D 的时间包含约简 |

## 实现与阶段 2 文档的差异（写论文时要同步）

1. **引理 2 的实现**：不细分边，改用等价判据“H 中边 $v\to u$ 支配 $u$ ⇔ $\mathrm{idom}(u)=v$ 且 $u$ 的其他前驱都被 $u$ 支配”，证明见 `dominators.py` 开头。
2. **R3** 推广到任意非目标节点（不限于入口）；**R4** 从“链压缩”推广为“不可删度 1 收缩”。
3. **D2' LP 引导补救（新增）**：所有动作收益都为 0 时，按 LP 松弛的 $x_e$ 排序逐条加边，直到风险下降。
   这是针对定理 2 那类冗余路径实例加的：在那类实例上 A 和 B2 的保护量恒为 0。
4. **D3** 改为“删掉冗余边 + 用省下的预算重新执行 D2”，而不是 1-交换。
5. D 同时运行“按性价比”和“按收益”两种选择规则，取更好的解（背包贪心的常用修正）；消融项 `D-ratio` 可以单独评估它的作用。

## 已知局限

- 成本必须是正整数（Dinic 的要求）；实数成本请按比例放大后取整。
- D 在约 1 万节点的图上每个预算点需要约 2 分钟（纯 Python 的支配树 + 每轮若干次最大流）。
- `experiments/synthetic.py` 里的图只用于验证流程，不代表真实数据，论文结论应基于你转换的数据集。
