# 阶段 1：文献调研 —— 预算约束下的 AD 攻击路径阻断

> 调研日期：2026-09-30
> 核实方式：逐篇用网络检索确认标题、作者、会议/期刊与链接。本环境无法打开 arXiv / ACM / AAAI / IJCAI 的全文页面，
> 因此“图规模、数据来源、算法细节”中凡是只凭摘要或检索摘要得到、未读全文的内容，一律标注 **【待核实】**。
> 进入阶段 2 前的第一件事：精读 A3、A9、A1 三篇全文，把【待核实】项逐条确认。

---

## 一、2020 年至今：AD 攻击图防御（核心相关）

几乎全部来自同一个团队：阿德莱德大学 Mingyu Guo、Hung X. Nguyen、Aneta/Frank Neumann 等人。这一点本身就是重要结论：**这个方向是一个团队深耕的“小领域”，要做出区别必须精确找到他们没覆盖的设定。**

| # | 标题 / 作者 / 发表处 / 年份 | 问题设定（目标、约束、规模） | 算法 | 数据 | 链接 |
|---|---|---|---|---|---|
| A1 | **Practical Fixed-Parameter Algorithms for Defending Active Directory Style Attack Graphs**；Mingyu Guo, Jialiang Li, Aneta Neumann, Frank Neumann, Hung Nguyen【作者列表待核实】；AAAI 2022 | 最短路边阻断（shortest-path edge interdiction），Stackelberg 博弈；单一目标（DA），多入口（入口由 nature 随机决定）；防御方预算内删边；**目标：最大化攻击者的期望最短路长度**。证明：即使最大攻击路径长度为常数，问题对预算参数仍然是 W[1]-hard | BudgetFPT（小预算、少入口）；基于树分解的 DP（无环、树宽小）；GCN 启发式（大图） | 合成 AD 图（DBCreator 等）【待核实】 | [arXiv:2112.13175](https://arxiv.org/abs/2112.13175) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/21167) |
| A2 | **Defending Active Directory by Combining Neural Network based Dynamic Program and Evolutionary Diversity Optimisation**；Diksha Goel, Max Ward-Graham, Aneta Neumann, Frank Neumann, Hung Nguyen, Mingyu Guo；GECCO 2022 | Stackelberg；攻击者**在被检测到之前**到达目标的概率最大化；防御方删除固定数量的边，最小化攻击者成功率 | 神经网络近似 DP（攻击方）+ 进化多样性优化 EDO（防御方） | 合成 AD 图【待核实】 | [arXiv:2204.03397](https://arxiv.org/abs/2204.03397) · [ACM](https://dl.acm.org/doi/10.1145/3512290.3528729) |
| A3 | **Scalable Edge Blocking Algorithms for Defending Active Directory Style Attack Graphs**；Mingyu Guo, Max Ward, Aneta Neumann, Frank Neumann, Hung Nguyen；AAAI 2023, 37(5): 5649–5656 | Stackelberg；多入口、单目标（DA）；防御方**只能从“可阻断边集合”中**删边，受预算限制；目标：最小化攻击者成功率。纯策略版可归约为单源单汇最短路边阻断（NP-hard）。**预算是“边数”还是“异质成本之和”：【待核实，关键】** | 利用 AD 图“近似树形”结构设计可扩展算法（核化 / DP / MIP 组合，细节【待核实】） | DBCreator、adsimulator 合成图【待核实】 | [arXiv:2212.04326](https://arxiv.org/abs/2212.04326) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25701) |
| A4 | **A Scalable Double Oracle Algorithm for Hardening Large Active Directory Systems**；Yumeng Zhang, Max Ward, Mingyu Guo, Hung Nguyen；ASIA CCS 2023 | 大规模 AD 的删边加固博弈（具体目标函数【待核实】） | Double Oracle | 大规模合成 AD 图【待核实】 | [ACM](https://dl.acm.org/doi/10.1145/3579856.3590343) |
| A5 | **Evolving Reinforcement Learning Environment to Minimize Learner's Achievable Reward: An Application on Hardening Active Directory Systems**；Diksha Goel, Aneta Neumann, Frank Neumann, Hung Nguyen, Mingyu Guo；GECCO 2023 | 防御方通过删边“改造环境”，最小化 RL 攻击者能获得的最大回报 | RL 攻击者 + 进化式环境优化 | 合成 AD 图【待核实】 | [arXiv:2304.03998](https://arxiv.org/abs/2304.03998) · [ACM](https://dl.acm.org/doi/10.1145/3583131.3590436) |
| A6 | **Near Optimal Strategies for Honeypots Placement in Dynamic and Large Active Directory Networks**；Huy Q. Ngo, Mingyu Guo, Hung Nguyen；AAMAS 2023（extended abstract？【待核实】） | 防御手段为蜜罐而非删边；动态、大规模图 | 【待核实】 | 【待核实】 | [ResearchGate](https://www.researchgate.net/publication/399817088_Near_Optimal_Strategies_for_Honeypots_Placement_in_Dynamic_and_Large_Active_Directory_Networks) |
| A7 | **Catch Me if You Can: Effective Honeypot Placement in Dynamic AD Attack Graphs**；Huy Quang Ngo, Mingyu Guo, Hung X. Nguyen；IEEE INFOCOM 2024 | Stackelberg；蜜罐阻止攻击者到达高价值目标；区分“看不到蜜罐”和“看得到蜜罐”两类攻击者；**动态图**（图随时间变化） | MIP；dyMIP(m)（合并 m 个快照的 MIP）；投票与聚类启发式 | 大规模动态合成 AD 图（数十万节点量级）【待核实】 | [arXiv:2312.16820](https://arxiv.org/abs/2312.16820) · [IEEE](https://ieeexplore.ieee.org/iel8/10621050/10621073/10621210.pdf) |
| A8 | **Limited Query Graph Connectivity Test**；Mingyu Guo, Jialiang Li, Aneta Neumann, Frank Neumann, Hung Nguyen；AAAI 2024, 38(18): 20718–20725 | 边状态（On/Off）未知，可查询；在最多 B 次查询内判定 s–t 连通性（找到 On 路径或 Off 割），最小化期望查询次数；动机为 AD 攻击路径确认 | 理论算法 + 实用启发式 | AD 图【待核实】 | [arXiv:2302.13036](https://arxiv.org/abs/2302.13036) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30059) |
| A9 | **Practical Anytime Algorithms for Judicious Partitioning of Active Directory Attack Graphs**；Yumeng Zhang, Max Ward, Hung Nguyen；IJCAI 2024: 7074–7081 | **删除有限预算的边，最大化被切断、无法到达管理员账户的入口节点数**（与本课题最接近）。成本是否异质、目标是否多个【待核实】 | Spiral 随时算法（基于有向图的源连通性观察）；对比 MCTS、ILP、最短路阻断 | 大规模模拟 AD 网络【待核实】 | [IJCAI](https://www.ijcai.org/proceedings/2024/782) |
| A10 | **Optimizing Cyber Defense in Dynamic Active Directories Through Reinforcement Learning**；Diksha Goel, Kristen Moore, Mingyu Guo, Derui Wang, Minjune Kim, Seyit Camtepe；ESORICS 2024 (LNCS 14982) | 动态 AD 上的删边防御，Stackelberg | RL 攻击者 + RL 辅助 EDO 防御者；“RL Training Facilitator”剪枝环境 | 动态合成 AD 图 | [arXiv:2406.19596](https://arxiv.org/abs/2406.19596) · [Springer](https://link.springer.com/chapter/10.1007/978-3-031-70879-4_17) · [代码](https://github.com/DrDikshaGoel/ACDC_AD_Esorics-code) |
| A11 | **Optimizing Cyber Response Time on Temporal Active Directory Networks Using Decoys**；Huy Q. Ngo, Mingyu Guo, Hung Nguyen；arXiv 2024（发表处【待核实】） | 时序 AD 图上布置诱饵，最大化防御方响应时间 | 【待核实】 | 【待核实】 | [arXiv:2403.18162](https://arxiv.org/abs/2403.18162) |
| A12 | **ADSynth: Synthesizing Realistic Active Directory Attack Graphs**；Nhu Long Nguyen 等（完整作者【待核实】）；IEEE/IFIP DSN 2024 | **数据工作**：基于元图（metagraph）按微软设计规范和常见误配置合成真实感 AD 图，可调节安全等级 | 元图生成模型 | 开源 | [IEEE](https://ieeexplore.ieee.org/document/10647035/) · [PDF](https://dsn2024uq.github.io/Proceedings/pdfs/DSN2024-6rvE3SSpzFYmysif75Dkid/410500a066/410500a066.pdf) · [GitHub](https://github.com/AUCyberLab/ADSynth) |
| A13 | **Hardening Active Directory Graphs via Evolutionary Diversity Optimization-based Policies**；Diksha Goel 等【作者、年份待核实】；ACM TELO | A2/A5 的期刊扩展【待核实】 | EDO | 【待核实】 | [ACM](https://dl.acm.org/doi/10.1145/3688401) |
| A14 | **Adaptive Wizard for Removing Cross-Tier Misconfigurations in Active Directory**；Huy Quang Ngo, Mingyu Guo, Hung X. Nguyen；IJCAI 2025 | “自适应路径删除问题”：每一步向管理员提出一条攻击路径，管理员从中选一条可删的边；最小化步数，保证最终割断；基于微软 **Tier 模型** 定义跨层误配置 | 自适应策略 / 优化 | 13 个 ADSynth 合成图 + 1 个真实组织 AD 图 | [arXiv:2505.01028](https://arxiv.org/abs/2505.01028) · [ML Anthology](https://mlanthology.org/ijcai/2025/ngo2025ijcai-adaptive/) |
| A15 | **Co-Evolutionary Defence of Active Directory Attack Graphs via GNN-Approximated Dynamic Programming**；作者与发表处【待核实】；arXiv 2025 | 攻防双方策略协同进化 | GNNDP + EDO | 【待核实】 | [arXiv:2505.11710](https://arxiv.org/abs/2505.11710) |
| A16 | **Practical Graph Optimisation and AI-Driven Models for Active Directory Security Hardening**；Huy Q. Ngo（博士论文，阿德莱德大学）；arXiv 2026-07 | 综述并扩展 A6/A7/A11/A14；作者自述的三大缺口：静态图假设、防御手段只限于删边、缺少管理员反馈 | — | — | [arXiv:2607.22009](https://arxiv.org/abs/2607.22009) |

### 其他近期相关（非该团队）

| # | 标题 / 作者 / 发表处 / 年份 | 与本课题的关系 | 链接 |
|---|---|---|---|
| B1 | **Securing Sideways: Thwarting Lateral Movement by Implementing Active Directory Tiering**；Tyler Schroder, Sohee Kim Park；arXiv 2025 | 实践/立场类文章：Tier 分层隔离；指出“自动把设备和账户分到各层”仍是开放问题 | [arXiv:2508.11812](https://arxiv.org/abs/2508.11812) |
| B2 | **Security Games on Series-Parallel Attack Graphs with Adaptive Attackers**；Russell Kai Min Tan, Hui Han Chin, Chun Kai Ling；arXiv 2026-09 | 通用攻击图（非 AD）；串并联图上攻击者最优策略为 Gittins 型指数策略，可多项式时间计算——说明“特殊图结构 → 多项式可解”这条思路仍然活跃 | [arXiv:2608.21259](https://arxiv.org/abs/2608.21259) |

---

## 二、奠基工作

| # | 标题 / 作者 / 发表处 / 年份 | 贡献 | 链接 |
|---|---|---|---|
| F1 | **Automated Generation and Analysis of Attack Graphs**；O. Sheyner, J. Haines, S. Jha, R. Lippmann, J. M. Wing；IEEE S&P 2002 | 基于符号模型检测自动生成攻击图 | [PDF](https://conferences.computer.org/sp/pdfs/sp/2002/02_08_01.pdf) |
| F2 | **Two Formal Analyses of Attack Graphs**；S. Jha, O. Sheyner, J. Wing；IEEE CSFW 2002 | 最小关键攻击集合（NP-hard），加固问题的早期形式化【链接待核实】 | 【待核实】 |
| F3 | **MulVAL: A Logic-based Network Security Analyzer**；Xinming Ou, Sudhakar Govindavajhala, Andrew W. Appel；USENIX Security 2005 | 基于 Datalog 的多主机、多阶段漏洞推理 | [USENIX](https://www.usenix.org/conference/14th-usenix-security-symposium/mulval-logic-based-network-security-analyzer) · [GitHub](https://github.com/risksense/mulval) |
| F4 | **A Scalable Approach to Attack Graph Generation**；Xinming Ou, Wayne F. Boyer, Miles A. McQueen；ACM CCS 2006 | 逻辑攻击图，规模为多项式级【链接待核实】 | 【待核实】 |
| F5 | **Minimum-Cost Network Hardening Using Attack Graphs**；Lingyu Wang, Steven Noel, Sushil Jajodia；Computer Communications 29(18), 2006 | 攻击图上的最小代价加固（初始条件的布尔表达式最小化） | [DOI:10.1016/j.comcom.2006.06.018](https://doi.org/10.1016/j.comcom.2006.06.018) |
| F6 | **Heat-ray: Combating Identity Snowball Attacks Using Machine Learning, Combinatorial Optimization and Attack Graphs**；John Dunagan, Alice X. Zheng, Daniel R. Simon；SOSP 2009 | **AD 身份滚雪球攻击的最早系统性防御**：稀疏割 + 机器学习推荐删边，数十万用户/机器规模，可攻击机器数减少 96% | [PDF](https://www.sigops.org/s/conferences/sosp/2009/papers/dunagan-sosp09.pdf) · [ACM](https://dl.acm.org/doi/10.1145/1629575.1629605) |
| F7 | **Optimal Network Security Hardening Using Attack Graph Games**；Karel Durkota, Viliam Lisý, Branislav Bošanský, Christopher Kiekintveld；IJCAI 2015 | 攻击图博弈 + 蜜罐 | [PDF](https://www.ijcai.org/Proceedings/15/Papers/080.pdf) |
| F8 | **Interdicting Attack Graphs to Protect Organizations from Cyber Attacks: A Bi-Level Defender–Attacker Model**；Apurba K. Nandi, Hugh R. Medal, Satish Vadlamani；Computers & Operations Research 75, 2016 | 攻击图双层网络阻断，内外层均为 MILP | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0305054816301113) · [作者 PDF](https://medalgroup.org/wp-content/uploads/2020/03/Paper2-27.pdf) |
| F9 | **Deterministic Network Interdiction**；R. Kevin Wood；Mathematical and Computer Modelling, 1993 | 网络阻断奠基；最大流阻断 NP-hard【链接待核实】 | 【待核实】 |
| F10 | **Shortest-Path Network Interdiction**；Eitan Israeli, R. Kevin Wood；Networks 40(2), 2002 | 最短路阻断的 MIP + Benders 分解 | [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/net.10039) · [DTIC PDF](https://apps.dtic.mil/sti/pdfs/ADA490133.pdf) |
| F11 | **Length-Bounded Cuts and Flows**；G. Baier, T. Erlebach, A. Hall, E. Köhler, P. Kolman, O. Pangrác, H. Schilling, M. Skutella；ACM TALG 7(1), 2010 | 割断所有长度 ≤ L 的 s–t 路径；复杂度与近似 | [PDF](https://www.cs.le.ac.uk/people/te17/papers/talg2010.pdf) |
| F12 | **Paths of Bounded Length and Their Cuts: Parameterized Complexity and Algorithms**；P. A. Golovach, D. M. Thilikos；IWPEC 2009 | 长度受限割的参数化复杂度 | [Springer](https://link.springer.com/chapter/10.1007/978-3-642-11269-0_17) |
| F13 | **Six Degrees of Domain Admin**（BloodHound 首发）；Andy Robbins, Rohan Vazarkar, Will Schroeder；DEF CON 24, 2016 | BloodHound：把 AD 建模为图，求到 DA 的攻击路径 | [幻灯片](https://media.defcon.org/DEF%20CON%2024/DEF%20CON%2024%20presentations/DEF%20CON%2024%20-%20Robbins-Vazarkar-Schroeder-Six-Degrees-of-Domain-Admin.pdf) |

---

## 三、已被研究的问题变体（坦白说：覆盖得相当全）

| 维度 | 已有工作 |
|---|---|
| 多入口、单目标（DA） | A1, A3, A9 |
| 部分边不可删（只能删“可阻断边集合”） | A3 |
| 攻击者有检测风险 / 成功概率 | A2, A3 |
| Stackelberg 博弈 | A1–A3, A5, A7, A10 |
| 最大化攻击者最短路长度（≈ 长度受限割） | A1（理论基础 F10–F12） |
| 最大化被切断的入口数（部分阻断） | A9 |
| 动态 / 时序图 | A7, A10, A11 |
| 蜜罐 / 诱饵 | A6, A7, A11, F7 |
| 管理员参与的交互式删边 | A14 |
| RL / 进化 / GNN 方法 | A2, A5, A10, A15 |
| 边状态不确定、按需查询 | A8 |

**对你在提示里给的例子逐一表态：**

- “部分边不可删除”：已做（A3）。
- “多个入口”：已做（A1、A3、A9）。
- “攻击者有检测风险”：已做（A2、A3）。
- “Stackelberg 博弈”：已经做得很多，**不建议**本科生在这里硬拼。
- “边删除成本异质”“多个目标节点”：在我能看到的摘要里没有出现，但**必须读完 A3、A9 全文才能下定论**【待核实】。

---

## 四、尚未被充分研究的设定（依据本次检索，需在阶段 2 开始时复核）

1. **G1 异质删边成本（背包型预算）。** 现实中删不同类型的边，代价差别很大：清理一条 `HasSession`（让管理员注销）几乎免费；撤销一条 `GenericAll` ACL 需要审计；删除 `MemberOf` 可能影响业务。已有工作大多用“删 k 条边”的基数预算（A3 是否支持成本【待核实】）。
2. **G2 多个 Tier-0 目标且价值不同。** 真实的 Tier-0 不止 DA，还包括 DC、Enterprise Admins、ADCS CA、Exchange、备份服务器等，重要程度不同。如果所有目标同等重要，加一个超级汇点就能退化为单目标（平凡）；**一旦目标带权、入口也带权，问题就变成“预算约束下最大化被切断的（入口, 目标）对的加权和”**，属于预算多割（budgeted multicut）类问题，不再能退化。
3. **G3 安全性与业务可用性的双目标。** 删边会破坏合法访问。可以求“删边代价—残余风险”的 Pareto 前沿。
4. **G4 配置级修复动作（一次动作删除一组边）。** 例如移除一条授予本地管理员的 GPO 链接，会同时删除大量 `AdminTo` 边；启用 LAPS 会消除一批本地管理员密码复用边。决策对象从“边”变成“边的集合”，是带权集合覆盖/击中集型的阻断问题，已有 AD 工作都以单边为决策单位。
5. **G5 删边与蜜罐联合部署**（A16 自己列为未来工作）。
6. **G6 BloodHound 采集不完整下的鲁棒阻断**（`HasSession` 是采样得到的，会漏边）：A7、A10 涉及动态性，但“不确定边集合下的鲁棒最坏情况”没有见到专门研究【待核实】。

---

## 五、推荐设定

**G1 + G2 合并：异质成本、部分可删、多入口—多 Tier-0 目标的加权可达性阻断**
（暂定英文名：*Cost-Aware Multi-Target Attack Path Interdiction in Active Directory*）

确定性、非博弈版本：给定预算 $B$、边删除成本 $c_e$（不可删边 $c_e=\infty$）、入口权重 $w_s$、目标价值 $v_t$，选择删边集合 $F$，满足 $\sum_{e\in F} c_e \le B$，最小化残余风险

$$
R(F) \;=\; \sum_{s\in S}\sum_{t\in T} w_s\, v_t \cdot \mathbb{1}\big[\, t \text{ 在 } G\setminus F \text{ 中从 } s \text{ 可达} \,\big].
$$

**理由：**

1. **与已有工作的差异可以说清楚**：它严格推广了 A9（单目标、按入口计数），也和 A3（单 DA、最小化成功概率）明确不同；动机来自真实的 Tier 模型（A14、B1）。
2. **三类算法天然成立**：贪心（按“单位成本的风险下降量”删边）、最小割（加超级源/汇，即 $B$ 足够时的最优解）、整数规划（PuLP 在小图上求精确解），正好对应你的实验计划。
3. **复杂度结论干净，本科生能写出完整证明**：单入口单目标时就是最小 $s$–$t$ 割（多项式可解）；单入口、多目标、目标经不相交路径可达时，问题恰好就是 0-1 背包，所以（弱）NP-hard；多入口多目标时可以从多割问题归约（阶段 2 细写）。这样“易解情形 vs. 困难情形”的边界很清楚。
4. **改进算法有 AD 特有的切入点**：强连通分量（SCC）缩点、嵌套组的 `MemberOf` 层级 DAG、Tier 分层带来的“层间瓶颈边”剪枝，以及支配树（dominator tree）识别必经边。
5. **个人电脑可以完成**：确定性模型，不需要 RL/GNN 训练；数据用 ADSynth（A12，开源）或自写生成器；精确 IP 只在小图上跑。
6. **工作量适合 3–4 个月**：形式化约 2 周，实现约 4–6 周，写作约 4 周。

**风险（必须提前说）：**

- 创新属于“问题变体 + 结构化算法”层面，是增量工作，**适合 workshop / 中文核心 / arXiv，不要期望 AAAI/IJCAI 主会**。
- 如果精读后发现 A3 或 A9 已经支持异质成本，那么差异点只剩“多目标加权”。这时建议把 **G4（配置级修复动作）** 并入作为主要创新点，这也是我的备选方案。
