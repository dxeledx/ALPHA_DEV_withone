好，我们就按**路线 A（AlphaZero/MCTS 风格）**把“BCI-IV 2a 四分类（左/右手/脚/舌 MI）通道选择”从头到尾严谨地设计成一个**单人规划搜索问题**。我会一边给出可行方案，一边把每一处“会翻车/会被审稿人质疑”的难点说透，并给出对应的工程与实验细节。
---

## 0) 数据集与评估“硬约束”先钉死（否则后面全白做）

BCI-IV 2a 官方描述明确：

* 9 个被试；每被试 **两个 session（不同天）**；每 session **6 runs**，每 run **48 trials**（每类 12），所以每 session **288 trials** ([BBCI][1])
* **22 EEG + 3 EOG**，采样 **250 Hz**；EOG 通道“提供用于伪迹处理，**不得用于分类**” ([BBCI][1])
* 官方竞赛评价给出 accuracy/kappa 时间序列，且“赢家指标是最大 kappa（X.KAP00）” ([BBCI][1])

这几条会直接决定：

1. 我们主指标建议以 **Cohen’s kappa** 为核心（兼报 accuracy）。
2. 必须把 **EOG 处理**写进方法（否则审稿人会说你模型在吃眼动伪迹）。
3. 通道选择不能碰测试 session（AxxE），否则就是信息泄漏。

---

## 1) 把“通道选择”写成 AlphaDev 式单人游戏（MDP）

### 1.1 环境定义：EEGChannelGame

对每个被试（或每个训练折）定义一局游戏：

* **状态 (s_t)**：当前已选通道集合 (C_t\subseteq{1..22})（二值 mask），外加“该被试训练数据上每个通道的统计摘要”（下面细讲）。
* **动作 (a_t)**：从未选通道里选一个加入；再加一个 **STOP** 动作（提前结束）。
* **转移**：(C_{t+1}=C_t\cup{a_t})；若 STOP 或到达预算 K 则终止。
* **终止奖励**（只在终止给）：
  [
  R(C)=\text{Perf}(C)-\lambda\cdot \text{Cost}(C)-\gamma\cdot \text{ArtifactRisk}(C)
  ]

  * Perf：建议用 **kappa**（主）+ accuracy（辅） ([BBCI][1])
  * Cost：通道数（或端到端推理延迟）
  * ArtifactRisk：与 EOG 的相关性/可解释的伪迹风险惩罚（关键！）

这个结构和 AlphaDev 完全同构：AlphaDev 的 state 是“当前程序+寄存器/内存状态”，action 是“追加一条指令”，reward 同时包含 correctness 与 latency 。

---

## 2) 路线 A 的最大难点：奖励又贵又噪，MCTS 会被拖死/被噪声误导

AlphaDev 之所以能做，是因为它对“昂贵且噪声的延迟测量”做了两件事：

* 用**代理（长度）**当性能 proxy（长度与延迟相关时） 
* 真测延迟时用“跨机器重复测量 + 取第 5 分位 + 置信区间”抗噪 
  并且还用了**双 value head**分别预测 correctness 与 latency，latency head 直接用真实测量做 Monte Carlo 目标监督 。

EEG 里“Perf=分类 kappa”天生更噪，所以我们要更激进地“工程化”：

### 2.1 奖励评估的三层分辨率（强烈建议）

* **Level-0（超快、低噪 proxy）**：冻结特征提取器 + 训练线性头

  * 例：对选中通道做 bandpower/协方差特征（或冻结的 EEG encoder 输出）→ Logistic/LDA
* **Level-1（中等成本、稳定）**：FBCSP +（r）LDA / SVM，5-fold CV
* **Level-2（最贵、最终报告用）**：EEGNet/Conformer/Transformer 端到端训练（多 seed）

**MCTS 在搜索时主要调用 L0/L1**，只有对“最有希望的少量子集”才做 L2，最后论文报 L2（并把 L1 当消融/辅助证据）。
这对应 AlphaDev 的“长度 proxy vs 真延迟测量”的策略 。

### 2.2 用“鲁棒奖励”替代单次 kappa（防止被偶然高分骗）

建议奖励用 **下置信界/分位数**，例如：

* 多 seed 训练得到 ({\kappa_i})，取 **第 20 分位**或 (\mu-\beta\sigma) 当 Perf
  这在 EEG 场景比只取均值更抗“训练随机性”。

你甚至可以借鉴 AlphaDev 的思路：它假设噪声大多“一边倒”，所以取第 5 分位 。EEG 的噪声不一定单边，但“取低分位/下界”对审稿人非常有说服力。

---

## 3) 状态怎么表示？——这是路线 A 能不能“真像 Nature”最关键的细节

AlphaDev 的表示网络由两块组成：Transformer 编码“程序结构”，再加 CPU state encoder 编码“程序对输入的动态影响” 。我们也做两块：

### 3.1 Channel-token（结构块）：把 22 个通道当成图上的 22 个 token

每个通道 token 的输入特征（全部**只用训练 session 的训练折**计算，避免泄漏）：

* **空间位置**：10–20 montage 的坐标/相对位置（可用公开坐标表）
* **频带统计**：mu(8–13)、beta(13–30) 的 log-variance、类间 Fisher score
* **相关/冗余**：与其它通道的相关系数/相干性摘要（可做成边特征）
* **伪迹风险**：EEG 与 EOG 的相关性（高则惩罚）——呼应官方“EOG 不得用于分类且需要去伪迹” ([BBCI][1])

编码器建议两种你们可二选一（都容易写进论文）：

* **Graph Transformer / GAT**：天然表达“通道间关系”
* **Set Transformer / DeepSets + pairwise bias**：保证置换不变性

### 3.2 Selection-context（动态块）：把“已选集合”汇聚成全局上下文

做一个 masked pooling 得到 (g(C_t))，用于：

* 估计当前子集还能提升多少（value）
* 判断下一个应选谁（policy）

> 这一块相当于 AlphaDev 的“CPU state encoder”——它告诉网络“当前部分解对任务已经造成了什么影响”。只是我们用的是“统计摘要”而不是逐样本运行程序。

---

## 4) MCTS 细节：怎么让它既能探索组合空间，又不被噪声拖垮

### 4.1 PUCT + 先验策略

沿用 AlphaZero：policy 给先验，PUCT 平衡探索利用。AlphaDev 也明确了 MCTS 的“先跟着 policy、再逐步偏向 value” 。

### 4.2 动作剪枝（必须做，但别剪过头）

AlphaDev 会剪掉非法/冗余动作来缩小动作空间 。EEG 里我们可以做“安全剪枝”：

* 已选通道不可再选（mask）
* 若你设定预算 K，深度最多 K（搜索树自然截断）
* 可选：**对称成对动作**（如 C3/C4 成对）作为加速版消融，但主方法建议保留单通道动作（避免被说强先验）

### 4.3 转置表（transposition）与缓存（决定算力是否可承受）

因为状态只由“集合”决定，顺序无关：

* 用 bitmask 作为 key
* 缓存：该 subset 的 (mean kappa, std, 训练次数, 最后一次评估时间)
* MCTS 节点共享：避免重复评估同一子集

这一步能把训练成本从“指数爆炸”拉回可做的范围。

---

## 5) 训练范式：AlphaZero-style 自博弈在这里怎么落地？

AlphaDev 的核心是“用搜索改进策略，再把搜索蒸馏进网络” 。我们照搬：

1. **采样任务**：随机挑一个被试 + 训练 session 的一个训练/验证划分
2. 从空集开始，重复：

   * 用当前网络做 MCTS，得到根节点访问计数分布 (\pi_{\text{MCTS}})
   * 采样/选择一个动作推进
3. 终局后用评估器算鲁棒奖励 (R)
4. 存 ((s_t,\pi_{\text{MCTS}},R)) 到回放池
5. 训练网络最小化：

   * policy loss：KL( (\pi_{\text{MCTS}}) || (\pi_\theta) )
   * value loss：((V_\theta(s)-R)^2)

并加一个关键改造（很像 AlphaDev 双头）：

* **双 value head**：一个预测 (\mathbb{E}[\kappa])，一个预测 (\text{Unc}(\kappa)) 或预测分位数（如 (\kappa_{0.2})）。
  AlphaDev 的双头（correctness+latency）在优化真实延迟时显著更好 ；我们这里对应的是“均值 vs 鲁棒下界”。

---

## 6) 伪迹与非平稳：EEG 特性必须正面处理，否则 RL 会学到“坏捷径”

BCI-IV 2a 明确提供 EOG 是为了做伪迹处理 ([BBCI][1])。如果你不处理，RL 很可能喜欢“前额通道”，因为眼动/注意相关成分能提升分类。

我们要让方法在论文里“站得住”，建议三层防线：

1. **预处理层**：对 EEG 做回归去 EOG（线性回归/ICA/SSP），并报告消融（有/无去伪迹）。
2. **奖励层**：加入 ArtifactRisk 惩罚（比如选中通道与 EOG 的相关性均值）。
3. **解释层**：画拓扑图 + 统计：最终子集集中在运动皮层附近、且与 EOG 相关性显著低于 baselines。

这三步组合起来，审稿人就很难说“你只是用眼动分类”。

---

## 7) 实验设计：怎么写才能“论文味道很足”

### 7.1 两种评价设置（建议都做）

* **Subject-dependent**：每被试在 AxxT 上选通道与训练，AxxE 上测试（不泄漏）
* **Leave-one-subject-out**（更有价值）：用 8 个被试训练 policy/value；对第 9 个被试只用其 AxxT 做 MCTS 选通道，然后 AxxE 测试。

### 7.2 结果呈现不要只报一个点：报 Pareto frontier

你们的卖点是“搜索找到更优 trade-off”。建议输出：

* K=4/6/8/10/12/16/22 的性能-通道数曲线
* 或固定 (\lambda) 扫一组得到 Pareto 前沿

### 7.3 基线要“卡死同预算”，否则对比不公平

同样的 K 下比较：

* Filter：MI/Fisher/ReliefF
* Wrapper：SFS/SBS、GA/PSO
* Embedded：L1/L0 稀疏门控、Attention-based（ECA/CBAM 类）
* 以及你们自己的 **MCTS 但不用学习先验**（纯 UCT）作为关键消融：证明“学习 + 搜索”确实有效

---

## 8) 你们最可能拿到的“强论文贡献点”（现实且能做出差异）

在 22 通道这种规模下，“准确率绝对 SOTA”不一定最容易，但下面三点非常容易做成高质量贡献：

1. **更少通道下保持/接近全通道性能**（尤其用 kappa 报告）
2. **跨 session/跨被试更稳**：用“下分位/下置信界”展示稳定性（这是 RL + 鲁棒目标的天然优势）
3. **伪迹免疫**：你们的方法在加入 ArtifactRisk 后仍能维持性能、且选通道更生理合理（这点很加分）

---

下面是一份**路线 A（AlphaDev/AlphaZero 风格：Transformer policy/value + MCTS + 多保真评估器）**在 **BCI Competition IV-2a（四分类、22 通道、250Hz）**上做 **EEGChannelGame（通道选择）**的「直接开工版」方案：把 **state/token 特征表、MCTS 超参、L0/L1/L2 评估器实现细节**一次写清，并把每个选择的**风险点**一起列出来。

---

## 0) 数据与基本约束（用来锁定时间窗/频带/评估指标）

* 2a 单次 trial：t=0 出现注视十字；t=2s 出现方向提示（四类 MI），要求持续想象到 t=6s 注视十字消失。([BBCI][1])
* EEG：**22 通道**，**250Hz**，采集时已 0.5–100Hz 带通并启用 50Hz notch；另有 3 个 EOG 仅用于去伪迹，**不得用于分类**。([BBCI][1])
* 竞赛官方评估会计算 accuracy & kappa；我们做论文更稳妥建议**主指标用 kappa（对类不平衡更稳）+ accuracy 辅助**。([BBCI][1])

> **落地默认时间窗**（后面 L1/L2 都会用）：
>
> * 主窗：**[3.0s, 6.0s]**（对应“通常从 cue 后 1s 开始取段”的经验设置 + 本数据 cue 在 2s，trial 到 6s）。([Frontiers][2])
> * 备选窗（用于消融/稳健性）：[2.5, 5.5]、[2.0, 6.0]（可能更易受视觉诱发/非平稳影响，见风险项）。

---

## 1) EEGChannelGame：状态/动作/回合终止（和 AlphaDev 对齐）

参考 AlphaDev 把“搜索对象”定义成 **(当前已构造对象 + 执行后环境状态)** 的思路（它的状态是 ⟨P_t, Z_t⟩：P_t 是当前程序，Z_t 是寄存器/内存状态）。
我们把 EEG 通道选择映射为：

* **P_t**：已选择通道集合与顺序（subset + order）
* **Z_t**：从训练集得到的“可用于决策的统计摘要”（每个通道的频带判别性、质量、与已选集合冗余等）

### 动作空间

* `a = 选择一个未选通道 i`（最多 22）
* `a = STOP`（提前终止，输出当前子集）

### 终止条件

* 选满 `B_max` 个通道（默认 10）或选择 `STOP`

> **奖励**：`R = κ_val - λ * (|S|/22)`

* κ_val：在验证集上用 L1/L2 得到的 kappa（或 accuracy）
* λ：稀疏惩罚系数（默认 0.05；后面列风险）

---

## 2) State / Token 设计：一张“每个 token 特征表”让你直接编码

### 2.1 Token 序列结构（Transformer 输入）

* `[CLS]`：全局 token（聚合用）
* `[CH_1]...[CH_22]`：每个 EEG 通道一个 token（固定顺序：与数据通道顺序一致）
* `[CTX]`：上下文 token（预算/步数/当前已选数量等）

> 这样做的直觉：把“通道选择”变成“在一串通道 token 上做注意力 + 输出对各通道的 prior”，再配合 MCTS 做规划。

---

### 2.2 **Token 特征表（建议 d_token = 64~128，全部按 subject 内 z-score 标准化）**

#### A) `[CH_i]`（每个通道一个 token）

| 特征组            | 具体特征（逐项可落地）                                                                  | 维度建议 | 计算方式（只用训练集）                             | 风险点                                   |                         |     |             |                       |
| -------------- | ---------------------------------------------------------------------------- | ---: | --------------------------------------- | ------------------------------------- | ----------------------- | --- | ----------- | --------------------- |
| 身份/空间          | `onehot(channel_id)` 或 `embedding(channel_id)`；可选：`(x,y,z)` 10-20 坐标         | 8~32 | id embedding 训练得到；坐标可查表                 | 坐标表不一致/通道命名混乱导致错位                     |                         |     |             |                       |
| 选择状态           | `is_selected`；`selected_order/ B_max`；`remaining_budget`                     |    3 | 环境内维护                                   | 若 order 特征过强，可能学到“固定顺序偏好”             |                         |     |             |                       |
| 质量/伪迹          | `log(var)`；`kurtosis`；`line_noise_ratio(48-52 / 8-40)`；`corr_with_EOG`(可选)   |  4~6 | 取主窗 [3,6]，对每 trial 统计再取均值               | EOG 回归/相关计算若用了 evaluation session 会泄漏 |                         |     |             |                       |
| 频带能量           | **9 个频带 log-bandpower**：4–8, 8–12, …, 36–40 Hz（与经典 FBCSP 一致）([Frontiers][2]) |    9 | bandpass → log-variance 或 log-power（主窗） | 频带太细导致样本少时方差大；滤波器相位/边界效应              |                         |     |             |                       |
| 类判别性（单通道）      | 对每个频带：`F-score(ANOVA)` 或 `mutual_info`（四分类）                                  |    9 | 用训练集标签算                                 | MI/ANOVA 在小样本上噪声大，易误导搜索               |                         |     |             |                       |
| 冗余（相对已选集合）     | `mean(                                                                       | corr | )`：与已选通道在各频带能量上的相关；`max(                | corr                                  | )`；`redundancy_penalty` | 3~6 | 预先算能量特征后做相关 | 冗余指标可能把“真正的协同信息”误判为冗余 |
| 快速“边际增益”代理（可选） | `Δproxy_i`：把 i 加入当前集合时 L0 proxy 的增益（见 L0）                                    |    1 | 运行时快速更新                                 | proxy 偏差会锁死搜索到局部最优                    |                         |     |             |                       |

> **最小可用版本**：身份(embedding)+选择状态+9 频带能量+9 判别性+冗余(3) → 大约 1* + 3 + 9 + 9 + 3 = 24~30 维，足够先跑通。

#### B) `[CLS]` 全局 token

| 特征                |   维度 | 说明                      | 风险点              |           |   |
| ----------------- | ---: | ----------------------- | ---------------- | --------- | - |
| `t/B_max`, `      |    S | /B_max`                 | 2                | 当前步数/已选个数 | 无 |
| `last_reward_est` |    1 | 上一步评估器给的 κ/acc 估计（若用）   | 引入评估噪声反馈回路       |           |   |
| `S_summary`       | 8~16 | 已选通道的能量/判别性均值与方差（按频带汇总） | 汇总统计过强→模型忽略单通道差异 |           |   |

#### C) `[CTX]` 上下文 token

| 特征                         |   维度 | 说明                        | 风险点                |
| -------------------------- | ---: | ------------------------- | ------------------ |
| `subject_id_embedding`(可选) | 8~16 | 做跨被试 meta-RL 时需要；先做单被试可不加 | 跨被试时容易学到“记忆被试”而非规律 |
| `window_id`(可选)            |    4 | 用多时间窗训练时标识当前窗             | 窗选择与通道选择耦合，搜索更难    |

---

## 3) MCTS 超参（模拟次数/PUCT 常数/rollout 截断）——给一套默认可跑 + 可写进论文的消融轴

AlphaDev/AlphaZero 典型：网络给 policy/value，MCTS 用 **PUCT**在树上平衡探索/利用，并用根节点访问次数分布做 improved policy。

### 3.1 关键公式（实现用）

对边 (s,a)：

* `U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
* `score = Q(s,a) + U(s,a)`

### 3.2 默认超参（建议你先用这一套把系统跑通）

* **每步模拟次数 `n_sim`**：**256**

  * 消融：128 / 256 / 512
  * 风险：512 会让训练成本爆炸；128 可能搜索不稳定（尤其奖励噪声大时）。
* **PUCT 常数 `c_puct`**：**1.5**

  * 消融：0.8 / 1.5 / 2.5
  * 风险：太小→过度依赖当前 Q（易陷局部）；太大→被 noisy prior 带偏。
* **根节点 Dirichlet 噪声**：`ε = 0.25`，`α = 0.3`（动作约 23 个时经验值）

  * 风险：α/ε 过大→训练早期几乎随机；过小→探索不足，通道子集很快塌缩。
* **rollout 截断（强烈建议：无随机 rollout，用 value + 多保真 evaluator）**

  * `depth_limit = B_max`（默认 10）
  * **截断评估策略**：到叶子就用 **V(s)**；若 V 不可靠则用 **L0/L1**补一刀（见下一节）
  * 风险：如果 value 早期很烂，完全不 rollout 会“搜索盲”；但随机 rollout 在此任务上会极慢且噪声巨大——所以用“多保真叶子评估”替代。
* **动作选择温度**：训练前 30% step 用 τ=1（采样），后 70% τ→0（argmax）

  * 风险：τ 太快降为 0 → 策略过早确定，后续难改。

---

## 4) L0 / L1 / L2 评估器：具体实现（FBCSP 频带、时间窗、分类器）+ 何时调用

你要把“奖励”做得既**可学**又**不被噪声支配**，关键就是多保真：

### L0：超快 proxy（每次扩展/模拟都能用）

**目的**：给 MCTS/value 一个“便宜但相关”的形状信号，避免早期完全靠噪声。

* **输入**：已选集合 S

* **输出**：`proxy_score(S)`（标量）

* **实现（推荐）**：

  1. 在主窗 [3,6] 上，对每通道计算 9 频带 log-bandpower（与 FBCSP 同一组频带）([Frontiers][2])
  2. 对每通道每频带算四分类判别性：`ANOVA F-score` 或 `mutual_info`
  3. `relevance(S)= sum_{i in S} mean_band(F_i)`
  4. `redundancy(S)= mean_{i≠j in S} |corr(bp_i, bp_j)|`
  5. `proxy = relevance - β*redundancy - λ*(|S|/22)`（β 默认 0.2）

* **风险点**

  * proxy 与真实 κ 的相关性不够 → MCTS 被“假信号”牵着走（必须做相关性检查：每隔 N 次用 L2 真评估校准）。
  * redundancy 用相关系数可能误杀协同信息（可消融：|corr| vs distance-corr vs HSIC）。

---

### L1：中等保真（用于：叶子节点价值修正 / 训练早期 reward）

**目的**：比 L0 更接近最终性能，但仍足够快，能在 RL 训练中频繁调用。

* **管线**：**FBCSP(OVR) + rLDA**

  * 多分类扩展：OVR / PW / DC 在 FBCSP 文献中都有；这里选 **OVR**：4 个二分类器，工程最简单。([Frontiers][2])

* **频带**：9 bands：4–8, 8–12, …, 36–40 Hz（直接按经典 FBCSP 配置）([Frontiers][2])

* **时间窗**：主窗 [3.0, 6.0]（“cue 后 1s 起”经验 + 数据 cue=2s）([Frontiers][2])

* **CSP 参数**：

  * 每个二分类器、每个频带：取 `m=2` 对 CSP（共 4 个空间滤波器）

* **特征**：log-variance（每频带×每 OVR 的 CSP 输出）

* **特征选择**：MIBIF（互信息 best individual feature）或简单 top-K（K=24~48）

* **分类器**：rLDA（强烈建议带 shrinkage）

* **验证方式**：训练 session 内 **3-fold CV**（比 10×10 省很多），输出平均 κ/acc 当作 reward

* **风险点**

  * CV 太浅（3-fold）→ reward 方差大；太深（10×10）→ 训练慢到不可用（FBCSP 论文里会做 10×10，但我们训练阶段不适合这么重）。([Frontiers][2])
  * OVR 会遇到类别不平衡（1 vs rest），rLDA 需 class_weight 或 balanced prior。

---

### L2：高保真（用于：论文主结果/最终评估/少量校准）

**目的**：对齐论文最终报告，给出可信提升。

* **管线（两条都要实现，写论文更稳）**

  1. **FBCSP(OVR) + rLDA**（和 L1 同结构，但评估更严格）
  2. **强基线深度模型**（例如 ShallowConvNet/EEGNet）在选通道后训练（用于证明“通道选择对深度模型也有效”）

* **评估协议（强烈建议对齐 2a 的 session-to-session）**：

  * 用带标签的 training session 训练/调参；最终在 evaluation session 测试（不参与搜索）。([Frontiers][2])

* **指标**：kappa 主报 + accuracy；同时报告 **通道数-性能曲线**（越少通道越重要）

* **风险点**

  * 如果你在 RL 搜索时“反复看 evaluation session”会发生严重数据泄漏（论文直接毙掉）。
  * 深度模型训练波动大，会掩盖通道选择增益（需固定种子 + 多次重复）。

---

## 5) 把关键“选择-风险”再集中列一遍（方便你写实验设计与消融）

1. **时间窗选 [3,6]**：符合“cue 后 1s 起”常用经验且适配本数据 cue=2s、MI 到 6s。([Frontiers][2])

   * 风险：对部分被试最佳窗可能更早/更晚 → 必须加窗口消融（2.5–5.5、2–6）。

2. **频带用 9×4Hz（4–40）**：经典 FBCSP 配置，覆盖 MI 常用范围且稳定。([Frontiers][2])

   * 风险：4Hz 过细会放大噪声 → 可消融为 6 bands（例如 6–12…）或用 learnable filterbank。

3. **MCTS n_sim=256, c_puct=1.5**：先保证可训练与可复现；并以 {128,256,512}、{0.8,1.5,2.5} 做消融轴。

   * 风险：奖励噪声大时 c_puct/噪声注入会决定成败（需要在训练早期更依赖 L0/L1 稳定叶子评估）。

4. **多保真 reward（L0/L1/L2）**：核心在于“训练可承受 + 最终可信”。FBCSP 文献也强调跨验证与 session-to-session。([Frontiers][2])

   * 风险：L0 proxy 偏差→学偏；解决：定期用 L2 校准相关性（画 proxy vs κ 的散点与皮尔逊相关）。

5. **EOG 只用于去伪迹**：数据说明明确 EOG 不可用于分类。([BBCI][1])

   * 风险：你若把 EOG 信息“间接编码进特征”（比如 token 里放 EOG 本身）也算违规；只能用于回归去伪迹。

---


下面给你一份“照着写就能跑”的**工程目录结构 + 伪代码接口**（PyTorch 风格），覆盖：

* `Environment / Evaluator(L0/L1/L2) / PolicyValueNet / MCTS / ReplayBuffer / TrainLoop`
* 每个模块的**输入输出张量形状**
* **缓存策略**（epoch、bandpower、每频带协方差矩阵、冗余相关矩阵、标准化统计量等）
* 关键实现细节（尤其“避免数据泄漏”的 fold 组织方式）

我按“先能跑通（L0+MCTS+网络）→ 再加 L1 → 最后上 L2 严格评估”的路线写，能保证你们先快速验证闭环，再逐步加重。

---

## 1) 工程目录结构（推荐）

```text
eeg_channel_game/
  configs/
    default.yaml
    mcts.yaml
    eval.yaml
    net.yaml

  data/
    raw/                      # 原始gdf/mat等
    processed/
      subj01/
        train_epochs.npz      # X_train: [N,22,T], y_train: [N]
        eval_epochs.npz
        meta.json             # sampling_rate, channel_names...
      ...
    cache/
      subj01/
        sessionT_cov_fb.npz   # cov: [B,N,22,22] (float32)
        sessionE_cov_fb.npz
        sessionT_bp.npz       # bandpower: [N,22,B] or [N,B,22]
        sessionE_bp.npz
        sessionT_quality.npz  # quality feats: [N,22,Q]
        # 注意：fold级统计单独缓存（见下）

  eeg/
    preprocess.py             # EOG回归/ICA等（只用训练session做拟合）
    epoching.py               # 切trial -> epochs
    filterbank.py             # 频带定义 & 滤波/PSD
    features.py               # bandpower/quality等特征抽取
    covariance.py             # 每频带协方差矩阵预计算
    splits.py                 # subject-dependent / LOSO splits

  game/
    env.py                    # EEGChannelGame Environment
    state_builder.py          # 把mask+fold stats -> token tensor
    reward.py                 # reward组合: perf - lambda*cost - artifactrisk等

  eval/
    evaluator_base.py
    evaluator_l0.py
    evaluator_l1_fbcsp.py
    evaluator_l2_deep.py
    metrics.py                # kappa, acc, CI/quantile等

  model/
    policy_value_net.py       # Transformer/GNN等
    heads.py                  # policy head + value head(s)
    positional.py             # channel id embedding/pos embedding
    normalization.py          # feature norm

  mcts/
    node.py
    mcts.py
    transposition.py
    batched_infer.py          # 叶子批量推理（强烈建议）

  rl/
    replay_buffer.py
    selfplay.py               # 用MCTS采样对局
    train_loop.py
    checkpoint.py

  utils/
    logging.py
    seed.py
    timers.py

  run_train.py
  run_eval.py
```

---

## 2) 数据与缓存：你们算力成败的关键

### 2.1 必存的“基础 epoch”

对每个 subject 的两个 session 各存一份：

* `X`: `float32 [N_trials, 22, T]`

  * 推荐 `T = 750`（3 秒窗 * 250Hz），窗为 **[3,6]**（cue=2s 后 1s 起）
* `y`: `int64 [N_trials]`，取值 `0..3`

> 这样后面无论 FBCSP 还是深度网络都能用同一份 epoch。

### 2.2 强烈建议预计算的 cache（一次算，后面飞快）

设 `B=9` 个频带（4–8, 8–12, …, 36–40 Hz）。

1. **Bandpower（L0/L1 状态与 proxy 都用）**

* `bp`: `float32 [N, 22, B]`
* 计算：每 trial、每通道、每频带的 log-bandpower（或 log-variance）

2. **Filterbank 协方差矩阵（L1 FBCSP 快速训练用）**

* `cov_fb`: `float32 [B, N, 22, 22]`
* 计算：对每频带滤波后，对每 trial 计算通道协方差（加微小对角线防奇异）

> **为什么要存 cov_fb？**
> FBCSP 的核心是类均值协方差 + 广义特征分解。你存了 `cov_fb` 后，每次评估一个通道子集 `S`，只需对协方差取子矩阵 `cov[:, :, S, S]` 并按类做平均，速度会快一个数量级。

3. **质量/伪迹相关特征（state 中的 quality）**

* `q`: `float32 [N, 22, Q]`，例如 `Q=4~6`（logvar、kurtosis、线噪比等）
* `corr_eog`：如果你做 EOG 回归，可存回归残差能量比例或与 EOG 的相关作为 artifact risk 代理

  * 注意：EOG 只能用于去伪迹与风险估计，不能作为分类输入

### 2.3 fold 级别统计（必须防泄漏）

通道选择的 state 里会用到“类判别性、冗余相关矩阵、特征标准化参数”等，这些**必须只用训练 fold**算。

建议缓存结构：

* 对每个 `(subject, split_id)` 生成 `FoldStats`：

  * `mu_feat, std_feat`: 用于 token 特征标准化
  * `fisher_or_mi`: `float32 [22, B]`（单通道判别性）
  * `redund_corr`: `float32 [22, 22]`（例如基于 bandpower 的平均 |corr|）
  * `class_cov_mean`: 可选 `float32 [B, 4, 22, 22]`（每类均值协方差，FBCSP更快）

> **Split_id** = subject-dependent 的 K 折编号，或者 LOSO 的“留出被试编号+内部折编号”。

---

## 3) Environment：EEGChannelGame（接口+张量形状）

### 3.1 环境对象

```python
class EEGChannelGame:
    def __init__(self, fold_data: FoldData, state_builder: StateBuilder,
                 evaluator: EvaluatorBase, B_max: int=10):
        self.fold_data = fold_data          # 只包含训练fold和验证fold引用
        self.state_builder = state_builder
        self.evaluator = evaluator
        self.B_max = B_max
        self.reset()

    def reset(self) -> dict:
        self.mask = np.zeros(22, dtype=np.int8)   # 0/1
        self.order = []                           # 已选通道顺序 list[int]
        self.t = 0
        return self._get_obs()

    def step(self, action: int) -> tuple[dict, float, bool, dict]:
        # action: 0..21 -> 选通道; 22 -> STOP
        info = {}
        if action == 22 or self.t >= self.B_max:
            done = True
        else:
            if self.mask[action] == 0:
                self.mask[action] = 1
                self.order.append(action)
                self.t += 1
            # 若选了重复通道，可选择：给负奖励或直接当无效动作（建议mask掉避免发生）
            done = (self.t >= self.B_max)

        if done:
            reward, perf_detail = self.evaluator.evaluate(self.mask, self.fold_data)
            info.update(perf_detail)
        else:
            reward = 0.0

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> dict:
        return self.state_builder.build(self.mask, self.order, self.t, self.fold_data)
```

### 3.2 观测 `obs` 的标准形状（给网络用）

建议 state_builder 输出：

* `tokens`: `float32 [T_tokens, D_in]`

  * `T_tokens = 24`（CLS + 22 通道 + CTX）
  * `D_in = 64`（先固定 64，后面可扩 128）
* `action_mask`: `bool [A]`，`A=23`（22+STOP）

  * 已选通道动作置 False；STOP 恒 True（也可规定至少选>=2才允许 STOP）
* `key`: `uint32/int64` bitmask（用于 transposition table）

---

## 4) StateBuilder：把 mask + FoldStats → token tensor（关键细节写死）

```python
class StateBuilder:
    def __init__(self, feat_spec: FeatureSpec, normalizer: FeatureNormalizer):
        self.spec = feat_spec
        self.norm = normalizer

    def build(self, mask: np.ndarray, order: list[int], t: int,
              fold_data: FoldData) -> dict:
        # 1) 从fold_data.stats中取预计算的 fold 级统计
        stats = fold_data.stats

        # 2) 生成每通道token特征: [22, D_ch_raw]
        ch_feats = self._channel_features(mask, order, t, stats)

        # 3) 生成CLS/CTX: [2, D_cls_raw]
        cls = self._cls_features(mask, t, stats)
        ctx = self._ctx_features(mask, t, stats)

        # 4) concat -> tokens: [24, D_raw] 然后 norm + linear proj 到 D_in
        tokens_raw = np.concatenate([cls[None], ch_feats, ctx[None]], axis=0)
        tokens = self.norm(tokens_raw, stats.mu_feat, stats.std_feat)  # 标准化
        tokens = self._project(tokens)  # 可选：线性层把 D_raw -> D_in

        action_mask = np.ones(23, dtype=bool)
        action_mask[:22] = (mask == 0)
        # 可选：限制最少选2个通道才允许STOP
        if mask.sum() < 2:
            action_mask[22] = False

        key = mask_to_bitmask(mask)  # int
        return {"tokens": tokens.astype(np.float32),
                "action_mask": action_mask,
                "key": key}
```

### 建议的通道 token 原始特征（对应你们上一轮的表，给一个“最小闭环版”）

* `is_selected`（1）
* `selected_order/B_max`（1，未选=0）
* `bp_mean[9]`：bandpower 在训练 fold 的均值（或当前 trial 的均值；推荐用 fold 均值更稳）
* `disc[9]`：fisher/MI（fold 内算）
* `redund_mean`、`redund_max`：与已选集合的冗余（基于 `stats.redund_corr`）
* `quality[Q]`：logvar、kurtosis、line_noise_ratio、artifact_risk 等（fold 内平均）

这样 `D_raw ≈ 1+1+9+9+2+Q`，取 `Q=4`，大约 26 维，后接线性投影到 `D_in=64`。

---

## 5) PolicyValueNet：输入/输出形状与接口

### 5.1 网络接口（PyTorch）

```python
class PolicyValueNet(nn.Module):
    def __init__(self, d_in=64, d_model=128, n_layers=4, n_heads=4, n_actions=23):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=4*d_model, batch_first=True),
            num_layers=n_layers
        )
        self.policy_head = nn.Linear(d_model, n_actions)
        self.value_head_mu = nn.Linear(d_model, 1)      # 预测均值
        self.value_head_q20 = nn.Linear(d_model, 1)     # 可选：预测0.2分位/下界

    def forward(self, tokens, action_mask=None):
        # tokens: float32 [B, T=24, D_in]
        x = self.in_proj(tokens)                         # [B,24,d_model]
        h = self.tr(x)                                   # [B,24,d_model]
        cls = h[:, 0, :]                                 # [B,d_model] 取CLS

        logits = self.policy_head(cls)                   # [B,23]
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        v_mu = self.value_head_mu(cls).squeeze(-1)       # [B]
        v_q20 = self.value_head_q20(cls).squeeze(-1)     # [B]
        return logits, v_mu, v_q20
```

### 5.2 训练目标（AlphaZero 风格）

对 replay 中每条样本 `(tokens, pi_mcts, z)`：

* `policy_loss = KL(softmax(logits), pi_mcts)`
* `value_loss = (v_q20 - z)^2` 或 `(v_mu - z)^2`（二选一或加权）
* `entropy_bonus`：可选（别太大）

> `z`：建议用“鲁棒 reward”（例如 L1 的多折均值下界），避免奖励噪声把 value 学坏。

---

## 6) MCTS：节点结构、transposition、批量推理（你们的核心壁垒）

### 6.1 Node（存统计量）

```python
@dataclass
class Node:
    key: int
    P: np.ndarray           # [23] prior prob
    N: np.ndarray           # [23] visit counts
    W: np.ndarray           # [23] total value
    Q: np.ndarray           # [23] mean value
    children: dict[int, int]  # action -> child_key
    is_expanded: bool
```

### 6.2 Transposition table（mask -> Node）

```python
class TranspositionTable:
    def __init__(self):
        self.table: dict[int, Node] = {}

    def get(self, key): ...
    def put(self, node): ...
```

### 6.3 MCTS 主流程（PUCT）

```python
class MCTS:
    def __init__(self, net: PolicyValueNet, n_sim=256, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25):
        ...

    def run(self, env: EEGChannelGame) -> np.ndarray:
        root_obs = env._get_obs()
        root_key = root_obs["key"]
        root = self._get_or_expand(root_key, root_obs, add_root_noise=True)

        for _ in range(self.n_sim):
            self._simulate(env, root_key)

        pi = root.N.astype(np.float32)
        pi /= (pi.sum() + 1e-8)               # [23]
        return pi

    def _simulate(self, env, root_key):
        # 1) copy env state (mask/order/t) 轻量复制，避免深拷贝大数据
        sim_state = env_snapshot(env)

        path = []  # list of (node_key, action)
        key = root_key

        # 2) selection
        while True:
            node = self.tt.get(key)
            if not node.is_expanded:
                break
            a = select_puct_action(node, c_puct=self.c_puct)
            path.append((key, a))
            key = self._next_key(sim_state, a)
            apply_action(sim_state, a)
            if is_terminal(sim_state):
                break

        # 3) expansion & evaluation
        if is_terminal(sim_state):
            v = self._terminal_value(sim_state)      # 这里一般=0，最终reward只在真实env终局算
        else:
            obs = build_obs_from_snapshot(sim_state) # tokens/action_mask/key
            # batched inference 强烈建议（见 6.4）
            logits, v_mu, v_q20 = self.net_infer(obs)
            P = softmax(logits)                      # [23]
            node = Node(key, P, zeros(23), zeros(23), zeros(23), {}, True)
            self.tt.put(node)
            v = float(v_q20)                         # 用鲁棒value更稳

        # 4) backup
        for k, a in reversed(path):
            node = self.tt.get(k)
            node.N[a] += 1
            node.W[a] += v
            node.Q[a] = node.W[a] / node.N[a]
```

### 6.4 批量推理（非常建议做）

每次模拟都单独跑一次网络会被 GPU 吞吐拖死。建议把“待扩展叶子”收集起来批量 forward。

做法：

* 在 `_simulate` 里遇到“未扩展 leaf”时，把 `obs` 放入 `pending` 列表并返回；
* 外层用循环：收集到 `batch_size` 或达到一定步数后统一 forward；
* 再把结果写回对应节点并继续 backup。

这能把推理成本降低很多，训练才能跑得动。

---

## 7) Evaluator：L0/L1/L2 统一接口 + 何时调用

### 7.1 统一接口

```python
class EvaluatorBase:
    def evaluate(self, mask: np.ndarray, fold_data: FoldData) -> tuple[float, dict]:
        """return reward_scalar, info_dict(perf, kappa, acc, n_ch, etc)"""
```

### 7.2 L0（proxy，快）

* 输入：`mask [22]`
* 输出：`proxy_score`（可当 reward，也可当 leaf value 校正）
* 计算：用 `FoldStats.fisher[22,B]` 与 `FoldStats.redund_corr[22,22]` 做

  * `relevance = sum_i mean_b fisher[i,b]`
  * `redund = mean_{i,j in S} |corr[i,j]|`
  * `reward = relevance - β*redund - λ*(|S|/22)`

> **缓存**：`fisher`、`redund_corr` 都在 fold stats 里；每次 evaluate 仅 O(|S|^2) 很小。

### 7.3 L1（FBCSP OVR + rLDA，中）

* 输入：`mask`
* 用 `cov_fb [B,N,22,22]` 取子矩阵 `[:, :, S, S]`
* OVR：4 个二分类，频带 9 个，每个频带 CSP `m=2` 对
* 验证：训练 fold 内 3-fold CV，输出 κ 均值（可加下界）

> **缓存**：

* `cov_fb` 已预计算；
* 如果你还缓存了 `class_cov_mean [B,4,22,22]`，则每个评估只需取子矩阵就能做 CSP（更快）。
* CV 时的“fold 内再分”建议固定随机种子并缓存 index，避免同一 mask 多次评估结果飘。

### 7.4 L2（严格评估，慢）

用于：周期性校准、最终报告。

* FBCSP 用更严格 CV 或直接 “train session 训练 → eval session 测试”
* Deep：EEGNet/ShallowConvNet（选通道后训练），多 seed 得到分位数/CI

---

## 8) ReplayBuffer：存什么才能既省内存又可复现

### 强烈建议不要把 `tokens[24,64]` 直接存满（很快爆）

改存“可重建的最小信息”：

* `key`（bitmask int）
* `order`（可选；如果 state 里包含 order 特征则必须存）
* `t`（步数）
* `split_id`（对应 fold stats）
* `pi_mcts`: `float16/float32 [23]`
* `z`: `float32`（回合终局 reward）

训练时再用 `StateBuilder.build(...)` 重建 tokens。

```python
class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.keys = np.zeros(capacity, dtype=np.int64)
        self.ts = np.zeros(capacity, dtype=np.int16)
        self.split_ids = np.zeros(capacity, dtype=np.int16)
        self.pi = np.zeros((capacity, 23), dtype=np.float16)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0; self.size = 0

    def add(self, key, t, split_id, pi, z): ...
    def sample(self, batch_size) -> dict: ...
```

---

## 9) TrainLoop：自博弈数据采样 + 网络训练 + 周期评估（闭环）

### 9.1 Self-play（这里其实是“单人局”）

```python
def play_one_game(env, mcts, temp_schedule):
    traj = []
    obs = env.reset()
    done = False
    while not done:
        pi = mcts.run(env)                    # [23]
        a = sample_from_pi(pi, temp=temp_schedule(env.t))
        traj.append((obs["key"], env.t, env.fold_data.split_id, pi))
        obs, r, done, info = env.step(a)

    # 终局 reward r
    for (key, t, split_id, pi) in traj:
        buffer.add(key, t, split_id, pi, r)
    return info
```

### 9.2 训练主循环（推荐“分阶段加重评估器”）

```python
for iter in range(num_iters):
    # 1) 采样若干局
    for g in range(games_per_iter):
        fold = sampler.sample_fold()             # subject + split
        env = EEGChannelGame(fold, state_builder, evaluator=current_eval)
        info = play_one_game(env, mcts, temp_schedule)

    # 2) 网络训练若干步
    for step in range(steps_per_iter):
        batch = buffer.sample(batch_size)
        tokens, action_mask = rebuild_tokens(batch)  # [B,24,64], [B,23]
        logits, v_mu, v_q20 = net(tokens, action_mask)
        loss = policy_kl(logits, batch.pi) + mse(v_q20, batch.z)
        optimize(loss)

    # 3) 周期性切换 evaluator / 校准
    if iter == switch_to_l1_iter:
        current_eval = L1Evaluator(...)
    if iter % l2_calib_every == 0:
        run_l2_calibration(top_k_masks_from_buffer)
```

**建议阶段计划：**

* Iter 0~X：用 L0 训练，快速让策略学会“不选垃圾通道、控制冗余”
* Iter X~Y：切到 L1（FBCSP）当终局 reward，策略开始对齐真实 κ
* 每隔 K iter：抽 top-N 子集跑 L2（严格），并记录 proxy/L1 与 L2 的相关性（这也是你们论文里的重要图）

---

## 10) 最后给你一份“默认超参”能直接开跑

* `B_max = 10`（先做 4/6/8/10/12 曲线再扩）
* MCTS：`n_sim=256, c_puct=1.5, alpha=0.3, eps=0.25`
* 网络：`D_in=64, d_model=128, layers=4, heads=4`
* buffer：200k
* 每 iter：`games=64, train_steps=200, batch=256`
* evaluator 切换：L0 先跑 20 iter → L1 再跑 60 iter
* L2 校准：每 10 iter 跑一次，top-20 masks

---

