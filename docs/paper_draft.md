# Paper Draft (v0.2): Risk-Sensitive Teacher-Guided RL for Cross-Session EEG Channel Subset Selection (One Policy for All Subjects)

> 项目仓库：`eeg_channel_game/`（通道选择环境 + RL/MCTS + 评估/基线）。
>
> 本文档是“论文初稿式”的写法：先把**科学问题（数学化）**、**主创新点**、**整体流程**、**实验与复现实验命令**讲清楚，后续再补相关工作与更严格的实验矩阵。

---

## 0. 我们的主结论（是否够 SCI 一区主创新？）

**可以够，但前提是：**

1. 你必须在 **All subjects × 多个 K** 上，稳定超过强基线（至少 `lr_weight`、`random_best_l1`、`GA`、`MI/Fisher`，并做预算对齐）。
2. 你必须把“为什么 EEG 通道选择多数负收益/负迁移”写成数学问题，并用你的方法去解决它，而不是“把多个技巧堆一起”。

本项目当前最有潜力成为 Q1 主创新的主线是（注意：不是“堆模块”，而是“为任务痛点写目标函数 + 设计可优化的近似”）：

> **风险敏感（risk-sensitive）的跨 session 通道子集选择** + **Teacher-guided 的构造式 RL（AlphaZero/MCTS）** + **无标签目标域（eval session）分布偏移惩罚/约束**。

这三者组合起来，能形成一个明确的数学目标（见第 2 节），并且在实现上与你现有代码完全一致（见第 4 节）。

---

## 1. 科学问题 → 数学问题（Problem Formulation）

### 1.1 数据设定（跨 session / domain shift）

对每个被试 \(s\)，存在两个 session：

- 训练域（source）：\(\mathcal{D}^T_s=\{(x^T_i, y^T_i)\}\)
- 测试域（target）：\(\mathcal{D}^E_s=\{x^E_j\}\)（**无标签可用作无监督对齐；标签 \(y^E\) 只用于最终报告**）

EEG trial：\(x\in\mathbb{R}^{C\times T}\)，BCI IV-2a 典型 \(C=22\)。

通道子集用集合 \(S\subseteq\{1,\dots,C\}\)，且 \(|S|=K\)。

### 1.2 目标：多数子集负收益（negative gain）下的稳健选择（适配 EEG 低 SNR）

令 \(f(\cdot;S)\) 是下游分类器（例如 FBCSP+LDA 或 ShallowConvNet），在目标域的评价指标（kappa）为：

\[
\kappa_s(S) = \kappa\big(f(\cdot;S)\ \text{trained on}\ \mathcal{D}^T_s,\ \text{tested on}\ \mathcal{D}^E_s\big)
\]

全通道基线 \(S_{\text{full}}\)（22 通道）：
\[
\kappa_s^{\text{full}}=\kappa_s(S_{\text{full}})
\]

我们关心的是真正“值得写论文”的目标：**固定预算 K 下，子集是否能稳定超过全通道**：
\[
\Delta_s(S)=\kappa_s(S)-\kappa_s^{\text{full}}
\]

但 EEG 低信噪比 + session shift 导致 \(\Delta_s(S)\) 的方差很大，且大多数 \(S\) 为负收益（你也在实验里看到“选子集经常变差”）。因此我们采用风险敏感目标（下分位/下置信界），让优化目标直接对齐“稳定提升而不是偶然提升”：

\[
\max_{|S|=K}\ Q_{\tau}\big(\Delta_s(S)\big)\quad (\tau=0.2\ \text{或 LCB})
\]

### 1.3 引入“无标签 target 域”约束（解决跨 session 负迁移；可作为主创新之一）

为了显式刻画跨 session 分布偏移（domain shift），我们允许使用 **无标签** \(\mathcal{D}^E_s\) 的特征分布，定义一个可计算的偏移度量：

\[
D_s(S)=D\big(\phi_S(X^T),\phi_S(X^E)\big)
\]

其中 \(\phi_S\) 可以是 bandpower / covariance 等不需要标签的表示。

最终目标可写为（主公式）：

\[
\max_{|S|=K}\ \underbrace{Q_{\tau}\big(\Delta_s(S)\big)}_{\text{稳健增益}}
-\eta \underbrace{D_s(S)}_{\text{跨 session 偏移惩罚}}
\]

> 备注：如果审稿人认为“不能看 target 数据”，则把 \(\eta=0\) 当作严格协议主结果，把 \(\eta>0\) 当作 UDA/无监督对齐增强消融。

### 1.4 进一步卖点：一个模型跨被试通用（但要用正确实验来证明）

代码实现里，policy/value 网络 \(\pi_\theta,V_\theta\) 是 **一个模型共享所有被试**（训练时从多个 subject 采样 episode）。因此我们的方法天然具备“跨被试迁移/通用策略”的潜力：

- **弱表述（当前默认能说）**：同一个 \(\theta\) 在所有 subjects 上做通道选择（不是每个被试训练一个 agent）。
- **强表述（SCI 更有说服力，建议补）**：做 **留一被试/留多被试** 的泛化实验：训练 subjects 集合 \(\mathcal{S}_{train}\)，测试未见 subjects 集合 \(\mathcal{S}_{test}\)，报告 \(\Delta_s(S)\) 与风险敏感指标在 \(\mathcal{S}_{test}\) 上是否仍成立。

数学上可写为跨被试期望目标：
\[
\max_{\theta}\ \mathbb{E}_{s\sim \mathcal{S}_{train}}\Big[\max_{|S|=K}\ Q_{\tau}\big(\Delta_s(S)\big)-\eta D_s(S)\Big]
\]
并在 \(s\sim \mathcal{S}_{test}\) 上评估泛化（policy 不再训练，仅计算被试的 fold stats 后直接选择）。

---

## 2. 主方法：Teacher-Guided Risk-Sensitive AlphaZero（建议命名）

### 2.1 构造式选择（sequential construction）——适配“组合选择 + 固定预算 K”

把通道选择写成一个序列决策：

- 状态：当前已选子集 \(S_t\)
- 动作：选择一个未选通道 \(a_t\in\{1,\dots,C\}\setminus S_t\) 或 STOP
- 终止：\(|S_t|=K\)（训练/评估对齐：训练时强制 exact budget，避免“训练允许 STOP、评测固定 K”导致天然吃亏）

### 2.2 训练 reward：用“可复现且不泄漏”的 proxy 对齐最终目标（为什么不是直接用 eval kappa）

训练阶段不能用 eval 标签，因此使用在训练 session 上可计算的 proxy：

- **L1 (FBCSP on 0train 内部 CV)**：\(\hat\kappa^{T\to T}_{\text{robust}}(S)\)
- 可选加上无标签 domain shift 惩罚：\(\hat D_s(S)\)
- 用 delta_full22 归一化：减去 full-22 的常数基线以对齐跨被试尺度

训练 reward（实现一致；注意这里是训练 session 内部 CV，不用 eval 标签）：
\[
r(S)=\Big(\hat\kappa_{\text{robust}}(S)-\hat\kappa_{\text{robust}}(S_{\text{full}})\Big) - \eta \hat D_s(S)
\]

**为什么这样设计是“适配任务”的（而不是堆技巧）**：

1. 直接优化 \(\Delta(S)\)（相对 full-22 的提升）等价于把 reward 与论文主指标对齐；同时减去 \(\hat\kappa(S_{\text{full}})\) 会把不同被试的 reward 尺度拉到可比水平，利于一个模型跨被试训练。
2. 用 robust 版本 \(\hat\kappa_{\text{robust}}\)（q20/LCB）对应 EEG 低 SNR 与高方差：我们希望策略学到“少翻车的子集”，而不是“偶然在某个 fold 很高”。
3. \(\hat D_s(S)\) 只用 eval session 的无标签特征，对齐“跨 session 负迁移”这个根因，同时不泄漏 eval 标签。

### 2.3 Teacher：用嵌入式特征选择提供先验（lr_weight）

用训练 session 的 bandpower 特征训练多类 Logistic Regression，得到每个通道的重要性分数 \(s_i\ge 0\)（权重范数聚合）：

\[
s_i = \left\|W_{:,i,:}\right\|_2
\]

构造 teacher 分布（只在合法动作上）：
\[
\pi_0(i\mid S_t)\propto \exp(\log(1+s_i)/T)
\]

Teacher 的使用方式（论文里可以写成 3 个模块；它们的“理由”都是同一个：EEG 子集空间里负收益居多，稀疏终局回报 + 昂贵评估会让纯 RL 冷启动崩，因此需要一个便宜的、任务相关的先验把搜索带到“更可能正收益”的区域；再逐步退火让 RL 学到组合协同而不是照抄 teacher）：

1. **MCTS prior 混合**：\(\pi_{\text{net}} \leftarrow (1-\eta_p)\pi_{\text{net}}+\eta_p\pi_0\)
2. **Leaf value bootstrap**（稀疏终局时稳定 MCTS）：\(v\leftarrow \alpha V_\theta(s)+(1-\alpha)\hat r_{\text{proxy}}(S)\)
3. **Teacher KL / imitation 正则**：
\[
\mathcal{L}=\mathcal{L}_{\pi}+\mathcal{L}_{V}+\lambda_{\text{teach}}\cdot \mathrm{CE}(\pi_0,\pi_\theta)
\]

### 2.4 风险敏感（risk-sensitive）

最终论文主结果建议报告 \(Q_{0.2}\) 或 LCB（而不是只报均值），并让训练/搜索目标与之对齐（例如在 L1 CV folds 上用 q20 或 mean-β·std）。

---

## 3. 整体流程（带图）

### 3.1 Pipeline 图（Mermaid）

```mermaid
flowchart LR
  A[MOABB 读取 BNCI2014_001] --> B[预处理: bandpass + epoch]
  B --> C[缓存: bandpower / quality / cov_fb]
  C --> D[FoldStats: fisher / MI / lr_weight / redundancy]
  D --> E[EEGChannelGame 环境]
  E --> F[MCTS: PUCT + leaf bootstrap + policy prior]
  F --> G[Policy/Value Transformer]
  G --> H[自博弈 self-play 生成 (s, pi, z)]
  H --> I[优化: policy loss + value loss + teacher loss]
  I --> G
  G --> J[run_pareto_curve: ours/uct/基线 across K]
  J --> K[0train->1test 报告 kappa/acc + Pareto]
```

### 3.2 训练/评估协议（必须写清楚）

- **训练阶段**：只用训练 session 标签（0train），不可触碰 eval 标签。
- **可选无监督**：允许使用 eval session 的无标签特征（例如 bandpower 均值）做 \(D_s(S)\)（需在论文里声明为 UDA 设定）。
- **最终评估**：0train 训练，下游在 1test 上测 kappa/acc（严格）。

---

## 4. 代码对应关系（你现在仓库里已有）

- 训练入口：`eeg_channel_game/run_train.py`
- Pareto 评估入口：`eeg_channel_game/run_pareto_curve.py`
- MCTS：`eeg_channel_game/mcts/mcts.py`
- Policy/Value Net：`eeg_channel_game/model/policy_value_net.py`（支持 `net.policy_mode=token`）
- FoldStats（含 lr_weight）：`eeg_channel_game/eeg/fold_stats.py`
- Teacher proxy：`eeg_channel_game/eval/evaluator_l0_lr_weight.py`
- Domain shift（无标签）惩罚：`eeg_channel_game/eval/evaluator_domain_shift.py`

---

## 4.1 实验开关清单：哪些已启用，哪些是可选

下面这个表格是为了回答“文档里写的东西现在到底有没有开”的疑问。

| 功能 | 为什么需要（对应任务痛点） | 配置键 | 默认 | `agent_bd_teacher_v1` 命令是否启用 |
|---|---|---|---:|---:|
| 多被试共享一个 agent | 把通道选择从“每人一个策略”变成“通用策略” | `data.subjects=[...]` | ✅ | ✅ |
| 固定预算对齐（exact K） | 训练/评测目标一致，避免 STOP 作弊 | `train.force_exact_budget=true` | ✅（在 `perf_multik_braindecode.yaml`） | ✅ |
| 多 K 训练 | 一次训练覆盖 Pareto 曲线 | `train.b_max_choices=[4,6,8,10,12,14]` | ✅（在 config） | ✅ |
| 相对 full-22 归一化 | reward 尺度跨被试可比、直接对齐“超越全通道” | `reward.normalize=delta_full22` | ✅（在 config） | ✅ |
| 风险敏感 robust kappa | 低 SNR/高方差下优化“稳定”而非“偶然” | `evaluator.l1_fbcsp.robust_mode=mean_std/q20` | ✅（mean_std） | ✅（mean_std, β=0.5） |
| L0 teacher proxy（lr_weight） | 冷启动引导到高收益区域 | `evaluator.phase_a=l0_lr_weight` | ❌（默认 `l0`） | ✅（override） |
| token policy head | 组合选择更直接地对每个通道打分 | `net.policy_mode=token` | ❌（默认 cls） | ✅（override） |
| leaf value bootstrap | 稀疏回报下稳定 MCTS 叶子估值 | `mcts.leaf_bootstrap.enabled=true` | ❌ | ✅（proxy=lr_weight, α 退火） |
| MCTS policy prior | 早期减少无效探索（负收益子集太多） | `mcts.policy_prior.enabled=true` | ❌ | ✅（η 退火到 0） |
| Teacher KL/CE 正则 | 让网络快速学到可用先验，再退火释放能力 | `train.teacher_kl.enabled=true` | ❌ | ✅（权重退火） |
| 无标签 domain shift 惩罚 | 显式对齐跨 session 分布偏移（不看 eval 标签） | `reward.domain_shift.enabled=true` | ❌ | ✅（η=0.05） |
| 通道数惩罚（subset size cost） | 只在“可变 K/成本敏感”设定需要；固定 K 时可关 | `reward.lambda_cost` | 0.05（config） | ✅（你 override 成 0，等价于关闭成本项） |

> 注：`policy_prior / leaf_bootstrap / teacher_kl` 主要用于训练阶段提升可学性；最终评估报告时我们更关注：在相同预算/协议下，最终搜索策略（ours）是否稳定超过强基线。

---

## 5. 复现实验命令（All subjects × All K × All methods）

### 5.1 评估（从某个 checkpoint 开始，全被试、全 K、全方法）

> 说明：`ours/uct` 的搜索非常耗时；建议开启 `--resume` 防断连，必要时先用较小 `mcts.n_sim` 和 `--ours-restarts` 做 quick check，再跑 full budget。

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_pareto_curve \
  --config eeg_channel_game/configs/perf_multik_braindecode.yaml \
  --override project.out_dir=runs/agent_bd_teacher_v1 reward.lambda_cost=0 net.policy_mode=token \
            reward.domain_shift.enabled=true reward.domain_shift.eta=0.05 reward.domain_shift.mode=bp_mean_l2 \
            mcts.n_sim=256 \
  --checkpoint runs/agent_bd_teacher_v1/checkpoints/best.pt \
  --k 4,6,8,10,12,14 \
  --methods ours,uct,fisher,mi,lr_weight,sfs_l1,random_best_l1,ga_l1,full22 \
  --random-n 200 --ga-restarts 2 --ga-pop 64 --ga-gens 30 \
  --ours-restarts 10 --ours-stochastic --ours-tau 0.8 \
  --uct-restarts 10 --uct-stochastic --uct-tau 0.8 \
  --tag eval_iterXXX_all --plot --resume
```

如果你的机器上 `cudaGetDeviceCount` 报错，可以加 `--device cpu`。

---

## 6. 你接下来要补的“Q1 必要实验”

1. **预算对齐**：让 `ours/uct/random/GA` 用相同数量的 evaluator 调用或相同 wall-time。
2. **消融**（至少 6 组）：
   - 无 teacher / 仅 policy prior / 仅 teacher loss / 仅 leaf bootstrap
   - 无 domain shift / 有 domain shift
   - mean vs q20（风险敏感）
3. **泛化解释**：报告 domain shift 指标与 \(\Delta(S)\) 的相关性（证明“负收益通道多”的根因是跨 session 偏移）。
