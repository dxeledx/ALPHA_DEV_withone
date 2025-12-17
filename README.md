# Alpha (EEG Channel Selection via RL/MCTS)

本项目目标：用 **强化学习 + 规划搜索（AlphaZero/AlphaDev 风格：policy/value + MCTS）**完成 **EEG 通道选择**，服务于 **BCI Competition IV-2a 四分类运动想象（MI）识别**，并最终形成可发表的完整实验链路（严谨协议、可复现、可视化与消融）。

> 设计文档：`design.md`（分阶段、风险点与实验写法都已写死，工程实现按它推进）。

---

## 1. 关键原则（写论文/避免翻车）

- **严格避免数据泄漏**：通道选择与模型训练只使用 `0train`（training session）；`1test`（evaluation session）只用于最终测试/报告。
- **EOG 不得用于分类**：EOG 仅用于去伪迹（回归/ICA 等）或“伪迹风险估计”，不能作为分类输入特征。
- **主指标建议 Cohen’s kappa**（同时报 accuracy），并报告 **通道数-性能曲线 / Pareto 前沿**。

---

## 2. 当前实现到哪里了？

核心代码在 `eeg_channel_game/`，已实现一个最小闭环：

- **数据准备（MOABB + 缓存）**：读取 BNCI2014_001，做 epoch（默认绝对 `[3,6]s`），训练 session 上拟合 EOG 回归并应用到 train/eval，保存 EEG-only epochs 与缓存（bandpower / quality / 可选 cov_fb）。
- **环境（EEGChannelGame）**：动作=选一个通道或 STOP；终局 reward=评估器输出（L0/L1/L2 可切换）。
- **MCTS（PUCT）**：policy/value 先验引导搜索，带根节点 Dirichlet 噪声。
- **Policy/Value 网络**：Transformer encoder + policy head + value head。
- **训练（self-play）**：MCTS 产生 `π_MCTS` 蒸馏到 policy；终局 reward 监督 value。
- **评估与可视化**：输出选中通道列表、topomap；提供一个快速基线（bandpower-logreg）与 **L2 深度模型**评估入口。

> 备注：当前训练默认用 L0（proxy）让闭环先跑通；L1/L2 会更慢但更“论文一致”。

---

## 3. 环境与依赖

推荐使用你已有的 conda 环境 `eeg`：

- `moabb==1.2.0`
- `mne==1.8.0`
- `braindecode==0.8`
- `torch`（已检测到 CUDA 可用）

运行示例统一用：

```bash
conda run -n eeg --no-capture-output <command...>
```

---

## 4. 快速开始（可复现的一条龙）

### 4.1 准备数据

默认配置：`eeg_channel_game/configs/default.yaml`

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_prepare_data \
  --config eeg_channel_game/configs/default.yaml \
  --subjects 1
```

如果你要启用 L1(FBCSP)/L2(Deep) 严格评估，建议 **不要**加 `--no-cov`（需要生成 `sessionT_cov_fb.npz`）。

> 数据会写入一个 **variant** 目录（默认会从 `fmin/fmax/time-window/eog` 自动生成，例如 `f4-40_t3-6_eog1`），用于窗口/预处理消融时避免覆盖：
>
> - `eeg_channel_game/data/processed/<variant>/subj01/...`
> - `eeg_channel_game/data/cache/<variant>/subj01/...`

### 4.2 训练（L0 reward，先跑通）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train \
  --config eeg_channel_game/configs/default.yaml \
  --override project.out_dir=runs/exp1
```

常用覆盖参数（示例）：

```bash
--override mcts.n_sim=256
--override game.b_max=10
--override train.num_iters=100
--override data.subjects=[1,2,3,4,5,6,7,8,9]
```

> `--override` 支持重复使用，也支持一次传多个覆盖项（例如：`--override project.out_dir=runs/x data.subjects=[1]`）。

> 结果保存提示：
> - 为避免误覆盖，`run_train` 检测到 `project.out_dir` 里已有 checkpoint 时会直接报错；请改 `project.out_dir`，或用 `--override train.resume=true` 续跑（或 `--override project.overwrite=true` 强制覆盖）。

#### 多被试共享策略（推荐）

用同一个 policy/value 网络在多个 subjects 的 `0train` 上联合训练（episode 随机采样 subject+fold），网络会通过 state 里的统计特征（fisher/bandpower/quality/EOG 相关/空间位置等）自动“条件化”到不同被试。

训练建议直接用 `eeg_channel_game/configs/perf.yaml`（默认 subjects=[1..9]，并包含 Phase A→B；默认启用 `reward.normalize=delta_full22` 做跨被试 reward 对齐）。

训练完成后，**不要**直接用 checkpoint 里的 `best_key` 当作“所有被试的最终子集”（它通常只对应某个 subject/fold 的一次 best）。
正确做法是：加载 checkpoint 网络，对每个 subject 重新跑一次搜索得到该 subject 的子集：

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_search \
  --config eeg_channel_game/configs/perf.yaml \
  --override project.out_dir=runs/perf \
  --checkpoint runs/perf/checkpoints/iter_079.pt \
  --split-mode full
```

想让推理更强（更像“规划搜索”的优势），可以做 best-of-N 的 multi-start（只按 `0train` 的 reward 选最优，不看 `1test`）：

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_search \
  --config eeg_channel_game/configs/perf.yaml \
  --override project.out_dir=runs/perf \
  --checkpoint runs/perf/checkpoints/iter_079.pt \
  --split-mode full \
  --restarts 8 --stochastic --tau 0.8
```

> 如果你希望多次运行 `run_search` 保留多份结果，使用 `--tag xxx`，输出会写到 `runs/<out>/search/xxx/`。

输出：

- `runs/perf/search/summary.csv`：每个 subject 的子集与 **FBCSP 0train→1test** 指标
- `runs/perf/search/subjXX_search.json`：详细信息（选中通道、搜索 reward 等）

#### 固定 K 的 Pareto 曲线（推荐写论文）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_pareto_curve \
  --config eeg_channel_game/configs/perf.yaml \
  --override project.out_dir=runs/perf \
  --checkpoint runs/perf/checkpoints/iter_079.pt \
  --k 4,6,8,10,12 \
  --methods ours,fisher,mi,full22 \
  --plot
```

`--methods` 可选：`ours` / `uct`(MCTS+uniform prior/value) / `fisher` / `mi` / `lr_weight` / `sfs_l1` / `ga_l1` / `random_best_l1` / `full22`。
如需保留多次曲线结果，使用 `--tag xxx`，输出写到 `runs/<out>/pareto/xxx/`。

生成：

- `runs/perf/pareto/pareto_summary.csv`：每个 K 的 mean/std/q20
- `runs/perf/pareto/pareto_kappa.png`：曲线图（mean±std）
- `runs/perf/pareto/pareto_acc.png`：accuracy 曲线图（mean±std）

### 4.3 评估 + 可视化（topomap + 快速基线）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_eval \
  --config eeg_channel_game/configs/default.yaml \
  --override project.out_dir=runs/exp1 \
  --subject 1
```

输出示例：

- `runs/exp1/figures/subj01_selected_topomap.png`
- `runs/exp1/checkpoints/iter_*.pt`

---

## 5. L0/L1/L2 评估器（reward 体系）

### L0（快，训练阶段默认）

`eeg_channel_game/eval/evaluator_l0.py`

- 用训练 fold 的 Fisher score + 冗余相关（bandpower corr） + 通道数惩罚 +（可选）伪迹惩罚。
- 其中 Fisher 会做 `log1p` 压缩并按选中通道取均值，避免原始 F 值重尾导致 reward 尺度不稳。
- 目的：给 MCTS/价值网络一个“便宜但有形状”的信号，让系统先闭环再升级。

### L1（中等成本，更接近传统 MI）

`eeg_channel_game/eval/evaluator_l1_fbcsp.py`

- FilterBank CSP(OVR) + shrinkage LDA，在训练 session 内部做 train/val（来自 FoldSampler 的 split）。
- 需要 `run_prepare_data` 时生成 `sessionT_cov_fb.npz`/`sessionE_cov_fb.npz`。
- 已支持 **鲁棒 reward**：`cv_folds>=2` 时输出 `mean/std/q20`，可用 `mean-β·std` 或 `q20` 对抗评估方差。

### L2（高保真，论文主结果/最终报告用）

`eeg_channel_game/eval/evaluator_l2_deep.py`

- 使用深度模型做严格评估（可选）：
  - Braindecode：`EEGNetv4` / `ShallowFBCSPNet`
  - 本项目集成：`vtransformer`（VTransformer V5.1 改造版：subset mask 门控 + channel-dropout + GroupNorm）
  - train：training session（内部 train/val split + early stopping）
  - test：evaluation session（`1test`）
  - 多 seed 输出 `mean/std/q20`（默认 q20 作为“鲁棒下界”更利于写论文）

运行（在 `run_eval` 里直接打开）：

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_eval \
  --config eeg_channel_game/configs/default.yaml \
  --override project.out_dir=runs/exp1 \
  --subject 1 \
  --l2 --l2-model vtransformer --l2-epochs 30 --l2-seeds 0,1,2
```

> 注意：**L2 会用到 eval session 标签**，因此不要在 RL 训练/搜索阶段把 L2 当 reward（否则论文会被认为泄漏）。

---

## 6. 目录结构

```
eeg_channel_game/
  configs/                 # yaml 配置
  data/
    processed/             # EEG-only epochs（按 subject）
    cache/                 # bandpower/quality/cov_fb/foldstats 等缓存
  eeg/                     # 数据准备与特征/协方差
  game/                    # 环境 + 状态构造（token）
  mcts/                    # PUCT + transposition table
  model/                   # policy/value 网络
  rl/                      # replay/self-play/train_loop
  eval/                    # L0/L1/L2 evaluator + metrics
  utils/                   # config/seed/bitmask/vis
```

---

## 7. 参考材料

- `alphadev-main/`：AlphaDev 论文相关代码与阅读材料（作为路线 A 的启发参考）
- `design.md`：本项目方法设计与实验写作模板（建议直接按它推进实现/消融）

---

## 8. 性能向配置与脚本（建议你现在就用）

### 8.1 两阶段训练（L0 → L1）

推荐从 `eeg_channel_game/configs/perf.yaml` 开始（已内置 `switch_to_l1_iter` 与 L1 鲁棒配置；默认不访问 `1test` 标签）：

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train --config eeg_channel_game/configs/perf.yaml
```

如果你的第一目标是把 **固定 K 的 Pareto 曲线**拉上去（K=4/6/8/10 等），建议用 `eeg_channel_game/configs/perf_multik.yaml`：

- `train.b_max_choices=[4,6,8,10]`：训练期每局随机采样一个预算 K
- `train.force_exact_budget=true`：每局强制“恰好选 K 个通道”再终止（更贴合固定-K评估）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train --config eeg_channel_game/configs/perf_multik.yaml
```

> 如果你确实需要训练期离线 L2 校准（仅做一致性检查，绝不用于训练），用 `eeg_channel_game/configs/perf_l2calib.yaml` 或手动覆盖打开：
> `--override train.allow_eval_labels=true train.l2_calib_every=10`

### 8.2 强基线（full-22，train→eval）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_baselines --config eeg_channel_game/configs/perf.yaml
```

### 8.3 固定 K 的对比表（示例：K=8/10，单被试）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_compare_k \
  --config eeg_channel_game/configs/default.yaml \
  --override project.out_dir=runs/exp1 \
  --subject 1 --k 8,10
```

> 如果你希望多次运行 `run_compare_k` 保留多份结果，使用 `--tag xxx`，输出会写到 `runs/<out>/compare_k/xxx/`。

输出表格会同时给出：

- **FBCSP(rLDA) 0train→1test**（更接近“真实测试性能”）
- **L2 Deep 0train→1test**（用于校准/最终叙事，前提是你把深度训练调到明显高于 chance）
