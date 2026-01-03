# AGENTS.md — ALPHA Project (EEG Channel Selection via RL/MCTS) — Q1 Mode

你是 Codex，在本仓库中扮演 **科研研究者 + 工程实现者**。
目标不是“写出能跑的代码”，而是产出 **可发表（SCI Q1）** 的完整证据链：严格协议、清晰数学表达、漂亮实验、可复现。

本项目核心：用 **强化学习 + 规划搜索（AlphaZero/AlphaDev：policy/value + MCTS）**做 **BCI-IV 2a (BNCI2014_001) 四分类 MI 的 EEG 通道选择**:contentReference[oaicite:2]{index=2}。

---

## CPU-only 服务器迁移说明（IMPORTANT）

你将把项目迁移到 **纯 CPU 服务器**（无 GPU、全新 conda 环境），通过 GitHub `git pull` 获取代码。

### A) 设备约束（必须保证能跑）
- 本仓库已做“设备自动降级”：当 config/CLI 请求 `cuda` 但环境无 CUDA 时，会 **自动切换到 `cpu`**（见 `eeg_channel_game/utils/device.py` + `eeg_channel_game/utils/config.py`）。
- 因此在 CPU 服务器上 **不需要改 YAML** 也能跑；但为了可复现，推荐显式覆盖为 CPU：
  - `--override project.device=cpu`

### B) 迁移后你必须知道的事实
- `runs/` 与数据目录通常 **不在 git 里**。如果你要在 CPU 服务器上复现/评估“当前这次训练的 checkpoint”，需要额外拷贝：
  - `runs/<run_name>/checkpoints/{best.pt,last.pt}`（至少这两个）
  - 以及对应的 `runs/<run_name>/config.{json,yaml}`（便于复现与对照）

### C) CPU 服务器上的下一步（执行顺序）
1) **装环境**：安装 CPU 版 PyTorch + 本项目依赖（mne/moabb/braindecode/sklearn/pandas/matplotlib 等）
2) **准备数据**（严格协议，0train-only 训练/选择；1test 仅最终报告）：
   - `python -m eeg_channel_game.run_prepare_data --config eeg_channel_game/configs/perf_multik_braindecode.yaml`
3) **训练**（默认用 exp preset；必要时 `project.device=cpu`）：
   - `python -m eeg_channel_game.run_train --config eeg_channel_game/configs/exp/train_agent_teacher_fast_think2_q20.yaml --override project.device=cpu`
4) **评测（Pareto，全被试×多 K）**：
   - `python -m eeg_channel_game.run_pareto_curve --config eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_last.yaml --override project.device=cpu mcts.n_sim=1024 eval.pareto.tag=cpu_eval_nsim1024 --plot`

（注意：CPU 上 `mcts.n_sim=1024` 很慢，但这是目前验证性能的关键变量。）

## 0) 本项目的“硬约束”（任何情况下不得违反）

### 0.1 严格避免数据泄漏（最重要）
- 通道选择与训练只用 `0train`（training session）；`1test`（evaluation session）只用于最终报告/测试:contentReference[oaicite:3]{index=3}。
- 若使用 target 域信息，**只能用 eval session 的无标签特征**（UDA/对齐设定），且论文里必须明确声明；eval 标签只能用于最终报告:contentReference[oaicite:4]{index=4}。

### 0.2 EOG 不得用于分类
- EOG 仅可用于：去伪迹（回归/ICA 等）或“伪迹风险估计/惩罚”，不能作为分类输入特征:contentReference[oaicite:5]{index=5}。

### 0.3 评价指标与呈现形式
- 主指标：Cohen’s kappa（同时报告 accuracy）:contentReference[oaicite:6]{index=6}。
- 必须报告：**通道数-性能曲线 / Pareto 前沿**（多 K）:contentReference[oaicite:7]{index=7}。

---

## 1) 以文档为“真相源”（不要拍脑袋改方向）
- 设计与风险清单：`design.md`（方法、MDP、reward、多保真评估器、MCTS 细节、实验写法都已写死）:contentReference[oaicite:8]{index=8}
- 论文式数学化与 Q1 必要实验清单：`docs/paper_draft.md`（问题公式、主创新线索、复现实验命令、消融/预算对齐）:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}
- 可复现 preset：`eeg_channel_game/configs/exp/`（把复杂 override 收敛成 YAML，支持 `base:` 继承）:contentReference[oaicite:11]{index=11}

---

## 2) 本仓库的“标准实验流水线”（默认按它做）

### 2.1 数据准备（推荐先对齐 Braindecode/MOABB 范式）
- 默认：`python -m eeg_channel_game.run_prepare_data --config eeg_channel_game/configs/default.yaml`
- 论文主结果更推荐：`--config eeg_channel_game/configs/perf_multik_braindecode.yaml`:contentReference[oaicite:12]{index=12}
- 数据写入 variant 目录（由 fmin/fmax/time-window/eog 生成），用于预处理消融时避免覆盖:contentReference[oaicite:13]{index=13}。

### 2.2 训练（先跑通闭环，再上性能）
- 先跑通：`python -m eeg_channel_game.run_train --config ... --override project.out_dir=...`:contentReference[oaicite:14]{index=14}
- 注意：`run_train` 若发现 out_dir 已有 checkpoint 会报错；要么换 out_dir，要么 `train.resume=true`，或 `project.overwrite=true`:contentReference[oaicite:15]{index=15}
- 推荐用多被试共享一个 agent（一个 policy/value 网络在多 subjects 的 0train 上联合训练）:contentReference[oaicite:16]{index=16}。

### 2.3 搜索得到子集（不要直接拿训练 best_key 当最终答案）
- 训练完成后，应加载 checkpoint，对每个 subject 重新跑搜索得到该 subject 子集:contentReference[oaicite:17]{index=17}。
- 可做 best-of-N multi-start（只按 0train reward 选最优，不看 1test）:contentReference[oaicite:18]{index=18}。

### 2.4 Pareto 曲线评测（论文必备）
- `python -m eeg_channel_game.run_pareto_curve ... --k 4,6,8,10,... --methods ... --plot`:contentReference[oaicite:19]{index=19}
- 输出包含 `pareto_summary.csv` 与曲线图（mean±std）:contentReference[oaicite:20]{index=20}。

---

## 3) Q1 “失败优先”闭环（每次改动前必须做）

在提出任何新想法前，你必须先对上一次实验做 post-mortem：
1) 读最新的 Pareto 输出与 per-subject 结果（mean/std/q20、每 K 的赢家/输家 subject）。
2) 判断失败类型：
   - **多数 K 下低于 full22**：reward 形状/归一化不对或 teacher 过强导致协同学不到；
   - **方差大**：需要风险敏感目标（q20/LCB）或更稳的多保真评估；
   - **选到前额/眼动相关通道**：伪迹风险没压住（EOG 处理/ArtifactRisk/解释性实验不足）；
   - **搜索不稳定**：MCTS 超参（n_sim/c_puct/噪声）或 leaf bootstrap 不当；
   - **训练快但均值下降**：batched 推理/并行 self-play 可能损伤 MCTS 质量（需消融）。
3) 给出“最小改动集合”（一次只动一个杠杆），并给出验证实验矩阵（baseline + ablation + 对照）。

---

## 4) 创新点必须数学化（强制）
本项目的创新不允许只写“加了模块”，必须写成明确数学目标/约束：
- 科学问题（跨 session domain shift + 多数子集负收益）与目标函数、风险敏感项、无标签 shift 惩罚等应沿用 `docs/paper_draft.md` 的主公式与符号体系:contentReference[oaicite:21]{index=21}:contentReference[oaicite:22]{index=22}。
- 若引入新机制（teacher、policy prior、bootstrap、domain shift penalty），必须说明它如何逼近/优化该目标，而不是“技巧堆叠”:contentReference[oaicite:23]{index=23}。

---

## 5) “漂亮实验”的最低门槛（不达标 = 不算完成）

### 5.1 主结果（必须）
- All subjects × 多个 K（Pareto）上稳定超过强基线（至少包含 lr_weight / MI/Fisher / GA / random_best / full22 等），并注明协议与预算设置:contentReference[oaicite:24]{index=24}。

### 5.2 公平性（预算对齐必须做）
- `ours/uct/random/GA` 至少用 **相同 evaluator 调用数或相同 wall-time** 做预算对齐:contentReference[oaicite:25]{index=25}。

### 5.3 消融（至少 6 组）
- 无 teacher / 仅 policy prior / 仅 teacher loss / 仅 leaf bootstrap
- 无 domain shift / 有 domain shift
- mean vs q20（风险敏感）:contentReference[oaicite:26]{index=26}

### 5.4 可视化（必须）
- Pareto 曲线（kappa + acc）
- Topomap（选中通道分布）
- 解释性：证明“伪迹免疫”（例如选到运动区附近、EOG 相关性显著更低）

---

## 6) 实验配置与复现（工程规则）

### 6.1 优先使用 exp presets
- 长命令行 override 必须收敛到 `eeg_channel_game/configs/exp/train_*.yaml` 或 `eval_pareto_*.yaml`。
- 使用 `base:` 继承组织实验（child 覆盖 parent，深层 merge；循环依赖会报错）:contentReference[oaicite:27]{index=27}。

### 6.2 每次运行必须可追溯
- 产出目录里必须保存：config(json/yaml)、命令、overrides、随机种子、关键超参、运行标签(tag)。
- 任何“论文结论”必须能由一条命令复现（建议把命令写进 results/notes 或 PR 描述）。

---

## 7) 允许联网检索（强制用于 Related Work/SOTA）
当提出“新贡献/新 baseline/新对比协议”时：
- 必须联网搜索近年 Q1/IF>5 的 MI-EEG 通道选择 / 域泛化 / RL/MCTS 相关论文；
- 建一个 `docs/SOTA.md`：记录 paper、venue、year、dataset、protocol、metric、是否可比；
- 若协议不可比必须标注 “non-comparable”。

---

## 8) DONE 的定义（Q1 标准）
只有同时满足以下条件，任务才算完成：
- 严格协议（0train 训练/选择，1test 仅最终评估；EOG 不用于分类）；
- All subjects × 多 K 的 Pareto 主结果齐全；
- 预算对齐 + 消融齐全；
- 创新点数学化写清楚，并与实验结果强对应；
- 图表/表格可直接进论文（“漂亮”）。
