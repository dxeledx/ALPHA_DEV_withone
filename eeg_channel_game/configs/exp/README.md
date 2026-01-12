# Experiment Presets (`eeg_channel_game/configs/exp/`)

这个目录的目标：把“长到看不懂的命令行 `--override ...`”收敛到少量可复现的 YAML preset。

## 1) YAML 继承（`base:`）

`eeg_channel_game/utils/config.py` 支持在 YAML 顶层写：

```yaml
base: ../perf_multik_braindecode.yaml
```

或多个 base（后面的覆盖前面的）：

```yaml
base:
  - ../perf_multik_braindecode.yaml
  - train_agent_teacher_v1.yaml
```

解析规则：
- base 路径相对“当前 YAML 文件所在目录”解析
- 合并方式是深层字典 merge（child 覆盖 parent）
- 有循环依赖会报错

## 2) 文件命名约定

- `train_*.yaml`：训练 preset（给 `run_train` 用）
- `eval_pareto_*.yaml`：全被试×多 K×多方法 Pareto 评测 preset（给 `run_pareto_curve` 用）

## 3) 现有 preset 一览（你应该选哪个）

### 训练

- `train_agent_teacher_fast_think2_q20_adv_lrmax_eegonly.yaml`（推荐：后续“纯 EEG 主结果”用这个）
  - 目的：在 `adv_lrmax` 基础上切换到 **纯 EEG 模式**（`data.include_eog=false`，不加载 EOG 通道）
  - 说明：用于避免“EOG 参与通道选择/分类”的争议；EOG 相关消融可在后续单独补

- `train_agent_teacher_fast_think2_q20_adv_lrmax_ds_eta0p2_arena.yaml`
  - 目的：在 `ds_eta0p2` 设置上加入 **AlphaZero-style accept/reject gate**，防止 teacher/prior 退火后训练漂移（越训越差）
  - 说明：该 gate 只用 0train（+可选无标签 eval 特征的 domain shift penalty），不会用 1test 标签；属于“训练稳定性”改进

- `train_agent_full_fast_v1.yaml`
  - 目的：验证“并行 self-play + batched MCTS”能否显著加速（较激进：`infer_batch_size=64`）
  - 风险：`infer_batch_size` 太大可能降低 MCTS 质量（性能波动/均值下降）

- `train_agent_full_fast_v1_long.yaml`
  - 目的：同上，但把 `num_iters` 拉长到 400（用于更充分训练）

- `train_agent_teacher_v1.yaml`
  - 目的：引入 teacher（`lr_weight`）+ leaf bootstrap + policy prior + teacher KL + domain-shift penalty
  - 适用：你当前任务“多数子集负收益 + 小 K 协同难 + 跨 session shift”

- `train_agent_teacher_think2.yaml`
  - 目的：在 teacher_v1 上加入 CTM 风格 “internal ticks”（`net.think_steps=2`）
  - 含义：同一状态 token 上让网络“多想几步”再输出 policy/value（提升决策质量的方向）

- `train_agent_teacher_fast_think2_q20.yaml`（推荐你当前先跑这个）
  - 目的：在 teacher_think2 上：
    - L1 reward 改成风险敏感：`robust_mode=q20`
    - 并行 self-play：`train.selfplay.num_workers=4`
    - batched MCTS 取折中：`infer_batch_size=16`（比 64 更稳）

### 评测（Pareto）

- `eval_pareto_agent_teacher_fast_think2_q20_adv_lrmax_best_eegonly.yaml` / `..._last_eegonly.yaml`
  - 目的：评测对应 `*_eegonly` 训练 run 的 `best/last` checkpoint（全被试×多 K×多方法）

- `eval_pareto_agent_full_fast_v1_last.yaml`
  - 目的：评测 `runs/agent_full_fast_v1` 的 `last` checkpoint
  - 默认会把基线缓存到 `results/baseline_cache/<variant>/...`，下次评测同样设置直接复用

- `eval_pareto_agent_teacher_fast_think2_q20_last.yaml`
  - 目的：评测 `runs/agent_bd_teacher_fast_think2_q20` 的 `last` checkpoint

- `eval_pareto_agent_teacher_fast_think2_q20_last_riemann.yaml`
  - 目的：同上，但额外加入一个 **Riemannian** 基线：`riemann_ts_lr`（covariance → tangent space → LogReg 权重得到 topK）

## 4) 如何运行（最短命令）

### 训练

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train \
  --config eeg_channel_game/configs/exp/train_agent_teacher_fast_think2_q20.yaml
```

### 评测（全被试×全K×全方法）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_pareto_curve \
  --config eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_last.yaml
```

## 5) 结果目录里会记录什么

- 训练：`runs/<out_dir>/config.{json,yaml}`（包含 `_meta.argv/config_path/overrides`）
- 评测：`runs/<out_dir>/pareto/<tag>/run_config.{json,yaml}` + `command.txt`

## 6) 仍然可以用 CLI override（少量、临时）

例如临时换输出目录：

```bash
... --override project.out_dir=runs/tmp_debug
```

建议：长期实验请新建一个 `train_*.yaml`/`eval_*.yaml`，避免命令散落在 shell history。
