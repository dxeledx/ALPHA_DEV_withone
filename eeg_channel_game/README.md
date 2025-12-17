# EEG Channel Game (BCI-IV 2a)

用 **AlphaZero/AlphaDev 风格（policy/value + MCTS）**做 **BCI Competition IV 2a 四分类 MI** 的 **EEG 通道选择**。

## 快速开始（建议在 conda 环境 `eeg`）

1) 准备数据（MOABB 读取 BNCI2014_001；epoch=绝对 `[3,6]` 秒；可选 EOG 回归去伪迹）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_prepare_data --config eeg_channel_game/configs/default.yaml
```

默认会把数据写到 `eeg_channel_game/data/{processed,cache}/<variant>/...`（variant 会从 `fmin/fmax/time-window/eog` 自动生成，例如 `f4-40_t3-6_eog1`），用于窗口/预处理消融时避免覆盖。

2) 训练（L0 proxy + MCTS + policy/value net；后续可切 L1/L2）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train --config eeg_channel_game/configs/default.yaml
```

可选：用 **L1(FBCSP+shrinkage LDA)** 做终局 reward（需要在准备数据时不加 `--no-cov` 以生成 `sessionT_cov_fb.npz`）：

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_train --config eeg_channel_game/configs/default.yaml --override evaluator.name=l1_fbcsp
```

3) 评估/可视化（输出选中通道拓扑图、以及性能-通道数曲线的占位接口）

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_eval --config eeg_channel_game/configs/default.yaml --subject 1
```

## L2（深度模型严格评估）

`run_eval` 内置 L2：在 training session 上训练深度模型（内部 train/val split + early stopping），在 evaluation session 上测试，并输出多 seed 的均值/方差/分位数。

```bash
conda run -n eeg --no-capture-output python -m eeg_channel_game.run_eval \
  --config eeg_channel_game/configs/default.yaml \
  --override project.out_dir=runs/exp1 \
  --subject 1 \
  --l2 --l2-model eegnetv4 --l2-epochs 30 --l2-seeds 0,1,2
```

> 注意：L2 会用到 eval session 标签，因此不要在 RL 训练阶段把 L2 当作 reward。

## 目录

- `eeg_channel_game/data/processed/`：每被试 `train/eval` session 的 epochs（清理后只保留 22 EEG 通道）
- `eeg_channel_game/data/cache/`：bandpower、冗余相关矩阵、FoldStats 等缓存
- `runs/`：训练日志、checkpoint、可视化输出

## 常用脚本

- 强基线（full-22，FBCSP + Deep）：`python -m eeg_channel_game.run_baselines --config eeg_channel_game/configs/perf.yaml`
- 固定 K 对比表（示例：K=8/10）：`python -m eeg_channel_game.run_compare_k --config eeg_channel_game/configs/default.yaml --override project.out_dir=runs/exp1 --subject 1 --k 8,10`（可加 `--tag xxx` 防止覆盖）
