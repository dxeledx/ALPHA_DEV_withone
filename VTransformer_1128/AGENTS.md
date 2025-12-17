# Agent Guidance (GNNT_Spatial_Only)

## 背景与目标
- 目标：在 MOABB 对齐前提下，迭代优化双流模型的空间 GNN 流；当前目录已关闭 Transformer 流，仅保留 GNN。
- 基线来源：`../GNNT/V251122` 是首次 MOABB 对齐版本；最优超参参考 `../最优模型存档/V5.1_MOABB/V5.1`（非对齐时期的最佳配置）。
- 数据：BCI IV 2a，固定种子 42；教师模型沿用旧缓存，但分层划分固定，不存在泄露。

## 现状快照
- 入口 `train_v5_1.py`：默认 `use_spatial=True`、`use_temporal=False`，包含 EA 对齐、PLV 图、KD、增强、内外层 CV。
- Trainer：withinsession 外层 5 折 + 内层 3 折超参搜索；教师 logits 在外层训练集上即时训练/缓存；Crosssession 支持 AdaBN。
- 模型：`LiteGNNT_V3_6`（Slice CNN + GAT + 注意力池化，可学习邻接）；图正则开关 `graph_reg`。
- 增强：`EEGAugmentation`，默认策略 `sign_flip/noise/scale/shift/channel_dropout/time_mask/freq_shift` 组合由配置控制。

## 约束与执行准则
- **超参**：后续实验直接使用 V5.1 最优组合（lr=1e-3，weight_decay=5e-3，label_smoothing=0.05，dropout=0.25，增强 multiplier=4 且策略 noise/scale/shift，KD alpha=0.3、T=3.0），不要再跑内层三折搜索。
- **被试范围**：只在 A02 运行与对比。当前脚本仅有 A01/A01_A02_A06/ALL 预设，需在 `SUBJECT_PRESETS` 中加 `A02_ONLY` 或硬编码 `selected_subjects=["A02"]`。
- **对齐规范**：保持 MOABB 划分、EA 对齐、8–32 Hz 滤波、PLV 阈值 0.8、不触碰教师缓存路径 `../GNNT/V251122/artifacts/teachers`。
- **种子与可重复性**：全局种子 42，不修改 StratifiedKFold/ShuffleSplit 的随机态。

## 立即待办
1) 为 `train_v5_1.py` 增加 A02-only 预设，便于单被试快速跑。  
2) 去掉/跳过内层超参搜索，改为锁定最优超参直训，减少计算。  
3) 跑出「未改进的单 GNN 流在 A02」的基线日志与结果存档到 `results/`，作为后续优化对照。  
4) 若做新改动，更新本 AGENTS 与 README，记录配置/差异点。

以后用中文与用户交流。
代码运行在conda虚拟环境SCDM中。