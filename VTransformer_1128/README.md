# VTransformer（LiteTransformer V5.1 单 Transformer 流）

基于 GNNT_Spatial_Only 精简出的纯 Transformer 流版本：彻底移除 GNN 分支，仅保留 EA + 滤波 + ALBERT 式权重共享 Transformer 与分类头，方便对双流架构的时间分支单独迭代。

## 目录速览
- `train_v5_1.py`：入口脚本，按 MOABB 标准执行 Withinsession 5-fold 与 Crosssession LOSO；默认仅启用 Transformer 流。
- `trainer_v5_1.py`：训练/评估逻辑，包含 EA + 8–32Hz 滤波、外层 CV、KD、AdaBN（仅 BN 刷新）、日志与摘要输出。
- `dataloader_v5_1.py`：BCI IV 2a `.mat` 数据读取（未滤波原始），保留旧版切片接口但本版本只用原始段。
- `models_lite_gnnt_v5_1.py`：仅保留 LiteTransformerStream 与包装分类器（类名兼容 `LiteGNNT_V3_6`）。
- `pipelines/augmentation.py` & `pipelines/teacher_utils.py`：数据增强与 4 种教师视角；教师缓存默认指向 `./artifacts/teachers/`。
- `artifacts/`、`logs/`、`results/`：教师 logits 缓存、训练日志、结果汇总输出位置。

## 数据与依赖
- 数据：BCI Competition IV 2a。
  - 若你有本地 `.mat`（如 `A01T.mat/A01E.mat`），可通过 `--data_path` 指定目录使用旧版加载方式。
  - 若本地 `.mat` 不存在，会自动改用 **MOABB** 的 `BNCI2014_001`（数据下载到 MOABB 默认路径即可）。
- 主要依赖：`torch`、`numpy`、`scipy`、`scikit-learn`、`tqdm`，可选 `PyEMD`（CEMD 教师）。
- 默认全局种子 `42`，EA 对齐 + 8–32 Hz Butterworth 滤波。

## 运行方式
```bash
# 典型用法（需保证 CUDA 可用）
python train_v5_1.py --subject_set A01 --data_path ../BCIIV2a_mat
```
- `--subject_set` 支持 `A01` / `A02_ONLY` / `A01_A02_A06` / `ALL`，也可用 `--subjects "A01 A02"` 自定义。
- 输出：日志 `logs/train_v5_1_<run_tag>_<timestamp>.log`，结果摘要 `results/V5.1_moabb_summary_<run_tag>_<subject_set>_<timestamp>.txt`。

## 流程摘要（当前实现）
1) 数据预处理：T/E 会话分别 EA 对齐；训练/验证/测试前统一 8–32Hz 滤波。  
2) Withinsession：外层 5 折；固定超参（无内层搜索）；教师 logits 在外层训练集上即训即用；支持多倍增强 + KD。  
3) Crosssession：训练集=非测试会话；早停验证 20%；测试前可选 AdaBN 刷新 BN 统计量。  
4) 模型：仅 Transformer 流（ALBERT 式权重共享 2 层）+ 线性分类头，无图构建/可学习邻接。

## 约定
- 采用固定超参：`lr=1e-3`，`weight_decay=1e-3`，`label_smoothing=0.0`，`dropout=0.25`，增强 `strategies=["noise","scale","shift"]`、`multiplier=4`，KD `alpha=0.3`、`T=3.0`。
- 教师模型沿用缓存（如有变更分层/数据路径，请关闭 `reuse_cached` 以重新计算 logits）。
