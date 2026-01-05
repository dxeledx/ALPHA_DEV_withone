# Experiment Report — 2026-01-04 — Reward `adv_lrmax` (plan)

## 1) Setting
- Protocol: BNCI2014_001 strict `0train` (train/selection) → `1test` (final report only)
- Dataset: `BNCI2014_001` (BCI-IV 2a), 4-class MI, subjects `1..9`
- Variant: `f4-38_t2-6_eog0` (`use_eog_regression=false`)
- Metrics: Cohen’s kappa (primary) + accuracy

## 2) Target (Q1)
- Improve Pareto: **beat strong baselines** (`ga_l1`, `lr_weight`, `random_best_l1`) across K, not only beat `full22` sometimes.
- Improve robustness: reduce tail failures (e.g., better `kappa_q20` / worst-subject).

## 3) Post-mortem (from previous run)
- Best evaluated checkpoint: `runs/agent_bd_teacher_fast_think2_q20/checkpoints/best.pt` (iter=437)
- Pareto eval: `runs/agent_bd_teacher_fast_think2_q20/pareto/eval_best_nsim1024/`
- Key outcomes (kappa_mean):
  - K=12: **ours 0.5478 (best overall)** > `ga_l1 0.5324` > `full22 0.5273`
  - K=14: ours 0.5365 < **`lr_weight 0.5478`** (ours ranked #2)
  - K=4/6/8/10: ours < `full22` and usually < `ga_l1`
- Robustness: even when mean improves, `kappa_q20` is still ≲ `full22` at K=12/14 → gains concentrate in some subjects.
- Failure signature (taxonomy): **domain shift / bottom-subject** + “small-K underperform”.

## 4) This iteration (one primary lever)
- Hypothesis:
  - `delta_full22` reward normalization is insufficient: it aligns scales to `full22`, but does **not** explicitly push the agent to beat strong deterministic baselines (notably `lr_weight@K` at large K).
  - Using a per-K stronger baseline for normalization will push learning toward subsets that outperform strong baselines, not just outperform `full22`.
- Primary lever: reward normalization only.
- What stays fixed:
  - state representation, network, MCTS, and training schedule (including `mcts.n_sim=1024`).
- Minimal change:
  - Add normalize mode `adv_lrmax`:
    - `r_adv(S) = r_raw(S) - max(r_raw(full22), r_raw(lr_weight_topK(|S|)))`
    - Baseline uses only `fold.stats.lr_weight` (0train-only), so no leakage.
- Risks & fallback:
  - Positive rewards may become rarer if baseline is strong → slower learning; mitigate by warmup (keep as-is first, then consider shaping if needed).

## 5) Runs
- Base commit (repo HEAD): `2bc28b9cbd05364955ee6daef6b26a91b1d8f2c8` (workspace has local changes)
- New preset:
  - `eeg_channel_game/configs/exp/train_agent_teacher_fast_think2_q20_adv_lrmax.yaml`

## 6) Commands
Train:
```bash
cd /home/wjx/workspace/RL/ALPHA

PYTHONPATH="$PWD/.vendor" \
MPLCONFIGDIR="$PWD/eeg_channel_game/data/mpl_cache" \
conda run -n rl --no-capture-output python3 -m eeg_channel_game.run_train \
  --config eeg_channel_game/configs/exp/train_agent_teacher_fast_think2_q20_adv_lrmax.yaml \
  --override project.device=cpu
```

Eval (Pareto):
```bash
cd /home/wjx/workspace/RL/ALPHA

PYTHONPATH="$PWD/.vendor" \
MPLCONFIGDIR="$PWD/eeg_channel_game/data/mpl_cache" \
conda run -n rl --no-capture-output python3 -m eeg_channel_game.run_pareto_curve \
  --config eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_adv_lrmax_best.yaml \
  --override project.device=cpu \
  --plot
```

