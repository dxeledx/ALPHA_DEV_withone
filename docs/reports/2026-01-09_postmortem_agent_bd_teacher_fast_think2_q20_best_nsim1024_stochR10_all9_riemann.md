# Post-mortem Report — runs/agent_bd_teacher_fast_think2_q20 — eval_best_nsim1024_stochR10_all9_riemann

## 1) Protocol & Reproducibility
- Pareto dir: `/home/wjx/workspace/RL/ALPHA/runs/agent_bd_teacher_fast_think2_q20/pareto/eval_best_nsim1024_stochR10_all9_riemann`
- Checkpoint: `runs/agent_bd_teacher_fast_think2_q20/checkpoints/best.pt`
- Subjects: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`
- K list: `[4, 6, 8, 10, 12, 14]`
- Methods: `['ours', 'uct', 'fisher', 'mi', 'lr_weight', 'riemann_ts_lr', 'sfs_l1', 'random_best_l1', 'ga_l1', 'full22']`
- Baseline cache: `/home/wjx/workspace/RL/ALPHA/results/baseline_cache/f4-38_t2-6_eog0/pareto_fbcsp_ded45bb53113.csv`

## 2) Main Findings (high-level)
- Ours peak (by kappa_mean) at **K=12**: kappa_mean=0.5499, acc_mean=0.6624.
- Ours surpasses `full22` on mean only at larger K, but tail robustness (`q20`) still lags due to regressions in a few subjects.
- Small-K regime (K=4/6/8/10): ours < `full22` and often < `ga_l1` (failure signature: compact subset search).

## 3) Key Curves (core methods, cleaner legends)
![kappa mean±std](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/pareto_kappa_mean_core.png)

![kappa q20](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/pareto_kappa_q20_core.png)

![acc mean±std](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/pareto_acc_mean_core.png)

![acc q20](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/pareto_acc_q20_core.png)

## 4) Ours vs Full22 (delta)
![delta vs full22 (kappa)](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/delta_vs_full22_kappa.png)

![delta vs full22 (acc)](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/delta_vs_full22_acc.png)

## 5) Per-subject View (why q20 is not improving)
![subject heatmap dkappa](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/heatmap_subject_dkappa_ours_vs_full22.png)

![subject bars kappa K=12](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_best_nsim1024_stochR10_all9_riemann/subject_bars_kappa_K12.png)

## 6) Numbers: Ours vs Full22 (per K)
| k | kappa_mean_ours | kappa_mean_full22 | dkappa_mean | kappa_q20_ours | kappa_q20_full22 | dkappa_q20 | acc_mean_ours | acc_mean_full22 | dacc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.3868 | 0.5273 | -0.1404 | 0.2574 | 0.3824 | -0.1250 | 0.5401 | 0.6454 | -0.1053 |
| 6 | 0.4794 | 0.5273 | -0.0478 | 0.3102 | 0.3824 | -0.0722 | 0.6096 | 0.6454 | -0.0359 |
| 8 | 0.4943 | 0.5273 | -0.0329 | 0.3657 | 0.3824 | -0.0167 | 0.6208 | 0.6454 | -0.0247 |
| 10 | 0.5103 | 0.5273 | -0.0170 | 0.3991 | 0.3824 | 0.0167 | 0.6327 | 0.6454 | -0.0127 |
| 12 | 0.5499 | 0.5273 | 0.0226 | 0.4204 | 0.3824 | 0.0380 | 0.6624 | 0.6454 | 0.0170 |
| 14 | 0.5180 | 0.5273 | -0.0093 | 0.3083 | 0.3824 | -0.0741 | 0.6385 | 0.6454 | -0.0069 |

## 7) Failure-first diagnosis (what to fix next)
- **Tail risk**: a few subjects regress even when mean improves → `q20` stagnates.
- **Small-K gap**: compact subset still fails to beat `full22`/`ga_l1`.
- Next lever (single): reward shaping/normalization toward beating stronger baselines (keep state/model/MCTS fixed).

## 8) Reproduce (eval command as recorded)
```bash
/home/wjx/workspace/RL/ALPHA/eeg_channel_game/run_pareto_curve.py \
  --config \
  eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_best_stochR10_riemann.yaml \
  --override \
  project.device=cpu \
  --override \
  mcts.n_sim=1024 \
  --tag \
  eval_best_nsim1024_stochR10_all9_riemann \
  --plot
```

