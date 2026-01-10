# Post-mortem Report — runs/agent_bd_teacher_fast_think2_q20_adv_lrmax — eval_adv_lrmax_best_stochR10_all9_riemann

## 1) Protocol & Reproducibility
- Pareto dir: `/home/wjx/workspace/RL/ALPHA/runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/pareto/eval_adv_lrmax_best_stochR10_all9_riemann`
- Checkpoint: `runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/checkpoints/best.pt`
- Subjects: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`
- K list: `[4, 6, 8, 10, 12, 14]`
- Methods: `['ours', 'uct', 'fisher', 'mi', 'lr_weight', 'riemann_ts_lr', 'sfs_l1', 'random_best_l1', 'ga_l1', 'full22']`
- Baseline cache: `/home/wjx/workspace/RL/ALPHA/results/baseline_cache/f4-38_t2-6_eog0/pareto_fbcsp_ded45bb53113.csv`

## 2) Main Findings (high-level)
- Ours peak (by kappa_mean) at **K=14**: kappa_mean=0.5520, acc_mean=0.6640.
- Ours surpasses `full22` on mean only at larger K, but tail robustness (`q20`) still lags due to regressions in a few subjects.
- Small-K regime (K=4/6/8/10): ours < `full22` and often < `ga_l1` (failure signature: compact subset search).

## 3) Key Curves (core methods, cleaner legends)
![kappa mean±std](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/pareto_kappa_mean_core.png)

![kappa q20](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/pareto_kappa_q20_core.png)

![acc mean±std](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/pareto_acc_mean_core.png)

![acc q20](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/pareto_acc_q20_core.png)

## 4) Ours vs Full22 (delta)
![delta vs full22 (kappa)](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/delta_vs_full22_kappa.png)

![delta vs full22 (acc)](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/delta_vs_full22_acc.png)

## 5) Per-subject View (why q20 is not improving)
![subject heatmap dkappa](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/heatmap_subject_dkappa_ours_vs_full22.png)

![subject bars kappa K=14](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_best_stochR10_all9_riemann/subject_bars_kappa_K14.png)

## 6) Numbers: Ours vs Full22 (per K)
| k | kappa_mean_ours | kappa_mean_full22 | dkappa_mean | kappa_q20_ours | kappa_q20_full22 | dkappa_q20 | acc_mean_ours | acc_mean_full22 | dacc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.4126 | 0.5273 | -0.1147 | 0.2704 | 0.3824 | -0.1120 | 0.5594 | 0.6454 | -0.0860 |
| 6 | 0.4727 | 0.5273 | -0.0545 | 0.2593 | 0.3824 | -0.1231 | 0.6046 | 0.6454 | -0.0409 |
| 8 | 0.4697 | 0.5273 | -0.0576 | 0.2787 | 0.3824 | -0.1037 | 0.6022 | 0.6454 | -0.0432 |
| 10 | 0.5237 | 0.5273 | -0.0036 | 0.3963 | 0.3824 | 0.0139 | 0.6427 | 0.6454 | -0.0027 |
| 12 | 0.5165 | 0.5273 | -0.0108 | 0.3741 | 0.3824 | -0.0083 | 0.6373 | 0.6454 | -0.0081 |
| 14 | 0.5520 | 0.5273 | 0.0247 | 0.4176 | 0.3824 | 0.0352 | 0.6640 | 0.6454 | 0.0185 |

## 7) Failure-first diagnosis (what to fix next)
- **Tail risk**: a few subjects regress even when mean improves → `q20` stagnates.
- **Small-K gap**: compact subset still fails to beat `full22`/`ga_l1`.
- Next lever (single): reward shaping/normalization toward beating stronger baselines (keep state/model/MCTS fixed).

## 8) Reproduce (eval command as recorded)
```bash
/home/wjx/workspace/RL/ALPHA/eeg_channel_game/run_pareto_curve.py \
  --config \
  eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_adv_lrmax_best_stochR10_riemann.yaml \
  --override \
  project.device=cpu \
  --tag \
  eval_adv_lrmax_best_stochR10_all9_riemann \
  --plot
```

