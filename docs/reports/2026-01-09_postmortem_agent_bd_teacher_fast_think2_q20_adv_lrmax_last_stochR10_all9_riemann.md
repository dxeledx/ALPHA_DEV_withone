# Post-mortem Report — runs/agent_bd_teacher_fast_think2_q20_adv_lrmax — eval_adv_lrmax_last_stochR10_all9_riemann

## 1) Protocol & Reproducibility
- Pareto dir: `/home/wjx/workspace/RL/ALPHA/runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/pareto/eval_adv_lrmax_last_stochR10_all9_riemann`
- Checkpoint: `runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/checkpoints/last.pt`
- Subjects: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`
- K list: `[4, 6, 8, 10, 12, 14]`
- Methods: `['ours', 'uct', 'fisher', 'mi', 'lr_weight', 'riemann_ts_lr', 'sfs_l1', 'random_best_l1', 'ga_l1', 'full22']`
- Baseline cache: `/home/wjx/workspace/RL/ALPHA/results/baseline_cache/f4-38_t2-6_eog0/pareto_fbcsp_ded45bb53113.csv`

## 2) Main Findings (high-level)
- Ours peak (by kappa_mean) at **K=14**: kappa_mean=0.5370, acc_mean=0.6528.
- Ours surpasses `full22` on mean only at larger K, but tail robustness (`q20`) still lags due to regressions in a few subjects.
- Small-K regime (K=4/6/8/10): ours < `full22` and often < `ga_l1` (failure signature: compact subset search).

## 3) Key Curves (core methods, cleaner legends)
![kappa mean±std](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/pareto_kappa_mean_core.png)

![kappa q20](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/pareto_kappa_q20_core.png)

![acc mean±std](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/pareto_acc_mean_core.png)

![acc q20](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/pareto_acc_q20_core.png)

## 4) Ours vs Full22 (delta)
![delta vs full22 (kappa)](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/delta_vs_full22_kappa.png)

![delta vs full22 (acc)](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/delta_vs_full22_acc.png)

## 5) Per-subject View (why q20 is not improving)
![subject heatmap dkappa](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/heatmap_subject_dkappa_ours_vs_full22.png)

![subject bars kappa K=14](figures/2026-01-09_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_stochR10_all9_riemann/subject_bars_kappa_K14.png)

## 6) Numbers: Ours vs Full22 (per K)
| k | kappa_mean_ours | kappa_mean_full22 | dkappa_mean | kappa_q20_ours | kappa_q20_full22 | dkappa_q20 | acc_mean_ours | acc_mean_full22 | dacc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.3637 | 0.5273 | -0.1636 | 0.2028 | 0.3824 | -0.1796 | 0.5228 | 0.6454 | -0.1227 |
| 6 | 0.4434 | 0.5273 | -0.0838 | 0.2731 | 0.3824 | -0.1093 | 0.5826 | 0.6454 | -0.0629 |
| 8 | 0.4763 | 0.5273 | -0.0509 | 0.2944 | 0.3824 | -0.0880 | 0.6073 | 0.6454 | -0.0382 |
| 10 | 0.4985 | 0.5273 | -0.0288 | 0.3500 | 0.3824 | -0.0324 | 0.6238 | 0.6454 | -0.0216 |
| 12 | 0.5134 | 0.5273 | -0.0139 | 0.3454 | 0.3824 | -0.0370 | 0.6350 | 0.6454 | -0.0104 |
| 14 | 0.5370 | 0.5273 | 0.0098 | 0.4296 | 0.3824 | 0.0472 | 0.6528 | 0.6454 | 0.0073 |

## 7) Failure-first diagnosis (what to fix next)
- **Tail risk**: a few subjects regress even when mean improves → `q20` stagnates.
- **Small-K gap**: compact subset still fails to beat `full22`/`ga_l1`.
- Next lever (single): reward shaping/normalization toward beating stronger baselines (keep state/model/MCTS fixed).

## 8) Reproduce (eval command as recorded)
```bash
/home/wjx/workspace/RL/ALPHA/eeg_channel_game/run_pareto_curve.py \
  --config \
  eeg_channel_game/configs/exp/eval_pareto_agent_teacher_fast_think2_q20_adv_lrmax_last.yaml \
  --override \
  project.device=cpu \
  --override \
  eval.pareto.methods=[ours,uct,fisher,mi,lr_weight,riemann_ts_lr,sfs_l1,random_best_l1,ga_l1,full22] \
  --override \
  eval.pareto.ours_restarts=10 \
  --override \
  eval.pareto.ours_stochastic=true \
  --override \
  eval.pareto.ours_tau=0.8 \
  --tag \
  eval_adv_lrmax_last_stochR10_all9_riemann \
  --plot
```

