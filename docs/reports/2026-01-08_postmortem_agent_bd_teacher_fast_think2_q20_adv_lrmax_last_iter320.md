# Post-mortem Report — runs/agent_bd_teacher_fast_think2_q20_adv_lrmax — eval_adv_lrmax_last_iter320

## 1) Protocol & Reproducibility
- Pareto dir: `/home/wjx/workspace/RL/ALPHA/runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/pareto/eval_adv_lrmax_last_iter320`
- Checkpoint: `runs/agent_bd_teacher_fast_think2_q20_adv_lrmax/checkpoints/last.pt`
- Subjects: `[1, 2, 3, 4, 5, 6, 7, 8, 9]`
- K list: `[4, 6, 8, 10, 12, 14]`
- Methods: `['ours', 'uct', 'fisher', 'mi', 'lr_weight', 'sfs_l1', 'random_best_l1', 'ga_l1', 'full22']`
- Baseline cache: `/home/wjx/workspace/RL/ALPHA/results/baseline_cache/f4-38_t2-6_eog0/pareto_fbcsp_ded45bb53113.csv`

## 2) Main Findings (high-level)
- Ours peak (by kappa_mean) at **K=14**: kappa_mean=0.5283, acc_mean=0.6462.
- Ours surpasses `full22` on mean only at larger K, but tail robustness (`q20`) still lags due to regressions in a few subjects.
- Small-K regime (K=4/6/8/10): ours < `full22` and often < `ga_l1` (failure signature: compact subset search).

## 3) Key Curves (core methods, cleaner legends)
![kappa mean±std](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/pareto_kappa_mean_core.png)

![kappa q20](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/pareto_kappa_q20_core.png)

![acc mean±std](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/pareto_acc_mean_core.png)

![acc q20](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/pareto_acc_q20_core.png)

## 4) Ours vs Full22 (delta)
![delta vs full22 (kappa)](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/delta_vs_full22_kappa.png)

![delta vs full22 (acc)](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/delta_vs_full22_acc.png)

## 5) Per-subject View (why q20 is not improving)
![subject heatmap dkappa](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/heatmap_subject_dkappa_ours_vs_full22.png)

![subject bars kappa K=14](figures/2026-01-08_agent_bd_teacher_fast_think2_q20_adv_lrmax_last_iter320/subject_bars_kappa_K14.png)

## 6) Numbers: Ours vs Full22 (per K)
| k | kappa_mean_ours | kappa_mean_full22 | dkappa_mean | kappa_q20_ours | kappa_q20_full22 | dkappa_q20 | acc_mean_ours | acc_mean_full22 | dacc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.4059 | 0.5273 | -0.1214 | 0.2417 | 0.3824 | -0.1407 | 0.5544 | 0.6454 | -0.0910 |
| 6 | 0.4753 | 0.5273 | -0.0520 | 0.3287 | 0.3824 | -0.0537 | 0.6065 | 0.6454 | -0.0390 |
| 8 | 0.4882 | 0.5273 | -0.0391 | 0.3315 | 0.3824 | -0.0509 | 0.6161 | 0.6454 | -0.0293 |
| 10 | 0.4851 | 0.5273 | -0.0422 | 0.3028 | 0.3824 | -0.0796 | 0.6138 | 0.6454 | -0.0316 |
| 12 | 0.5216 | 0.5273 | -0.0057 | 0.3880 | 0.3824 | 0.0056 | 0.6412 | 0.6454 | -0.0042 |
| 14 | 0.5283 | 0.5273 | 0.0010 | 0.3750 | 0.3824 | -0.0074 | 0.6462 | 0.6454 | 0.0008 |

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
  eval.pareto.tag=eval_adv_lrmax_last_iter320 \
  --override \
  mcts.n_sim=1024 \
  --plot
```

