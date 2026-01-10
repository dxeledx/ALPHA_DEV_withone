# SOTA / Related Work (MI-EEG Channel Selection & Graph/Connectivity)

This file is a **living** literature log for MI-EEG channel selection (focus: **BCI Competition IV-2a / BNCI2014_001**) with special attention to **graph / edge / connectivity** modeling and **protocol comparability**.

> Our project’s strict protocol: **train/selection on 0train only; 1test only for final report** (no label leakage).
> When a paper uses different splits (e.g., within-session CV, different datasets, binary tasks), mark it **non-comparable**.

## Graph / Connectivity-aware channel selection (core to our “edge” discussion)

| Paper | Venue | Year | Edge/Graph signal | Dataset | Tasks | Protocol | Metrics | Comparable to ours? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sun et al., “Graph Convolution Neural Network Based End-to-End Channel Selection and Classification for Motor Imagery BCI” (DOI: 10.1109/TII.2022.3227736) | IEEE TII | 2023 | Learnable adjacency (GCN), edge-/aggregation-selection | BCICIV-2a + others | multi-class MI | end-to-end supervised | acc (+ robustness) | **Partly** | Check exact split (within-session CV vs session transfer). Uses learned adjacency to represent inter-channel relations. |
| Varsehi & Firoozabadi, “... using Granger causality” (DOI: 10.1016/j.neunet.2020.11.002) | Neural Networks | 2020 | Directed GC (MVGC) graph + thresholding | PhysioNet MI | binary MI | subject-specific | acc/sens/spec | **No** | Different dataset + binary tasks; still a strong example of *directed edges* guiding channel selection. |
| Liang et al., “Novel channel selection model based on GCN for motor imagery” (PMCID: PMC10542066) | Cogn Neurodyn | 2023 | Connectivity graph + node classification (GCN-CS) | 3 MI datasets | MI | supervised | acc | **Unknown** | Check whether BCICIV-2a is included and what split is used. |

## Covariance / Riemannian geometry (often implicitly “edge-aware” via covariance matrices)

| Paper | Venue | Year | Key idea | Dataset | Comparable? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Barachant et al., “Multiclass BCI classification by Riemannian geometry” (PMID: 22010143) | IEEE TBME | 2012 | Covariance SPD manifold; MDRM / tangent-space | BCI Comp IV-2a | **Partly** | Not channel selection per se, but covariance encodes inter-channel relations; good baseline family. |

## TODO (to fill before paper submission)
- Verify for each paper: preprocessing, time window, filter banks, subject split, session split, and whether results are directly comparable.
- Add recent Q1 works on MI-EEG **graph-based** modeling (ST-GCN variants, MI-based adjacency, attention graph, etc.) and mark comparability.

