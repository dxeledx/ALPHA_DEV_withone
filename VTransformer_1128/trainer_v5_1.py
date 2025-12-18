# -*- coding: utf-8 -*-
"""LiteTransformer V5.1 训练/评估工具 (Transformer 单流版本)。"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import butter, filtfilt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from euclidean_alignment import EuclideanAlignment
from pipelines.teacher_utils import (
    teacher_csp_lda,
    teacher_riemannian_lr,
    teacher_cemd_lr,
    teacher_stft_lr,
)


# ============================================================
# BatchNorm 自适应
# ============================================================


def adapt_batch_norm(model, data_loader, device="cuda", num_batches=None, verbose=False):
    """AdaBN - 在目标分布上刷新 BN 统计量。"""

    model.eval()
    bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for bn in bn_layers:
        bn.train()

    if not bn_layers:
        if verbose:
            print("  ⚠️ 模型中没有BatchNorm层，跳过AdaBN")
        return

    with torch.no_grad():
        batch_count = 0
        for batch in data_loader:
            if len(batch) == 3:
                X_raw, _, _ = batch
            else:
                X_raw, _ = batch
            X_raw = X_raw.to(device)
            _ = model(X_raw)

            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break

    for bn in bn_layers:
        bn.eval()

    if verbose:
        print(f"  ✅ AdaBN (动量更新): 使用 {batch_count} 个batch更新 {len(bn_layers)} 个BN层")


def _set_internal_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_filter_kwargs(filter_cfg=None):
    defaults = {"fs": 250, "t_low": 8.0, "t_high": 32.0, "filter_order": 4}
    if filter_cfg:
        defaults.update({k: filter_cfg[k] for k in defaults if k in filter_cfg})
    return defaults


def _filter_raw(X, filter_cfg=None):
    """对原始EEG执行8-32Hz滤波，返回 (N,1,C,T) 张量。"""
    cfg = _resolve_filter_kwargs(filter_cfg)
    b, a = butter(cfg["filter_order"], [cfg["t_low"], cfg["t_high"]], btype="band", fs=cfg["fs"])
    X_filtered = np.zeros_like(X, dtype=np.float64)
    for trial in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_filtered[trial, ch, :] = filtfilt(b, a, X[trial, ch, :])
    X_filtered = X_filtered[:, np.newaxis, :, :]
    return X_filtered


def _augment_dataset_with_teacher_logits(X_raw, y, y_teacher, augmenter,
                                         multiplier, strategies, filter_cfg):
    if augmenter is None or multiplier <= 1:
        X_filt = _filter_raw(X_raw, filter_cfg)
        return X_filt, y, y_teacher

    X_raw_list, y_list = [], []
    y_teacher_list = [] if y_teacher is not None else None

    for i in range(len(X_raw)):
        X_raw_list.append(X_raw[i])
        y_list.append(y[i])
        if y_teacher is not None:
            y_teacher_list.append(y_teacher[i])

        for _ in range(multiplier - 1):
            X_aug = augmenter.apply_augmentation(X_raw[i:i+1], strategies)
            X_raw_list.append(X_aug[0])
            y_list.append(y[i])
            if y_teacher is not None:
                y_teacher_list.append(y_teacher[i])

    X_aug = np.array(X_raw_list)
    y_aug = np.array(y_list)
    y_teacher_aug = np.array(y_teacher_list) if y_teacher is not None else None

    X_filt = _filter_raw(X_aug, filter_cfg)
    return X_filt, y_aug, y_teacher_aug


def _prepare_training_loader(X, y, teacher_logits, batch_size, filter_cfg,
                             augmenter=None, aug_multiplier=1, aug_strategies=None,
                             shuffle=True):
    if augmenter is not None and aug_multiplier > 1 and aug_strategies:
        Xf, y_final, y_teacher_final = _augment_dataset_with_teacher_logits(
            X, y, teacher_logits, augmenter, aug_multiplier, aug_strategies, filter_cfg
        )
    else:
        Xf = _filter_raw(X, filter_cfg)
        y_final = y
        y_teacher_final = teacher_logits

    tensors = [
        torch.FloatTensor(Xf),  # (N,1,C,T)
        torch.LongTensor(y_final),
    ]
    if y_teacher_final is not None:
        tensors.append(torch.FloatTensor(y_teacher_final))

    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4,
    )


def _build_dataloader_filtered(X, y, batch_size, filter_cfg, shuffle):
    Xf = _filter_raw(X, filter_cfg)
    dataset = TensorDataset(
        torch.FloatTensor(Xf),  # (N,1,C,T)
        torch.LongTensor(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4,
    )


def _get_outer_teacher_dir(cache_dir, subject_id, session_label, outer_idx):
    fold_dir = Path(cache_dir) / subject_id / session_label / f"outer_{outer_idx+1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return fold_dir


def _train_or_load_outer_teacher_logits(subject_id,
                                        session_label,
                                        outer_idx,
                                        train_indices,
                                        X_train_outer,
                                        y_train_outer,
                                        teacher_cfg):
    fold_dir = _get_outer_teacher_dir(teacher_cfg["cache_dir"], subject_id, session_label, outer_idx)
    logits_file = fold_dir / "teacher_logits_avg.npz"
    if logits_file.exists():
        data = np.load(logits_file)
        cached_idx = data.get("indices")
        reuse_cached = teacher_cfg.get("reuse_cached", False)
        if reuse_cached:
            if cached_idx is not None and not np.array_equal(cached_idx, train_indices):
                print(f"  ⚠️ 使用缓存Teacher (Subject {subject_id}{session_label} Outer{outer_idx+1}) | 索引未校验匹配，存在折分不一致风险")
            else:
                print(f"  🔁 使用缓存Teacher (Subject {subject_id}{session_label} Outer{outer_idx+1}) | 忽略索引检查")
            return data["logits"]
        if cached_idx is not None and np.array_equal(cached_idx, train_indices):
            print(f"  🔁 使用缓存Teacher (Subject {subject_id}{session_label} Outer{outer_idx+1})")
            return data["logits"]

    print(f"  🧑‍🏫 训练教师组 (Subject {subject_id}{session_label} Outer{outer_idx+1})")
    teacher_funcs = [
        ("csp_lda", teacher_csp_lda),
        ("riemannian_lr", teacher_riemannian_lr),
        ("cemd_lr", teacher_cemd_lr),
        ("stft_lr", teacher_stft_lr),
    ]
    logits_list = []
    teacher_metrics = []
    for name, fn in teacher_funcs:
        print(f"    ▶ {name}")
        if name in {"csp_lda", "cemd_lr", "stft_lr"}:
            logits = fn(X_train_outer, y_train_outer, fs=teacher_cfg.get("fs", 250))
        else:
            logits = fn(X_train_outer, y_train_outer)
        acc = float(np.mean(np.argmax(logits, axis=1) == y_train_outer))
        print(f"      Acc={acc*100:.2f}%")
        teacher_metrics.append({"name": name, "acc": acc})
        logits_list.append(logits)

    avg_logits = np.mean(np.stack(logits_list, axis=0), axis=0)
    ensemble_acc = float(np.mean(np.argmax(avg_logits, axis=1) == y_train_outer))
    print(f"    ⭐ Ensemble Acc={ensemble_acc*100:.2f}%")
    np.savez_compressed(logits_file, logits=avg_logits, indices=train_indices)

    meta = {
        "subject": subject_id,
        "session": session_label,
        "outer_fold": outer_idx + 1,
        "train_size": int(len(X_train_outer)),
        "teachers": [name for name, _ in teacher_funcs],
        "teacher_metrics": teacher_metrics,
        "ensemble_accuracy": ensemble_acc,
    }
    with open(fold_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    info_path = fold_dir / "teacher_logits_info.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Teacher Ensemble - Subject {subject_id} Session {session_label} Outer {outer_idx+1}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train samples: {len(X_train_outer)}\n")
        f.write("教师准确率:\n")
        for tm in teacher_metrics:
            f.write(f"  {tm['name']}: {tm['acc']*100:.2f}%\n")
        f.write(f"平均准确率: {ensemble_acc*100:.2f}%\n")

    return avg_logits


def _evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                X_raw, y_batch, _ = batch
            else:
                X_raw, y_batch = batch
            X_raw = X_raw.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_raw)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
    return np.array(labels), np.array(preds)


def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"acc": acc, "f1": f1, "kappa": kappa}


def _train_with_validation(model_factory,
                           train_loader,
                           val_loader,
                           device,
                           lr=1e-3,
                           weight_decay=5e-3,
                           max_epochs=300,
                           label_smoothing=0.05,
                           early_stop_patience=32,
                           log_prefix="",
                           seed=None,
                           kd_alpha=0.0,
                           kd_temperature=1.0):
    if seed is not None:
        _set_internal_seed(seed)

    model = model_factory().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_metrics_acc = {"acc": 0.0, "f1": 0.0, "kappa": 0.0}
    best_state_acc = None
    best_epoch_acc = 0

    best_val_loss = float("inf")
    best_metrics_loss = {"acc": 0.0, "f1": 0.0, "kappa": 0.0}
    best_state_loss = None
    best_epoch_loss = 0

    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            if len(batch) == 3:
                X_raw, y_batch, y_teacher_batch = batch
                y_teacher_batch = y_teacher_batch.to(device)
            else:
                X_raw, y_batch = batch
                y_teacher_batch = None

            X_raw = X_raw.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_raw)
            loss_ce = F.cross_entropy(logits, y_batch, label_smoothing=label_smoothing)

            if y_teacher_batch is not None and kd_alpha > 0:
                T = kd_temperature
                loss_kd = F.kl_div(
                    F.log_softmax(logits / T, dim=-1),
                    F.softmax(y_teacher_batch / T, dim=-1),
                    reduction='batchmean'
                ) * (T * T)
                loss = (1 - kd_alpha) * loss_ce + kd_alpha * loss_kd
            else:
                loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds) if train_labels else 0.0

        # 验证集指标与loss
        model.eval()
        val_preds, val_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                if len(val_batch) == 3:
                    X_raw_val, y_val_batch, _ = val_batch
                else:
                    X_raw_val, y_val_batch = val_batch
                X_raw_val = X_raw_val.to(device)
                y_val_batch = y_val_batch.to(device)
                logits_val = model(X_raw_val)
                loss_val = F.cross_entropy(logits_val, y_val_batch, label_smoothing=label_smoothing)
                val_loss_total += loss_val.item()
                val_preds.extend(torch.argmax(logits_val, dim=1).cpu().numpy())
                val_labels.extend(y_val_batch.cpu().numpy())
        val_loss_avg = val_loss_total / max(1, len(val_loader))
        val_metrics = _compute_metrics(np.array(val_labels), np.array(val_preds))

        avg_loss = epoch_loss / max(1, len(train_loader))
        if epoch == 0:
            print(f"{log_prefix}Epoch | TrLoss | TrAcc | ValLoss | ValAcc | ValF1 | ValKappa | Flag")
        flag = []
        if val_metrics["acc"] >= best_metrics_acc["acc"]:
            flag.append("A")
        if val_loss_avg <= best_val_loss:
            flag.append("L")
        flag_str = "/".join(flag)
        print(
            f"{log_prefix}{epoch+1:03d} | {avg_loss:.4f} | {train_acc:.4f} | "
            f"{val_loss_avg:.4f} | {val_metrics['acc']:.4f} | {val_metrics['f1']:.4f} | "
            f"{val_metrics['kappa']:.4f} | {flag_str}"
        )

        if val_metrics["acc"] > best_metrics_acc["acc"]:
            best_metrics_acc = val_metrics
            best_state_acc = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch_acc = epoch + 1
            print(
                f"{log_prefix}⭐ 新最佳 (Val Acc) Epoch {best_epoch_acc}: "
                f"Train Acc={train_acc:.4f}, Val Acc={val_metrics['acc']:.4f}, "
                f"Val F1={val_metrics['f1']:.4f}, Val Kappa={val_metrics['kappa']:.4f}"
            )
        else:
            patience_counter += 1

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_metrics_loss = val_metrics
            best_state_loss = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch_loss = epoch + 1
            print(
                f"{log_prefix}⭐ 新最佳 (Val Loss) Epoch {best_epoch_loss}: "
                f"Val Loss={val_loss_avg:.4f}, Val Acc={val_metrics['acc']:.4f}"
            )

        if patience_counter >= early_stop_patience:
            print(f"{log_prefix}⏹ 早停于 Epoch {epoch+1}, 最佳验证准确率: {best_metrics_acc['acc']:.4f}")
            break

    if best_state_acc is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_acc.items()})

    print(
        f"{log_prefix}✅ 训练完成 | Best(Acc) Epoch={best_epoch_acc}, "
        f"Val Acc={best_metrics_acc['acc']:.4f}, F1={best_metrics_acc['f1']:.4f}, "
        f"Kappa={best_metrics_acc['kappa']:.4f} | "
        f"Best(Loss) Epoch={best_epoch_loss}, Val Loss={best_val_loss:.4f}"
    )

    return model, best_metrics_acc, best_epoch_acc, best_metrics_loss, best_epoch_loss, best_state_loss


def evaluate_within_session(model_factory_fn,
                            subject_id,
                            session_label,
                            X_session_raw,
                            y_session,
                            train_epochs=300,
                            batch_size=64,
                            lr=1e-3,
                            weight_decay=5e-3,
                            device="cuda",
                            random_state=42,
                            label_smoothing=0.05,
                            early_stop_patience=28,
                            filter_cfg=None,
                            outer_folds=5,
                            augmenter=None,
                            aug_cfg=None,
                            kd_cfg=None,
                            teacher_cfg=None):
    aug_cfg = aug_cfg or {"multiplier": 1, "strategies": []}
    kd_cfg = kd_cfg or {"alpha": 0.0, "temperature": 1.0}
    teacher_cfg = teacher_cfg or {"cache_dir": "artifacts/teachers", "fs": 250}

    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    test_metrics_per_fold_acc = []
    test_metrics_per_fold_loss = []
    all_test_labels_acc, all_test_preds_acc = [], []
    all_test_labels_loss, all_test_preds_loss = [], []
    hyperparam_history = []

    for outer_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_session_raw, y_session)):
        print(f"\n{'='*80}")
        print(f"Withinsession Session {session_label} - Outer Fold {outer_idx+1}/{outer_folds}")
        print(f"{'='*80}")

        X_train_outer_raw = X_session_raw[train_idx]
        y_train_outer = y_session[train_idx]
        X_test_outer_raw = X_session_raw[test_idx]
        y_test_outer = y_session[test_idx]

        ea = EuclideanAlignment()
        ea.fit(X_train_outer_raw)
        X_train_outer = ea.transform(X_train_outer_raw)
        X_test_outer = ea.transform(X_test_outer_raw)

        teacher_logits_outer = _train_or_load_outer_teacher_logits(
            subject_id,
            session_label,
            outer_idx,
            train_idx,
            X_train_outer,
            y_train_outer,
            teacher_cfg,
        )

        best_hp = {"lr": lr, "weight_decay": weight_decay, "label_smoothing": label_smoothing}

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1,
                                     random_state=random_state + outer_idx)
        train_final_idx, val_final_idx = next(sss.split(X_train_outer, y_train_outer))
        X_train_final = X_train_outer[train_final_idx]
        y_train_final = y_train_outer[train_final_idx]
        X_val_final = X_train_outer[val_final_idx]
        y_val_final = y_train_outer[val_final_idx]
        y_teacher_final = teacher_logits_outer[train_final_idx]

        model_factory = model_factory_fn()

        train_loader = _prepare_training_loader(
            X_train_final,
            y_train_final,
            y_teacher_final,
            batch_size,
            filter_cfg,
            augmenter=augmenter,
            aug_multiplier=aug_cfg.get("multiplier", 1),
            aug_strategies=aug_cfg.get("strategies", []),
            shuffle=True,
        )
        val_loader = _build_dataloader_filtered(
            X_val_final, y_val_final, batch_size, filter_cfg, shuffle=False
        )
        test_loader = _build_dataloader_filtered(
            X_test_outer, y_test_outer, batch_size, filter_cfg, shuffle=False
        )

        final_lr = best_hp.get("lr", lr)
        final_wd = best_hp.get("weight_decay", weight_decay)
        final_ls = best_hp.get("label_smoothing", label_smoothing)

        (model,
         best_metrics_acc,
         best_epoch_acc,
         best_metrics_loss,
         best_epoch_loss,
         best_state_loss) = _train_with_validation(
            model_factory,
            train_loader,
            val_loader,
            device=device,
            lr=final_lr,
            weight_decay=final_wd,
            max_epochs=train_epochs,
            label_smoothing=final_ls,
            early_stop_patience=early_stop_patience,
            log_prefix=f"[Session {session_label} Outer{outer_idx+1} Final] ",
            seed=random_state + outer_idx,
            kd_alpha=kd_cfg.get("alpha", 0.0),
            kd_temperature=kd_cfg.get("temperature", 1.0),
        )

        y_test_true, y_test_pred = _evaluate_model(model, test_loader, device=device)
        outer_metrics_acc = _compute_metrics(y_test_true, y_test_pred)

        if best_state_loss is not None:
            model_loss = model_factory()
            model_loss.load_state_dict({k: v.to(device) for k, v in best_state_loss.items()})
            model_loss = model_loss.to(device)
            y_test_true_loss, y_test_pred_loss = _evaluate_model(model_loss, test_loader, device=device)
        else:
            y_test_true_loss, y_test_pred_loss = y_test_true, y_test_pred
        outer_metrics_loss = _compute_metrics(y_test_true_loss, y_test_pred_loss)

        print(f"  📊 Outer Fold {outer_idx+1} 测试 (ValAcc选择): "
              f"Acc={outer_metrics_acc['acc']*100:.2f}%, "
              f"F1={outer_metrics_acc['f1']*100:.2f}%, "
              f"Kappa={outer_metrics_acc['kappa']*100:.2f}% | Best Epoch={best_epoch_acc}")
        print(f"  📊 Outer Fold {outer_idx+1} 测试 (ValLoss选择): "
              f"Acc={outer_metrics_loss['acc']*100:.2f}%, "
              f"F1={outer_metrics_loss['f1']*100:.2f}%, "
              f"Kappa={outer_metrics_loss['kappa']*100:.2f}% | Best Epoch={best_epoch_loss}")

        all_test_labels_acc.extend(y_test_true.tolist())
        all_test_preds_acc.extend(y_test_pred.tolist())
        all_test_labels_loss.extend(y_test_true_loss.tolist())
        all_test_preds_loss.extend(y_test_pred_loss.tolist())
        test_metrics_per_fold_acc.append({
            "outer_fold": outer_idx + 1,
            "metrics": outer_metrics_acc,
            "best_hyperparams": best_hp,
            "best_epoch": best_epoch_acc,
        })
        test_metrics_per_fold_loss.append({
            "outer_fold": outer_idx + 1,
            "metrics": outer_metrics_loss,
            "best_hyperparams": best_hp,
            "best_epoch": best_epoch_loss,
        })
        del model

    mean_acc_acc = np.mean([m["metrics"]["acc"] for m in test_metrics_per_fold_acc])
    mean_f1_acc = np.mean([m["metrics"]["f1"] for m in test_metrics_per_fold_acc])
    mean_kappa_acc = np.mean([m["metrics"]["kappa"] for m in test_metrics_per_fold_acc])
    std_acc_acc = np.std([m["metrics"]["acc"] for m in test_metrics_per_fold_acc])

    mean_acc_loss = np.mean([m["metrics"]["acc"] for m in test_metrics_per_fold_loss])
    mean_f1_loss = np.mean([m["metrics"]["f1"] for m in test_metrics_per_fold_loss])
    mean_kappa_loss = np.mean([m["metrics"]["kappa"] for m in test_metrics_per_fold_loss])
    std_acc_loss = np.std([m["metrics"]["acc"] for m in test_metrics_per_fold_loss])

    labels = [0, 1, 2, 3]
    cm_acc = confusion_matrix(all_test_labels_acc, all_test_preds_acc, labels=labels)
    cm_loss = confusion_matrix(all_test_labels_loss, all_test_preds_loss, labels=labels)

    print("\n混淆矩阵 (Withinsession Outer 测试, ValAcc选择)")
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    print(f"真实↓ 预测→   {class_names[0]:12s} {class_names[1]:12s} {class_names[2]:12s} {class_names[3]:12s}   准确率")
    print("-" * 80)
    for i in range(len(class_names)):
        row_acc = cm_acc[i, i] / cm_acc[i].sum() if cm_acc[i].sum() > 0 else 0
        print(f"{class_names[i]:12s}   {cm_acc[i,0]:4d}         {cm_acc[i,1]:4d}         {cm_acc[i,2]:4d}         {cm_acc[i,3]:4d}        {row_acc:.2%}")

    print("\n混淆矩阵 (Withinsession Outer 测试, ValLoss选择)")
    print(f"真实↓ 预测→   {class_names[0]:12s} {class_names[1]:12s} {class_names[2]:12s} {class_names[3]:12s}   准确率")
    print("-" * 80)
    for i in range(len(class_names)):
        row_acc = cm_loss[i, i] / cm_loss[i].sum() if cm_loss[i].sum() > 0 else 0
        print(f"{class_names[i]:12s}   {cm_loss[i,0]:4d}         {cm_loss[i,1]:4d}         {cm_loss[i,2]:4d}         {cm_loss[i,3]:4d}        {row_acc:.2%}")

    summary = {
        "session": session_label,
        "mean_acc": mean_acc_acc,
        "std_acc": std_acc_acc,
        "mean_f1": mean_f1_acc,
        "mean_kappa": mean_kappa_acc,
        "fold_metrics_acc": test_metrics_per_fold_acc,
        "fold_metrics_loss": test_metrics_per_fold_loss,
        "confusion_matrix": cm_acc,
        "confusion_matrix_loss": cm_loss,
        "hyperparam_history": hyperparam_history,
        "mean_acc_loss": mean_acc_loss,
        "std_acc_loss": std_acc_loss,
        "mean_f1_loss": mean_f1_loss,
        "mean_kappa_loss": mean_kappa_loss,
    }
    print(f"\n📊 Session {session_label} Withinsession(Outer Test) 平均 (ValAcc选择): "
          f"Acc={mean_acc_acc*100:.2f}% ± {std_acc_acc*100:.2f}%, "
          f"F1={mean_f1_acc*100:.2f}%, "
          f"Kappa={mean_kappa_acc*100:.2f}%")
    print(f"📊 Session {session_label} Withinsession(Outer Test) 平均 (ValLoss选择): "
          f"Acc={mean_acc_loss*100:.2f}% ± {std_acc_loss*100:.2f}%, "
          f"F1={mean_f1_loss*100:.2f}%, "
          f"Kappa={mean_kappa_loss*100:.2f}%")
    return summary


def evaluate_cross_session_leave_one(session_data,
                                     model_factory_fn,
                                     train_epochs=300,
                                     batch_size=64,
                                     lr=1e-3,
                                     weight_decay=5e-3,
                                     device="cuda",
                                     random_state=42,
                                     label_smoothing=0.05,
                                     early_stop_patience=75,
                                     enable_adabn=True,
                                     filter_cfg=None,
                                     only_test_session=None):
    results = {}
    filter_kwargs = _resolve_filter_kwargs(filter_cfg)

    for idx, (test_session, data) in enumerate(session_data.items()):
        if only_test_session is not None and test_session != only_test_session:
            continue
        X_test_raw = data["X"]
        y_test = data["y"]
        session_seed = random_state + idx * 1000

        train_sessions = [s for s in session_data.keys() if s != test_session]
        if not train_sessions:
            print(f"⚠️ 跳过 {test_session}，无可用训练session")
            continue

        X_train_full_raw = np.concatenate([session_data[s]["X"] for s in train_sessions], axis=0)
        y_train_full = np.concatenate([session_data[s]["y"] for s in train_sessions], axis=0)

        ea_cross = EuclideanAlignment()
        ea_cross.fit(X_train_full_raw)
        X_train_full = ea_cross.transform(X_train_full_raw)
        X_test = ea_cross.transform(X_test_raw)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=session_seed)
        train_idx, val_idx = next(splitter.split(X_train_full, y_train_full))

        X_train_fold = X_train_full[train_idx]
        y_train_fold = y_train_full[train_idx]
        X_val_fold = X_train_full[val_idx]
        y_val_fold = y_train_full[val_idx]

        _set_internal_seed(session_seed)
        train_loader = _prepare_training_loader(
            X_train_fold, y_train_fold, None, batch_size, filter_kwargs,
            augmenter=None, aug_multiplier=1, aug_strategies=[], shuffle=True
        )
        val_loader = _build_dataloader_filtered(
            X_val_fold, y_val_fold, batch_size, filter_kwargs, shuffle=False
        )
        test_loader = _build_dataloader_filtered(
            X_test, y_test, batch_size, filter_kwargs, shuffle=False
        )

        print("\n" + "=" * 80)
        print(f"Crosssession 训练 (Test Session: {test_session})")
        print("=" * 80)

        model_factory = model_factory_fn()

        model, best_val_metrics, best_epoch, _, _, _ = _train_with_validation(
            model_factory,
            train_loader,
            val_loader,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=train_epochs,
            label_smoothing=label_smoothing,
            early_stop_patience=early_stop_patience,
            log_prefix=f"[Crosssession {test_session}] ",
            seed=session_seed,
            kd_alpha=0.0,
            kd_temperature=1.0,
        )

        print(f"✅ {test_session} 验证性能: Acc={best_val_metrics['acc']*100:.2f}%, "
              f"F1={best_val_metrics['f1']*100:.2f}%, "
              f"Kappa={best_val_metrics['kappa']*100:.2f}%, "
              f"Best Epoch={best_epoch}")

        if enable_adabn:
            adapt_batch_norm(model, test_loader, device=device, num_batches=5, verbose=True)

        y_test_true, y_test_pred = _evaluate_model(model, test_loader, device=device)
        test_metrics = _compute_metrics(y_test_true, y_test_pred)
        cm_test = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1, 2, 3])

        print(f"\n📊 Crosssession 测试性能 (Test {test_session}): "
              f"Acc={test_metrics['acc']*100:.2f}%, "
              f"F1={test_metrics['f1']*100:.2f}%, "
              f"Kappa={test_metrics['kappa']*100:.2f}%")
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
        print("混淆矩阵 (Crosssession 测试)")
        print(f"真实↓ 预测→   {class_names[0]:12s} {class_names[1]:12s} {class_names[2]:12s} {class_names[3]:12s}   准确率")
        print("-" * 80)
        for i in range(len(class_names)):
            row_acc = cm_test[i, i] / cm_test[i].sum() if cm_test[i].sum() > 0 else 0
            print(f"{class_names[i]:12s}   {cm_test[i,0]:4d}         {cm_test[i,1]:4d}         {cm_test[i,2]:4d}         {cm_test[i,3]:4d}        {row_acc:.2%}")

        results[test_session] = {
            "val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
            "test_confusion_matrix": cm_test,
            "train_sessions": train_sessions,
        }

    return results
