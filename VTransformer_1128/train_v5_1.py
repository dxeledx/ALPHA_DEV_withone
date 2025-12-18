"""
LiteGNNT V5.1 - MOABB评估脚本

功能:
1. 按照MOABB标准执行 Withinsession (5-fold) 与 Crosssession (Leave-One-Session-Out)。
2. 支持选择被试集合: A01 / A01+A02+A06 / 全部九个。
3. 输出Accuracy、F1、Cohen's Kappa，并保存日志与摘要。
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ALL_SUBJECTS = [f"A0{i}" for i in range(1, 10)]
SUBJECT_PRESETS = {
    "A01": ["A01"],
    "A02_ONLY": ["A02"],
    "A01_A02_A06": ["A01", "A02", "A06"],
    "ALL": ALL_SUBJECTS,
}

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
DEFAULT_DATA_PATH = str(REPO_ROOT / "BCIIV2a_mat")


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


CLASS_NAMES = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']


def format_confusion_matrix(cm):
    header = f"真实↓ 预测→   {CLASS_NAMES[0]:12s} {CLASS_NAMES[1]:12s} {CLASS_NAMES[2]:12s} {CLASS_NAMES[3]:12s}   准确率"
    lines = [header, "-" * 80]
    for i, name in enumerate(CLASS_NAMES):
        row_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        lines.append(f"{name:12s}   {cm[i,0]:4d}         {cm[i,1]:4d}         {cm[i,2]:4d}         {cm[i,3]:4d}        {row_acc:.2%}")
    return "\n".join(lines)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    print(f"✅ 随机种子已设置: {seed}")


import dataloader_v5_1 as dataloader
import trainer_v5_1 as trainer
from models_lite_gnnt_v5_1 import LiteGNNT_V3_6 as LiteGNNT_V5_1
from pipelines.augmentation import EEGAugmentation


def derive_run_tag(model_cfg):
    use_spatial = model_cfg.get("use_spatial", True)
    use_temporal = model_cfg.get("use_temporal", True)
    if use_spatial and use_temporal:
        return "dual"
    if use_spatial:
        return "gnn"
    if use_temporal:
        return "transformer"
    return "unknown"


config = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model": {
        "signal_length": 1000,
        "t_dim": 16,
        "t_heads": 2,
        "t_layers": 2,
        "dropout": 0.25,
        "use_spatial": False,
        "use_temporal": True,
    },
    "train": {
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "epoch": 300,
        "patience": 75,
        "label_smoothing": 0.0,
        "inner_folds": 1,  # 固定超参，跳过内层搜索
        "outer_folds": 5,
        "hyperparams": [
            {"lr": 1e-3, "weight_decay": 1e-3, "label_smoothing": 0.0},
        ],
    },
    "filter": {"t_low": 8.0, "t_high": 32.0, "fs": 250, "filter_order": 4},
    "augmentation": {
        "strategies": ["noise", "scale", "shift"],
        "multiplier": 4,
        "aug_prob": 0.5,
        "noise_level": 0.05,
        "scale_range": (0.9, 1.1),
        "shift_range": (-50, 50),
        "drop_prob": 0.1,
    },
    "kd": {"alpha": 0.3, "temperature": 3.0},
    "teachers": {
        "cache_dir": str(CURRENT_DIR / "artifacts" / "teachers"),
        "fs": 250,
        "reuse_cached": True,  # 直接复用缓存教师logits
    },
}

config["augmenter"] = EEGAugmentation(
    aug_prob=config["augmentation"]["aug_prob"],
    noise_level=config["augmentation"]["noise_level"],
    scale_range=config["augmentation"]["scale_range"],
    shift_range=config["augmentation"]["shift_range"],
    drop_prob=config["augmentation"]["drop_prob"],
)
os.makedirs(config["teachers"]["cache_dir"], exist_ok=True)

if "run_tag" not in config:
    config["run_tag"] = derive_run_tag(config["model"])


parser = argparse.ArgumentParser(description="LiteGNNT V5.1 - MOABB训练")
parser.add_argument(
    "--subject_set",
    choices=list(SUBJECT_PRESETS.keys()),
    default="A01",
    help="选择运行的被试集合",
)
parser.add_argument(
    "--subjects",
    type=str,
    default="",
    help="自定义被试列表，逗号或空格分隔，如 \"A02\" 或 \"A01 A02\"；若提供则优先于 --subject_set",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=DEFAULT_DATA_PATH,
    help="BCIIV2a数据目录（若找不到 A01T.mat/A01E.mat，将自动改用 MOABB BNCI2014_001）",
)
parser.add_argument(
    "--enable_crosssession",
    action="store_true",
    help="启用 Crosssession LOSO 评估（默认关闭，当前仅关注 Withinsession）",
)
args = parser.parse_args()
if args.subjects.strip():
    raw_list = args.subjects.replace(",", " ").split()
    selected_subjects = []
    for sid in raw_list:
        sid = sid.strip()
        if not sid:
            continue
        if sid not in ALL_SUBJECTS:
            raise ValueError(f"无效被试ID: {sid}，可选: {ALL_SUBJECTS}")
        selected_subjects.append(sid)
    if not selected_subjects:
        raise ValueError("自定义被试列表为空，请检查 --subjects 参数。")
    subject_tag = "CUSTOM"
else:
    selected_subjects = SUBJECT_PRESETS[args.subject_set]
    subject_tag = args.subject_set
subject_slug = "-".join(selected_subjects)

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/train_v5_1_{config['run_tag']}_{subject_tag}_{subject_slug}_{timestamp}.log"
logger = Logger(log_file)
sys.stdout = logger

print(f"[INFO] LiteGNNT V5.1 MOABB评估 | 日志: {log_file} | 被试: {selected_subjects} | 模式: {config['run_tag']} | 选择方式: {subject_tag}")

set_seed(config["seed"])


def build_model_factory(model_cfg):
    def factory():
        return LiteGNNT_V5_1(
            n_channels=22,
            n_classes=4,
            **model_cfg,
        )

    return factory


def run_subject(subject_id, data_path, cfg):
    print(f"\n[INFO] Subject {subject_id} 开始")
    set_seed(cfg["seed"])

    X_train, y_train, X_eval, y_eval = dataloader.load_train_eval_separately(
        data_path, subject_id, trial_start_offset=2.0, trial_length=4.0
    )

    # 打印模型参数量（总参数/可训练参数）
    tmp_model = build_model_factory(cfg["model"])()
    total_params = sum(p.numel() for p in tmp_model.parameters())
    trainable_params = sum(p.numel() for p in tmp_model.parameters() if p.requires_grad)
    print(f"[INFO] 模型参数量: 总参数={total_params:,} | 可训练参数={trainable_params:,}")
    del tmp_model

    def model_factory_fn():
        return build_model_factory(cfg["model"])

    within_T = trainer.evaluate_within_session(
        model_factory_fn,
        subject_id=subject_id,
        session_label="T",
        X_session_raw=X_train,
        y_session=y_train,
        train_epochs=cfg["train"]["epoch"],
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        device=cfg["device"],
        random_state=cfg["seed"],
        label_smoothing=cfg["train"]["label_smoothing"],
        early_stop_patience=cfg["train"]["patience"],
        filter_cfg=cfg["filter"],
        augmenter=cfg["augmenter"],
        aug_cfg=cfg["augmentation"],
        kd_cfg=cfg["kd"],
        teacher_cfg=cfg["teachers"],
    )

    within_E = trainer.evaluate_within_session(
        model_factory_fn,
        subject_id=subject_id,
        session_label="E",
        X_session_raw=X_eval,
        y_session=y_eval,
        train_epochs=cfg["train"]["epoch"],
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        device=cfg["device"],
        random_state=cfg["seed"],
        label_smoothing=cfg["train"]["label_smoothing"],
        early_stop_patience=cfg["train"]["patience"],
        filter_cfg=cfg["filter"],
        augmenter=cfg["augmenter"],
        aug_cfg=cfg["augmentation"],
        kd_cfg=cfg["kd"],
        teacher_cfg=cfg["teachers"],
    )

    session_dict = {
        'T': {
            'X': X_train,
            'y': y_train,
        },
        'E': {
            'X': X_eval,
            'y': y_eval,
        }
    }

    cross = {}
    if args.enable_crosssession:
        cross = trainer.evaluate_cross_session_leave_one(
            session_data=session_dict,
            model_factory_fn=model_factory_fn,
            train_epochs=cfg["train"]["epoch"],
            batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            device=cfg["device"],
            random_state=cfg["seed"],
            label_smoothing=cfg["train"]["label_smoothing"],
            early_stop_patience=cfg["train"]["patience"],
            enable_adabn=True,
            filter_cfg=cfg["filter"],
        )

    print(f"[DONE] Subject {subject_id} 完成")
    return {
        "subject": subject_id,
        "within_T": within_T,
        "within_E": within_E,
        "cross": cross,
    }



subject_results = []
for subj in selected_subjects:
    result = run_subject(subj, args.data_path, config)
    subject_results.append(result)

subject_stats = []
for res in subject_results:
    sessions = [s for s in [res["within_T"], res["within_E"]] if s]
    if sessions:
        subject_stats.append({
            "subject": res["subject"],
            "mean_acc": float(np.mean([s["mean_acc"] for s in sessions])),
            "mean_f1": float(np.mean([s["mean_f1"] for s in sessions])),
            "mean_kappa": float(np.mean([s["mean_kappa"] for s in sessions])),
            "mean_acc_loss": float(np.mean([s.get("mean_acc_loss", 0.0) for s in sessions])),
            "mean_f1_loss": float(np.mean([s.get("mean_f1_loss", 0.0) for s in sessions])),
            "mean_kappa_loss": float(np.mean([s.get("mean_kappa_loss", 0.0) for s in sessions])),
        })

dataset_level = {
    "mean_acc": float(np.mean([s["mean_acc"] for s in subject_stats])) if subject_stats else 0.0,
    "mean_f1": float(np.mean([s["mean_f1"] for s in subject_stats])) if subject_stats else 0.0,
    "mean_kappa": float(np.mean([s["mean_kappa"] for s in subject_stats])) if subject_stats else 0.0,
    "mean_acc_loss": float(np.mean([s["mean_acc_loss"] for s in subject_stats])) if subject_stats else 0.0,
    "mean_f1_loss": float(np.mean([s["mean_f1_loss"] for s in subject_stats])) if subject_stats else 0.0,
    "mean_kappa_loss": float(np.mean([s["mean_kappa_loss"] for s in subject_stats])) if subject_stats else 0.0,
}

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
summary_file = os.path.join(
    results_dir, f"V5.1_moabb_summary_{config['run_tag']}_{subject_tag}_{subject_slug}_{timestamp}.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("LiteGNNT V5.1 - MOABB评估结果汇总\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"时间戳: {timestamp}\n")
    f.write(f"被试集合: {selected_subjects}\n\n")
    for res in subject_results:
        f.write(f"Subject {res['subject']}\n")
        wT = res["within_T"]
        f.write(
            f"  Within T (ValAcc): Acc={wT['mean_acc']*100:.2f}% ± {wT['std_acc']*100:.2f}%, "
            f"F1={wT['mean_f1']*100:.2f}%, Kappa={wT['mean_kappa']*100:.2f}%\n"
        )
        f.write(
            f"  Within T (ValLoss): Acc={wT['mean_acc_loss']*100:.2f}% ± {wT['std_acc_loss']*100:.2f}%, "
            f"F1={wT['mean_f1_loss']*100:.2f}%, Kappa={wT['mean_kappa_loss']*100:.2f}%\n"
        )
        f.write("  Within T 混淆矩阵 (ValAcc):\n")
        f.write(format_confusion_matrix(wT['confusion_matrix']) + "\n")
        f.write("  Within T 混淆矩阵 (ValLoss):\n")
        f.write(format_confusion_matrix(wT['confusion_matrix_loss']) + "\n")
        if res["within_E"]:
            wE = res["within_E"]
            f.write(
                f"  Within E (ValAcc): Acc={wE['mean_acc']*100:.2f}% ± {wE['std_acc']*100:.2f}%, "
                f"F1={wE['mean_f1']*100:.2f}%, Kappa={wE['mean_kappa']*100:.2f}%\n"
            )
            f.write(
                f"  Within E (ValLoss): Acc={wE['mean_acc_loss']*100:.2f}% ± {wE['std_acc_loss']*100:.2f}%, "
                f"F1={wE['mean_f1_loss']*100:.2f}%, Kappa={wE['mean_kappa_loss']*100:.2f}%\n"
            )
            f.write("  Within E 混淆矩阵 (ValAcc):\n")
            f.write(format_confusion_matrix(wE['confusion_matrix']) + "\n")
            f.write("  Within E 混淆矩阵 (ValLoss):\n")
            f.write(format_confusion_matrix(wE['confusion_matrix_loss']) + "\n")
        else:
            f.write("  Within E: 无可用评估session\n")
        cross = res["cross"]
        if cross:
            for session_name, metrics in cross.items():
                f.write(
                    f"  Cross (Train {','.join(metrics['train_sessions'])} → Test {session_name}): "
                    f"Acc={metrics['test_metrics']['acc']*100:.2f}%, "
                    f"F1={metrics['test_metrics']['f1']*100:.2f}%, "
                    f"Kappa={metrics['test_metrics']['kappa']*100:.2f}%\n"
                )
                f.write("    测试混淆矩阵:\n")
                f.write(format_confusion_matrix(metrics['test_confusion_matrix']) + "\n")
        else:
            f.write("  Crosssession: 已关闭\n")
        f.write("\n")

    f.write("Subject 平均结果:\n")
    for stats in subject_stats:
        f.write(
            f"  {stats['subject']}: "
            f"Acc(ValAcc)={stats['mean_acc']*100:.2f}%, F1={stats['mean_f1']*100:.2f}%, Kappa={stats['mean_kappa']*100:.2f}% | "
            f"Acc(ValLoss)={stats['mean_acc_loss']*100:.2f}%, F1={stats['mean_f1_loss']*100:.2f}%, Kappa={stats['mean_kappa_loss']*100:.2f}%\n"
        )
    f.write("\nDataset-level 平均 (9被试×2 session 平均):\n")
    f.write(
        f"  Acc(ValAcc)={dataset_level['mean_acc']*100:.2f}%, "
        f"F1(ValAcc)={dataset_level['mean_f1']*100:.2f}%, "
        f"Kappa(ValAcc)={dataset_level['mean_kappa']*100:.2f}%\n"
    )
    f.write(
        f"  Acc(ValLoss)={dataset_level['mean_acc_loss']*100:.2f}%, "
        f"F1(ValLoss)={dataset_level['mean_f1_loss']*100:.2f}%, "
        f"Kappa(ValLoss)={dataset_level['mean_kappa_loss']*100:.2f}%\n"
    )

print(f"\n✅ 结果已保存: {summary_file}")
print(f"🎉 完成 {len(subject_results)} 个被试的评估 | 日志: {log_file}")
print("Subject 平均结果:")
for stats in subject_stats:
    print(
        f"  {stats['subject']}: "
        f"Acc(ValAcc)={stats['mean_acc']*100:.2f}% | F1={stats['mean_f1']*100:.2f}% | Kappa={stats['mean_kappa']*100:.2f}% || "
        f"Acc(ValLoss)={stats['mean_acc_loss']*100:.2f}% | F1={stats['mean_f1_loss']*100:.2f}% | Kappa={stats['mean_kappa_loss']*100:.2f}%"
    )
print(
    f"Dataset-level 平均 (ValAcc): Acc={dataset_level['mean_acc']*100:.2f}%, "
    f"F1={dataset_level['mean_f1']*100:.2f}%, Kappa={dataset_level['mean_kappa']*100:.2f}%"
)
print(
    f"Dataset-level 平均 (ValLoss): Acc={dataset_level['mean_acc_loss']*100:.2f}%, "
    f"F1={dataset_level['mean_f1_loss']*100:.2f}%, Kappa={dataset_level['mean_kappa_loss']*100:.2f}%"
)

logger.close()
sys.stdout = logger.terminal
