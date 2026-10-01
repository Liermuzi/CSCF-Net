#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSCF-Net FINAL 5-Seed Training
==============================

Final strengthened model:
    28D -> Transformer -> four phase extractors -> CAIM
        -> explicit cyclic CCM
        -> Planning / Execution / Monitoring / Reflection heads
        -> Overall fusion head

CCM:
    Reflection -> Planning -> Execution -> Monitoring -> Reflection

Loss:
    L = KL_soft + 0.2 * MSE_score + 0.1 * consistency

Consistency:
    MSE(
        independent score-head prediction,
        0.2*p_low + 0.5*p_mid + 0.8*p_high
    )

Important:
- Uses TRAIN/VAL only. TEST is never touched here.
- Runs five fixed seeds: 42, 52, 62, 72, 82.
- Best checkpoint is selected independently for each seed by VAL total loss.
- TRAIN mean/std are computed once and reused for VAL and later TEST.
- The original Planning≈Reflection probability-equality penalty is NOT used;
  cyclic dependence is represented structurally by CCM.
"""

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

# ============================================================
# CONFIG -- 修改路径即可
# ============================================================

X_TRAIN_PATH = r"D:\研究生毕设\SRL\feature_extract\raw\软标签new3对应的28维原始特征\train_28d.npy"
X_VAL_PATH   = r"D:\研究生毕设\SRL\feature_extract\raw\软标签new3对应的28维原始特征\val_28d.npy"

Y_TRAIN_PATH = r"D:\研究生毕设\SRL\feature_extract\soft label\new3\train_labels20new3.npy"
Y_VAL_PATH   = r"D:\研究生毕设\SRL\feature_extract\soft label\new3\val_labels20new3.npy"

CONF_TRAIN_PATH = r"D:\研究生毕设\SRL\feature_extract\soft label\new3\train_confnew3.npy"
CONF_VAL_PATH   = r"D:\研究生毕设\SRL\feature_extract\soft label\new3\val_confnew3.npy"

OUTPUT_DIR = (
    r"D:\研究生毕设\SRL\28维，最好性能\大修版\原标签原特征提取"
    r"\cscf_ccm_consistency_5seed_final"
)

SEEDS = [42, 52, 62, 72, 82]

BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 6
NUM_WORKERS = 0

FEAT_DIM = 28
HIDDEN_DIM = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CCM_CYCLES = 2
CCM_DROPOUT = 0.15

W_KL = 1.0
W_SCORE = 0.2
W_CONSISTENCY = 0.10

SCORE_ANCHORS = (0.2, 0.5, 0.8)
LABEL_SCORE_TOL = 1e-3
EPS = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact original label/model order:
OUTPUT_NAMES = [
    "overall",
    "planning",
    "execution",
    "monitoring",
    "reflection",
]

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ============================================================
# Data
# ============================================================

def audit_label_score_relation(y20, split_name):
    if y20.ndim != 2 or y20.shape[1] != 20:
        raise ValueError(
            f"{split_name}: expected y [N,20], got {y20.shape}"
        )

    soft = y20[:, :15].reshape(-1, 5, 3)
    score = y20[:, 15:20]

    anchors = np.asarray(
        SCORE_ANCHORS,
        dtype=np.float64
    ).reshape(1, 1, 3)

    implied = np.sum(
        soft.astype(np.float64) * anchors,
        axis=-1
    )

    gap = np.abs(
        score.astype(np.float64) - implied
    )

    print(
        f"[{split_name} label audit] "
        f"mean|score-E[p]|={gap.mean():.8f} | "
        f"max={gap.max():.8f}"
    )

    if gap.max() > LABEL_SCORE_TOL:
        raise ValueError(
            f"{split_name}: score/probability relation failed. "
            f"max gap={gap.max():.8f} > {LABEL_SCORE_TOL}"
        )


class SRLDataset(Dataset):
    def __init__(
        self,
        X,
        y20,
        confidence,
        mean=None,
        std=None,
    ):
        if X.ndim == 3:
            X = X[:, -1, :]

        if X.ndim != 2 or X.shape[1] != FEAT_DIM:
            raise ValueError(
                f"Expected X [N,{FEAT_DIM}], got {X.shape}"
            )

        if y20.ndim != 2 or y20.shape[1] != 20:
            raise ValueError(
                f"Expected y20 [N,20], got {y20.shape}"
            )

        if confidence.ndim != 2 or confidence.shape[1] != 5:
            raise ValueError(
                f"Expected confidence [N,5], got {confidence.shape}"
            )

        if not (
            len(X) == len(y20) == len(confidence)
        ):
            raise ValueError(
                "X/y/confidence length mismatch."
            )

        if mean is None:
            self.mean = np.mean(
                X,
                axis=0,
                keepdims=True
            ).astype(np.float32)

            self.std = (
                np.std(
                    X,
                    axis=0,
                    keepdims=True
                )
                + 1e-8
            ).astype(np.float32)
        else:
            self.mean = np.asarray(
                mean,
                dtype=np.float32
            )
            self.std = np.asarray(
                std,
                dtype=np.float32
            )

        self.X = torch.tensor(
            (X - self.mean) / self.std,
            dtype=torch.float32
        )

        self.y20 = torch.tensor(
            y20,
            dtype=torch.float32
        )

        self.confidence = torch.tensor(
            confidence,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y20[idx],
            self.confidence[idx],
        )

# ============================================================
# CAIM
# ============================================================

class CrossPhaseAttention(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=4,
        dropout=0.15,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Keep the exact shared-LayerNorm behavior of the validated diagnostic model.
        self.norm = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, phase_reps):
        attended, attn_weights = self.attention(
            phase_reps,
            phase_reps,
            phase_reps,
        )

        attended = self.dropout(attended)

        out = self.norm(
            phase_reps + attended
        )

        ffn_out = self.ffn(out)

        out = self.norm(
            out + ffn_out
        )

        return out, attn_weights

# ============================================================
# Explicit cyclic CCM
# ============================================================

class CyclicCCM(nn.Module):
    """
    Four-state cyclic GRU refinement.

    Phase order:
        0 Planning
        1 Execution
        2 Monitoring
        3 Reflection

    Predecessor relation:
        R -> P
        P -> E
        E -> M
        M -> R

    Every new state in a cycle is computed from the PREVIOUS cycle states,
    avoiding within-cycle order artifacts.
    """

    def __init__(
        self,
        hidden_dim,
        num_cycles=2,
        dropout=0.15,
    ):
        super().__init__()

        self.num_cycles = num_cycles

        self.cells = nn.ModuleList([
            nn.GRUCell(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
            )
            for _ in range(4)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(4)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, phase_states):
        if (
            phase_states.ndim != 3
            or phase_states.size(1) != 4
        ):
            raise ValueError(
                f"CCM expects [B,4,H], got {phase_states.shape}"
            )

        states = [
            phase_states[:, i, :]
            for i in range(4)
        ]

        for _ in range(self.num_cycles):
            old = states

            predecessor = [
                old[3],  # R -> P
                old[0],  # P -> E
                old[1],  # E -> M
                old[2],  # M -> R
            ]

            new_states = []

            for i in range(4):
                updated = self.cells[i](
                    predecessor[i],
                    old[i],
                )

                updated = self.norms[i](
                    updated
                )

                updated = self.dropout(
                    updated
                )

                new_states.append(
                    updated
                )

            states = new_states

        return torch.stack(
            states,
            dim=1
        )

# ============================================================
# Model
# ============================================================

class ImprovedSRLModel(nn.Module):
    def __init__(
        self,
        feat_dim=28,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.15,
        ccm_cycles=2,
    ):
        super().__init__()

        self.emb = nn.Linear(
            feat_dim,
            hidden_dim
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.phase_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(4)
        ])

        self.cross_phase_attention = CrossPhaseAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
        )

        self.ccm = CyclicCCM(
            hidden_dim=hidden_dim,
            num_cycles=ccm_cycles,
            dropout=CCM_DROPOUT,
        )

        self.global_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.overall_head = nn.Linear(hidden_dim, 3)
        self.planning_head = nn.Linear(hidden_dim, 3)
        self.execution_head = nn.Linear(hidden_dim, 3)
        self.monitoring_head = nn.Linear(hidden_dim, 3)
        self.reflection_head = nn.Linear(hidden_dim, 3)

        self.overall_score_head = nn.Linear(hidden_dim, 1)
        self.planning_score_head = nn.Linear(hidden_dim, 1)
        self.execution_score_head = nn.Linear(hidden_dim, 1)
        self.monitoring_score_head = nn.Linear(hidden_dim, 1)
        self.reflection_score_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, -1, :]

        x = self.emb(x).unsqueeze(1)
        encoded = self.encoder(x)
        seq_repr = encoded.squeeze(1)

        phase_reps = [
            extractor(seq_repr)
            for extractor
            in self.phase_extractors
        ]

        phase_stack = torch.stack(
            phase_reps,
            dim=1
        )

        interacted_phases, attn_weights = self.cross_phase_attention(
            phase_stack
        )

        ccm_phases = self.ccm(
            interacted_phases
        )

        planning_repr = ccm_phases[:, 0, :]
        execution_repr = ccm_phases[:, 1, :]
        monitoring_repr = ccm_phases[:, 2, :]
        reflection_repr = ccm_phases[:, 3, :]

        overall_repr = self.global_extractor(
            torch.cat(
                [
                    planning_repr,
                    execution_repr,
                    monitoring_repr,
                    reflection_repr,
                ],
                dim=-1,
            )
        )

        logits_list = [
            self.overall_head(overall_repr),
            self.planning_head(planning_repr),
            self.execution_head(execution_repr),
            self.monitoring_head(monitoring_repr),
            self.reflection_head(reflection_repr),
        ]

        score_preds = [
            self.overall_score_head(overall_repr),
            self.planning_score_head(planning_repr),
            self.execution_score_head(execution_repr),
            self.monitoring_score_head(monitoring_repr),
            self.reflection_score_head(reflection_repr),
        ]

        return {
            "logits_list": logits_list,
            "score_preds": score_preds,
            "attention_weights": attn_weights,
        }

# ============================================================
# Score relation + loss
# ============================================================

def expected_score_from_logits(logits):
    probs = F.softmax(
        logits,
        dim=-1
    )

    anchors = torch.tensor(
        SCORE_ANCHORS,
        dtype=probs.dtype,
        device=probs.device,
    ).view(1, 3)

    return torch.sum(
        probs * anchors,
        dim=-1,
        keepdim=True,
    )


class SRLCombinedLoss(nn.Module):
    def forward(
        self,
        logits_list,
        score_preds,
        soft_label_list,
        score_labels,
        confidence,
    ):
        loss_kl = torch.zeros(
            (),
            device=logits_list[0].device,
        )

        loss_score = torch.zeros(
            (),
            device=logits_list[0].device,
        )

        loss_consistency = torch.zeros(
            (),
            device=logits_list[0].device,
        )

        for i, (
            logits,
            score_pred,
            soft_target,
            score_target,
        ) in enumerate(
            zip(
                logits_list,
                score_preds,
                soft_label_list,
                score_labels,
            )
        ):
            target = torch.clamp(
                soft_target,
                min=EPS,
            )

            target = target / torch.sum(
                target,
                dim=-1,
                keepdim=True,
            )

            log_pred = F.log_softmax(
                logits,
                dim=-1,
            )

            kl_each = torch.sum(
                target * (
                    torch.log(target)
                    - log_pred
                ),
                dim=-1,
            )

            c = confidence[:, i]

            loss_kl = (
                loss_kl
                + torch.mean(
                    c * kl_each
                )
            )

            score_pred_1d = score_pred.squeeze(-1)
            score_target_1d = score_target.squeeze(-1)

            score_mse_each = (
                score_pred_1d
                - score_target_1d
            ) ** 2

            loss_score = (
                loss_score
                + torch.mean(
                    c * score_mse_each
                )
            )

            implied_score = expected_score_from_logits(
                logits
            ).squeeze(-1)

            consistency_each = (
                score_pred_1d
                - implied_score
            ) ** 2

            # Internal mathematical consistency:
            # do not confidence-weight this term.
            loss_consistency = (
                loss_consistency
                + torch.mean(
                    consistency_each
                )
            )

        total = (
            W_KL * loss_kl
            + W_SCORE * loss_score
            + W_CONSISTENCY * loss_consistency
        )

        if not torch.isfinite(total):
            raise FloatingPointError(
                "Non-finite training loss encountered."
            )

        return {
            "total": total,
            "kl": loss_kl,
            "score": loss_score,
            "consistency": loss_consistency,
        }

# ============================================================
# Validation metrics
# ============================================================

def safe_spearman(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    if (
        np.std(y_true) <= EPS
        or np.std(y_pred) <= EPS
    ):
        return float("nan")

    result = spearmanr(
        y_true,
        y_pred
    )

    return float(
        result.statistic
        if hasattr(result, "statistic")
        else result[0]
    )


def safe_kendall(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64
    )

    if (
        np.std(y_true) <= EPS
        or np.std(y_pred) <= EPS
    ):
        return float("nan")

    result = kendalltau(
        y_true,
        y_pred
    )

    return float(
        result.statistic
        if hasattr(result, "statistic")
        else result[0]
    )


@torch.no_grad()
def evaluate_validation(
    model,
    loader,
    criterion,
):
    model.eval()

    true_probs = {
        name: []
        for name in OUTPUT_NAMES
    }
    pred_probs = {
        name: []
        for name in OUTPUT_NAMES
    }
    true_scores = {
        name: []
        for name in OUTPUT_NAMES
    }
    head_scores = {
        name: []
        for name in OUTPUT_NAMES
    }
    prob_scores = {
        name: []
        for name in OUTPUT_NAMES
    }

    loss_sum = {
        "total": 0.0,
        "kl": 0.0,
        "score": 0.0,
        "consistency": 0.0,
    }

    n = 0

    for x, y, conf in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        conf = conf.to(DEVICE)

        outputs = model(x)

        logits_list = outputs["logits_list"]
        score_preds = outputs["score_preds"]

        soft_label_list = torch.split(
            y[:, :15],
            3,
            dim=-1,
        )

        score_labels = torch.split(
            y[:, 15:20],
            1,
            dim=-1,
        )

        parts = criterion(
            logits_list=logits_list,
            score_preds=score_preds,
            soft_label_list=soft_label_list,
            score_labels=score_labels,
            confidence=conf,
        )

        bs = x.size(0)
        n += bs

        for key in loss_sum:
            loss_sum[key] += (
                parts[key].item() * bs
            )

        for (
            name,
            logits,
            score_pred,
            target_prob,
            target_score,
        ) in zip(
            OUTPUT_NAMES,
            logits_list,
            score_preds,
            soft_label_list,
            score_labels,
        ):
            probs = F.softmax(
                logits,
                dim=-1,
            )

            prob_score = expected_score_from_logits(
                logits
            )

            true_probs[name].append(
                target_prob.cpu().numpy()
            )
            pred_probs[name].append(
                probs.cpu().numpy()
            )
            true_scores[name].append(
                target_score.cpu().numpy().reshape(-1)
            )
            head_scores[name].append(
                score_pred.cpu().numpy().reshape(-1)
            )
            prob_scores[name].append(
                prob_score.cpu().numpy().reshape(-1)
            )

    metrics = {}

    for name in OUTPUT_NAMES:
        yt_prob = np.concatenate(
            true_probs[name],
            axis=0,
        )
        yp_prob = np.concatenate(
            pred_probs[name],
            axis=0,
        )
        yt_score = np.concatenate(
            true_scores[name],
            axis=0,
        )
        yp_head = np.concatenate(
            head_scores[name],
            axis=0,
        )
        yp_prob_score = np.concatenate(
            prob_scores[name],
            axis=0,
        )

        true_cls = np.argmax(
            yt_prob,
            axis=1,
        )
        pred_cls = np.argmax(
            yp_prob,
            axis=1,
        )

        metrics[name] = {
            "accuracy": float(
                accuracy_score(
                    true_cls,
                    pred_cls,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    true_cls,
                    pred_cls,
                    average="weighted",
                    zero_division=0,
                )
            ),
            # observed-class macro F1
            "macro_f1": float(
                f1_score(
                    true_cls,
                    pred_cls,
                    average="macro",
                    zero_division=0,
                )
            ),
            # sklearn BA = mean recall over classes present in y_true
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    true_cls,
                    pred_cls,
                )
            ),
            "head_mae": float(
                mean_absolute_error(
                    yt_score,
                    yp_head,
                )
            ),
            "head_rmse": float(
                math.sqrt(
                    mean_squared_error(
                        yt_score,
                        yp_head,
                    )
                )
            ),
            "head_spearman": safe_spearman(
                yt_score,
                yp_head,
            ),
            "head_kendall": safe_kendall(
                yt_score,
                yp_head,
            ),
            "prob_score_mae": float(
                mean_absolute_error(
                    yt_score,
                    yp_prob_score,
                )
            ),
            "prob_score_rmse": float(
                math.sqrt(
                    mean_squared_error(
                        yt_score,
                        yp_prob_score,
                    )
                )
            ),
            "prob_score_spearman": safe_spearman(
                yt_score,
                yp_prob_score,
            ),
            "prob_score_kendall": safe_kendall(
                yt_score,
                yp_prob_score,
            ),
            "head_vs_probscore_mae": float(
                mean_absolute_error(
                    yp_head,
                    yp_prob_score,
                )
            ),
        }

    macro = {}

    for key in [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "balanced_accuracy",
        "head_mae",
        "head_rmse",
        "head_spearman",
        "head_kendall",
        "prob_score_mae",
        "prob_score_rmse",
        "prob_score_spearman",
        "prob_score_kendall",
        "head_vs_probscore_mae",
    ]:
        values = np.asarray(
            [
                metrics[name][key]
                for name in OUTPUT_NAMES
            ],
            dtype=np.float64,
        )

        values = values[
            np.isfinite(values)
        ]

        macro[key] = (
            float(np.mean(values))
            if len(values) > 0
            else float("nan")
        )

    losses = {
        key: value / max(n, 1)
        for key, value in loss_sum.items()
    }

    return losses, metrics, macro

# ============================================================
# Windows / Unicode-safe torch serialization
# ============================================================

def safe_torch_save(obj, path):
    """
    Robust torch.save for Windows paths containing Chinese/non-ASCII chars.

    Older PyTorch builds may fail when a Unicode Windows path is passed
    directly to torch.save(), even though pathlib successfully created
    the directory. Opening the file with Python first avoids that C++
    path-handling issue.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.parent.exists():
        raise RuntimeError(
            f"Checkpoint parent directory still does not exist: {path.parent}"
        )

    with open(path, "wb") as f:
        torch.save(obj, f)


# ============================================================
# Saving helpers
# ============================================================

def save_csv(rows, path):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# Main
# ============================================================

def main():
    out_root = Path(
        OUTPUT_DIR
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("CSCF-Net FINAL 5-SEED TRAINING")
    print(f"DEVICE={DEVICE}")
    print(f"SEEDS={SEEDS}")
    print(f"CCM_CYCLES={CCM_CYCLES}")
    print(
        f"LOSS: KL={W_KL}, SCORE={W_SCORE}, "
        f"CONSISTENCY={W_CONSISTENCY}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Load once
    # --------------------------------------------------------
    X_train = np.load(
        X_TRAIN_PATH
    )
    X_val = np.load(
        X_VAL_PATH
    )

    y_train = np.load(
        Y_TRAIN_PATH
    )
    y_val = np.load(
        Y_VAL_PATH
    )

    conf_train = np.load(
        CONF_TRAIN_PATH
    )
    conf_val = np.load(
        CONF_VAL_PATH
    )

    print(
        f"TRAIN: X={X_train.shape}, y={y_train.shape}, conf={conf_train.shape}"
    )
    print(
        f"VAL:   X={X_val.shape}, y={y_val.shape}, conf={conf_val.shape}"
    )

    audit_label_score_relation(
        y_train,
        "TRAIN",
    )
    audit_label_score_relation(
        y_val,
        "VAL",
    )

    # --------------------------------------------------------
    # TRAIN-only normalization -- same for all seeds
    # --------------------------------------------------------
    train_dataset = SRLDataset(
        X_train,
        y_train,
        conf_train,
    )

    val_dataset = SRLDataset(
        X_val,
        y_val,
        conf_val,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    np.savez(
        out_root
        / "train_normalization_stats.npz",
        mean=train_dataset.mean,
        std=train_dataset.std,
    )

    # Save fixed experiment config.
    config = {
        "seeds": SEEDS,
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "feat_dim": FEAT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "ccm_cycles": CCM_CYCLES,
        "ccm_cycle": (
            "Reflection->Planning->Execution->Monitoring->Reflection"
        ),
        "w_kl": W_KL,
        "w_score": W_SCORE,
        "w_consistency": W_CONSISTENCY,
        "score_anchors": SCORE_ANCHORS,
        "selection_rule": (
            "best checkpoint selected independently by validation total loss"
        ),
        "test_used_in_training": False,
    }

    with open(
        out_root
        / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    summaries = []

    # --------------------------------------------------------
    # Five independent runs
    # --------------------------------------------------------
    for seed in SEEDS:
        print("\n" + "=" * 100)
        print(f"SEED {seed}")
        print("=" * 100)

        set_seed(seed)

        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[Path check] seed_dir exists={seed_dir.exists()} | "
            f"{seed_dir}"
        )

        if not seed_dir.exists():
            raise RuntimeError(
                f"Failed to create seed directory: {seed_dir}"
            )

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

        model = ImprovedSRLModel(
            feat_dim=FEAT_DIM,
            hidden_dim=HIDDEN_DIM,
            ccm_cycles=CCM_CYCLES,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.5,
        )

        criterion = SRLCombinedLoss()

        best_val = float("inf")
        best_epoch = -1
        no_improve = 0
        history = []

        model_path = (
            seed_dir
            / f"best_model_ccm_consistency_seed{seed}.pth"
        )

        for epoch_idx in range(EPOCHS):
            epoch = epoch_idx + 1

            # ---------------- TRAIN ----------------
            model.train()

            running = {
                "total": 0.0,
                "kl": 0.0,
                "score": 0.0,
                "consistency": 0.0,
            }

            n_train = 0

            pbar = tqdm(
                train_loader,
                desc=f"Seed {seed} Epoch {epoch}/{EPOCHS}",
                dynamic_ncols=True,
            )

            for x, y, conf in pbar:
                x = x.to(
                    DEVICE,
                    non_blocking=True,
                )
                y = y.to(
                    DEVICE,
                    non_blocking=True,
                )
                conf = conf.to(
                    DEVICE,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(x)

                logits_list = outputs[
                    "logits_list"
                ]
                score_preds = outputs[
                    "score_preds"
                ]

                soft_label_list = torch.split(
                    y[:, :15],
                    3,
                    dim=-1,
                )

                score_labels = torch.split(
                    y[:, 15:20],
                    1,
                    dim=-1,
                )

                parts = criterion(
                    logits_list=logits_list,
                    score_preds=score_preds,
                    soft_label_list=soft_label_list,
                    score_labels=score_labels,
                    confidence=conf,
                )

                loss = parts["total"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                optimizer.step()

                bs = x.size(0)
                n_train += bs

                for key in running:
                    running[key] += (
                        parts[key].item()
                        * bs
                    )

                pbar.set_postfix(
                    total=f"{parts['total'].item():.4f}",
                    cons=f"{parts['consistency'].item():.4f}",
                )

            train_parts = {
                key: value / max(n_train, 1)
                for key, value in running.items()
            }

            # ---------------- VAL ----------------
            val_losses, val_metrics, val_macro = evaluate_validation(
                model,
                val_loader,
                criterion,
            )

            scheduler.step(
                val_losses["total"]
            )

            lr = optimizer.param_groups[0]["lr"]

            row = {
                "seed": seed,
                "epoch": epoch,
                "lr": lr,

                "train_total": train_parts["total"],
                "train_kl": train_parts["kl"],
                "train_score": train_parts["score"],
                "train_consistency": train_parts["consistency"],

                "val_total": val_losses["total"],
                "val_kl": val_losses["kl"],
                "val_score": val_losses["score"],
                "val_consistency": val_losses["consistency"],

                "val_macro_prob_spearman": (
                    val_macro["prob_score_spearman"]
                ),
                "val_macro_prob_kendall": (
                    val_macro["prob_score_kendall"]
                ),
                "val_macro_prob_mae": (
                    val_macro["prob_score_mae"]
                ),
                "val_macro_head_prob_mae": (
                    val_macro["head_vs_probscore_mae"]
                ),
            }

            for name in OUTPUT_NAMES:
                row[
                    f"{name}_acc"
                ] = val_metrics[name]["accuracy"]

                row[
                    f"{name}_macro_f1"
                ] = val_metrics[name]["macro_f1"]

                row[
                    f"{name}_ba"
                ] = val_metrics[name]["balanced_accuracy"]

                row[
                    f"{name}_prob_rho"
                ] = val_metrics[name]["prob_score_spearman"]

                row[
                    f"{name}_prob_tau"
                ] = val_metrics[name]["prob_score_kendall"]

            history.append(row)

            save_csv(
                history,
                seed_dir
                / "training_history.csv",
            )

            print(
                f"[Seed {seed} Epoch {epoch}] "
                f"Train={train_parts['total']:.6f} | "
                f"Val={val_losses['total']:.6f} | "
                f"Cons={val_losses['consistency']:.6f} | "
                f"Macro rho={val_macro['prob_score_spearman']:.4f} | "
                f"Macro tau={val_macro['prob_score_kendall']:.4f} | "
                f"LR={lr:.2e}"
            )

            # ---------------- Best checkpoint ----------------
            if val_losses[
                "total"
            ] < best_val:
                best_val = val_losses[
                    "total"
                ]
                best_epoch = epoch
                no_improve = 0

                safe_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val,
                        "seed": seed,
                        "config": {
                            "feat_dim": FEAT_DIM,
                            "hidden_dim": HIDDEN_DIM,
                            "ccm_cycles": CCM_CYCLES,
                            "w_kl": W_KL,
                            "w_score": W_SCORE,
                            "w_consistency": W_CONSISTENCY,
                            "score_anchors": SCORE_ANCHORS,
                        },
                    },
                    model_path,
                )

                with open(
                    seed_dir
                    / "best_validation_metrics.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "val_losses": val_losses,
                            "val_metrics": val_metrics,
                            "val_macro_output": val_macro,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                print(
                    f"Saved new best checkpoint: {model_path}"
                )

            else:
                no_improve += 1

                if no_improve >= PATIENCE:
                    print(
                        f"Early stopping seed={seed}; "
                        f"best epoch={best_epoch}, "
                        f"best val={best_val:.6f}"
                    )
                    break

        summary = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "model_path": str(model_path),
        }

        with open(
            seed_dir
            / "seed_summary.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        summaries.append(summary)

    # --------------------------------------------------------
    # Final training summary
    # --------------------------------------------------------
    with open(
        out_root
        / "training_5seed_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_csv(
        summaries,
        out_root
        / "training_5seed_summary.csv",
    )

    print("\n" + "=" * 100)
    print("ALL FIVE SEEDS COMPLETED")
    print("=" * 100)

    for row in summaries:
        print(
            f"seed={row['seed']} | "
            f"best_epoch={row['best_epoch']} | "
            f"best_val={row['best_val_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
