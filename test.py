#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSCF-Net FINAL 5-Seed Test
==========================


import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import spearmanr, kendalltau, t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

import matplotlib.pyplot as plt

# ============================================================
# CONFIG -- 修改路径即可
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 256
NUM_WORKERS = 0

FEAT_DIM = 28
HIDDEN_DIM = 64
EPS = 1e-8

CCM_CYCLES = 2
CCM_DROPOUT = 0.15
SCORE_ANCHORS = (0.2, 0.5, 0.8)
LABEL_SCORE_TOL = 1e-3

SEEDS = [42, 52, 62, 72, 82]

X_TEST_PATH = (
    r"\test_28d.npy"

)

Y_TEST_PATH = (

    r"\test_labels20new3.npy"
)

TRAINING_ROOT = (

    r"\model"
)

TRAIN_NORM_STATS_PATH = (
    TRAINING_ROOT
    + r"\train_normalization_stats.npz"
)

MODEL_PATHS = {
    seed: (
        TRAINING_ROOT
        + fr"\seed_{seed}\best_model_seed{seed}.pth"
    )
    for seed in SEEDS
}

OUTPUT_DIR = (
    TRAINING_ROOT
    + r"\final_5seed_test"
)

REQUIRE_ALL_MODELS = True
SAVE_CONFUSION_PLOTS = True

# ============================================================
# Output definitions
# ============================================================

# Exact model / label order.
MODEL_OUTPUT_ORDER = [
    "overall",
    "planning",
    "execution",
    "monitoring",
    "reflection",
]

REPORT_ORDER = [
    "planning",
    "execution",
    "monitoring",
    "reflection",
    "overall",
]

CLASS_NAMES = [
    "Low",
    "Medium",
    "High",
]

# ============================================================
# Model -- EXACT match to final training
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


class CyclicCCM(nn.Module):
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
                old[3],
                old[0],
                old[1],
                old[2],
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
            for extractor in self.phase_extractors
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
# Dataset + label audit
# ============================================================

class SRLTestDataset(Dataset):
    def __init__(
        self,
        X,
        y20,
        mean,
        std,
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

        if len(X) != len(y20):
            raise ValueError(
                "X/y length mismatch."
            )

        self.X = torch.tensor(
            (X - mean) / std,
            dtype=torch.float32,
        )

        self.y20 = torch.tensor(
            y20,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.y20[idx],
        )


def audit_label_score_relation(y20):
    soft = y20[:, :15].reshape(
        -1,
        5,
        3,
    )

    score = y20[:, 15:20]

    anchors = np.asarray(
        SCORE_ANCHORS,
        dtype=np.float64,
    ).reshape(1, 1, 3)

    implied = np.sum(
        soft.astype(np.float64) * anchors,
        axis=-1,
    )

    gap = np.abs(
        score.astype(np.float64) - implied
    )

    print(
        "[TEST label audit] "
        f"mean|score-E[p]|={gap.mean():.8f} | "
        f"max={gap.max():.8f}"
    )

    if gap.max() > LABEL_SCORE_TOL:
        raise ValueError(
            "TEST labels violate the configured probability-score relation. "
            f"max gap={gap.max():.8f}"
        )

# ============================================================
# Probability-derived score
# ============================================================

def expected_score_from_logits(logits):
    probs = F.softmax(
        logits,
        dim=-1,
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

# ============================================================
# Correlations
# ============================================================

def safe_spearman(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if (
        np.std(y_true) <= EPS
        or np.std(y_pred) <= EPS
    ):
        return float("nan")

    result = spearmanr(
        y_true,
        y_pred,
    )

    return float(
        result.statistic
        if hasattr(result, "statistic")
        else result[0]
    )


def safe_kendall(y_true, y_pred):
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if (
        np.std(y_true) <= EPS
        or np.std(y_pred) <= EPS
    ):
        return float("nan")

    result = kendalltau(
        y_true,
        y_pred,
    )

    return float(
        result.statistic
        if hasattr(result, "statistic")
        else result[0]
    )

# ============================================================
# Complete Test inference
# ============================================================

@torch.no_grad()
def collect_predictions(
    model,
    loader,
):
    model.eval()

    true_probs = {
        name: []
        for name in MODEL_OUTPUT_ORDER
    }

    pred_probs = {
        name: []
        for name in MODEL_OUTPUT_ORDER
    }

    true_scores = {
        name: []
        for name in MODEL_OUTPUT_ORDER
    }

    head_scores = {
        name: []
        for name in MODEL_OUTPUT_ORDER
    }

    prob_scores = {
        name: []
        for name in MODEL_OUTPUT_ORDER
    }

    for x, y20 in loader:
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y20 = y20.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(x)

        logits_list = outputs["logits_list"]
        score_preds = outputs["score_preds"]

        target_prob_list = torch.split(
            y20[:, :15],
            3,
            dim=-1,
        )

        target_score_list = torch.split(
            y20[:, 15:20],
            1,
            dim=-1,
        )

        for (
            name,
            logits,
            score_head,
            target_prob,
            target_score,
        ) in zip(
            MODEL_OUTPUT_ORDER,
            logits_list,
            score_preds,
            target_prob_list,
            target_score_list,
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
                score_head.cpu().numpy().reshape(-1)
            )

            prob_scores[name].append(
                prob_score.cpu().numpy().reshape(-1)
            )

    for name in MODEL_OUTPUT_ORDER:
        true_probs[name] = np.concatenate(
            true_probs[name],
            axis=0,
        )

        pred_probs[name] = np.concatenate(
            pred_probs[name],
            axis=0,
        )

        true_scores[name] = np.concatenate(
            true_scores[name],
            axis=0,
        )

        head_scores[name] = np.concatenate(
            head_scores[name],
            axis=0,
        )

        prob_scores[name] = np.concatenate(
            prob_scores[name],
            axis=0,
        )

    return (
        true_probs,
        pred_probs,
        true_scores,
        head_scores,
        prob_scores,
    )

# ============================================================
# Metrics
# ============================================================

def evaluate_one_output(
    true_probs,
    pred_probs,
    true_scores,
    head_scores,
    prob_scores,
):
    true_probs = np.asarray(
        true_probs,
        dtype=np.float64,
    )

    pred_probs = np.asarray(
        pred_probs,
        dtype=np.float64,
    )

    true_scores = np.asarray(
        true_scores,
        dtype=np.float64,
    ).reshape(-1)

    head_scores = np.asarray(
        head_scores,
        dtype=np.float64,
    ).reshape(-1)

    prob_scores = np.asarray(
        prob_scores,
        dtype=np.float64,
    ).reshape(-1)

    true_cls = np.argmax(
        true_probs,
        axis=1,
    )

    pred_cls = np.argmax(
        pred_probs,
        axis=1,
    )

    # --------------------------------------------------------
    # Primary classification metrics: standard observed-class definitions
    # --------------------------------------------------------
    accuracy = float(
        accuracy_score(
            true_cls,
            pred_cls,
        )
    )

    weighted_f1 = float(
        f1_score(
            true_cls,
            pred_cls,
            average="weighted",
            zero_division=0,
        )
    )

    # Primary Macro-F1 uses the same evaluable class set as BA:
    # classes that are actually present in y_true for this output.
    # This avoids sklearn's default union(y_true, y_pred) behavior from
    # silently adding a predicted-only class when its true support is zero.
    observed_labels = np.unique(
        true_cls
    )

    macro_f1 = float(
        f1_score(
            true_cls,
            pred_cls,
            labels=observed_labels,
            average="macro",
            zero_division=0,
        )
    )

    # sklearn BA = macro-average recall over classes present in y_true.
    balanced_accuracy = float(
        balanced_accuracy_score(
            true_cls,
            pred_cls,
        )
    )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            true_cls,
            pred_cls,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        true_cls,
        pred_cls,
        labels=[0, 1, 2],
    )

    # --------------------------------------------------------
    # Fixed-3-class sensitivity diagnostics
    # --------------------------------------------------------
    macro_f1_fixed3_zero = float(
        f1_score(
            true_cls,
            pred_cls,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )

    balanced_accuracy_fixed3_zero = float(
        np.mean(
            recall
        )
    )

    # --------------------------------------------------------
    # Soft target fit
    # --------------------------------------------------------
    brier = float(
        np.mean(
            np.sum(
                (
                    pred_probs - true_probs
                ) ** 2,
                axis=1,
            )
        )
    )

    target_safe = np.clip(
        true_probs,
        EPS,
        1.0,
    )

    pred_safe = np.clip(
        pred_probs,
        EPS,
        1.0,
    )

    kl = float(
        np.mean(
            np.sum(
                target_safe
                * (
                    np.log(target_safe)
                    - np.log(pred_safe)
                ),
                axis=1,
            )
        )
    )

    # --------------------------------------------------------
    # Independent score head
    # --------------------------------------------------------
    head_mae = float(
        mean_absolute_error(
            true_scores,
            head_scores,
        )
    )

    head_rmse = float(
        math.sqrt(
            mean_squared_error(
                true_scores,
                head_scores,
            )
        )
    )

    head_spearman = safe_spearman(
        true_scores,
        head_scores,
    )

    head_kendall = safe_kendall(
        true_scores,
        head_scores,
    )

    # --------------------------------------------------------
    # PRIMARY continuous score = probability-derived
    # --------------------------------------------------------
    prob_score_mae = float(
        mean_absolute_error(
            true_scores,
            prob_scores,
        )
    )

    prob_score_rmse = float(
        math.sqrt(
            mean_squared_error(
                true_scores,
                prob_scores,
            )
        )
    )

    prob_score_spearman = safe_spearman(
        true_scores,
        prob_scores,
    )

    prob_score_kendall = safe_kendall(
        true_scores,
        prob_scores,
    )

    head_vs_probscore_mae = float(
        mean_absolute_error(
            head_scores,
            prob_scores,
        )
    )

    result = {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,

        "macro_f1_fixed3_zero": macro_f1_fixed3_zero,
        "balanced_accuracy_fixed3_zero": (
            balanced_accuracy_fixed3_zero
        ),
        "n_observed_classes": int(
            len(observed_labels)
        ),
        "observed_class_indices": [
            int(x)
            for x in observed_labels.tolist()
        ],

        "brier_target_fit": brier,
        "kl_target_fit": kl,

        "head_mae": head_mae,
        "head_rmse": head_rmse,
        "head_spearman": head_spearman,
        "head_kendall": head_kendall,

        "prob_score_mae": prob_score_mae,
        "prob_score_rmse": prob_score_rmse,
        "prob_score_spearman": prob_score_spearman,
        "prob_score_kendall": prob_score_kendall,

        "head_vs_probscore_mae": head_vs_probscore_mae,

        "confusion_matrix": cm.tolist(),
        "per_class": {},
    }

    for i, class_name in enumerate(
        CLASS_NAMES
    ):
        result[
            "per_class"
        ][
            class_name
        ] = {
            "precision": float(
                precision[i]
            ),
            "recall": float(
                recall[i]
            ),
            "f1": float(
                class_f1[i]
            ),
            "support": int(
                support[i]
            ),
            "present_in_test": bool(
                support[i] > 0
            ),
        }

    return result

# ============================================================
# Pooled and Macro-output
# ============================================================

def evaluate_pooled_5n(
    true_probs,
    pred_probs,
    true_scores,
    head_scores,
    prob_scores,
    per_output_metrics,
):
    pooled = evaluate_one_output(
        np.concatenate(
            [
                true_probs[name]
                for name in REPORT_ORDER
            ],
            axis=0,
        ),
        np.concatenate(
            [
                pred_probs[name]
                for name in REPORT_ORDER
            ],
            axis=0,
        ),
        np.concatenate(
            [
                true_scores[name]
                for name in REPORT_ORDER
            ],
            axis=0,
        ),
        np.concatenate(
            [
                head_scores[name]
                for name in REPORT_ORDER
            ],
            axis=0,
        ),
        np.concatenate(
            [
                prob_scores[name]
                for name in REPORT_ORDER
            ],
            axis=0,
        ),
    )

    # Direct 5N correlations are diagnostic only.
    pooled[
        "head_spearman_5n_diagnostic"
    ] = pooled[
        "head_spearman"
    ]

    pooled[
        "head_kendall_5n_diagnostic"
    ] = pooled[
        "head_kendall"
    ]

    pooled[
        "prob_score_spearman_5n_diagnostic"
    ] = pooled[
        "prob_score_spearman"
    ]

    pooled[
        "prob_score_kendall_5n_diagnostic"
    ] = pooled[
        "prob_score_kendall"
    ]

    # Reported pooled rank = macro-average of five output-wise coefficients.
    for key in [
        "head_spearman",
        "head_kendall",
        "prob_score_spearman",
        "prob_score_kendall",
    ]:
        vals = np.asarray(
            [
                per_output_metrics[name][key]
                for name in REPORT_ORDER
            ],
            dtype=np.float64,
        )

        vals = vals[
            np.isfinite(vals)
        ]

        pooled[key] = (
            float(np.mean(vals))
            if len(vals) > 0
            else float("nan")
        )

    return pooled


def evaluate_macro_output(
    per_output_metrics,
):
    keys = [
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "balanced_accuracy",
        "macro_f1_fixed3_zero",
        "balanced_accuracy_fixed3_zero",
        "brier_target_fit",
        "kl_target_fit",
        "head_mae",
        "head_rmse",
        "head_spearman",
        "head_kendall",
        "prob_score_mae",
        "prob_score_rmse",
        "prob_score_spearman",
        "prob_score_kendall",
        "head_vs_probscore_mae",
    ]

    result = {}

    for key in keys:
        vals = np.asarray(
            [
                per_output_metrics[name][key]
                for name in REPORT_ORDER
            ],
            dtype=np.float64,
        )

        vals = vals[
            np.isfinite(vals)
        ]

        result[key] = (
            float(np.mean(vals))
            if len(vals) > 0
            else float("nan")
        )

    return result

# ============================================================
# Windows / Unicode-safe torch serialization
# ============================================================

def safe_torch_load(path, map_location):
    """
    Robust torch.load for Windows paths containing Chinese/non-ASCII chars.

    Uses Python's file handle instead of passing the Unicode path directly
    into PyTorch's lower-level path handling.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    with open(path, "rb") as f:
        return torch.load(
            f,
            map_location=map_location,
        )


# ============================================================
# Model loading
# ============================================================

def load_model(
    model_path,
):
    checkpoint = safe_torch_load(
        model_path,
        map_location=DEVICE,
    )

    if isinstance(
        checkpoint,
        dict,
    ):
        config = checkpoint.get(
            "config",
            {},
        )

        if config:
            saved_cycles = config.get(
                "ccm_cycles",
                CCM_CYCLES,
            )

            if saved_cycles != CCM_CYCLES:
                raise ValueError(
                    f"CCM_CYCLES mismatch: "
                    f"test={CCM_CYCLES}, checkpoint={saved_cycles}"
                )

            saved_anchors = tuple(
                config.get(
                    "score_anchors",
                    SCORE_ANCHORS,
                )
            )

            if saved_anchors != tuple(
                SCORE_ANCHORS
            ):
                raise ValueError(
                    f"SCORE_ANCHORS mismatch: "
                    f"test={SCORE_ANCHORS}, checkpoint={saved_anchors}"
                )

    model = ImprovedSRLModel(
        feat_dim=FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        ccm_cycles=CCM_CYCLES,
    ).to(
        DEVICE
    )

    state_dict = (
        checkpoint["model_state_dict"]
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        )
        else checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    return model, checkpoint

# ============================================================
# Plot
# ============================================================

def save_confusion_plot(
    cm,
    title,
    save_path,
):
    cm = np.asarray(
        cm,
        dtype=np.int64,
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    im = ax.imshow(
        cm
    )

    ax.set_xticks(
        [0, 1, 2]
    )
    ax.set_yticks(
        [0, 1, 2]
    )

    ax.set_xticklabels(
        CLASS_NAMES
    )
    ax.set_yticklabels(
        CLASS_NAMES
    )

    ax.set_xlabel(
        "Predicted class"
    )
    ax.set_ylabel(
        "Target argmax class"
    )
    ax.set_title(
        title
    )

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        im,
        ax=ax,
    )

    fig.tight_layout()
    fig.savefig(
        save_path,
        dpi=250,
    )
    plt.close(fig)

# ============================================================
# Per-seed evaluation
# ============================================================

def evaluate_seed(
    seed,
    model_path,
    test_loader,
    output_root,
):
    seed_dir = (
        output_root
        / f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, checkpoint = load_model(
        model_path
    )

    (
        true_probs,
        pred_probs,
        true_scores,
        head_scores,
        prob_scores,
    ) = collect_predictions(
        model,
        test_loader,
    )

    metrics = {}

    for name in REPORT_ORDER:
        metrics[name] = evaluate_one_output(
            true_probs[name],
            pred_probs[name],
            true_scores[name],
            head_scores[name],
            prob_scores[name],
        )

    # Diagnostic only: concatenate all five outputs to 5N records.
    # This is mathematically valid, but nonlinear metrics such as Macro-F1
    # and BA can be strongly altered by output-specific class-support mixing.
    pooled_5n = evaluate_pooled_5n(
        true_probs=true_probs,
        pred_probs=pred_probs,
        true_scores=true_scores,
        head_scores=head_scores,
        prob_scores=prob_scores,
        per_output_metrics=metrics,
    )

    # PRIMARY paper summary:
    # equal-weight arithmetic mean across the five outputs.
    # This makes the summary estimand consistent for ACC/W-F1/M-F1/BA/
    # Brier/KL/MAE/RMSE/Spearman/Kendall.
    pooled_output_macro = evaluate_macro_output(
        metrics
    )

    result = {
        "seed": seed,
        "checkpoint": {
            "path": str(model_path),
            "epoch": (
                checkpoint.get("epoch")
                if isinstance(checkpoint, dict)
                else None
            ),
            "val_loss": (
                checkpoint.get("val_loss")
                if isinstance(checkpoint, dict)
                else None
            ),
        },
        "per_output": metrics,

        # Primary summary used in the paper-facing table.
        "pooled_output_macro": pooled_output_macro,

        # Backward-compatible alias.
        "macro_output": pooled_output_macro,

        # Diagnostic / continuity result only.
        "pooled_5n_diagnostic": pooled_5n,
        "pooled_5n": pooled_5n,
    }

    with open(
        seed_dir
        / "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save raw predictions for paired bootstrap / later statistical tests.
    save_dict = {}

    for name in REPORT_ORDER:
        save_dict[
            f"{name}_true_probs"
        ] = true_probs[name]

        save_dict[
            f"{name}_pred_probs"
        ] = pred_probs[name]

        save_dict[
            f"{name}_true_scores"
        ] = true_scores[name]

        save_dict[
            f"{name}_head_scores"
        ] = head_scores[name]

        save_dict[
            f"{name}_prob_scores"
        ] = prob_scores[name]

    np.savez_compressed(
        seed_dir
        / "test_predictions.npz",
        **save_dict,
    )

    # Per-class CSV
    class_rows = []

    for name in REPORT_ORDER + [
        "pooled_5n"
    ]:
        m = (
            pooled_5n
            if name == "pooled_5n"
            else metrics[name]
        )

        for class_name in CLASS_NAMES:
            c = m[
                "per_class"
            ][
                class_name
            ]

            class_rows.append(
                {
                    "seed": seed,
                    "output": name,
                    "class": class_name,
                    "precision": c["precision"],
                    "recall": c["recall"],
                    "f1": c["f1"],
                    "support": c["support"],
                    "present_in_test": c["present_in_test"],
                }
            )

    with open(
        seed_dir
        / "per_class_metrics.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                class_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            class_rows
        )

    # Confusion matrices
    if SAVE_CONFUSION_PLOTS:
        for name in REPORT_ORDER:
            save_confusion_plot(
                metrics[name][
                    "confusion_matrix"
                ],
                f"Confusion Matrix - {name.capitalize()}",
                seed_dir
                / f"confusion_{name}.png",
            )

        save_confusion_plot(
            pooled_5n[
                "confusion_matrix"
            ],
            "Confusion Matrix - Pooled 5N",
            seed_dir
            / "confusion_pooled_5n.png",
        )

    return result

# ============================================================
# Multi-seed aggregation
# ============================================================

PRIMARY_SCALAR_KEYS = [
    "accuracy",
    "weighted_f1",
    "macro_f1",
    "balanced_accuracy",
    "brier_target_fit",
    "kl_target_fit",
    "prob_score_mae",
    "prob_score_rmse",
    "prob_score_spearman",
    "prob_score_kendall",
]

DIAGNOSTIC_SCALAR_KEYS = [
    "macro_f1_fixed3_zero",
    "balanced_accuracy_fixed3_zero",
    "head_mae",
    "head_rmse",
    "head_spearman",
    "head_kendall",
    "head_vs_probscore_mae",
]

ALL_SCALAR_KEYS = (
    PRIMARY_SCALAR_KEYS
    + DIAGNOSTIC_SCALAR_KEYS
)


def summarize_values(values):
    arr = np.asarray(
        values,
        dtype=np.float64,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    n = len(arr)

    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    mean = float(
        np.mean(arr)
    )

    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "sd": None,
            "ci95_low": None,
            "ci95_high": None,
        }

    sd = float(
        np.std(
            arr,
            ddof=1,
        )
    )

    critical = float(
        t.ppf(
            0.975,
            df=n - 1,
        )
    )

    margin = (
        critical
        * sd
        / math.sqrt(n)
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
    }


def get_metric_block(
    seed_result,
    output_name,
):
    if output_name in {
        "pooled",
        "pooled_output_macro",
        "macro_output",
    }:
        return seed_result[
            "pooled_output_macro"
        ]

    if output_name in {
        "pooled_5n",
        "pooled_5n_diagnostic",
    }:
        return seed_result[
            "pooled_5n_diagnostic"
        ]

    return seed_result[
        "per_output"
    ][
        output_name
    ]


def aggregate_multiseed(
    all_results,
    output_root,
):
    outputs = (
        REPORT_ORDER
        + [
            "pooled_output_macro",
            "pooled_5n_diagnostic",
        ]
    )

    # --------------------------------------------------------
    # Raw seed-level metrics
    # --------------------------------------------------------
    raw_rows = []

    for seed, result in all_results.items():
        for output_name in outputs:
            m = get_metric_block(
                result,
                output_name,
            )

            row = {
                "seed": seed,
                "output": output_name,
            }

            for key in ALL_SCALAR_KEYS:
                row[key] = m[key]

            raw_rows.append(row)

    with open(
        output_root
        / "multiseed_raw_metrics.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                raw_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(raw_rows)

    # --------------------------------------------------------
    # Long mean / SD / 95CI
    # --------------------------------------------------------
    summary_rows = []

    for output_name in outputs:
        for key in ALL_SCALAR_KEYS:
            values = [
                get_metric_block(
                    result,
                    output_name,
                )[key]
                for result
                in all_results.values()
            ]

            s = summarize_values(
                values
            )

            summary_rows.append(
                {
                    "output": output_name,
                    "metric": key,
                    "n_seeds": s["n"],
                    "mean": s["mean"],
                    "sd": s["sd"],
                    "ci95_low": s["ci95_low"],
                    "ci95_high": s["ci95_high"],
                }
            )

    with open(
        output_root
        / "multiseed_summary_mean_sd_ci95.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Primary paper-facing table
    #
    # Includes 5 actual outputs + Pooled-5N.
    # Macro-output exported separately as sensitivity.
    # --------------------------------------------------------
    paper_outputs = (
        REPORT_ORDER
        + [
            "pooled_output_macro"
        ]
    )

    paper_rows = []

    for output_name in paper_outputs:
        row = {
            "Output": (
                "Pooled (Output-Macro)"
                if output_name == "pooled_output_macro"
                else output_name.capitalize()
            )
        }

        mapping = [
            (
                "accuracy",
                "ACC (%) ↑",
                True,
                2,
            ),
            (
                "weighted_f1",
                "Weighted-F1 (%) ↑",
                True,
                2,
            ),
            (
                "macro_f1",
                "Macro-F1 (%) ↑",
                True,
                2,
            ),
            (
                "balanced_accuracy",
                "BA (%) ↑",
                True,
                2,
            ),
            (
                "brier_target_fit",
                "Brier ↓",
                False,
                4,
            ),
            (
                "kl_target_fit",
                "KL ↓",
                False,
                4,
            ),
            (
                "prob_score_mae",
                "MAE ↓",
                False,
                4,
            ),
            (
                "prob_score_rmse",
                "RMSE ↓",
                False,
                4,
            ),
            (
                "prob_score_spearman",
                "Spearman ρ ↑",
                False,
                4,
            ),
            (
                "prob_score_kendall",
                "Kendall τ ↑",
                False,
                4,
            ),
        ]

        for key, label, is_percent, decimals in mapping:
            values = [
                get_metric_block(
                    result,
                    output_name,
                )[key]
                for result
                in all_results.values()
            ]

            s = summarize_values(
                values
            )

            scale = (
                100.0
                if is_percent
                else 1.0
            )

            mean = (
                s["mean"] * scale
            )

            sd = (
                s["sd"] * scale
            )

            ci_low = (
                s["ci95_low"] * scale
            )

            ci_high = (
                s["ci95_high"] * scale
            )

            row[label] = (
                f"{mean:.{decimals}f} ± "
                f"{sd:.{decimals}f}"
            )

            row[
                label + " 95% CI"
            ] = (
                f"[{ci_low:.{decimals}f}, "
                f"{ci_high:.{decimals}f}]"
            )

        paper_rows.append(row)

    with open(
        output_root
        / "paper_table_primary_mean_sd.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                paper_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            paper_rows
        )

    # --------------------------------------------------------
    # Separate 5N pooled diagnostic table.
    # Do not mix this row into the primary paper table.
    # --------------------------------------------------------
    pooled5n_rows = []

    for key in ALL_SCALAR_KEYS:
        values = [
            get_metric_block(
                result,
                "pooled_5n_diagnostic",
            )[key]
            for result in all_results.values()
        ]

        s = summarize_values(
            values
        )

        pooled5n_rows.append(
            {
                "metric": key,
                "mean": s["mean"],
                "sd": s["sd"],
                "ci95_low": s["ci95_low"],
                "ci95_high": s["ci95_high"],
                "interpretation": (
                    "5N concatenation diagnostic; not the primary Pooled summary"
                ),
            }
        )

    with open(
        output_root
        / "pooled_5n_diagnostic_mean_sd_ci95.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                pooled5n_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            pooled5n_rows
        )

    # --------------------------------------------------------
    # Macro-output + fixed-class sensitivity table
    # --------------------------------------------------------
    sensitivity_rows = []

    for output_name in outputs:
        mrow = {
            "Output": output_name,
        }

        for key in [
            "macro_f1",
            "macro_f1_fixed3_zero",
            "balanced_accuracy",
            "balanced_accuracy_fixed3_zero",
        ]:
            values = [
                get_metric_block(
                    result,
                    output_name,
                )[key]
                for result
                in all_results.values()
            ]

            s = summarize_values(
                values
            )

            mrow[key] = (
                f"{s['mean'] * 100:.2f} ± "
                f"{s['sd'] * 100:.2f}"
            )

        sensitivity_rows.append(
            mrow
        )

    with open(
        output_root
        / "classification_definition_sensitivity.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                sensitivity_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            sensitivity_rows
        )

    # --------------------------------------------------------
    # Test-target class support audit (same targets for every seed)
    # --------------------------------------------------------
    first_result = next(
        iter(
            all_results.values()
        )
    )

    support_rows = []

    for output_name in REPORT_ORDER:
        block = first_result[
            "per_output"
        ][
            output_name
        ]

        for class_name in CLASS_NAMES:
            c = block[
                "per_class"
            ][
                class_name
            ]

            support_rows.append(
                {
                    "output": output_name,
                    "class": class_name,
                    "support": c["support"],
                    "present_in_test": c["present_in_test"],
                    "warning": (
                        "ABSENT CLASS: class-wise recall/F1 cannot be empirically "
                        "evaluated on this Test split"
                        if c["support"] == 0
                        else (
                            "EXTREMELY SPARSE CLASS"
                            if c["support"] < 10
                            else ""
                        )
                    ),
                }
            )

    with open(
        output_root
        / "test_target_class_support_audit.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                support_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            support_rows
        )

    # --------------------------------------------------------
    # Aggregate class-wise metrics across seeds
    # --------------------------------------------------------
    class_rows = []

    for output_name in (
        REPORT_ORDER
        + ["pooled_5n_diagnostic"]
    ):
        for class_name in CLASS_NAMES:
            # Support identical across seeds, but read from first run.
            first_result = next(
                iter(
                    all_results.values()
                )
            )

            first_block = get_metric_block(
                first_result,
                output_name,
            )

            support = first_block[
                "per_class"
            ][
                class_name
            ][
                "support"
            ]

            for metric_name in [
                "precision",
                "recall",
                "f1",
            ]:
                values = []

                for result in all_results.values():
                    block = get_metric_block(
                        result,
                        output_name,
                    )

                    values.append(
                        block[
                            "per_class"
                        ][
                            class_name
                        ][
                            metric_name
                        ]
                    )

                s = summarize_values(
                    values
                )

                class_rows.append(
                    {
                        "output": output_name,
                        "class": class_name,
                        "support": support,
                        "metric": metric_name,
                        "mean": s["mean"],
                        "sd": s["sd"],
                        "ci95_low": s["ci95_low"],
                        "ci95_high": s["ci95_high"],
                    }
                )

    with open(
        output_root
        / "per_class_multiseed_summary.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                class_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            class_rows
        )

# ============================================================
# Main
# ============================================================

def main():
    output_root = Path(
        OUTPUT_DIR
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("CSCF-Net FINAL 5-SEED TEST")
    print(f"DEVICE={DEVICE}")
    print(f"SEEDS={SEEDS}")
    print("=" * 100)

    # --------------------------------------------------------
    # Path checks
    # --------------------------------------------------------
    required_paths = [
        ("X_TEST", X_TEST_PATH),
        ("Y_TEST", Y_TEST_PATH),
        ("TRAIN_NORM", TRAIN_NORM_STATS_PATH),
    ]

    for label, path in required_paths:
        exists = Path(path).exists()

        print(
            f"{label:12s} exists={exists} | {path}"
        )

        if not exists:
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    available_models = {}

    for seed, path in MODEL_PATHS.items():
        exists = Path(path).exists()

        print(
            f"MODEL seed={seed} exists={exists} | {path}"
        )

        if exists:
            available_models[seed] = path

    if (
        REQUIRE_ALL_MODELS
        and len(available_models)
        != len(MODEL_PATHS)
    ):
        raise FileNotFoundError(
            "REQUIRE_ALL_MODELS=True, but one or more "
            "final seed checkpoints are missing."
        )

    if not available_models:
        raise FileNotFoundError(
            "No model checkpoint found."
        )

    # --------------------------------------------------------
    # Test data + audit
    # --------------------------------------------------------
    X_test = np.load(
        X_TEST_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    print(
        f"X_test={X_test.shape}"
    )

    print(
        f"y_test={y_test.shape}"
    )

    audit_label_score_relation(
        y_test
    )

    # --------------------------------------------------------
    # TRAIN normalization only
    # --------------------------------------------------------
    norm = np.load(
        TRAIN_NORM_STATS_PATH,
        allow_pickle=False,
    )

    mean = np.asarray(
        norm["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        norm["std"],
        dtype=np.float32,
    )

    test_dataset = SRLTestDataset(
        X_test,
        y_test,
        mean,
        std,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Each seed evaluated independently
    # --------------------------------------------------------
    all_results = {}

    for seed, model_path in available_models.items():
        print("\n" + "-" * 100)
        print(f"Evaluating seed {seed}")
        print("-" * 100)

        result = evaluate_seed(
            seed=seed,
            model_path=model_path,
            test_loader=test_loader,
            output_root=output_root,
        )

        all_results[
            seed
        ] = result

        o = result[
            "per_output"
        ][
            "overall"
        ]

        p = result[
            "pooled_output_macro"
        ]

        print(
            f"Seed {seed} | "
            f"Overall ACC={o['accuracy'] * 100:.2f}% | "
            f"rho={o['prob_score_spearman']:.4f} | "
            f"tau={o['prob_score_kendall']:.4f}"
        )

        print(
            f"Seed {seed} | "
            f"Pooled ACC={p['accuracy'] * 100:.2f}% | "
            f"rho={p['prob_score_spearman']:.4f} | "
            f"tau={p['prob_score_kendall']:.4f}"
        )

    # --------------------------------------------------------
    # Multi-seed summary
    # --------------------------------------------------------
    aggregate_multiseed(
        all_results,
        output_root,
    )

    # --------------------------------------------------------
    # Save protocol
    # --------------------------------------------------------
    protocol = {
        "real_outputs": REPORT_ORDER,
        "overall": (
            "dedicated integrated model output"
        ),
        "pooled_primary": (
            "equal-weight arithmetic mean across Planning, Execution, "
            "Monitoring, Reflection, and Overall. This is the paper-facing "
            "Pooled summary."
        ),
        "pooled_5n_diagnostic": (
            "concatenation of all five outputs to 5N user-output records; "
            "retained only as a diagnostic/continuity statistic, not used "
            "for the paper-facing Pooled row."
        ),
        "hard_target": (
            "argmax of soft target probability"
        ),
        "hard_prediction": (
            "argmax of model predicted probability"
        ),
        "primary_macro_f1": (
            "macro-F1 over labels actually present in y_true for each output; "
            "predicted-only absent classes are not silently added to the "
            "averaging set. Fixed 3-class Macro-F1 is reported separately "
            "as a sensitivity statistic."
        ),
        "primary_balanced_accuracy": (
            "sklearn balanced accuracy; mean recall over classes present in y_true"
        ),
        "fixed3_sensitivity": (
            "also reports fixed Low/Medium/High Macro-F1 and BA with "
            "missing-class contribution set to zero"
        ),
        "soft_target_fit": [
            "Brier target-fit",
            "KL(target || prediction)",
        ],
        "primary_continuous_score": (
            "0.2*p_low + 0.5*p_mid + 0.8*p_high "
            "from predicted probabilities"
        ),
        "primary_score_metrics": [
            "MAE",
            "RMSE",
            "Spearman rho",
            "Kendall tau-b",
        ],
        "secondary_score_diagnostic": (
            "independent score-head prediction"
        ),
        "pooled_metric_rule": (
            "all primary Pooled metrics are macro-averaged across the five "
            "outputs, so ACC/W-F1/M-F1/BA/Brier/KL/MAE/RMSE/rank metrics "
            "share the same equal-output estimand"
        ),
        "direct_5n_metrics": (
            "saved only as diagnostics because class-support mixing can "
            "inflate/alter nonlinear classification metrics and heterogeneous "
            "score distributions can inflate direct rank correlations"
        ),
        "normalization": (
            "TRAIN mean/std only"
        ),
        "test_confidence_weighting": False,
        "nonlinear_batch_averaging": False,
        "multiseed_summary": (
            "mean, sample SD, t-based 95% CI across seeds"
        ),
    }

    with open(
        output_root
        / "evaluation_protocol.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            protocol,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("FINAL TEST COMPLETED")
    print("=" * 100)
    print(
        output_root
        / "paper_table_primary_mean_sd.csv"
    )
    print(
        output_root
        / "pooled_5n_diagnostic_mean_sd_ci95.csv"
    )
    print(
        output_root
        / "multiseed_summary_mean_sd_ci95.csv"
    )
    print(
        output_root
        / "classification_definition_sensitivity.csv"
    )
    print(
        output_root
        / "per_class_multiseed_summary.csv"
    )
    print(
        output_root
        / "evaluation_protocol.json"
    )


if __name__ == "__main__":
    main()
