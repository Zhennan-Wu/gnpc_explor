#!/usr/bin/env python3
"""Improved CLOUDS continuous-time anchored factor simulation pipeline.

Major changes from gem7:
- Exact observed-data log likelihood from Kalman innovations.
- Held-out subject likelihood as the default model-selection score.
- Correct covariance-scaled PCA warm start and one cached low-rank PCA per data set.
- Magnitude-aware anchor scoring with a global Hungarian assignment.
- Identifiable correlation, SPD, and skew-symmetric parameterizations.
- Identifiable diagonal mode with Omega fixed to the identity.
- Observation intercepts and covariate-dependent latent levels and slopes.
- Fresh multistart scoring, best-epoch checkpointing, and nonfinite rollback.
- Stable Cholesky solves, relative jitter, covariance symmetrization, and bounds.
- Thread-aware multiprocessing, queue-based logging, and JSON/CSV outputs.
- Optional misspecification stressors in the simulator.

The implementation is generalized EM by default: Kalman smoothing is the E-step,
while Adam approximately maximizes the expected complete-data likelihood. Optional
regularization turns the fit into generalized MAP-EM. Observed-data likelihood is
used for fresh checkpointing and model selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import logging
import logging.handlers
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Set conservative numerical-library defaults before importing NumPy, SciPy, or Torch.
_DEFAULT_THREADS = max(1, int(os.environ.get("CLOUDS_THREADS_PER_WORKER", "1")))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_DEFAULT_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_DEFAULT_THREADS))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import linear_sum_assignment

LOGGER = logging.getLogger("clouds")
LOG_2PI = math.log(2.0 * math.pi)

try:
    torch.set_num_threads(_DEFAULT_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch permits setting inter-op threads only before parallel work starts.
    pass
torch.set_flush_denormal(True)


# -----------------------------------------------------------------------------
# Configuration and result containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorConfig:
    """Optional regularization. Defaults are zero so BIC uses an ML fit."""

    loading_l2: float = 0.0
    intercept_l2: float = 0.0
    mean_l2: float = 0.0
    log_psi_l2: float = 0.0
    correlation_l2: float = 0.0
    dynamics_l1: float = 0.0
    diagonal_rate_l2: float = 0.0

    @classmethod
    def weak_map(cls) -> "PriorConfig":
        """A conservative regularization profile for difficult real data."""
        return cls(
            loading_l2=5e-4,
            intercept_l2=1e-4,
            mean_l2=5e-3,
            log_psi_l2=5e-4,
            correlation_l2=5e-3,
            dynamics_l1=2e-2,
            diagonal_rate_l2=5e-3,
        )

    def is_zero(self) -> bool:
        return all(value == 0.0 for value in dataclasses.asdict(self).values())


@dataclass(frozen=True)
class FitConfig:
    total_em_epochs: int = 30
    burn_in_epochs: int = 5
    warmup_epochs: int = 8
    m_step_iters: int = 10
    learning_rate: float = 1e-2
    spatial_lr_scale: float = 5e-2
    n_starts: int = 3
    gradient_clip: float = 2.0
    early_stopping_patience: int = 8
    min_improvement: float = 1e-4
    max_epoch_failures: int = 3
    prior: PriorConfig = field(default_factory=PriorConfig)

    def validate(self) -> None:
        if self.total_em_epochs < self.burn_in_epochs:
            raise ValueError("total_em_epochs must be at least burn_in_epochs")
        if self.burn_in_epochs < 1 or self.m_step_iters < 1 or self.n_starts < 1:
            raise ValueError("burn_in_epochs, m_step_iters, and n_starts must be positive")
        if self.learning_rate <= 0.0 or self.spatial_lr_scale <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.max_epoch_failures < 0:
            raise ValueError("max_epoch_failures must be nonnegative")


@dataclass(frozen=True)
class SelectionConfig:
    validation_fraction: float = 0.20
    metric: str = "validation_nll"
    bic_sample_unit: str = "subjects"
    include_anchor_search_penalty: bool = True

    def validate(self) -> None:
        valid_metrics = {
            "validation_nll",
            "bic",
            "bic_anchor",
            "bic_subjects",
            "bic_visits",
            "bic_entries",
            "bic_anchor_subjects",
            "bic_anchor_visits",
            "bic_anchor_entries",
        }
        if self.metric not in valid_metrics:
            raise ValueError(f"Unknown selection metric: {self.metric}")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if self.bic_sample_unit not in {"subjects", "visits", "entries"}:
            raise ValueError("bic_sample_unit must be subjects, visits, or entries")
        if self.metric == "validation_nll" and self.validation_fraction <= 0.0:
            raise ValueError("validation_nll requires validation_fraction > 0")


@dataclass
class PosteriorStats:
    mean: torch.Tensor
    covariance: torch.Tensor
    lag_one_covariance: torch.Tensor


@dataclass
class PCAWorkspace:
    column_means: torch.Tensor
    feature_variances: torch.Tensor
    right_vectors: torch.Tensor
    singular_values: torch.Tensor
    n_rows: int
    valid_anchor_features: torch.Tensor
    rotated_cache: Dict[int, torch.Tensor] = field(default_factory=dict)

    def covariance_loadings(self, k: int) -> torch.Tensor:
        if k < 1 or k > self.right_vectors.shape[1]:
            raise ValueError(f"Requested k={k}, available rank={self.right_vectors.shape[1]}")
        denominator = math.sqrt(max(self.n_rows - 1, 1))
        return self.right_vectors[:, :k] * (self.singular_values[:k] / denominator)

    def rotated_loadings(self, k: int) -> torch.Tensor:
        if k not in self.rotated_cache:
            self.rotated_cache[k] = varimax_rotation(self.covariance_loadings(k))
        return self.rotated_cache[k]


@dataclass
class FitResult:
    smoothed_stats: List[PosteriorStats]
    history: List[Dict[str, Any]]
    train_log_likelihood: float
    train_log_posterior: float


# -----------------------------------------------------------------------------
# Stable linear algebra and constrained transforms
# -----------------------------------------------------------------------------
def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def stable_cholesky(
    matrix: torch.Tensor,
    initial_relative_jitter: float = 1e-6,
    max_relative_jitter: float = 1e-1,
) -> torch.Tensor:
    """Cholesky factorization with scale-aware jitter and an eigenvalue fallback."""
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("stable_cholesky requires square matrices")
    matrix = symmetrize(matrix)
    if not bool(torch.isfinite(matrix).all().item()):
        raise FloatingPointError("stable_cholesky received a nonfinite matrix")
    n = matrix.shape[-1]
    eye = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    minimum_scale = 1e-6 if matrix.dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-12
    scale = diagonal.abs().mean(dim=-1).clamp_min(minimum_scale)

    jitters = [0.0]
    jitter = initial_relative_jitter
    while jitter <= max_relative_jitter * (1.0 + 1e-12):
        jitters.append(jitter)
        jitter *= 10.0

    for relative_jitter in jitters:
        candidate = matrix + relative_jitter * scale[..., None, None] * eye
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        if bool(torch.all(info == 0).item()) and bool(torch.isfinite(factor).all().item()):
            return factor

    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    floor = initial_relative_jitter * scale[..., None]
    projected_values = torch.maximum(eigenvalues, floor)
    projected = eigenvectors @ torch.diag_embed(projected_values) @ eigenvectors.transpose(-1, -2)
    projected = symmetrize(projected)
    return torch.linalg.cholesky(projected)


def cholesky_logdet(factor: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(dim=-1)


def spd_inverse(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cholesky_inverse(stable_cholesky(matrix))


def spd_solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    factor = stable_cholesky(matrix)
    vector_rhs = rhs.dim() == matrix.dim() - 1
    rhs_matrix = rhs.unsqueeze(-1) if vector_rhs else rhs
    solution = torch.cholesky_solve(rhs_matrix, factor)
    return solution.squeeze(-1) if vector_rhs else solution


def softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp_min(torch.finfo(value.dtype).tiny)
    # Stable for both small and large values; unlike torch.where with expm1,
    # this does not evaluate an overflowing inactive branch.
    return value + torch.log(-torch.expm1(-value))


def bounded_from_raw(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def raw_from_bounded(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    scaled = ((value - lower) / (upper - lower)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(scaled) - torch.log1p(-scaled)


def num_correlation_parameters(k: int) -> int:
    return k * (k - 1) // 2


def num_lower_triangular_parameters(k: int) -> int:
    return k * (k + 1) // 2


def correlation_cholesky_from_raw(raw: torch.Tensor, k: int, margin: float = 1e-5) -> torch.Tensor:
    """Unique row-wise partial-correlation parameterization of a Cholesky factor."""
    expected = num_correlation_parameters(k)
    if raw.numel() != expected:
        raise ValueError(f"Expected {expected} correlation parameters, got {raw.numel()}")
    rows: List[torch.Tensor] = []
    cursor = 0
    for row_index in range(k):
        entries: List[torch.Tensor] = []
        product = torch.ones((), dtype=raw.dtype, device=raw.device)
        for _ in range(row_index):
            partial = (1.0 - margin) * torch.tanh(raw[cursor])
            cursor += 1
            entries.append(partial * product)
            product = product * torch.sqrt((1.0 - partial.square()).clamp_min(margin))
        entries.append(product)
        if row_index + 1 < k:
            entries.extend(
                [torch.zeros((), dtype=raw.dtype, device=raw.device) for _ in range(k - row_index - 1)]
            )
        rows.append(torch.stack(entries))
    return torch.stack(rows)


def raw_from_correlation(correlation: torch.Tensor, margin: float = 1e-5) -> torch.Tensor:
    correlation = symmetrize(correlation)
    k = correlation.shape[-1]
    factor = stable_cholesky(correlation)
    values: List[torch.Tensor] = []
    for row_index in range(k):
        product = torch.ones((), dtype=correlation.dtype, device=correlation.device)
        for column_index in range(row_index):
            partial = factor[row_index, column_index] / product.clamp_min(margin)
            partial = (partial / (1.0 - margin)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            values.append(torch.atanh(partial))
            product = product * torch.sqrt((1.0 - ((1.0 - margin) * partial).square()).clamp_min(margin))
    if not values:
        return torch.empty(0, dtype=correlation.dtype, device=correlation.device)
    return torch.stack(values)


def lower_cholesky_from_raw(raw: torch.Tensor, k: int, diagonal_floor: float = 1e-4) -> torch.Tensor:
    expected = num_lower_triangular_parameters(k)
    if raw.numel() != expected:
        raise ValueError(f"Expected {expected} lower-triangular parameters, got {raw.numel()}")
    matrix = torch.zeros(k, k, dtype=raw.dtype, device=raw.device)
    cursor = 0
    for row_index in range(k):
        for column_index in range(row_index + 1):
            value = raw[cursor]
            cursor += 1
            if row_index == column_index:
                value = diagonal_floor + F.softplus(value)
            matrix[row_index, column_index] = value
    return matrix


def raw_from_lower_cholesky(factor: torch.Tensor, diagonal_floor: float = 1e-4) -> torch.Tensor:
    k = factor.shape[-1]
    values: List[torch.Tensor] = []
    for row_index in range(k):
        for column_index in range(row_index + 1):
            value = factor[row_index, column_index]
            if row_index == column_index:
                value = softplus_inverse((value - diagonal_floor).clamp_min(1e-6))
            values.append(value)
    return torch.stack(values)


def skew_symmetric_from_raw(raw: torch.Tensor, k: int) -> torch.Tensor:
    expected = num_correlation_parameters(k)
    if raw.numel() != expected:
        raise ValueError(f"Expected {expected} skew parameters, got {raw.numel()}")
    lower = torch.zeros(k, k, dtype=raw.dtype, device=raw.device)
    cursor = 0
    for row_index in range(1, k):
        for column_index in range(row_index):
            lower[row_index, column_index] = raw[cursor]
            cursor += 1
    return lower - lower.transpose(-1, -2)


# -----------------------------------------------------------------------------
# Input validation, PCA, and anchor discovery
# -----------------------------------------------------------------------------
def validate_subjects(
    subjects_data: Sequence[Mapping[str, torch.Tensor]],
    obs_dim: Optional[int] = None,
    covar_dim: Optional[int] = None,
) -> Tuple[int, int]:
    if not subjects_data:
        raise ValueError("subjects_data must contain at least one subject")

    inferred_obs_dim: Optional[int] = obs_dim
    inferred_covar_dim: Optional[int] = covar_dim
    reference_dtype: Optional[torch.dtype] = None
    reference_device: Optional[torch.device] = None
    for subject_index, subject in enumerate(subjects_data):
        for required_key in ("x", "u", "t"):
            if required_key not in subject:
                raise KeyError(f"Subject {subject_index} is missing key {required_key!r}")
        x_obs = subject["x"]
        covariates = subject["u"]
        times = subject["t"]
        if x_obs.dim() != 2 or times.dim() != 1 or covariates.dim() != 1:
            raise ValueError(f"Subject {subject_index}: x must be 2D, u and t must be 1D")
        if not x_obs.is_floating_point() or not times.is_floating_point() or not covariates.is_floating_point():
            raise TypeError(f"Subject {subject_index}: x, u, and t must be floating-point tensors")
        if times.dtype != x_obs.dtype or covariates.dtype != x_obs.dtype:
            raise TypeError(f"Subject {subject_index}: x, u, and t must share one dtype")
        if times.device != x_obs.device or covariates.device != x_obs.device:
            raise ValueError(f"Subject {subject_index}: x, u, and t must share one device")
        if reference_dtype is None:
            reference_dtype = x_obs.dtype
            reference_device = x_obs.device
        elif x_obs.dtype != reference_dtype or x_obs.device != reference_device:
            raise ValueError("All subjects must share one dtype and device")
        if x_obs.shape[0] != times.numel():
            raise ValueError(f"Subject {subject_index}: x rows must equal number of times")
        if times.numel() < 1:
            raise ValueError(f"Subject {subject_index}: at least one visit is required")
        if not torch.isfinite(times).all() or not torch.isfinite(covariates).all():
            raise ValueError(f"Subject {subject_index}: times and covariates must be finite")
        if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1]).item()):
            raise ValueError(f"Subject {subject_index}: visit times must be strictly increasing")
        if torch.isinf(x_obs).any():
            raise ValueError(f"Subject {subject_index}: x may contain NaN but not infinity")
        if not bool((~torch.isnan(x_obs)).any().item()):
            raise ValueError(f"Subject {subject_index}: at least one measurement must be observed")

        inferred_obs_dim = inferred_obs_dim or x_obs.shape[1]
        inferred_covar_dim = inferred_covar_dim if inferred_covar_dim is not None else covariates.numel()
        if x_obs.shape[1] != inferred_obs_dim:
            raise ValueError(f"Subject {subject_index}: inconsistent observation dimension")
        if covariates.numel() != inferred_covar_dim:
            raise ValueError(f"Subject {subject_index}: inconsistent covariate dimension")

    assert inferred_obs_dim is not None and inferred_covar_dim is not None
    return inferred_obs_dim, inferred_covar_dim



def validate_covariate_design(
    subjects_data: Sequence[Mapping[str, torch.Tensor]],
    tolerance: Optional[float] = None,
) -> None:
    """Reject constant or collinear covariates that break mean identifiability."""
    if not subjects_data:
        raise ValueError("subjects_data must not be empty")
    covariate_dim = subjects_data[0]["u"].numel()
    if covariate_dim == 0:
        return
    design = torch.stack([subject["u"] for subject in subjects_data], dim=0)
    if design.shape[0] <= covariate_dim:
        raise ValueError(
            "The number of training subjects must exceed the number of covariates"
        )
    centered = design - design.mean(dim=0, keepdim=True)
    rank_tensor = (
        torch.linalg.matrix_rank(centered)
        if tolerance is None
        else torch.linalg.matrix_rank(centered, tolerance)
    )
    rank = int(rank_tensor.item())
    if rank < covariate_dim:
        raise ValueError(
            "Covariates must be nonconstant and full rank after centering; "
            "remove intercept, constant, or collinear columns"
        )

def prepare_pca_workspace(
    subjects_data: Sequence[Mapping[str, torch.Tensor]],
    max_rank: int,
    oversample: int = 8,
    n_iter: int = 4,
    seed: int = 0,
) -> PCAWorkspace:
    obs_dim, _ = validate_subjects(subjects_data)
    if max_rank < 1:
        raise ValueError("max_rank must be positive")
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        x_all = torch.cat([subject["x"] for subject in subjects_data], dim=0)
        observed = ~torch.isnan(x_all)
        counts = observed.sum(dim=0)
        sums = torch.where(observed, x_all, torch.zeros_like(x_all)).sum(dim=0)
        means = sums / counts.clamp_min(1)
        means = torch.where(counts > 0, means, torch.zeros_like(means))
        centered = torch.where(observed, x_all - means, torch.zeros_like(x_all))
        variances = centered.square().sum(dim=0) / (counts - 1).clamp_min(1)
        variances = torch.where(counts > 1, variances, torch.ones_like(variances))
        variances = variances.clamp_min(1e-6)

        min_dimension = min(centered.shape)
        if max_rank > min_dimension:
            raise ValueError(f"max_rank={max_rank} exceeds available matrix rank bound {min_dimension}")
        q = min(max_rank + oversample, min_dimension)
        if min_dimension <= 128 or q == min_dimension:
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            right_vectors = vh[:q, :].transpose(0, 1).contiguous()
            singular_values = singular_values[:q].contiguous()
        else:
            _, singular_values, right_vectors = torch.pca_lowrank(
                centered,
                q=q,
                center=False,
                niter=n_iter,
            )
            right_vectors = right_vectors.contiguous()
            singular_values = singular_values.contiguous()

        valid_anchor_features = counts >= 2
        if int(valid_anchor_features.sum().item()) < max_rank:
            raise ValueError("Too few nondegenerate observed features for anchor discovery")

    return PCAWorkspace(
        column_means=means,
        feature_variances=variances,
        right_vectors=right_vectors,
        singular_values=singular_values,
        n_rows=x_all.shape[0],
        valid_anchor_features=valid_anchor_features,
    )


def varimax_rotation(loadings: torch.Tensor, tolerance: float = 1e-6, max_iterations: int = 250) -> torch.Tensor:
    with torch.no_grad():
        n_features, n_factors = loadings.shape
        rotation = torch.eye(n_factors, dtype=loadings.dtype, device=loadings.device)
        previous_objective = 0.0
        for _ in range(max_iterations):
            rotated = loadings @ rotation
            column_sums = rotated.square().sum(dim=0)
            gradient = loadings.transpose(0, 1) @ (
                rotated.pow(3) - (rotated @ torch.diag(column_sums)) / max(n_features, 1)
            )
            left, singular_values, right_t = torch.linalg.svd(gradient, full_matrices=False)
            rotation = left @ right_t
            objective = float(singular_values.sum().item())
            if previous_objective > 0.0 and objective <= previous_objective * (1.0 + tolerance):
                break
            previous_objective = objective
        return loadings @ rotation


def discover_anchor_items(workspace: PCAWorkspace, k: int) -> List[int]:
    """Select distinct anchors by global assignment using signal and purity."""
    with torch.no_grad():
        rotated = workspace.rotated_loadings(k)
        squared = rotated.square()
        communality = squared.sum(dim=1, keepdim=True).clamp_min(1e-12)
        purity_share = squared / communality
        scores = rotated.abs() * purity_share
        scores = scores.clone()
        scores[~workspace.valid_anchor_features, :] = -1e12
        row_indices, column_indices = linear_sum_assignment(-scores.cpu().numpy())
        anchors_by_factor = [-1] * k
        for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
            anchors_by_factor[column_index] = row_index
        if any(anchor < 0 for anchor in anchors_by_factor):
            raise RuntimeError("Global anchor assignment did not cover all factors")
        return anchors_by_factor


# -----------------------------------------------------------------------------
# CLOUDS model
# -----------------------------------------------------------------------------
class CLOUDS(nn.Module):
    """Continuous-time anchored latent factor model.

    Observation model:
        x_i(t) = obs_intercept + Lambda f_i(t) + epsilon_i(t)

    Latent mean model:
        mu_i(t) = Phi_level u_i + t * (alpha_slope + Phi_slope u_i)

    In exact mode, Omega is a uniquely parameterized correlation matrix and
    Gamma = (S + B) Omega^{-1}, where S is SPD and B is skew-symmetric. This
    guarantees Gamma Omega + Omega Gamma^T = 2S > 0. In diagonal mode Omega is
    fixed to I to remove the otherwise unresolved factor-scale invariance.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        covar_dim: int,
        anchor_items: Sequence[int],
        theta_mode: str = "exact",
        covariance_floor: float = 1e-4,
        anchor_floor: float = 1e-4,
        min_log_psi: float = -9.0,
        max_log_psi: float = 7.0,
        min_log_rho: float = -7.0,
        max_log_rho: float = 2.0,
    ) -> None:
        super().__init__()
        if theta_mode not in {"exact", "diagonal"}:
            raise ValueError("theta_mode must be 'exact' or 'diagonal'")
        if obs_dim < 1 or latent_dim < 1 or covar_dim < 0:
            raise ValueError("obs_dim and latent_dim must be positive; covar_dim must be nonnegative")
        if obs_dim < latent_dim:
            raise ValueError("obs_dim must be at least latent_dim")
        if len(anchor_items) != latent_dim or len(set(anchor_items)) != latent_dim:
            raise ValueError("Exactly one unique anchor item is required per latent factor")
        if min(anchor_items) < 0 or max(anchor_items) >= obs_dim:
            raise ValueError("Anchor indices are out of range")

        self.D = int(obs_dim)
        self.K = int(latent_dim)
        self.C_dim = int(covar_dim)
        self.theta_mode = theta_mode
        self.covariance_floor = float(covariance_floor)
        self.anchor_floor = float(anchor_floor)
        self.min_log_psi = float(min_log_psi)
        self.max_log_psi = float(max_log_psi)
        self.min_log_rho = float(min_log_rho)
        self.max_log_rho = float(max_log_rho)

        if theta_mode == "exact":
            self.correlation_raw = nn.Parameter(torch.zeros(num_correlation_parameters(self.K)))
            initial_s_factor = 0.5 * torch.eye(self.K)
            self.s_cholesky_raw = nn.Parameter(raw_from_lower_cholesky(initial_s_factor))
            self.skew_raw = nn.Parameter(torch.zeros(num_correlation_parameters(self.K)))
        else:
            initial_log_rho = torch.full((self.K,), -2.0)
            self.rho_raw = nn.Parameter(raw_from_bounded(initial_log_rho, min_log_rho, max_log_rho))

        # Identifiable latent mean: covariate-dependent level plus covariate and
        # population slope. The observation intercept handles the population level.
        self.Phi_level = nn.Parameter(torch.zeros(self.K, self.C_dim))
        self.Phi_slope = nn.Parameter(torch.zeros(self.K, self.C_dim))
        self.alpha_slope = nn.Parameter(torch.zeros(self.K))

        self.lambda_raw = nn.Parameter(torch.zeros(self.D, self.K))
        self.obs_intercept = nn.Parameter(torch.zeros(self.D))
        self.log_psi_raw = nn.Parameter(torch.zeros(self.D))

        anchor_index = torch.tensor(anchor_items, dtype=torch.long)
        anchor_columns = torch.arange(self.K, dtype=torch.long)
        self.register_buffer("anchor_idx", anchor_index)
        self.register_buffer("anchor_cols", anchor_columns)

        structural_mask = torch.ones(self.D, self.K)
        structural_mask[anchor_index, :] = 0.0
        structural_mask[anchor_index, anchor_columns] = 1.0
        self.register_buffer("structural_mask", structural_mask)

        positivity_mask = torch.zeros(self.D, self.K, dtype=torch.bool)
        positivity_mask[anchor_index, anchor_columns] = True
        self.register_buffer("positivity_mask", positivity_mask)

    @property
    def Lambda(self) -> torch.Tensor:
        unconstrained = self.lambda_raw * self.structural_mask
        positive = self.anchor_floor + F.softplus(self.lambda_raw)
        return torch.where(self.positivity_mask, positive, unconstrained)

    @property
    def log_psi(self) -> torch.Tensor:
        return bounded_from_raw(self.log_psi_raw, self.min_log_psi, self.max_log_psi)

    @property
    def psi(self) -> torch.Tensor:
        return torch.exp(self.log_psi)

    def get_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = self.lambda_raw.dtype
        device = self.lambda_raw.device
        identity = torch.eye(self.K, dtype=dtype, device=device)

        if self.theta_mode == "exact":
            omega_cholesky = correlation_cholesky_from_raw(self.correlation_raw, self.K)
            omega = symmetrize(omega_cholesky @ omega_cholesky.transpose(-1, -2))

            s_cholesky = lower_cholesky_from_raw(self.s_cholesky_raw, self.K)
            symmetric_part = 0.5 * (s_cholesky @ s_cholesky.transpose(-1, -2))
            symmetric_part = symmetric_part + self.covariance_floor * identity
            skew_part = skew_symmetric_from_raw(self.skew_raw, self.K)
            gamma = spd_solve(omega, (symmetric_part + skew_part).transpose(-1, -2)).transpose(-1, -2)
            return gamma, omega, symmetric_part

        log_rho = bounded_from_raw(self.rho_raw, self.min_log_rho, self.max_log_rho)
        rho = torch.exp(log_rho)
        gamma = torch.diag(rho)
        omega = identity
        symmetric_part = torch.diag(rho)
        return gamma, omega, symmetric_part

    @torch.no_grad()
    def get_identifiable_parameters(self) -> Dict[str, torch.Tensor]:
        gamma, omega, symmetric_part = self.get_dynamics()
        return {
            "Omega_corr": omega.detach().clone(),
            "Gamma": gamma.detach().clone(),
            "S": symmetric_part.detach().clone(),
            "Lambda": self.Lambda.detach().clone(),
            "obs_intercept": self.obs_intercept.detach().clone(),
            "Phi_level": self.Phi_level.detach().clone(),
            "Phi_slope": self.Phi_slope.detach().clone(),
            "alpha_slope": self.alpha_slope.detach().clone(),
            "psi": self.psi.detach().clone(),
        }

    def get_subject_matrices(
        self,
        gamma: torch.Tensor,
        omega: torch.Tensor,
        covariates: torch.Tensor,
        times: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if times.numel() <= 1:
            empty_matrix = torch.empty(0, self.K, self.K, dtype=times.dtype, device=times.device)
            empty_vector = torch.empty(0, self.K, dtype=times.dtype, device=times.device)
            level = self.Phi_level @ covariates
            slope = self.Phi_slope @ covariates + self.alpha_slope
            mean_trajectory = level.unsqueeze(0) + times.unsqueeze(1) * slope.unsqueeze(0)
            return empty_matrix, empty_vector, empty_matrix, mean_trajectory

        delta_times = times[1:] - times[:-1]
        if self.theta_mode == "diagonal":
            rho = torch.diagonal(gamma)
            decays = torch.exp(-delta_times.unsqueeze(1) * rho.unsqueeze(0))
            transitions = torch.diag_embed(decays)
            transition_covariances = torch.diag_embed(1.0 - decays.square())
        else:
            transitions = torch.linalg.matrix_exp(
                -gamma.unsqueeze(0) * delta_times[:, None, None]
            )
            omega_batch = omega.unsqueeze(0).expand(delta_times.numel(), self.K, self.K)
            transition_covariances = omega_batch - transitions @ omega_batch @ transitions.transpose(-1, -2)
            transition_covariances = symmetrize(transition_covariances)

        level = self.Phi_level @ covariates
        slope = self.Phi_slope @ covariates + self.alpha_slope
        mean_trajectory = level.unsqueeze(0) + times.unsqueeze(1) * slope.unsqueeze(0)
        shifts = mean_trajectory[1:] - (
            transitions @ mean_trajectory[:-1].unsqueeze(-1)
        ).squeeze(-1)
        return transitions, shifts, transition_covariances, mean_trajectory

    def _measurement_update(
        self,
        observation: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_covariance: torch.Tensor,
        lambda_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prior_factor = stable_cholesky(prior_covariance)
        prior_covariance_stable = symmetrize(
            prior_factor @ prior_factor.transpose(-1, -2)
        )
        valid = ~torch.isnan(observation)
        if not bool(valid.any().item()):
            zero = torch.zeros((), dtype=observation.dtype, device=observation.device)
            return prior_mean, prior_covariance_stable, zero

        loading = lambda_matrix[valid, :]
        residual = observation[valid] - self.obs_intercept[valid] - loading @ prior_mean
        log_psi_valid = self.log_psi[valid]
        inverse_psi = torch.exp(-log_psi_valid)

        prior_inverse = torch.cholesky_inverse(prior_factor)
        observation_information = loading.transpose(0, 1) @ (inverse_psi.unsqueeze(1) * loading)
        posterior_information = symmetrize(prior_inverse + observation_information)
        posterior_information_factor = stable_cholesky(posterior_information)
        posterior_covariance = symmetrize(torch.cholesky_inverse(posterior_information_factor))

        information_vector = loading.transpose(0, 1) @ (inverse_psi * residual)
        posterior_mean = prior_mean + posterior_covariance @ information_vector

        logdet_innovation = (
            log_psi_valid.sum()
            + cholesky_logdet(prior_factor)
            + cholesky_logdet(posterior_information_factor)
        )
        quadratic = residual @ (inverse_psi * residual) - information_vector @ (
            posterior_covariance @ information_vector
        )
        quadratic = quadratic.clamp_min(0.0)
        log_likelihood = -0.5 * (
            valid.sum().to(observation.dtype) * LOG_2PI + logdet_innovation + quadratic
        )
        return posterior_mean, posterior_covariance, log_likelihood

    def kalman_inference(
        self,
        x_obs: torch.Tensor,
        transitions: torch.Tensor,
        shifts: torch.Tensor,
        transition_covariances: torch.Tensor,
        mean_trajectory: torch.Tensor,
        omega: torch.Tensor,
        lambda_matrix: torch.Tensor,
        smooth: bool = True,
    ) -> Tuple[Optional[PosteriorStats], torch.Tensor]:
        n_visits = x_obs.shape[0]
        dtype = x_obs.dtype
        device = x_obs.device

        predicted_means = torch.zeros(n_visits, self.K, dtype=dtype, device=device)
        predicted_covariances = torch.zeros(n_visits, self.K, self.K, dtype=dtype, device=device)
        filtered_means = torch.zeros_like(predicted_means)
        filtered_covariances = torch.zeros_like(predicted_covariances)

        predicted_means[0] = mean_trajectory[0]
        omega_factor = stable_cholesky(omega)
        predicted_covariances[0] = symmetrize(
            omega_factor @ omega_factor.transpose(-1, -2)
        )
        filtered_means[0], filtered_covariances[0], log_likelihood = self._measurement_update(
            x_obs[0], predicted_means[0], predicted_covariances[0], lambda_matrix
        )

        for visit_index in range(1, n_visits):
            transition_index = visit_index - 1
            predicted_means[visit_index] = (
                transitions[transition_index] @ filtered_means[visit_index - 1]
                + shifts[transition_index]
            )
            q_factor = stable_cholesky(transition_covariances[transition_index])
            q_stable = symmetrize(q_factor @ q_factor.transpose(-1, -2))
            predicted_covariance_raw = symmetrize(
                transitions[transition_index]
                @ filtered_covariances[visit_index - 1]
                @ transitions[transition_index].transpose(-1, -2)
                + q_stable
            )
            predicted_factor = stable_cholesky(predicted_covariance_raw)
            predicted_covariances[visit_index] = symmetrize(
                predicted_factor @ predicted_factor.transpose(-1, -2)
            )
            (
                filtered_means[visit_index],
                filtered_covariances[visit_index],
                visit_log_likelihood,
            ) = self._measurement_update(
                x_obs[visit_index],
                predicted_means[visit_index],
                predicted_covariances[visit_index],
                lambda_matrix,
            )
            log_likelihood = log_likelihood + visit_log_likelihood

        if not smooth:
            return None, log_likelihood

        smoothed_means = filtered_means.clone()
        smoothed_covariances = filtered_covariances.clone()
        lag_one_covariances = torch.zeros_like(filtered_covariances)

        for visit_index in range(n_visits - 2, -1, -1):
            transition = transitions[visit_index]
            # J = P_filt A^T P_pred^{-1}, computed using a stable solve.
            smoother_gain = spd_solve(
                predicted_covariances[visit_index + 1],
                transition @ filtered_covariances[visit_index],
            ).transpose(-1, -2)
            smoothed_means[visit_index] = filtered_means[visit_index] + smoother_gain @ (
                smoothed_means[visit_index + 1] - predicted_means[visit_index + 1]
            )
            smoothed_covariances[visit_index] = symmetrize(
                filtered_covariances[visit_index]
                + smoother_gain
                @ (smoothed_covariances[visit_index + 1] - predicted_covariances[visit_index + 1])
                @ smoother_gain.transpose(-1, -2)
            )
            lag_one_covariances[visit_index + 1] = (
                smoothed_covariances[visit_index + 1] @ smoother_gain.transpose(-1, -2)
            )

        return (
            PosteriorStats(smoothed_means, smoothed_covariances, lag_one_covariances),
            log_likelihood,
        )

    def e_step(self, subjects_data: Sequence[Mapping[str, torch.Tensor]]) -> List[PosteriorStats]:
        stats: List[PosteriorStats] = []
        with torch.no_grad():
            gamma, omega, _ = self.get_dynamics()
            lambda_matrix = self.Lambda
            for subject in subjects_data:
                transitions, shifts, q_matrices, mean_trajectory = self.get_subject_matrices(
                    gamma, omega, subject["u"], subject["t"]
                )
                subject_stats, _ = self.kalman_inference(
                    subject["x"],
                    transitions,
                    shifts,
                    q_matrices,
                    mean_trajectory,
                    omega,
                    lambda_matrix,
                    smooth=True,
                )
                assert subject_stats is not None
                stats.append(subject_stats)
        return stats

    @torch.no_grad()
    def observed_log_likelihood_tensor(
        self, subjects_data: Sequence[Mapping[str, torch.Tensor]]
    ) -> torch.Tensor:
        gamma, omega, _ = self.get_dynamics()
        lambda_matrix = self.Lambda
        total = torch.zeros((), dtype=lambda_matrix.dtype, device=lambda_matrix.device)
        for subject in subjects_data:
            transitions, shifts, q_matrices, mean_trajectory = self.get_subject_matrices(
                gamma, omega, subject["u"], subject["t"]
            )
            _, subject_log_likelihood = self.kalman_inference(
                subject["x"],
                transitions,
                shifts,
                q_matrices,
                mean_trajectory,
                omega,
                lambda_matrix,
                smooth=False,
            )
            total = total + subject_log_likelihood
        return total

    def parameter_log_prior(self, prior: PriorConfig, include_spatial: bool = True) -> torch.Tensor:
        log_prior = torch.zeros((), dtype=self.lambda_raw.dtype, device=self.lambda_raw.device)
        mean_parameters = torch.cat(
            [self.Phi_level.reshape(-1), self.Phi_slope.reshape(-1), self.alpha_slope]
        )
        log_prior = log_prior - 0.5 * prior.mean_l2 * mean_parameters.square().sum()

        if self.theta_mode == "exact":
            log_prior = log_prior - 0.5 * prior.correlation_l2 * self.correlation_raw.square().sum()
            log_prior = log_prior - prior.dynamics_l1 * self.skew_raw.abs().sum()
            s_factor = lower_cholesky_from_raw(self.s_cholesky_raw, self.K)
            log_prior = log_prior - prior.dynamics_l1 * torch.tril(s_factor, diagonal=-1).abs().sum()
        else:
            log_rho = bounded_from_raw(self.rho_raw, self.min_log_rho, self.max_log_rho)
            log_prior = log_prior - 0.5 * prior.diagonal_rate_l2 * log_rho.square().sum()

        if include_spatial:
            active_loadings = self.Lambda[self.structural_mask.bool()]
            log_prior = log_prior - 0.5 * prior.loading_l2 * active_loadings.square().sum()
            log_prior = log_prior - 0.5 * prior.intercept_l2 * self.obs_intercept.square().sum()
            log_prior = log_prior - 0.5 * prior.log_psi_l2 * self.log_psi.square().sum()
        return log_prior

    def expected_complete_log_posterior(
        self,
        subjects_data: Sequence[Mapping[str, torch.Tensor]],
        smoothed_stats: Sequence[PosteriorStats],
        prior: PriorConfig,
        include_observation: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, omega, _ = self.get_dynamics()
        lambda_matrix = self.Lambda if include_observation else None
        dtype = self.lambda_raw.dtype
        device = self.lambda_raw.device

        observation_log_likelihood = torch.zeros((), dtype=dtype, device=device)
        latent_log_likelihood = torch.zeros((), dtype=dtype, device=device)
        omega_factor = stable_cholesky(omega)
        omega_logdet = cholesky_logdet(omega_factor)

        for subject, stats in zip(subjects_data, smoothed_stats):
            x_obs = subject["x"]
            transitions, shifts, q_matrices, mean_trajectory = self.get_subject_matrices(
                gamma, omega, subject["u"], subject["t"]
            )

            if include_observation:
                if lambda_matrix is None:
                    raise RuntimeError("Observation loadings were not constructed")
                observed = ~torch.isnan(x_obs)
                centered_observation = torch.where(
                    observed,
                    x_obs - self.obs_intercept.unsqueeze(0),
                    torch.zeros_like(x_obs),
                )
                predicted = stats.mean @ lambda_matrix.transpose(0, 1)
                covariance_times_loading = stats.covariance @ lambda_matrix.transpose(0, 1)
                predictive_variance = (
                    covariance_times_loading * lambda_matrix.transpose(0, 1).unsqueeze(0)
                ).sum(dim=1).clamp_min(0.0)
                expected_squared_error = (centered_observation - predicted).square() + predictive_variance
                observation_terms = (
                    LOG_2PI
                    + self.log_psi.unsqueeze(0)
                    + expected_squared_error * torch.exp(-self.log_psi).unsqueeze(0)
                )
                observation_log_likelihood = observation_log_likelihood - 0.5 * (
                    observed * observation_terms
                ).sum()

            initial_error = stats.mean[0] - mean_trajectory[0]
            initial_second_moment = stats.covariance[0] + torch.outer(initial_error, initial_error)
            initial_solve = torch.cholesky_solve(initial_second_moment, omega_factor)
            initial_trace = torch.diagonal(initial_solve).sum().clamp_min(0.0)
            latent_log_likelihood = latent_log_likelihood - 0.5 * (
                self.K * LOG_2PI + omega_logdet + initial_trace
            )

            if x_obs.shape[0] > 1:
                current_mean = stats.mean[1:]
                previous_mean = stats.mean[:-1]
                current_covariance = stats.covariance[1:]
                previous_covariance = stats.covariance[:-1]
                cross_covariance = stats.lag_one_covariance[1:]

                mean_error = current_mean - (
                    transitions @ previous_mean.unsqueeze(-1)
                ).squeeze(-1) - shifts
                covariance_error = (
                    current_covariance
                    + transitions @ previous_covariance @ transitions.transpose(-1, -2)
                    - cross_covariance @ transitions.transpose(-1, -2)
                    - transitions @ cross_covariance.transpose(-1, -2)
                )
                second_moment = symmetrize(covariance_error) + (
                    mean_error.unsqueeze(-1) @ mean_error.unsqueeze(-2)
                )

                q_factor = stable_cholesky(q_matrices)
                q_logdet = cholesky_logdet(q_factor)
                q_solve = torch.cholesky_solve(second_moment, q_factor)
                trace_terms = torch.diagonal(q_solve, dim1=-2, dim2=-1).sum(dim=-1).clamp_min(0.0)
                latent_log_likelihood = latent_log_likelihood - 0.5 * (
                    self.K * LOG_2PI + q_logdet + trace_terms
                ).sum()

        log_prior = self.parameter_log_prior(prior, include_spatial=include_observation)
        log_posterior = observation_log_likelihood + latent_log_likelihood + log_prior
        return log_posterior, observation_log_likelihood, latent_log_likelihood

    @torch.no_grad()
    def observed_scores(
        self,
        subjects_data: Sequence[Mapping[str, torch.Tensor]],
        prior: PriorConfig,
    ) -> Tuple[float, float]:
        log_likelihood = self.observed_log_likelihood_tensor(subjects_data)
        log_posterior = log_likelihood + self.parameter_log_prior(prior, include_spatial=True)
        return float(log_likelihood.item()), float(log_posterior.item())

    def pca_warm_start(self, workspace: PCAWorkspace) -> None:
        with torch.no_grad():
            rotated = workspace.rotated_loadings(self.K).clone()
            diagonal_values = rotated[self.anchor_idx, self.anchor_cols]
            signs = torch.where(diagonal_values >= 0.0, torch.ones_like(diagonal_values), -torch.ones_like(diagonal_values))
            rotated = rotated * signs.unsqueeze(0)

            initial_lambda = rotated * self.structural_mask
            positive_anchor_values = rotated[self.anchor_idx, self.anchor_cols].abs().clamp_min(
                self.anchor_floor + 1e-3
            )

            raw_lambda = initial_lambda.clone()
            raw_lambda[self.anchor_idx, self.anchor_cols] = softplus_inverse(
                positive_anchor_values - self.anchor_floor
            )
            self.lambda_raw.copy_(raw_lambda)
            self.obs_intercept.copy_(workspace.column_means)

            effective_lambda = self.Lambda
            explained_variance = effective_lambda.square().sum(dim=1)
            minimum_noise = torch.maximum(
                0.05 * workspace.feature_variances,
                torch.full_like(workspace.feature_variances, math.exp(self.min_log_psi + 0.5)),
            )
            initial_psi = torch.maximum(
                workspace.feature_variances - explained_variance,
                minimum_noise,
            )
            initial_log_psi = torch.log(initial_psi).clamp(
                self.min_log_psi + 1e-3,
                self.max_log_psi - 1e-3,
            )
            self.log_psi_raw.copy_(
                raw_from_bounded(initial_log_psi, self.min_log_psi, self.max_log_psi)
            )

            self.Phi_level.zero_()
            self.Phi_slope.zero_()
            self.alpha_slope.zero_()
            if self.theta_mode == "exact":
                self.correlation_raw.zero_()

    def reset_temporal_parameters(self, seed: int) -> None:
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            nn.init.normal_(self.Phi_level, mean=0.0, std=0.05)
            nn.init.normal_(self.Phi_slope, mean=0.0, std=0.05)
            nn.init.normal_(self.alpha_slope, mean=0.0, std=0.05)
            if self.theta_mode == "exact":
                nn.init.normal_(self.correlation_raw, mean=0.0, std=0.08)
                nn.init.normal_(self.skew_raw, mean=0.0, std=0.05)
                factor = torch.tril(torch.randn(self.K, self.K) * 0.04)
                factor.diagonal().copy_(0.45 + 0.08 * torch.rand(self.K))
                self.s_cholesky_raw.copy_(raw_from_lower_cholesky(factor))
            else:
                initial_log_rho = -2.0 + 0.15 * torch.randn(self.K)
                self.rho_raw.copy_(
                    raw_from_bounded(initial_log_rho, self.min_log_rho, self.max_log_rho)
                )

    @torch.no_grad()
    def sanitize_parameters(self) -> None:
        bounds = {
            "lambda_raw": (-20.0, 20.0),
            "obs_intercept": (-1e4, 1e4),
            "log_psi_raw": (-12.0, 12.0),
            "Phi_level": (-50.0, 50.0),
            "Phi_slope": (-50.0, 50.0),
            "alpha_slope": (-50.0, 50.0),
            "correlation_raw": (-6.0, 6.0),
            "s_cholesky_raw": (-12.0, 12.0),
            "skew_raw": (-20.0, 20.0),
            "rho_raw": (-12.0, 12.0),
        }
        for name, parameter in self.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Parameter {name} became nonfinite")
            lower, upper = bounds.get(name, (-1e6, 1e6))
            parameter.clamp_(lower, upper)

    def parameter_count(self, count_observation_intercept: bool = True) -> int:
        free_loadings = self.D * self.K - self.K * (self.K - 1)
        observation_parameters = free_loadings + self.D
        if count_observation_intercept:
            observation_parameters += self.D
        mean_parameters = 2 * self.K * self.C_dim + self.K
        if self.theta_mode == "exact":
            temporal_parameters = (
                num_correlation_parameters(self.K)
                + num_lower_triangular_parameters(self.K)
                + num_correlation_parameters(self.K)
            )
        else:
            # Omega is fixed to I, so only the K decay rates are free.
            temporal_parameters = self.K
        return int(observation_parameters + mean_parameters + temporal_parameters)

    @torch.no_grad()
    def information_criteria(
        self,
        subjects_data: Sequence[Mapping[str, torch.Tensor]],
        include_anchor_search_penalty: bool = True,
    ) -> Dict[str, float]:
        log_likelihood = float(self.observed_log_likelihood_tensor(subjects_data).item())
        parameter_count = self.parameter_count(count_observation_intercept=True)
        sample_sizes = {
            "subjects": len(subjects_data),
            "visits": sum(subject["x"].shape[0] for subject in subjects_data),
            "entries": count_observed_entries(subjects_data),
        }
        criteria: Dict[str, float] = {
            "observed_log_likelihood": log_likelihood,
            "parameter_count": float(parameter_count),
            "aic": -2.0 * log_likelihood + 2.0 * parameter_count,
        }
        anchor_penalty = 0.0
        if include_anchor_search_penalty:
            # Number of ordered distinct anchor assignments: D! / (D-K)!.
            anchor_penalty = 2.0 * sum(math.log(self.D - index) for index in range(self.K))
        criteria["anchor_search_penalty"] = anchor_penalty

        for unit, sample_size in sample_sizes.items():
            n_effective = max(int(sample_size), 2)
            criteria[f"n_effective_{unit}"] = float(n_effective)
            bic = -2.0 * log_likelihood + parameter_count * math.log(n_effective)
            criteria[f"bic_{unit}"] = bic
            criteria[f"bic_anchor_{unit}"] = bic + anchor_penalty
        return criteria

    def _clone_state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.state_dict().items()}

    def _m_step(
        self,
        subjects_data: Sequence[Mapping[str, torch.Tensor]],
        smoothed_stats: Sequence[PosteriorStats],
        optimizer: optim.Optimizer,
        parameters_to_clip: Sequence[nn.Parameter],
        config: FitConfig,
        include_observation: bool,
    ) -> None:
        if include_observation:
            denominator = max(count_observed_entries(subjects_data), 1)
        else:
            denominator = max(
                sum(subject["x"].shape[0] for subject in subjects_data) * self.K,
                1,
            )

        for _ in range(config.m_step_iters):
            optimizer.zero_grad(set_to_none=True)
            log_posterior, _, _ = self.expected_complete_log_posterior(
                subjects_data,
                smoothed_stats,
                config.prior,
                include_observation=include_observation,
            )
            loss = -log_posterior / denominator
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite M-step loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters_to_clip,
                config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            self.sanitize_parameters()

    def fit_em_multistart(
        self,
        subjects_data: Sequence[Mapping[str, torch.Tensor]],
        workspace: PCAWorkspace,
        config: FitConfig,
        seed: int = 0,
    ) -> FitResult:
        config.validate()
        validate_subjects(subjects_data, self.D, self.C_dim)
        validate_covariate_design(subjects_data)
        self.pca_warm_start(workspace)

        spatial_parameter_names = ("lambda_raw", "obs_intercept", "log_psi_raw")
        named_parameters = dict(self.named_parameters())
        spatial_parameters = [named_parameters[name] for name in spatial_parameter_names]
        temporal_parameters = [
            parameter
            for name, parameter in named_parameters.items()
            if name not in spatial_parameter_names
        ]
        spatial_initial_state = {
            name: named_parameters[name].detach().clone() for name in spatial_parameter_names
        }

        best_start_score = -math.inf
        best_start_state: Optional[Dict[str, torch.Tensor]] = None
        history: List[Dict[str, Any]] = []

        for start_index in range(config.n_starts):
            local_best_score = -math.inf
            local_best_state: Optional[Dict[str, torch.Tensor]] = None
            try:
                with torch.no_grad():
                    for name in spatial_parameter_names:
                        named_parameters[name].copy_(spatial_initial_state[name])
                self.reset_temporal_parameters(seed + 1009 * (start_index + 1))
                for parameter in spatial_parameters:
                    parameter.requires_grad_(False)
                burn_optimizer = optim.Adam(temporal_parameters, lr=config.learning_rate)

                for epoch_index in range(config.burn_in_epochs):
                    smoothed_stats = self.e_step(subjects_data)
                    self._m_step(
                        subjects_data,
                        smoothed_stats,
                        burn_optimizer,
                        temporal_parameters,
                        config,
                        include_observation=False,
                    )
                    log_likelihood, log_posterior = self.observed_scores(subjects_data, config.prior)
                    history.append(
                        {
                            "phase": "burn_in",
                            "start": start_index,
                            "epoch": epoch_index,
                            "observed_log_likelihood": log_likelihood,
                            "observed_log_posterior": log_posterior,
                        }
                    )
                    if log_posterior > local_best_score:
                        local_best_score = log_posterior
                        local_best_state = self._clone_state_dict()

                if local_best_state is None:
                    raise RuntimeError("No valid burn-in checkpoint was produced")
                if local_best_score > best_start_score:
                    best_start_score = local_best_score
                    best_start_state = local_best_state
            except Exception as error:
                if local_best_state is not None and local_best_score > best_start_score:
                    best_start_score = local_best_score
                    best_start_state = local_best_state
                LOGGER.warning(
                    "Start %d/%d failed for K=%d after preserving its best valid checkpoint: %s",
                    start_index + 1,
                    config.n_starts,
                    self.K,
                    error,
                )

        if best_start_state is None:
            raise RuntimeError("All multistart burn-in runs failed")
        self.load_state_dict(best_start_state)

        for parameter in spatial_parameters:
            parameter.requires_grad_(True)
        temporal_optimizer = optim.Adam(temporal_parameters, lr=config.learning_rate)
        joint_optimizer = optim.Adam(
            [
                {"params": temporal_parameters, "lr": config.learning_rate},
                {"params": spatial_parameters, "lr": config.learning_rate * config.spatial_lr_scale},
            ]
        )

        best_score = best_start_score
        best_state = self._clone_state_dict()
        epochs_without_improvement = 0
        epoch_failures = 0
        continuation_epochs = config.total_em_epochs - config.burn_in_epochs

        for epoch_index in range(continuation_epochs):
            joint_phase = epoch_index >= config.warmup_epochs
            if epoch_index == config.warmup_epochs:
                # Do not let a long temporal-only warmup stop the fit before
                # the spatial parameters have had an opportunity to update.
                epochs_without_improvement = 0
            for parameter in spatial_parameters:
                parameter.requires_grad_(joint_phase)
            active_optimizer = joint_optimizer if joint_phase else temporal_optimizer
            active_parameters = temporal_parameters + (spatial_parameters if joint_phase else [])

            checkpoint_before_epoch = self._clone_state_dict()
            try:
                smoothed_stats = self.e_step(subjects_data)
                self._m_step(
                    subjects_data,
                    smoothed_stats,
                    active_optimizer,
                    active_parameters,
                    config,
                    include_observation=joint_phase,
                )
                log_likelihood, log_posterior = self.observed_scores(subjects_data, config.prior)
            except Exception as error:
                self.load_state_dict(checkpoint_before_epoch)
                active_optimizer.state.clear()
                for parameter_group in active_optimizer.param_groups:
                    parameter_group["lr"] *= 0.5
                epoch_failures += 1
                LOGGER.warning(
                    "Continuation epoch %d failed for K=%d; restored the last "
                    "valid checkpoint and halved the active learning rate: %s",
                    epoch_index,
                    self.K,
                    error,
                )
                history.append(
                    {
                        "phase": "joint" if joint_phase else "temporal_warmup",
                        "start": None,
                        "epoch": epoch_index,
                        "status": "rolled_back",
                        "error": str(error),
                    }
                )
                if epoch_failures > config.max_epoch_failures:
                    LOGGER.warning(
                        "K=%d exceeded max_epoch_failures=%d; returning the best checkpoint",
                        self.K,
                        config.max_epoch_failures,
                    )
                    break
                continue

            history.append(
                {
                    "phase": "joint" if joint_phase else "temporal_warmup",
                    "start": None,
                    "epoch": epoch_index,
                    "observed_log_likelihood": log_likelihood,
                    "observed_log_posterior": log_posterior,
                }
            )

            if log_posterior > best_score + config.min_improvement:
                best_score = log_posterior
                best_state = self._clone_state_dict()
                if joint_phase:
                    epochs_without_improvement = 0
            elif joint_phase:
                epochs_without_improvement += 1
                if (
                    config.early_stopping_patience > 0
                    and epochs_without_improvement >= config.early_stopping_patience
                ):
                    break

        self.load_state_dict(best_state)
        for parameter in spatial_parameters:
            parameter.requires_grad_(True)
        final_stats = self.e_step(subjects_data)
        train_log_likelihood, train_log_posterior = self.observed_scores(subjects_data, config.prior)
        return FitResult(
            smoothed_stats=final_stats,
            history=history,
            train_log_likelihood=train_log_likelihood,
            train_log_posterior=train_log_posterior,
        )


# -----------------------------------------------------------------------------
# Data utilities and simulation
# -----------------------------------------------------------------------------
def count_observed_entries(subjects_data: Sequence[Mapping[str, torch.Tensor]]) -> int:
    return int(sum((~torch.isnan(subject["x"])).sum().item() for subject in subjects_data))


def split_subjects(
    subjects_data: Sequence[Mapping[str, torch.Tensor]],
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Mapping[str, torch.Tensor]], List[Mapping[str, torch.Tensor]]]:
    if validation_fraction <= 0.0:
        return list(subjects_data), []
    n_subjects = len(subjects_data)
    if n_subjects < 3:
        raise ValueError("At least three subjects are needed for a validation split")
    n_validation = min(max(1, int(round(n_subjects * validation_fraction))), n_subjects - 2)
    generator = np.random.default_rng(seed)
    indices = generator.permutation(n_subjects)
    validation_set = set(indices[:n_validation].tolist())
    training = [subject for index, subject in enumerate(subjects_data) if index not in validation_set]
    validation = [subject for index, subject in enumerate(subjects_data) if index in validation_set]
    return training, validation


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def simulate_ad_cohort_stress(
    n_subjects: int,
    obs_dim: int,
    latent_dim: int,
    covar_dim: int,
    theta_mode: str = "exact",
    seed: int = 42,
    missing_rate: float = 0.0,
    noise_scale: float = 1.0,
    missing_mechanism: str = "mcar",
    informative_missing_strength: float = 0.0,
    outlier_rate: float = 0.0,
    outlier_scale: float = 8.0,
    nonlinear_strength: float = 0.0,
    correlated_noise_strength: float = 0.0,
    batch_effect_scale: float = 0.0,
    anchor_cross_loading: float = 0.0,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    if theta_mode not in {"exact", "diagonal"}:
        raise ValueError("theta_mode must be exact or diagonal")
    if missing_mechanism not in {"mcar", "latent"}:
        raise ValueError("missing_mechanism must be mcar or latent")
    if n_subjects < 1 or obs_dim < 1 or latent_dim < 1 or covar_dim < 0:
        raise ValueError("n_subjects, obs_dim, and latent_dim must be positive")
    if obs_dim < latent_dim:
        raise ValueError("obs_dim must be at least latent_dim")
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    if not 0.0 <= outlier_rate < 1.0:
        raise ValueError("outlier_rate must be in [0, 1)")
    if noise_scale <= 0.0 or outlier_scale <= 0.0:
        raise ValueError("noise_scale and outlier_scale must be positive")
    if correlated_noise_strength < 0.0 or batch_effect_scale < 0.0:
        raise ValueError("correlated-noise and batch-effect strengths must be nonnegative")
    if nonlinear_strength < 0.0 or anchor_cross_loading < 0.0:
        raise ValueError("nonlinear and anchor-cross-loading strengths must be nonnegative")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        true_anchors = torch.randperm(obs_dim)[:latent_dim].tolist()
        identity = torch.eye(latent_dim)

        if theta_mode == "diagonal":
            rho_true = torch.linspace(0.03, 0.16, latent_dim)
            gamma_true = torch.diag(rho_true)
            omega_true = identity
            symmetric_true = torch.diag(rho_true)
        else:
            correlation_raw = 0.25 * torch.randn(num_correlation_parameters(latent_dim))
            omega_cholesky = correlation_cholesky_from_raw(correlation_raw, latent_dim)
            omega_true = symmetrize(omega_cholesky @ omega_cholesky.transpose(-1, -2))

            s_factor = torch.tril(0.15 * torch.randn(latent_dim, latent_dim))
            s_factor.diagonal().copy_(0.35 + 0.25 * torch.rand(latent_dim))
            symmetric_true = 0.5 * (s_factor @ s_factor.transpose(-1, -2)) + 1e-4 * identity
            skew_raw = 0.08 * torch.randn(num_correlation_parameters(latent_dim))
            skew_true = skew_symmetric_from_raw(skew_raw, latent_dim)
            gamma_true = spd_solve(
                omega_true, (symmetric_true + skew_true).transpose(-1, -2)
            ).transpose(-1, -2)

        phi_level_true = 0.35 * torch.randn(latent_dim, covar_dim)
        phi_slope_true = 0.25 * torch.randn(latent_dim, covar_dim)
        alpha_slope_true = 0.35 * torch.randn(latent_dim)
        quadratic_true = nonlinear_strength * torch.randn(latent_dim)

        lambda_true = 0.45 * torch.randn(obs_dim, latent_dim)
        for factor_index, feature_index in enumerate(true_anchors):
            lambda_true[feature_index, :] = (
                anchor_cross_loading * torch.randn(latent_dim)
            )
            lambda_true[feature_index, factor_index] = 0.5 + F.softplus(0.4 * torch.randn(()))

        observation_intercept_true = 0.5 * torch.randn(obs_dim)
        log_psi_true = 2.0 * math.log(noise_scale) + 0.15 * torch.randn(obs_dim)
        psi_true = torch.exp(log_psi_true)
        common_noise_loading = torch.randn(obs_dim)
        common_noise_loading = common_noise_loading / common_noise_loading.square().mean().sqrt().clamp_min(1e-6)
        batch_loading = torch.randn(obs_dim)
        batch_loading = batch_loading / batch_loading.square().mean().sqrt().clamp_min(1e-6)

        omega_factor = stable_cholesky(omega_true)
        subjects_data: List[Dict[str, torch.Tensor]] = []
        for _ in range(n_subjects):
            n_visits = int(torch.randint(3, 6, (1,)).item())
            baseline_age = 55.0 + 20.0 * torch.rand(())
            raw_intervals = 1.5 + 3.5 * torch.rand(n_visits - 1)
            raw_times = torch.cat(
                [baseline_age.reshape(1), baseline_age + torch.cumsum(raw_intervals, dim=0)]
            )
            times = (raw_times - 70.0) / 10.0
            covariates = torch.randn(covar_dim)

            level = phi_level_true @ covariates
            slope = phi_slope_true @ covariates + alpha_slope_true
            mean_trajectory = level.unsqueeze(0) + times.unsqueeze(1) * slope.unsqueeze(0)
            if nonlinear_strength != 0.0:
                mean_trajectory = mean_trajectory + times.square().unsqueeze(1) * quadratic_true.unsqueeze(0)

            latent_states = torch.zeros(n_visits, latent_dim)
            latent_states[0] = mean_trajectory[0] + omega_factor @ torch.randn(latent_dim)
            for visit_index in range(1, n_visits):
                delta_time = times[visit_index] - times[visit_index - 1]
                transition = torch.linalg.matrix_exp(-gamma_true * delta_time)
                q_matrix = symmetrize(
                    omega_true - transition @ omega_true @ transition.transpose(-1, -2)
                )
                q_factor = stable_cholesky(q_matrix)
                latent_states[visit_index] = (
                    transition @ latent_states[visit_index - 1]
                    + mean_trajectory[visit_index]
                    - transition @ mean_trajectory[visit_index - 1]
                    + q_factor @ torch.randn(latent_dim)
                )

            independent_noise = torch.randn(n_visits, obs_dim) * torch.sqrt(psi_true).unsqueeze(0)
            correlated_noise = (
                correlated_noise_strength
                * torch.randn(n_visits, 1)
                * common_noise_loading.unsqueeze(0)
            )
            subject_batch = batch_effect_scale * torch.randn(()) * batch_loading.unsqueeze(0)
            observations = (
                observation_intercept_true.unsqueeze(0)
                + latent_states @ lambda_true.transpose(0, 1)
                + independent_noise
                + correlated_noise
                + subject_batch
            )

            if outlier_rate > 0.0:
                outlier_mask = torch.rand_like(observations) < outlier_rate
                observations = observations + outlier_mask * (
                    outlier_scale * noise_scale * torch.randn_like(observations)
                )

            if missing_rate > 0.0:
                if missing_mechanism == "mcar":
                    missing_probabilities = torch.full_like(observations, missing_rate)
                else:
                    visit_signal = latent_states[:, 0]
                    visit_signal = (
                        visit_signal - visit_signal.mean()
                    ) / visit_signal.std(unbiased=False).clamp_min(1e-6)
                    logits = _logit(missing_rate) + informative_missing_strength * visit_signal.unsqueeze(1)
                    missing_probabilities = torch.sigmoid(logits).expand_as(observations)
                missing_mask = torch.rand_like(observations) < missing_probabilities
                if bool(missing_mask.all().item()):
                    missing_mask[0, 0] = False
                observations = observations.masked_fill(missing_mask, float("nan"))

            subjects_data.append(
                {
                    "x": observations,
                    "u": covariates,
                    "t": times,
                    "F_true": latent_states,
                }
            )

    true_parameters = {
        "anchors": torch.tensor(true_anchors, dtype=torch.long),
        "Lambda": lambda_true,
        "Gamma": gamma_true,
        "Omega_corr": omega_true,
        "S": symmetric_true,
        "obs_intercept": observation_intercept_true,
        "Phi_level": phi_level_true,
        "Phi_slope": phi_slope_true,
        "alpha_slope": alpha_slope_true,
        "psi": psi_true,
    }
    return subjects_data, true_parameters


# -----------------------------------------------------------------------------
# Recovery metrics
# -----------------------------------------------------------------------------
def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_vector = np.asarray(first, dtype=float).reshape(-1)
    second_vector = np.asarray(second, dtype=float).reshape(-1)
    first_centered = first_vector - first_vector.mean()
    second_centered = second_vector - second_vector.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator <= 1e-12:
        return float("nan")
    return float(first_centered @ second_centered / denominator)


def _column_correlation_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_centered = first - first.mean(axis=0, keepdims=True)
    second_centered = second - second.mean(axis=0, keepdims=True)
    first_norm = np.linalg.norm(first_centered, axis=0, keepdims=True).T
    second_norm = np.linalg.norm(second_centered, axis=0, keepdims=True)
    denominator = np.maximum(first_norm @ second_norm, 1e-12)
    return first_centered.T @ second_centered / denominator


def _relative_rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    denominator = np.sqrt(np.mean(np.square(truth)))
    if denominator <= 1e-12:
        return float(np.sqrt(np.mean(np.square(estimate - truth))))
    return float(np.sqrt(np.mean(np.square(estimate - truth))) / denominator)


def _subspace_similarity(first: np.ndarray, second: np.ndarray) -> float:
    q_first, _ = np.linalg.qr(first)
    q_second, _ = np.linalg.qr(second)
    singular_values = np.linalg.svd(q_first.T @ q_second, compute_uv=False)
    return float(np.mean(np.clip(singular_values, 0.0, 1.0)))


def align_and_score_recovery(
    model: CLOUDS,
    training_data: Sequence[Mapping[str, torch.Tensor]],
    training_stats: Sequence[PosteriorStats],
    true_parameters: Mapping[str, torch.Tensor],
    discovered_anchors: Sequence[int],
    validation_data: Optional[Sequence[Mapping[str, torch.Tensor]]] = None,
    validation_stats: Optional[Sequence[PosteriorStats]] = None,
) -> Dict[str, Any]:
    """Align on training trajectories, then score train and held-out subjects."""
    identifiable = model.get_identifiable_parameters()
    true_states_train = torch.cat(
        [subject["F_true"] for subject in training_data], dim=0
    ).cpu().numpy()
    estimated_states_train = torch.cat(
        [stats.mean for stats in training_stats], dim=0
    ).cpu().numpy()

    correlation_matrix = _column_correlation_matrix(true_states_train, estimated_states_train)
    true_indices, estimated_indices = linear_sum_assignment(-np.abs(correlation_matrix))
    if not np.array_equal(true_indices, np.arange(model.K)):
        order = np.argsort(true_indices)
        estimated_indices = estimated_indices[order]
    signs = np.where(
        correlation_matrix[np.arange(model.K), estimated_indices] >= 0.0,
        1.0,
        -1.0,
    )

    def align_states(estimated_states: np.ndarray) -> np.ndarray:
        return estimated_states[:, estimated_indices] * signs

    def state_metrics(
        true_states: np.ndarray,
        estimated_states: np.ndarray,
    ) -> Tuple[List[float], float, float]:
        aligned = align_states(estimated_states)
        correlations = [
            _safe_correlation(true_states[:, factor], aligned[:, factor])
            for factor in range(model.K)
        ]
        return (
            correlations,
            float(np.nanmean(correlations)),
            _safe_correlation(true_states, aligned),
        )

    train_factor_correlations, train_factor_mean, train_factor_flat = state_metrics(
        true_states_train,
        estimated_states_train,
    )

    true_state_groups = [true_states_train]
    estimated_state_groups = [estimated_states_train]
    validation_factor_correlations: Optional[List[float]] = None
    validation_factor_mean: Optional[float] = None
    validation_factor_flat: Optional[float] = None
    if validation_data and validation_stats:
        true_states_validation = torch.cat(
            [subject["F_true"] for subject in validation_data], dim=0
        ).cpu().numpy()
        estimated_states_validation = torch.cat(
            [stats.mean for stats in validation_stats], dim=0
        ).cpu().numpy()
        (
            validation_factor_correlations,
            validation_factor_mean,
            validation_factor_flat,
        ) = state_metrics(true_states_validation, estimated_states_validation)
        true_state_groups.append(true_states_validation)
        estimated_state_groups.append(estimated_states_validation)

    true_states_all = np.concatenate(true_state_groups, axis=0)
    estimated_states_all = np.concatenate(estimated_state_groups, axis=0)
    factor_correlations, factor_mean, factor_flat = state_metrics(
        true_states_all,
        estimated_states_all,
    )

    lambda_estimate = identifiable["Lambda"].cpu().numpy()
    lambda_aligned = lambda_estimate[:, estimated_indices] * signs

    def align_square(parameter: torch.Tensor) -> np.ndarray:
        estimate = parameter.cpu().numpy()
        permuted = estimate[np.ix_(estimated_indices, estimated_indices)]
        return signs[:, None] * permuted * signs[None, :]

    gamma_aligned = align_square(identifiable["Gamma"])
    omega_aligned = align_square(identifiable["Omega_corr"])
    symmetric_aligned = align_square(identifiable["S"])

    phi_level_aligned = (
        identifiable["Phi_level"].cpu().numpy()[estimated_indices, :] * signs[:, None]
    )
    phi_slope_aligned = (
        identifiable["Phi_slope"].cpu().numpy()[estimated_indices, :] * signs[:, None]
    )
    alpha_slope_aligned = (
        identifiable["alpha_slope"].cpu().numpy()[estimated_indices] * signs
    )

    true_anchors = true_parameters["anchors"].cpu().numpy()
    discovered = np.asarray(discovered_anchors)
    aligned_discovered = discovered[estimated_indices]

    metrics: Dict[str, Any] = {
        "factor_correlations": factor_correlations,
        "factor_correlation_mean": factor_mean,
        "factor_correlation_flat": factor_flat,
        "train_factor_correlations": train_factor_correlations,
        "train_factor_correlation_mean": train_factor_mean,
        "train_factor_correlation_flat": train_factor_flat,
        "validation_factor_correlations": validation_factor_correlations,
        "validation_factor_correlation_mean": validation_factor_mean,
        "validation_factor_correlation_flat": validation_factor_flat,
        "loading_correlation": _safe_correlation(
            true_parameters["Lambda"].cpu().numpy(), lambda_aligned
        ),
        "loading_relative_rmse": _relative_rmse(
            lambda_aligned, true_parameters["Lambda"].cpu().numpy()
        ),
        "loading_subspace_similarity": _subspace_similarity(
            true_parameters["Lambda"].cpu().numpy(), lambda_aligned
        ),
        "gamma_correlation": _safe_correlation(
            true_parameters["Gamma"].cpu().numpy(), gamma_aligned
        ),
        "gamma_relative_rmse": _relative_rmse(
            gamma_aligned, true_parameters["Gamma"].cpu().numpy()
        ),
        "omega_correlation": _safe_correlation(
            true_parameters["Omega_corr"].cpu().numpy(), omega_aligned
        ),
        "omega_relative_rmse": _relative_rmse(
            omega_aligned, true_parameters["Omega_corr"].cpu().numpy()
        ),
        "symmetric_dynamics_correlation": _safe_correlation(
            true_parameters["S"].cpu().numpy(), symmetric_aligned
        ),
        "symmetric_dynamics_relative_rmse": _relative_rmse(
            symmetric_aligned, true_parameters["S"].cpu().numpy()
        ),
        "phi_level_correlation": _safe_correlation(
            true_parameters["Phi_level"].cpu().numpy(), phi_level_aligned
        ),
        "phi_slope_correlation": _safe_correlation(
            true_parameters["Phi_slope"].cpu().numpy(), phi_slope_aligned
        ),
        "alpha_slope_correlation": _safe_correlation(
            true_parameters["alpha_slope"].cpu().numpy(), alpha_slope_aligned
        ),
        "psi_correlation": _safe_correlation(
            true_parameters["psi"].cpu().numpy(), identifiable["psi"].cpu().numpy()
        ),
        "intercept_correlation": _safe_correlation(
            true_parameters["obs_intercept"].cpu().numpy(),
            identifiable["obs_intercept"].cpu().numpy(),
        ),
        "anchor_recovery_rate": float(np.mean(aligned_discovered == true_anchors)),
        "estimated_factor_order": estimated_indices.tolist(),
        "estimated_factor_signs": signs.tolist(),
    }
    return metrics


# -----------------------------------------------------------------------------
# Worker and pipeline orchestration
# -----------------------------------------------------------------------------

def validate_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    selection_config: SelectionConfig,
) -> None:
    required = {"name", "N", "D", "K", "test_Ks", "C"}
    names: List[str] = []
    for scenario_index, scenario in enumerate(scenarios):
        missing = required.difference(scenario)
        if missing:
            raise KeyError(
                f"Scenario {scenario_index} is missing required keys: {sorted(missing)}"
            )
        name = str(scenario["name"])
        names.append(name)
        n_subjects = int(scenario["N"])
        obs_dim = int(scenario["D"])
        true_k = int(scenario["K"])
        covar_dim = int(scenario["C"])
        test_ks = [int(value) for value in scenario["test_Ks"]]
        if n_subjects < 1 or obs_dim < 1 or true_k < 1 or covar_dim < 0:
            raise ValueError(f"Scenario {name!r} has invalid dimensions")
        if selection_config.validation_fraction > 0.0 and n_subjects < 3:
            raise ValueError(f"Scenario {name!r} needs at least three subjects for holdout")
        if selection_config.validation_fraction > 0.0:
            n_validation = min(
                max(1, int(round(n_subjects * selection_config.validation_fraction))),
                n_subjects - 2,
            )
        else:
            n_validation = 0
        n_training = n_subjects - n_validation
        if n_training <= covar_dim:
            raise ValueError(
                f"Scenario {name!r} needs more training subjects than covariates"
            )
        if not test_ks or len(set(test_ks)) != len(test_ks):
            raise ValueError(f"Scenario {name!r} must have unique candidate dimensions")
        if true_k not in test_ks:
            raise ValueError(f"Scenario {name!r} must include the true K in test_Ks")
        if min(test_ks) < 1 or max(test_ks + [true_k]) > obs_dim:
            raise ValueError(f"Scenario {name!r} has a latent dimension outside [1, D]")
        minimum_visit_rows = 3 * n_training
        if max(test_ks) > minimum_visit_rows:
            raise ValueError(
                f"Scenario {name!r} can request more factors than the minimum training visit count"
            )
        if str(scenario.get("theta_mode", "exact")) not in {"exact", "diagonal"}:
            raise ValueError(f"Scenario {name!r} has an invalid theta_mode")
        missing_rate = float(scenario.get("miss", 0.0))
        if not 0.0 <= missing_rate < 1.0:
            raise ValueError(f"Scenario {name!r} has an invalid missing rate")
        if float(scenario.get("noise", 1.0)) <= 0.0:
            raise ValueError(f"Scenario {name!r} must have positive observation noise")
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique")

def _worker_initializer(log_queue: mp.Queue, threads_per_worker: int) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)
    torch.set_num_threads(max(1, threads_per_worker))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _selection_score(
    selection_config: SelectionConfig,
    validation_log_likelihood: Optional[float],
    validation_entries: int,
    information_criteria: Mapping[str, float],
) -> float:
    if selection_config.metric == "validation_nll":
        if validation_log_likelihood is None or validation_entries <= 0:
            raise ValueError("Validation likelihood is unavailable")
        return -validation_log_likelihood / validation_entries
    if selection_config.metric == "bic":
        return float(information_criteria[f"bic_{selection_config.bic_sample_unit}"])
    if selection_config.metric == "bic_anchor":
        return float(information_criteria[f"bic_anchor_{selection_config.bic_sample_unit}"])
    return float(information_criteria[selection_config.metric])


def run_single_discovery_sweep(
    task_id: int,
    scenario: Mapping[str, Any],
    run_index: int,
    seed: int,
    fit_config: FitConfig,
    selection_config: SelectionConfig,
) -> Dict[str, Any]:
    LOGGER.info(
        "[Worker %d] %s | run=%d | true K=%d",
        task_id,
        scenario["name"],
        run_index,
        scenario["K"],
    )
    try:
        true_k = int(scenario["K"])
        subjects_data, true_parameters = simulate_ad_cohort_stress(
            n_subjects=int(scenario["N"]),
            obs_dim=int(scenario["D"]),
            latent_dim=true_k,
            covar_dim=int(scenario["C"]),
            theta_mode=str(scenario.get("theta_mode", "exact")),
            seed=seed,
            missing_rate=float(scenario.get("miss", 0.0)),
            noise_scale=float(scenario.get("noise", 1.0)),
            missing_mechanism=str(scenario.get("missing_mechanism", "mcar")),
            informative_missing_strength=float(scenario.get("informative_missing_strength", 0.0)),
            outlier_rate=float(scenario.get("outlier_rate", 0.0)),
            outlier_scale=float(scenario.get("outlier_scale", 8.0)),
            nonlinear_strength=float(scenario.get("nonlinear_strength", 0.0)),
            correlated_noise_strength=float(scenario.get("correlated_noise_strength", 0.0)),
            batch_effect_scale=float(scenario.get("batch_effect_scale", 0.0)),
            anchor_cross_loading=float(scenario.get("anchor_cross_loading", 0.0)),
        )
        training_data, validation_data = split_subjects(
            subjects_data,
            selection_config.validation_fraction,
            seed + 313,
        )
        workspace = prepare_pca_workspace(
            training_data,
            max_rank=max(scenario["test_Ks"]),
            seed=seed + 719,
        )

        sweep_results: List[Dict[str, Any]] = []
        best_score = math.inf
        best_k: Optional[int] = None

        for candidate_k in scenario["test_Ks"]:
            started = time.perf_counter()
            discovered_anchors = discover_anchor_items(workspace, int(candidate_k))
            model = CLOUDS(
                obs_dim=int(scenario["D"]),
                latent_dim=int(candidate_k),
                covar_dim=int(scenario["C"]),
                anchor_items=discovered_anchors,
                theta_mode=str(scenario.get("theta_mode", "exact")),
            ).to(
                dtype=training_data[0]["x"].dtype,
                device=training_data[0]["x"].device,
            )
            fit_result = model.fit_em_multistart(
                training_data,
                workspace,
                fit_config,
                seed=seed + 10000 * int(candidate_k),
            )
            information_criteria = model.information_criteria(
                training_data,
                include_anchor_search_penalty=selection_config.include_anchor_search_penalty,
            )

            validation_log_likelihood: Optional[float] = None
            validation_entries = count_observed_entries(validation_data)
            if validation_data:
                with torch.no_grad():
                    validation_log_likelihood = float(
                        model.observed_log_likelihood_tensor(validation_data).item()
                    )
            candidate_score = _selection_score(
                selection_config,
                validation_log_likelihood,
                validation_entries,
                information_criteria,
            )

            recovery_metrics: Optional[Dict[str, Any]] = None
            if int(candidate_k) == true_k:
                training_recovery_stats = fit_result.smoothed_stats
                with torch.no_grad():
                    validation_recovery_stats = (
                        model.e_step(validation_data) if validation_data else None
                    )
                recovery_metrics = align_and_score_recovery(
                    model,
                    training_data,
                    training_recovery_stats,
                    true_parameters,
                    discovered_anchors,
                    validation_data=validation_data,
                    validation_stats=validation_recovery_stats,
                )

            elapsed = time.perf_counter() - started
            result = {
                "candidate_k": int(candidate_k),
                "anchors": discovered_anchors,
                "selection_metric": selection_config.metric,
                "selection_score": candidate_score,
                "train_log_likelihood": fit_result.train_log_likelihood,
                "train_log_posterior": fit_result.train_log_posterior,
                "validation_log_likelihood": validation_log_likelihood,
                "validation_nll_per_entry": (
                    -validation_log_likelihood / validation_entries
                    if validation_log_likelihood is not None and validation_entries > 0
                    else None
                ),
                "information_criteria": information_criteria,
                "recovery": recovery_metrics,
                "elapsed_seconds": elapsed,
                "epochs_recorded": len(fit_result.history),
                "fit_history": fit_result.history,
            }
            sweep_results.append(result)

            if candidate_score < best_score:
                best_score = candidate_score
                best_k = int(candidate_k)

        return {
            "status": "success",
            "task_id": task_id,
            "scenario_name": scenario["name"],
            "run_index": run_index,
            "seed": seed,
            "true_k": true_k,
            "best_k": best_k,
            "best_score": best_score,
            "n_train_subjects": len(training_data),
            "n_validation_subjects": len(validation_data),
            "sweep": sweep_results,
        }
    except Exception as error:
        return {
            "status": "error",
            "task_id": task_id,
            "scenario_name": scenario["name"],
            "run_index": run_index,
            "seed": seed,
            "error_msg": f"{error}\n{traceback.format_exc()}",
        }


def build_default_scenarios(include_misspecification: bool = True) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = [
        {"name": "1a. Low D, Low K", "N": 150, "D": 500, "K": 4, "test_Ks": [3, 4, 5], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "1b. Low D, Mid K", "N": 200, "D": 500, "K": 8, "test_Ks": [7, 8, 9], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "2a. Std D, Low K", "N": 150, "D": 2000, "K": 4, "test_Ks": [3, 4, 5], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "2b. Std D, Mid K", "N": 200, "D": 2000, "K": 8, "test_Ks": [7, 8, 9], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "2c. Std D, High K", "N": 250, "D": 2000, "K": 12, "test_Ks": [10, 12, 14], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "3a. High D, Low K", "N": 150, "D": 5000, "K": 4, "test_Ks": [3, 4, 5], "C": 2, "miss": 0.10, "noise": 1.0},
        {"name": "3b. High D, High K", "N": 250, "D": 5000, "K": 12, "test_Ks": [10, 12, 14], "C": 2, "miss": 0.10, "noise": 1.0},
    ]
    if include_misspecification:
        scenarios.extend(
            [
                {
                    "name": "4a. Informative missing and outliers",
                    "N": 180,
                    "D": 1000,
                    "K": 6,
                    "test_Ks": [5, 6, 7],
                    "C": 2,
                    "miss": 0.15,
                    "noise": 1.0,
                    "missing_mechanism": "latent",
                    "informative_missing_strength": 0.8,
                    "outlier_rate": 0.005,
                    "outlier_scale": 8.0,
                },
                {
                    "name": "4b. Nonlinear mean and correlated noise",
                    "N": 180,
                    "D": 1000,
                    "K": 6,
                    "test_Ks": [5, 6, 7],
                    "C": 2,
                    "miss": 0.10,
                    "noise": 1.0,
                    "nonlinear_strength": 0.15,
                    "correlated_noise_strength": 0.35,
                    "batch_effect_scale": 0.20,
                },
                {
                    "name": "4c. Imperfect anchor variables",
                    "N": 180,
                    "D": 1000,
                    "K": 6,
                    "test_Ks": [5, 6, 7],
                    "C": 2,
                    "miss": 0.10,
                    "noise": 1.0,
                    "anchor_cross_loading": 0.20,
                },
            ]
        )
    return scenarios


def build_quick_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "name": "Quick smoke scenario",
            "N": 18,
            "D": 40,
            "K": 2,
            "test_Ks": [1, 2, 3],
            "C": 2,
            "miss": 0.10,
            "noise": 0.8,
        }
    ]


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _flatten_record(record: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_record(value, full_key))
        elif isinstance(value, list):
            flat[full_key] = json.dumps(_json_ready(value), separators=(",", ":"))
        else:
            flat[full_key] = _json_ready(value)
    return flat


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = [_flatten_record(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row})
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def summarize_results(
    scenarios: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for scenario in scenarios:
        scenario_results = [
            result
            for result in results
            if result.get("scenario_name") == scenario["name"] and result.get("status") == "success"
        ]
        success_count = len(scenario_results)
        failure_count = sum(
            1
            for result in results
            if result.get("scenario_name") == scenario["name"] and result.get("status") == "error"
        )
        selection_counts = {
            int(candidate): sum(result.get("best_k") == int(candidate) for result in scenario_results)
            for candidate in scenario["test_Ks"]
        }

        for candidate in scenario["test_Ks"]:
            candidate_rows = [
                sweep_row
                for result in scenario_results
                for sweep_row in result["sweep"]
                if sweep_row["candidate_k"] == int(candidate)
            ]
            if not candidate_rows:
                continue
            summary: Dict[str, Any] = {
                "scenario": scenario["name"],
                "true_k": int(scenario["K"]),
                "candidate_k": int(candidate),
                "successful_runs": success_count,
                "failed_runs": failure_count,
                "selection_rate": (
                    selection_counts[int(candidate)] / success_count if success_count > 0 else None
                ),
                "mean_selection_score": float(np.mean([row["selection_score"] for row in candidate_rows])),
                "mean_elapsed_seconds": float(np.mean([row["elapsed_seconds"] for row in candidate_rows])),
                "mean_train_log_likelihood": float(
                    np.mean([row["train_log_likelihood"] for row in candidate_rows])
                ),
                "mean_validation_nll_per_entry": None,
            }
            validation_values = [
                row["validation_nll_per_entry"]
                for row in candidate_rows
                if row["validation_nll_per_entry"] is not None
            ]
            if validation_values:
                summary["mean_validation_nll_per_entry"] = float(np.mean(validation_values))

            recovery_rows = [row["recovery"] for row in candidate_rows if row["recovery"] is not None]
            if recovery_rows:
                for metric in (
                    "factor_correlation_mean",
                    "train_factor_correlation_mean",
                    "validation_factor_correlation_mean",
                    "loading_correlation",
                    "loading_relative_rmse",
                    "loading_subspace_similarity",
                    "gamma_correlation",
                    "gamma_relative_rmse",
                    "omega_correlation",
                    "omega_relative_rmse",
                    "symmetric_dynamics_correlation",
                    "symmetric_dynamics_relative_rmse",
                    "phi_level_correlation",
                    "phi_slope_correlation",
                    "alpha_slope_correlation",
                    "anchor_recovery_rate",
                    "psi_correlation",
                    "intercept_correlation",
                ):
                    values = [row[metric] for row in recovery_rows if row.get(metric) is not None]
                    finite_values = [value for value in values if np.isfinite(value)]
                    summary[f"mean_{metric}"] = (
                        float(np.mean(finite_values)) if finite_values else None
                    )
                    summary[f"sd_{metric}"] = (
                        float(np.std(finite_values)) if finite_values else None
                    )
            summaries.append(summary)
    return summaries


def _configure_queue_logging(output_directory: Path) -> Tuple[mp.Queue, logging.handlers.QueueListener, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "clouds_simulation.log"
    formatter = logging.Formatter("%(asctime)s [%(processName)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    context = mp.get_context("spawn")
    log_queue = context.Queue()
    listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        stream_handler,
        respect_handler_level=True,
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)
    listener.start()
    return log_queue, listener, log_path


def run_comprehensive_pipeline_stress_test(
    scenarios: Sequence[Mapping[str, Any]],
    n_runs: int,
    fit_config: FitConfig,
    selection_config: SelectionConfig,
    output_directory: Path,
    max_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    base_seed: int = 2000,
) -> Dict[str, Any]:
    fit_config.validate()
    selection_config.validate()
    if not scenarios:
        raise ValueError("At least one scenario is required")
    validate_scenarios(scenarios, selection_config)
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be positive")
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive when provided")

    os.environ["CLOUDS_THREADS_PER_WORKER"] = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    torch.set_num_threads(threads_per_worker)

    output_directory = Path(output_directory)
    log_queue, listener, log_path = _configure_queue_logging(output_directory)
    try:
        if selection_config.metric.startswith("bic") and not fit_config.prior.is_zero():
            LOGGER.warning(
                "BIC is being evaluated at a MAP estimate because nonzero priors are enabled; "
                "use the default zero priors for a standard ML-based BIC calculation"
            )
        tasks: List[Dict[str, Any]] = []
        task_id = 0
        for scenario_index, scenario in enumerate(scenarios):
            for run_index in range(n_runs):
                tasks.append(
                    {
                        "task_id": task_id,
                        "scenario": dict(scenario),
                        "run_index": run_index,
                        "seed": base_seed + 10000 * scenario_index + 101 * run_index,
                    }
                )
                task_id += 1

        cpu_count = os.cpu_count() or 1
        capacity = max(1, cpu_count // threads_per_worker)
        safe_default_workers = max(1, int(os.environ.get("CLOUDS_MAX_WORKERS", "8")))
        requested_workers = max_workers if max_workers is not None else safe_default_workers
        worker_count = min(len(tasks), requested_workers, capacity)
        LOGGER.info("Starting %d tasks with %d workers and %d thread(s) per worker", len(tasks), worker_count, threads_per_worker)
        LOGGER.info("Selection metric: %s; validation fraction: %.3f", selection_config.metric, selection_config.validation_fraction)

        context = mp.get_context("spawn")
        results: List[Dict[str, Any]] = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_initializer,
            initargs=(log_queue, threads_per_worker),
        ) as executor:
            future_to_task = {
                executor.submit(
                    run_single_discovery_sweep,
                    task["task_id"],
                    task["scenario"],
                    task["run_index"],
                    task["seed"],
                    fit_config,
                    selection_config,
                ): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "error",
                        "task_id": task["task_id"],
                        "scenario_name": task["scenario"]["name"],
                        "run_index": task["run_index"],
                        "seed": task["seed"],
                        "error_msg": f"Executor failure: {error}\n{traceback.format_exc()}",
                    }
                results.append(result)
                if result["status"] == "error":
                    LOGGER.error("Task %d failed: %s", result["task_id"], result["error_msg"])
                else:
                    LOGGER.info(
                        "Task %d completed: %s selected K=%s",
                        result["task_id"],
                        result["scenario_name"],
                        result["best_k"],
                    )

        summaries = summarize_results(scenarios, results)
        LOGGER.info("Scenario summary (rates use successful runs only):")
        for summary in summaries:
            LOGGER.info(
                "%s | true K=%d | candidate K=%d | selected=%.1f%% | score=%.6g | time=%.2fs",
                summary["scenario"],
                summary["true_k"],
                summary["candidate_k"],
                100.0 * (summary["selection_rate"] or 0.0),
                summary["mean_selection_score"],
                summary["mean_elapsed_seconds"],
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_json_path = output_directory / f"clouds_raw_results_{timestamp}.json"
        sweep_csv_path = output_directory / f"clouds_sweep_results_{timestamp}.csv"
        summary_csv_path = output_directory / f"clouds_summary_{timestamp}.csv"

        payload = {
            "created_at": datetime.now().isoformat(),
            "fit_config": fit_config,
            "selection_config": selection_config,
            "scenarios": list(scenarios),
            "results": results,
            "summary": summaries,
        }
        raw_json_path.write_text(
            json.dumps(_json_ready(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        sweep_rows = [
            {
                "task_id": result["task_id"],
                "scenario_name": result["scenario_name"],
                "run_index": result.get("run_index"),
                "seed": result.get("seed"),
                "true_k": result.get("true_k"),
                "best_k": result.get("best_k"),
                **sweep_row,
            }
            for result in results
            if result.get("status") == "success"
            for sweep_row in result["sweep"]
        ]
        _write_csv(sweep_csv_path, sweep_rows)
        _write_csv(summary_csv_path, summaries)

        successful_tasks = sum(result["status"] == "success" for result in results)
        failed_tasks = sum(result["status"] == "error" for result in results)
        LOGGER.info("Completed pipeline: %d success, %d failure", successful_tasks, failed_tasks)
        LOGGER.info("Raw JSON: %s", raw_json_path)
        LOGGER.info("Sweep CSV: %s", sweep_csv_path)
        LOGGER.info("Summary CSV: %s", summary_csv_path)
        return {
            "log": str(log_path),
            "raw_json": str(raw_json_path),
            "sweep_csv": str(sweep_csv_path),
            "summary_csv": str(summary_csv_path),
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
        }
    finally:
        listener.stop()
        for handler in listener.handlers:
            handler.close()
        logging.getLogger().handlers.clear()
        log_queue.close()
        log_queue.join_thread()


# -----------------------------------------------------------------------------
# Lightweight numerical self-checks
# -----------------------------------------------------------------------------
def run_self_checks() -> None:
    torch.manual_seed(123)

    # Correlation parameterization has unit diagonal and is positive definite.
    raw = torch.randn(num_correlation_parameters(4)) * 0.2
    factor = correlation_cholesky_from_raw(raw, 4)
    correlation = factor @ factor.transpose(-1, -2)
    if not torch.allclose(torch.diagonal(correlation), torch.ones(4), atol=1e-5):
        raise AssertionError("Correlation parameterization failed unit-diagonal check")
    if torch.linalg.eigvalsh(correlation).min() <= 0.0:
        raise AssertionError("Correlation parameterization is not positive definite")

    # Information-form innovation likelihood agrees with direct observation-space algebra.
    model = CLOUDS(5, 2, 1, anchor_items=[0, 1], theta_mode="exact")
    with torch.no_grad():
        model.obs_intercept.copy_(torch.randn(5) * 0.1)
        target_log_psi = torch.log(torch.tensor([0.8, 1.1, 0.7, 1.3, 0.9]))
        model.log_psi_raw.copy_(raw_from_bounded(target_log_psi, model.min_log_psi, model.max_log_psi))
        model.lambda_raw.normal_(0.0, 0.2)
    observation = torch.randn(5)
    observation[3] = float("nan")
    prior_mean = torch.randn(2)
    prior_covariance = torch.tensor([[1.2, 0.2], [0.2, 0.9]])
    _, _, information_log_likelihood = model._measurement_update(
        observation, prior_mean, prior_covariance, model.Lambda
    )
    valid = ~torch.isnan(observation)
    loading = model.Lambda[valid]
    residual = observation[valid] - model.obs_intercept[valid] - loading @ prior_mean
    innovation = torch.diag(model.psi[valid]) + loading @ prior_covariance @ loading.transpose(0, 1)
    innovation_factor = stable_cholesky(innovation)
    direct_log_likelihood = -0.5 * (
        valid.sum() * LOG_2PI
        + cholesky_logdet(innovation_factor)
        + residual @ torch.cholesky_solve(residual.unsqueeze(1), innovation_factor).squeeze(1)
    )
    if not torch.allclose(information_log_likelihood, direct_log_likelihood, atol=2e-5, rtol=2e-5):
        raise AssertionError("Innovation likelihood check failed")

    # Exact dynamics satisfy the continuous Lyapunov positivity condition.
    gamma, omega, symmetric_part = model.get_dynamics()
    diffusion = symmetrize(gamma @ omega + omega @ gamma.transpose(-1, -2))
    if torch.linalg.eigvalsh(diffusion).min() <= 0.0:
        raise AssertionError("Exact dynamics failed positivity check")
    if not torch.allclose(diffusion, 2.0 * symmetric_part, atol=1e-5, rtol=1e-5):
        raise AssertionError("Exact dynamics failed Lyapunov identity check")

    # Diagonal mode fixes Omega to I, removing the factor-scale invariance.
    diagonal_model = CLOUDS(5, 2, 1, anchor_items=[0, 1], theta_mode="diagonal")
    diagonal_gamma, diagonal_omega, _ = diagonal_model.get_dynamics()
    if not torch.allclose(diagonal_omega, torch.eye(2), atol=1e-7):
        raise AssertionError("Diagonal mode did not fix Omega to identity")
    if not torch.all(torch.diagonal(diagonal_gamma) > 0.0):
        raise AssertionError("Diagonal decay rates must be positive")

    expected_temporal = (3 * model.K * model.K - model.K) // 2
    expected_total = (
        model.D * model.K
        - model.K * (model.K - 1)
        + 2 * model.D
        + 2 * model.K * model.C_dim
        + model.K
        + expected_temporal
    )
    if model.parameter_count() != expected_total:
        raise AssertionError("Exact-mode parameter count is inconsistent")


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Replicates per scenario")
    parser.add_argument("--workers", type=int, default=None, help="Maximum worker processes")
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("clouds_results"))
    parser.add_argument(
        "--selection-metric",
        choices=(
            "validation_nll",
            "bic",
            "bic_anchor",
            "bic_subjects",
            "bic_visits",
            "bic_entries",
            "bic_anchor_subjects",
            "bic_anchor_visits",
            "bic_anchor_entries",
        ),
        default="validation_nll",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help="Subject holdout fraction; defaults to 0.20 for validation NLL and 0 for BIC",
    )
    parser.add_argument(
        "--bic-sample-unit",
        choices=("subjects", "visits", "entries"),
        default="subjects",
        help="Effective sample size used by the bic and bic_anchor aliases",
    )
    parser.add_argument("--base-seed", type=int, default=2000)
    parser.add_argument("--baseline-only", action="store_true", help="Exclude misspecification scenarios")
    parser.add_argument(
        "--weak-map-priors",
        action="store_true",
        help="Use weak regularization instead of the default maximum-likelihood fit",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small end-to-end smoke scenario")
    parser.add_argument("--self-check", action="store_true", help="Run numerical checks and exit")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(argv)
    run_self_checks()
    if arguments.self_check:
        print("All CLOUDS numerical self-checks passed.")
        return 0

    prior = PriorConfig.weak_map() if arguments.weak_map_priors else PriorConfig()

    if arguments.quick:
        scenarios = build_quick_scenarios()
        fit_config = FitConfig(
            total_em_epochs=4,
            burn_in_epochs=1,
            warmup_epochs=1,
            m_step_iters=2,
            n_starts=2,
            early_stopping_patience=3,
            prior=prior,
        )
        n_runs = 1
    else:
        scenarios = build_default_scenarios(include_misspecification=not arguments.baseline_only)
        fit_config = FitConfig(prior=prior)
        n_runs = arguments.runs

    validation_fraction = arguments.validation_fraction
    if validation_fraction is None:
        validation_fraction = 0.20 if arguments.selection_metric == "validation_nll" else 0.0
    selection_config = SelectionConfig(
        validation_fraction=validation_fraction,
        metric=arguments.selection_metric,
        bic_sample_unit=arguments.bic_sample_unit,
        include_anchor_search_penalty=True,
    )
    artifacts = run_comprehensive_pipeline_stress_test(
        scenarios=scenarios,
        n_runs=n_runs,
        fit_config=fit_config,
        selection_config=selection_config,
        output_directory=arguments.output_dir,
        max_workers=arguments.workers,
        threads_per_worker=arguments.threads_per_worker,
        base_seed=arguments.base_seed,
    )
    print(json.dumps(artifacts, indent=2))
    return 0 if artifacts["successful_tasks"] > 0 else 1


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())