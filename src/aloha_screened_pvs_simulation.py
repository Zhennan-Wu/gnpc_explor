#!/usr/bin/env python3
"""Screened PVS/LOVE initialization for ALOHA, with simulation experiments.

This module is intentionally an extension layer around the original ALOHA/CLOUDS
implementation supplied by the user.  It does not duplicate the Kalman/RTS or
EM core.  Instead it adds the proposed missing-middle pipeline:

    cross-fitted mean removal
        -> subject-balanced residual moments
        -> one high-recall low-rank screen
        -> approximate-replicate/PVS structure recovery on s << p features
        -> signed core anchors and halo variables
        -> exact fixed-structure ALOHA refit on all p features.

The PVS implementation follows the q=2 correlation-profile score and the
successive Schur-complement pruning idea from Bing, Bunea and Wegkamp (2023),
but is adapted to a screened feature set and subject-level longitudinal blocks.
It is research code: all thresholds are exposed, diagnostics are written, and
failure to find an identified structure is reported rather than silently relaxed.

Example
-------
python aloha_screened_pvs_simulation.py \
    --aloha-core '/path/to/clouds_anchor_simu_multi_core_chat8(1).py' \
    --experiment smoke

python aloha_screened_pvs_simulation.py \
    --aloha-core '/path/to/clouds_anchor_simu_multi_core_chat8(1).py' \
    --experiment structure --scenario baseline --replicates 100 \
    --output-dir results_screened_pvs

python aloha_screened_pvs_simulation.py \
    --aloha-core '/path/to/clouds_anchor_simu_multi_core_chat8(1).py' \
    --experiment end-to-end --scenario baseline --replicates 5 \
    --fit-profile fast --output-dir results_screened_pvs
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover
    fcluster = linkage = squareform = None


# -----------------------------------------------------------------------------
# Core loading and generic helpers
# -----------------------------------------------------------------------------


def load_aloha_core(path: str):
    """Load the supplied ALOHA/CLOUDS Python file without requiring a module name."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"ALOHA core file not found: {source}")
    spec = importlib.util.spec_from_file_location("aloha_reference_core", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import specification for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    return np.random.default_rng(int(seed))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().numpy())
    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(str(key))
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(json_safe(row.get(k))) if isinstance(row.get(k), (list, dict, tuple)) else json_safe(row.get(k)) for k in keys})


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    """PSD-project a symmetric matrix and renormalize it to unit diagonal."""
    A = np.asarray(matrix, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, floor, None)
    B = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(B), floor, None))
    B = B / np.outer(d, d)
    B = 0.5 * (B + B.T)
    np.fill_diagonal(B, 1.0)
    return B


def psd_projection(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    A = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, floor, None)
    return 0.5 * (((vecs * vals) @ vecs.T) + ((vecs * vals) @ vecs.T).T)


def sign_invariant_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    den = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(abs(np.dot(a, b)) / den)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# -----------------------------------------------------------------------------
# Configuration and returned structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    name: str = "baseline"
    n_subjects: int = 160
    obs_dim: int = 2000
    latent_dim: int = 4
    covar_dim: int = 2
    visit_min: int = 5
    visit_max: int = 8
    gap_year_min: float = 1.5
    gap_year_max: float = 5.0
    dynamics_mode: str = "exact"
    omega_rho: float = 0.35
    gamma_target_rate: float = 1.0
    gamma_skew_strength: float = 0.06
    anchors_per_factor: int = 4
    positive_anchors_per_factor: int = 3
    halos_per_factor: int = 2
    halo_cross_loading: float = 0.10
    dense_replicate_groups: int = 0
    dense_replicates_per_group: int = 3
    dense_replicate_scale: float = 0.45
    loading_scale: float = 0.28
    nonanchor_zero_fraction: float = 0.15
    pure_noise_fraction: float = 0.03
    anchor_log_scale_sd: float = 0.35
    noise_scale: float = 0.8
    noise_log_sd: float = 0.35
    residual_block_rho: float = 0.0
    item_missing_rate: float = 0.10
    visit_missing_rate: float = 0.05
    observation_intercept_scale: float = 0.25
    include_latent_level: bool = False
    seed: int = 1000


@dataclass(frozen=True)
class StructureConfig:
    crossfit_folds: int = 2
    mean_ridge: float = 1e-4
    feature_chunk_size: int = 512
    max_screen_rank: int = 16
    spectral_rank_c0: float = 1.0
    screen_rank_padding: int = 4
    screen_size: int = 320
    direction_cells: int = 24
    top_per_cell: int = 10
    global_top_features: int = 40
    min_feature_coverage: float = 0.45
    min_embedding_strength_quantile: float = 0.05
    pure_noise_max_quantile: float = 0.60
    pure_noise_grid_size: int = 25
    min_candidates_after_noise_screen: int = 24
    delta_multipliers: Tuple[float, ...] = (0.65, 0.80, 1.00, 1.20, 1.45, 1.75)
    delta_base_constant: float = 0.15
    delta_selection: str = "split_stability"  # split_stability, split_reconstruction, or theory
    rank_method: str = "split_prediction"  # or threshold
    rank_penalty: float = 0.01
    rank_threshold_multiplier: float = 0.25
    min_parallel_group_size: int = 3
    parallel_grouping: str = "neighborhood"  # neighborhood/complete are robust; graph reproduces Algorithm 1
    max_parallel_group_size: int = 8
    neighborhood_clique_multiplier: float = 1.20
    max_group_fraction: float = 0.20
    min_group_common_diagonal: float = 0.06
    split_group_jaccard: float = 0.35
    split_threshold_multiplier: float = 1.35
    core_anchors_per_factor: int = 3
    halo_score_multiplier: float = 1.75
    bootstrap_replicates: int = 30
    bootstrap_full_screen: bool = False
    bootstrap_seed: int = 7000
    require_all_core_anchors: bool = True
    covariance_psd_floor: float = 1e-7


@dataclass
class ResidualizedData:
    residuals: np.ndarray
    observed: np.ndarray
    design: np.ndarray
    subject_ids: np.ndarray
    row_weights: np.ndarray
    row_subject_position: List[Tuple[int, int]]
    feature_coverage: np.ndarray
    feature_scale: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelStructure:
    delta: float
    candidate_indices: np.ndarray
    correlation: np.ndarray
    covariance: np.ndarray
    scores: np.ndarray
    parallel_groups_local: List[List[int]]
    parallel_groups_global: List[List[int]]
    representatives_local: List[int]
    representative_matrix: np.ndarray
    estimated_rank: int
    selected_group_ids: List[int]
    core_groups: List[List[int]]
    anchor_signs: List[List[int]]
    halo_groups: List[List[int]]
    screening_rank: int
    screening_indices: np.ndarray
    bootstrap_stability: Optional[np.ndarray] = None
    bootstrap_successful: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Longitudinal DGP with core anchors, halos, and dense replicate groups
# -----------------------------------------------------------------------------


def toeplitz_correlation(k: int, rho: float, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(k)
    return torch.tensor(float(rho), dtype=dtype).pow(torch.abs(idx[:, None] - idx[None, :]))


def generate_loading_structure(cfg: SimulationConfig, rng: np.random.Generator, dtype: torch.dtype) -> Tuple[torch.Tensor, Dict[str, Any]]:
    D, K = cfg.obs_dim, cfg.latent_dim
    required = (
        K * cfg.anchors_per_factor
        + K * cfg.halos_per_factor
        + cfg.dense_replicate_groups * cfg.dense_replicates_per_group
    )
    n_pure_noise = int(round(cfg.pure_noise_fraction * D))
    if required + n_pure_noise > D:
        raise ValueError("obs_dim is too small for the requested structural groups.")

    Lambda = torch.zeros(D, K, dtype=dtype)
    core_groups: List[List[int]] = []
    anchor_signs: List[List[int]] = []
    halo_groups: List[List[int]] = []
    dense_groups: List[List[int]] = []
    dense_directions: List[List[float]] = []
    cursor = 0

    for r in range(K):
        group = list(range(cursor, cursor + cfg.anchors_per_factor))
        cursor += cfg.anchors_per_factor
        signs = [1] * min(cfg.positive_anchors_per_factor, cfg.anchors_per_factor)
        signs += [-1] * (cfg.anchors_per_factor - len(signs))
        if len(signs) > 1:
            tail = signs[1:]
            rng.shuffle(tail)
            signs[1:] = tail
        signs[0] = 1
        for idx, sign in zip(group, signs):
            scale = float(np.exp(rng.normal(0.0, cfg.anchor_log_scale_sd)))
            Lambda[idx, r] = sign * scale
        core_groups.append(group)
        anchor_signs.append([int(s) for s in signs])

    for r in range(K):
        group = list(range(cursor, cursor + cfg.halos_per_factor))
        cursor += cfg.halos_per_factor
        for idx in group:
            row = np.zeros(K, dtype=float)
            dominant = float(np.exp(rng.normal(-0.15, 0.25)))
            row[r] = dominant
            choices = [x for x in range(K) if x != r]
            if choices:
                cross_count = min(2, len(choices))
                cross = rng.choice(choices, size=cross_count, replace=False)
                row[cross] = rng.normal(0.0, cfg.halo_cross_loading, size=cross_count)
            Lambda[idx] = torch.tensor(row, dtype=dtype)
        halo_groups.append(group)

    for _ in range(cfg.dense_replicate_groups):
        group = list(range(cursor, cursor + cfg.dense_replicates_per_group))
        cursor += cfg.dense_replicates_per_group
        direction = rng.normal(size=K)
        # Force a genuinely dense/noncanonical direction.
        if K >= 2:
            keep = np.argsort(np.abs(direction))[-min(K, 3):]
            mask = np.zeros(K, dtype=bool)
            mask[keep] = True
            direction[~mask] = 0.0
        direction /= max(np.linalg.norm(direction), 1e-12)
        if np.max(np.abs(direction)) > 0.92 and K >= 2:
            direction = direction + 0.35 * np.roll(direction, 1)
            direction /= np.linalg.norm(direction)
        for idx in group:
            scale = cfg.dense_replicate_scale * float(np.exp(rng.normal(-0.10, 0.20)))
            sign = 1.0 if rng.random() > 0.35 else -1.0
            Lambda[idx] = torch.tensor(sign * scale * direction, dtype=dtype)
        dense_groups.append(group)
        dense_directions.append(direction.tolist())

    pure_noise_indices = list(range(D - n_pure_noise, D)) if n_pure_noise > 0 else []
    remaining_end = D - n_pure_noise
    for idx in range(cursor, remaining_end):
        # Ordinary nonanchors are deliberately weaker than core anchors.  Their
        # direction can be sparse, but each row is normalized before assigning a
        # bounded total loading norm.  This makes the well-specified baseline
        # satisfy the PVS relative-scale condition while separate stress suites
        # can violate it through dense replicate groups or weak anchors.
        row = rng.normal(size=K)
        zero_mask = rng.random(K) < cfg.nonanchor_zero_fraction
        row[zero_mask] = 0.0
        if np.count_nonzero(np.abs(row) > 1e-10) < min(2, K):
            chosen = rng.choice(K, size=min(2, K), replace=False)
            row[chosen] = rng.normal(size=len(chosen))
        norm = np.linalg.norm(row)
        row = row / max(norm, 1e-12)
        if K > 1 and np.max(np.abs(row)) > 0.90:
            largest = int(np.argmax(np.abs(row)))
            second = int(np.argsort(np.abs(row))[-2])
            row[second] += np.sign(row[largest]) * 0.40
            row /= max(np.linalg.norm(row), 1e-12)
        total_scale = float(rng.uniform(0.08, max(cfg.loading_scale, 0.081)))
        Lambda[idx] = torch.tensor(total_scale * row, dtype=dtype)

    structure = {
        "core_groups": core_groups,
        "anchor_signs": anchor_signs,
        "halo_groups": halo_groups,
        "dense_replicate_groups": dense_groups,
        "dense_directions": dense_directions,
        "pure_noise_indices": pure_noise_indices,
        "all_core_indices": sorted(i for g in core_groups for i in g),
        "all_halo_indices": sorted(i for g in halo_groups for i in g),
        "all_dense_replicate_indices": sorted(i for g in dense_groups for i in g),
    }
    return Lambda, structure


def simulate_structured_aloha_cohort(core: Any, cfg: SimulationConfig) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Simulate the final proposed dynamic factor model.

    The latent process follows the same exact OU transition used by the supplied
    ALOHA core.  The measurement matrix is extended to contain arbitrary-scale
    signed pure anchors, near-pure halo variables, and optional dense parallel
    groups that force the PVS pruning problem.
    """
    if cfg.dynamics_mode not in {"exact", "diagonal"}:
        raise ValueError("dynamics_mode must be exact or diagonal")
    if cfg.visit_min < 2 or cfg.visit_max < cfg.visit_min:
        raise ValueError("visit_min must be at least two and no larger than visit_max")
    rng = set_seed(cfg.seed)
    dtype = torch.get_default_dtype()
    K, D, C = cfg.latent_dim, cfg.obs_dim, cfg.covar_dim
    eye = torch.eye(K, dtype=dtype)

    if cfg.dynamics_mode == "diagonal":
        rates = torch.linspace(0.35, 1.25, K, dtype=dtype)
        Omega = eye.clone()
        Gamma = torch.diag(rates)
        diffusion = torch.diag(torch.sqrt(2.0 * rates))
    else:
        Omega = toeplitz_correlation(K, cfg.omega_rho, dtype)
        Ls = torch.tril(0.20 * torch.randn(K, K, dtype=dtype) + 0.70 * eye)
        S = core.symmetrize(0.5 * (Ls @ Ls.T)) + 1e-5 * eye
        raw = cfg.gamma_skew_strength * torch.randn(K, K, dtype=dtype)
        skew = raw - raw.T
        Gamma_base = torch.linalg.solve(Omega.T, (S + skew).T).T
        rates = torch.real(torch.linalg.eigvals(Gamma_base)).clamp_min(1e-6)
        scale = cfg.gamma_target_rate / float(torch.median(rates).item())
        Gamma = Gamma_base * scale
        diffusion = torch.linalg.cholesky(core.symmetrize(2.0 * S * scale))

    Phi = 0.45 * torch.randn(K, C, dtype=dtype)
    alpha = 0.35 * torch.randn(K, dtype=dtype)
    Phi_level = 0.25 * torch.randn(K, C, dtype=dtype) if cfg.include_latent_level else torch.zeros(K, C, dtype=dtype)
    level_bias = torch.zeros(K, dtype=dtype)

    Lambda, structure = generate_loading_structure(cfg, rng, dtype)
    obs_intercept = cfg.observation_intercept_scale * torch.randn(D, dtype=dtype)
    psi = (cfg.noise_scale ** 2) * torch.exp(cfg.noise_log_sd * torch.randn(D, dtype=dtype))

    # Optional block residual correlation is a deliberate misspecification suite.
    block_size = min(12, D)
    block_chol: Optional[torch.Tensor] = None
    if cfg.residual_block_rho > 0:
        B = (1.0 - cfg.residual_block_rho) * torch.eye(block_size, dtype=dtype) + cfg.residual_block_rho * torch.ones(block_size, block_size, dtype=dtype)
        B = torch.diag(torch.sqrt(psi[:block_size])) @ B @ torch.diag(torch.sqrt(psi[:block_size]))
        block_chol = core.safe_cholesky(B, jitter=1e-10)

    L_Omega = core.safe_cholesky(Omega, jitter=1e-12)
    subjects: List[Dict[str, torch.Tensor]] = []
    for subject_id in range(cfg.n_subjects):
        J = int(rng.integers(cfg.visit_min, cfg.visit_max + 1))
        baseline = float(rng.uniform(55.0, 75.0))
        gaps = rng.uniform(cfg.gap_year_min, cfg.gap_year_max, size=J - 1)
        ages = np.concatenate([[baseline], baseline + np.cumsum(gaps)])
        t_model = torch.tensor((ages - 70.0) / 10.0, dtype=dtype)
        x_cov = torch.tensor(rng.normal(size=C), dtype=dtype)
        u = x_cov.unsqueeze(0).expand(J, C).clone()
        slope = Phi @ x_cov + alpha
        level = Phi_level @ x_cov + level_bias
        mu = level.unsqueeze(0) + t_model.unsqueeze(1) * slope.unsqueeze(0)

        latent = torch.zeros(J, K, dtype=dtype)
        latent[0] = mu[0] + L_Omega @ torch.randn(K, dtype=dtype)
        for j in range(1, J):
            dt = t_model[j] - t_model[j - 1]
            if cfg.dynamics_mode == "diagonal":
                rates_diag = torch.diag(Gamma)
                A = torch.diag(torch.exp(-rates_diag * dt))
                q = -torch.expm1(-2.0 * rates_diag * dt)
                Q = torch.diag(q.clamp_min(torch.finfo(dtype).tiny))
            else:
                A = torch.linalg.matrix_exp(-Gamma * dt)
                Q = core.symmetrize(Omega - A @ Omega @ A.T)
            LQ = core.safe_cholesky(Q, jitter=1e-12)
            latent[j] = mu[j] + A @ (latent[j - 1] - mu[j - 1]) + LQ @ torch.randn(K, dtype=dtype)

        eps = torch.randn(J, D, dtype=dtype) * torch.sqrt(psi).unsqueeze(0)
        if block_chol is not None:
            eps[:, :block_size] = torch.randn(J, block_size, dtype=dtype) @ block_chol.T
        Y = obs_intercept.unsqueeze(0) + latent @ Lambda.T + eps

        if cfg.visit_missing_rate > 0:
            for j in range(1, J):
                if rng.random() < cfg.visit_missing_rate:
                    Y[j] = float("nan")
        if cfg.item_missing_rate > 0:
            miss = torch.tensor(rng.random((J, D)) < cfg.item_missing_rate)
            Y[miss] = float("nan")

        subjects.append({
            "x": Y,
            "u": u,
            "x_xi": x_cov,
            "t_model": t_model,
            "t_dyn": t_model,
            "t_trend": t_model,
            "t_age": torch.tensor(ages, dtype=dtype),
            "t": t_model,
            "F_true": latent,
            "subject_id": torch.tensor(subject_id),
        })

    truth: Dict[str, Any] = {
        "Lambda": Lambda,
        "Gamma": Gamma,
        "Omega": Omega,
        "G": diffusion,
        "Phi": Phi,
        "alpha": alpha,
        "Phi_level": Phi_level,
        "level_bias": level_bias,
        "obs_intercept": obs_intercept,
        "psi": psi,
        **structure,
        "simulation_config": dataclasses.asdict(cfg),
    }
    return subjects, truth


# -----------------------------------------------------------------------------
# Cross-fitted featurewise mean removal
# -----------------------------------------------------------------------------


def build_longitudinal_design(subjects: Sequence[Mapping[str, torch.Tensor]], include_level: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    designs: List[np.ndarray] = []
    subject_ids: List[int] = []
    weights: List[float] = []
    positions: List[Tuple[int, int]] = []
    for i, subject in enumerate(subjects):
        Y = subject["x"]
        J = int(Y.shape[0])
        t = subject.get("t_trend", subject.get("t_model", subject["t"])).detach().cpu().numpy().astype(float)
        if "x_xi" in subject:
            z = subject["x_xi"].detach().cpu().numpy().astype(float).reshape(-1)
        else:
            u = subject.get("u")
            z = u[0].detach().cpu().numpy().astype(float).reshape(-1) if u is not None else np.empty(0)
        for j in range(J):
            parts = [1.0]
            if include_level and z.size:
                parts.extend(z.tolist())
            parts.append(float(t[j]))
            if z.size:
                parts.extend((float(t[j]) * z).tolist())
            designs.append(np.asarray(parts, dtype=float))
            subject_ids.append(i)
            weights.append(1.0 / max(J, 1))
            positions.append((i, j))
    return np.vstack(designs), np.asarray(subject_ids, dtype=int), np.asarray(weights, dtype=float), positions


def stack_measurements(subjects: Sequence[Mapping[str, torch.Tensor]]) -> np.ndarray:
    return np.vstack([s["x"].detach().cpu().numpy().astype(float) for s in subjects])


def batched_featurewise_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    chunk_size: int,
) -> np.ndarray:
    """Fit one weighted ridge regression per feature, in feature chunks."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    n, q = X.shape
    p = Y.shape[1]
    beta = np.zeros((q, p), dtype=float)
    xx = np.einsum("na,nb->nab", X, X, optimize=True)
    eye = np.eye(q)
    for start in range(0, p, max(1, int(chunk_size))):
        stop = min(p, start + max(1, int(chunk_size)))
        Yc = Y[:, start:stop]
        mask = np.isfinite(Yc).astype(float)
        Y0 = np.nan_to_num(Yc, nan=0.0)
        wm = weights[:, None] * mask
        A = np.einsum("nf,nab->fab", wm, xx, optimize=True)
        b = np.einsum("nf,na,nf->fa", wm, X, Y0, optimize=True)
        counts = wm.sum(axis=0)
        A += float(ridge) * eye[None, :, :]
        try:
            sol = np.linalg.solve(A, b[..., None]).squeeze(-1)
        except np.linalg.LinAlgError:
            sol = np.stack([np.linalg.pinv(Af) @ bf for Af, bf in zip(A, b)], axis=0)
        sol[counts <= 0] = 0.0
        beta[:, start:stop] = sol.T
    return beta


def crossfit_residualize(
    subjects: Sequence[Mapping[str, torch.Tensor]],
    *,
    n_folds: int,
    ridge: float,
    chunk_size: int,
    seed: int,
    include_level: bool,
) -> ResidualizedData:
    Y = stack_measurements(subjects)
    design, subject_ids, row_weights, positions = build_longitudinal_design(subjects, include_level)
    observed = np.isfinite(Y)
    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    n_folds = max(1, min(int(n_folds), len(shuffled)))
    fold_map = {int(s): int(k % n_folds) for k, s in enumerate(shuffled)}
    folds = np.asarray([fold_map[int(s)] for s in subject_ids], dtype=int)
    residuals = np.full_like(Y, np.nan, dtype=float)
    coefficients: List[np.ndarray] = []

    if n_folds == 1:
        beta = batched_featurewise_ridge(design, Y, row_weights, ridge, chunk_size)
        pred = design @ beta
        residuals[observed] = (Y - pred)[observed]
        coefficients.append(beta)
    else:
        for fold in range(n_folds):
            train = folds != fold
            test = folds == fold
            beta = batched_featurewise_ridge(design[train], Y[train], row_weights[train], ridge, chunk_size)
            pred = design[test] @ beta
            sub_obs = observed[test]
            out = np.full_like(Y[test], np.nan, dtype=float)
            out[sub_obs] = (Y[test] - pred)[sub_obs]
            residuals[test] = out
            coefficients.append(beta)

    obs_weight = np.sum(row_weights[:, None] * observed, axis=0)
    total_weight = float(np.sum(row_weights))
    coverage = obs_weight / max(total_weight, 1e-12)
    centered = residuals.copy()
    means = np.divide(
        np.nansum(row_weights[:, None] * centered, axis=0),
        np.maximum(obs_weight, 1e-12),
    )
    centered = centered - means[None, :]
    var = np.divide(
        np.nansum(row_weights[:, None] * np.nan_to_num(centered, nan=0.0) ** 2, axis=0),
        np.maximum(obs_weight, 1e-12),
    )
    scale = np.sqrt(np.maximum(var, 1e-10))

    return ResidualizedData(
        residuals=residuals,
        observed=observed,
        design=design,
        subject_ids=subject_ids,
        row_weights=row_weights,
        row_subject_position=positions,
        feature_coverage=coverage,
        feature_scale=scale,
        metadata={
            "n_rows": int(Y.shape[0]),
            "n_subjects": int(len(unique_subjects)),
            "n_features": int(Y.shape[1]),
            "design_dim": int(design.shape[1]),
            "crossfit_folds": int(n_folds),
            "coefficient_fits": len(coefficients),
        },
    )


# -----------------------------------------------------------------------------
# Spectral high-recall screen
# -----------------------------------------------------------------------------


def weighted_standardized_matrix(data: ResidualizedData) -> Tuple[np.ndarray, np.ndarray]:
    R = data.residuals.copy()
    w = data.row_weights
    obs = np.isfinite(R)
    obs_weight = np.sum(w[:, None] * obs, axis=0)
    means = np.divide(np.nansum(w[:, None] * R, axis=0), np.maximum(obs_weight, 1e-12))
    centered = R - means[None, :]
    var = np.divide(
        np.nansum(w[:, None] * np.nan_to_num(centered, nan=0.0) ** 2, axis=0),
        np.maximum(obs_weight, 1e-12),
    )
    scale = np.sqrt(np.maximum(var, 1e-10))
    Z = centered / scale[None, :]
    Z = np.nan_to_num(Z, nan=0.0)
    total_weight = max(float(np.sum(w)), 1e-12)
    # Correct first-order attenuation from missing cells after zero filling.
    missing_adjustment = np.sqrt(total_weight / np.maximum(obs_weight, 1e-12))
    Xw = np.sqrt(w)[:, None] * Z * missing_adjustment[None, :]
    return Xw, scale


def adaptive_spectral_rank(singular_values: np.ndarray, n_rows: int, p: int, max_rank: int, c0: float) -> int:
    s2 = np.asarray(singular_values, dtype=float) ** 2
    total = float(np.sum(s2))
    max_rank = min(int(max_rank), len(s2), max(1, n_rows - 1), p)
    mu = float(c0) * (n_rows + p)
    values = []
    for k in range(max_rank + 1):
        denom = n_rows * p - mu * k
        if denom <= 0:
            values.append(float("inf"))
        else:
            values.append((total - float(np.sum(s2[:k]))) / denom)
    return max(1, int(np.argmin(values)))


def truncated_embedding(core: Any, data: ResidualizedData, cfg: StructureConfig) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    Xw, _ = weighted_standardized_matrix(data)
    max_rank = min(cfg.max_screen_rank, min(Xw.shape))
    Xt = torch.tensor(Xw, dtype=torch.get_default_dtype())
    U, S, Vh = core.truncated_svd_matrix(Xt, max_rank, randomized_threshold=2_000_000)
    singular = S.detach().cpu().numpy()
    rank_hat = adaptive_spectral_rank(singular, data.metadata["n_subjects"], Xw.shape[1], max_rank, cfg.spectral_rank_c0)
    screen_rank = min(max_rank, max(rank_hat + cfg.screen_rank_padding, 2 * rank_hat))
    embedding = (Vh[:screen_rank].T * S[:screen_rank]).detach().cpu().numpy()
    strength = np.linalg.norm(embedding, axis=1)
    return embedding, strength, int(screen_rank), {
        "spectral_rank_hat": int(rank_hat),
        "screen_rank": int(screen_rank),
        "singular_values": singular.tolist(),
    }


def diversified_directional_screen(
    embedding: np.ndarray,
    strength: np.ndarray,
    coverage: np.ndarray,
    cfg: StructureConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    p, r = embedding.shape
    valid = np.flatnonzero(
        (coverage >= cfg.min_feature_coverage)
        & np.isfinite(strength)
        & (strength > 1e-12)
    )
    if valid.size == 0:
        raise RuntimeError("No features pass the coverage/signal screen.")
    strength_cut = np.quantile(strength[valid], cfg.min_embedding_strength_quantile)
    valid = valid[strength[valid] >= strength_cut]
    if valid.size < cfg.core_anchors_per_factor * 2:
        raise RuntimeError("Too few features remain after the permissive screen.")

    directions = embedding[valid] / np.maximum(strength[valid, None], 1e-12)
    n_cells = min(cfg.direction_cells, max(1, cfg.screen_size // max(cfg.top_per_cell, 1)), valid.size)
    rel_strength = strength[valid] / max(float(np.median(strength[valid])), 1e-12)
    centers_local: List[int] = [int(np.argmax(strength[valid]))]
    min_distance = 1.0 - np.abs(directions @ directions[centers_local[0]])
    for _ in range(1, n_cells):
        objective = min_distance * np.sqrt(np.clip(rel_strength, 0.05, 20.0))
        objective[centers_local] = -np.inf
        nxt = int(np.argmax(objective))
        if not np.isfinite(objective[nxt]):
            break
        centers_local.append(nxt)
        dist = 1.0 - np.abs(directions @ directions[nxt])
        min_distance = np.minimum(min_distance, dist)

    center_dirs = directions[np.asarray(centers_local)]
    assignments = np.argmax(np.abs(directions @ center_dirs.T), axis=1)
    selected: List[int] = []
    per_cell: Dict[int, List[int]] = {}
    for cell in range(len(centers_local)):
        members_local = np.flatnonzero(assignments == cell)
        if members_local.size == 0:
            continue
        order = members_local[np.argsort(-strength[valid[members_local]])]
        chosen = valid[order[: cfg.top_per_cell]].tolist()
        per_cell[cell] = [int(x) for x in chosen]
        selected.extend(chosen)

    global_order = valid[np.argsort(-strength[valid])]
    selected.extend(global_order[: cfg.global_top_features].tolist())
    # Stable unique order, then cap by a score while preserving directional representatives.
    unique = list(dict.fromkeys(int(x) for x in selected))
    mandatory = set(int(valid[c]) for c in centers_local)
    if len(unique) > cfg.screen_size:
        keep_mandatory = [x for x in unique if x in mandatory]
        remaining = [x for x in unique if x not in mandatory]
        remaining.sort(key=lambda x: (strength[x] * math.sqrt(max(coverage[x], 1e-12))), reverse=True)
        unique = keep_mandatory + remaining[: max(0, cfg.screen_size - len(keep_mandatory))]
    elif len(unique) < min(cfg.screen_size, valid.size):
        for x in global_order:
            xi = int(x)
            if xi not in unique:
                unique.append(xi)
                if len(unique) >= cfg.screen_size:
                    break

    result = np.asarray(sorted(unique), dtype=int)
    return result, {
        "valid_features": int(valid.size),
        "strength_cut": float(strength_cut),
        "direction_cells_requested": int(cfg.direction_cells),
        "direction_cells_used": int(len(centers_local)),
        "screen_size": int(result.size),
        "cell_members": per_cell,
    }


# -----------------------------------------------------------------------------
# Subject-balanced covariance/correlation and PVS
# -----------------------------------------------------------------------------


def subject_balanced_covariance(
    residualized: ResidualizedData,
    indices: Sequence[int],
    *,
    subject_subset: Optional[Sequence[int]] = None,
    psd_floor: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    idx = np.asarray(indices, dtype=int)
    X = residualized.residuals[:, idx]
    w = residualized.row_weights.copy()
    if subject_subset is not None:
        allowed = np.isin(residualized.subject_ids, np.asarray(subject_subset, dtype=int))
        X = X[allowed]
        w = w[allowed]
    mask = np.isfinite(X)
    if not np.any(mask):
        raise RuntimeError("No observed residuals for candidate features.")
    obs_weight = np.sum(w[:, None] * mask, axis=0)
    means = np.divide(np.nansum(w[:, None] * X, axis=0), np.maximum(obs_weight, 1e-12))
    centered = X - means[None, :]
    filled = np.nan_to_num(centered, nan=0.0)
    weighted = filled * np.sqrt(w)[:, None]
    numerator = weighted.T @ weighted
    denom = mask.astype(float).T @ (w[:, None] * mask.astype(float))
    covariance = np.divide(numerator, np.maximum(denom, 1e-12))
    covariance = 0.5 * (covariance + covariance.T)
    diag = np.clip(np.diag(covariance), 1e-10, None)
    correlation = covariance / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(correlation, 1.0)
    correlation = nearest_correlation(correlation, floor=psd_floor)
    covariance_psd = correlation * np.sqrt(np.outer(diag, diag))
    n_subjects = len(np.unique(residualized.subject_ids if subject_subset is None else np.asarray(subject_subset)))
    return covariance_psd, correlation, {
        "min_pair_weight": float(np.min(denom)),
        "median_pair_weight": float(np.median(denom)),
        "min_feature_weight": float(np.min(obs_weight)),
        "n_subjects": int(n_subjects),
    }


def offdiagonal_row_norm(correlation: np.ndarray) -> np.ndarray:
    R = np.asarray(correlation, dtype=float).copy()
    np.fill_diagonal(R, 0.0)
    return np.linalg.norm(R, axis=1)


def _masked_correlation_loss(source: np.ndarray, target: np.ndarray, keep: np.ndarray) -> float:
    masked = np.asarray(source, dtype=float).copy()
    drop = np.flatnonzero(~keep)
    if drop.size:
        masked[drop, :] = 0.0
        masked[:, drop] = 0.0
    np.fill_diagonal(masked, 1.0)
    mask = ~np.eye(masked.shape[0], dtype=bool)
    return float(np.sqrt(np.mean((masked[mask] - target[mask]) ** 2)))


def select_pure_noise_prescreen(
    full_correlation: np.ndarray,
    first_correlation: np.ndarray,
    second_correlation: np.ndarray,
    cfg: StructureConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Data-driven version of the PVS pure-noise row-norm pre-screen.

    The approximate-replicate paper recommends removing variables whose
    off-diagonal correlation profile is indistinguishable from zero.  We choose
    the removed fraction by symmetric split-sample prediction, then apply that
    quantile to the full-sample row norms.
    """
    n = full_correlation.shape[0]
    norm_full = offdiagonal_row_norm(full_correlation)
    norm_first = offdiagonal_row_norm(first_correlation)
    norm_second = offdiagonal_row_norm(second_correlation)
    max_q = float(np.clip(cfg.pure_noise_max_quantile, 0.0, 0.95))
    grid_n = max(2, int(cfg.pure_noise_grid_size))
    quantiles = np.linspace(0.0, max_q, grid_n)
    records: List[Dict[str, Any]] = []
    for q in quantiles:
        t1 = float(np.quantile(norm_first, q))
        t2 = float(np.quantile(norm_second, q))
        keep1 = norm_first > t1
        keep2 = norm_second > t2
        loss12 = _masked_correlation_loss(first_correlation, second_correlation, keep1)
        loss21 = _masked_correlation_loss(second_correlation, first_correlation, keep2)
        records.append({
            "quantile": float(q),
            "threshold_first": t1,
            "threshold_second": t2,
            "kept_first": int(np.sum(keep1)),
            "kept_second": int(np.sum(keep2)),
            "loss_first_to_second": loss12,
            "loss_second_to_first": loss21,
            "mean_loss": 0.5 * (loss12 + loss21),
        })
    best = min(records, key=lambda r: (float(r["mean_loss"]), float(r["quantile"])))
    q_hat = float(best["quantile"])
    threshold_full = float(np.quantile(norm_full, q_hat))
    keep = norm_full > threshold_full
    minimum = min(
        n,
        max(
            int(cfg.min_candidates_after_noise_screen),
            2 * int(cfg.core_anchors_per_factor),
            3 * int(cfg.max_screen_rank),
        ),
    )
    if int(np.sum(keep)) < minimum:
        order = np.argsort(-norm_full)
        keep = np.zeros(n, dtype=bool)
        keep[order[:minimum]] = True
    return keep, {
        "selected_quantile": q_hat,
        "threshold_full": threshold_full,
        "kept": int(np.sum(keep)),
        "removed": int(n - np.sum(keep)),
        "row_norm_min": float(np.min(norm_full)),
        "row_norm_median": float(np.median(norm_full)),
        "row_norm_max": float(np.max(norm_full)),
        "path": records,
    }


def pvs_s2_scores(correlation: np.ndarray) -> np.ndarray:
    """Closed-form q=2 approximate-replicate score for every feature pair."""
    R = np.asarray(correlation, dtype=float)
    s = R.shape[0]
    if s < 3:
        raise ValueError("At least three candidate features are required.")
    gram = R @ R.T
    row_sq = np.diag(gram)
    rij2 = R ** 2
    vii = row_sq[:, None] - 1.0 - rij2
    vjj = row_sq[None, :] - 1.0 - rij2
    vij = gram - 2.0 * R
    minv = np.minimum(vii, vjj)
    denom = np.maximum(vii * vjj, 1e-15)
    sin_sq = np.clip(1.0 - (vij ** 2) / denom, 0.0, 1.0)
    score_sq = np.maximum(minv, 0.0) * sin_sq / max(s - 2, 1)
    score = np.sqrt(np.maximum(score_sq, 0.0))
    score[~np.isfinite(score)] = np.inf
    np.fill_diagonal(score, 0.0)
    return score


def parallel_groups_from_scores(
    scores: np.ndarray,
    threshold: float,
    min_group_size: int,
    *,
    method: str = "complete",
    max_group_size: Optional[int] = None,
    clique_multiplier: float = 1.20,
) -> List[List[int]]:
    """Convert pairwise PVS scores into candidate replicate groups.

    ``method="graph"`` reproduces the transitive merge in Algorithm 1 of the
    approximate-replicate paper.  ``method="neighborhood"`` is the default
    screened finite-sample adaptation: it builds small mutually compatible
    local groups and avoids long near-parallel chains.  ``method="complete"``
    supplies a disjoint complete-linkage alternative.  At the population level,
    where exact parallelism is an equivalence relation, the methods agree.
    """
    D = np.asarray(scores, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("scores must be a square matrix")
    s = D.shape[0]
    min_group_size = max(2, int(min_group_size))
    method = str(method).lower()

    if method == "graph":
        uf = UnionFind(s)
        rows, cols = np.where(np.triu(D <= threshold, k=1))
        for i, j in zip(rows.tolist(), cols.tolist()):
            uf.union(int(i), int(j))
        groups: Dict[int, List[int]] = {}
        for i in range(s):
            groups.setdefault(uf.find(i), []).append(i)
        out = [sorted(g) for g in groups.values() if len(g) >= min_group_size]
        out.sort(key=lambda g: (g[0], len(g)))
        return out

    if method == "neighborhood":
        cap = s if max_group_size is None else max(min_group_size, int(max_group_size))
        candidate_sets: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        for seed in range(s):
            close = [j for j in range(s) if j != seed and float(D[seed, j]) <= threshold]
            close.sort(key=lambda j: (float(D[seed, j]), j))
            group = [seed]
            for j in close:
                if len(group) >= cap:
                    break
                if max(float(D[j, k]) for k in group) <= float(threshold) * float(clique_multiplier):
                    group.append(j)
            if len(group) < min_group_size:
                continue
            key = tuple(sorted(group))
            within = [float(D[i, j]) for a, i in enumerate(key) for j in key[a + 1 :]]
            candidate_sets[key] = (float(np.median(within)), float(np.max(within)))
        ordered = sorted(
            candidate_sets,
            key=lambda key: (candidate_sets[key][0], candidate_sets[key][1], -len(key), key),
        )
        return [list(key) for key in ordered]

    if method != "complete":
        raise ValueError("parallel_grouping must be 'neighborhood', 'complete', or 'graph'")

    # Complete-linkage uses the largest within-cluster pair score as the merge
    # criterion.  It is deterministic after adding an infinitesimal index-based
    # tie break, and is computationally modest because it is run only on s << p.
    finite = D[np.isfinite(D) & (~np.eye(s, dtype=bool))]
    replacement = max(float(np.max(finite)) if finite.size else 1.0, threshold * 10.0, 1.0)
    dist = np.where(np.isfinite(D), np.maximum(D, 0.0), replacement)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    if fcluster is not None and linkage is not None and squareform is not None and s >= 2:
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="complete", optimal_ordering=False)
        labels = fcluster(Z, t=float(threshold), criterion="distance")
        groups_map: Dict[int, List[int]] = {}
        for i, label in enumerate(labels.tolist()):
            groups_map.setdefault(int(label), []).append(i)
        out = [sorted(g) for g in groups_map.values() if len(g) >= min_group_size]
        out.sort(key=lambda g: (g[0], len(g)))
        return out

    # Deterministic fallback: greedily grow complete-linkage cliques from pairs
    # with the smallest score.  This branch is mainly for SciPy-free systems.
    unused = set(range(s))
    out: List[List[int]] = []
    pair_order = sorted(
        (float(dist[i, j]), i, j) for i in range(s) for j in range(i + 1, s)
    )
    for value, i, j in pair_order:
        if value > threshold:
            break
        if i not in unused or j not in unused:
            continue
        group = [i, j]
        candidates = sorted(unused - {i, j}, key=lambda x: max(float(dist[x, y]) for y in group))
        for x in candidates:
            if max(float(dist[x, y]) for y in group) <= threshold:
                group.append(x)
        if len(group) >= min_group_size:
            out.append(sorted(group))
            unused.difference_update(group)
    out.sort(key=lambda g: (g[0], len(g)))
    return out


def estimate_common_diagonal(correlation: np.ndarray, scores: np.ndarray, group: Sequence[int], i: int) -> float:
    partners = [j for j in group if j != i]
    if not partners:
        return 0.0
    j = min(partners, key=lambda x: float(scores[i, x]))
    keep = [k for k in range(correlation.shape[0]) if k not in {i, j}]
    ni = float(np.linalg.norm(correlation[i, keep]))
    nj = float(np.linalg.norm(correlation[j, keep]))
    if nj <= 1e-12:
        return 0.0
    return max(0.0, abs(float(correlation[i, j])) * ni / nj)


def group_representatives(correlation: np.ndarray, groups: Sequence[Sequence[int]]) -> List[int]:
    row_energy = np.sum(correlation ** 2, axis=1) - 1.0
    return [int(max(group, key=lambda i: float(row_energy[i]))) for group in groups]


def representative_common_matrix(
    correlation: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    representatives: Sequence[int],
) -> np.ndarray:
    G = len(groups)
    M = np.zeros((G, G), dtype=float)
    for a, i in enumerate(representatives):
        M[a, a] = estimate_common_diagonal(correlation, scores, groups[a], i)
        for b in range(a + 1, G):
            j = representatives[b]
            M[a, b] = M[b, a] = correlation[i, j]
    return psd_projection(M, floor=0.0)


def rank_k_psd(matrix: np.ndarray, k: int) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    k = min(max(int(k), 0), len(vals))
    if k == 0:
        return np.zeros_like(matrix)
    return (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T


def estimate_rank_split_prediction(
    M1: np.ndarray,
    M2: np.ndarray,
    *,
    penalty: float,
    n_subjects: int,
) -> Tuple[int, List[float]]:
    G = M1.shape[0]
    losses = []
    scale = max(float(np.linalg.norm(M2, ord="fro") ** 2), 1e-12)
    for k in range(1, G + 1):
        fit = rank_k_psd(M1, k)
        loss = float(np.linalg.norm(M2 - fit, ord="fro") ** 2 / scale)
        loss += float(penalty) * k * math.log(max(G, 2)) / max(n_subjects, 1)
        losses.append(loss)
    return int(np.argmin(losses) + 1), losses


def estimate_rank_threshold(M: np.ndarray, delta: float, multiplier: float) -> Tuple[int, List[float]]:
    vals = np.linalg.eigvalsh(0.5 * (M + M.T))[::-1]
    threshold = float(multiplier) * (math.sqrt(max(M.shape[0], 1)) * delta + M.shape[0] * delta * delta)
    rank = int(np.sum(vals >= threshold))
    return max(1, rank), vals.tolist()


def denoised_theta_on_parallel_set(
    correlation: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, List[int], Dict[int, float]]:
    H = sorted(i for group in groups for i in group)
    pos = {idx: k for k, idx in enumerate(H)}
    Theta = correlation[np.ix_(H, H)].copy()
    diag_map: Dict[int, float] = {}
    for group in groups:
        for i in group:
            diag_map[i] = estimate_common_diagonal(correlation, scores, group, i)
            Theta[pos[i], pos[i]] = diag_map[i]
    Theta = psd_projection(Theta, floor=0.0)
    return Theta, H, diag_map


def schur_complement_group_pruning(
    Theta: np.ndarray,
    H: Sequence[int],
    groups: Sequence[Sequence[int]],
    rank: int,
) -> Tuple[List[int], List[int], List[float]]:
    group_of = {idx: gid for gid, group in enumerate(groups) for idx in group}
    selected_indices: List[int] = []
    selected_groups: List[int] = []
    values: List[float] = []
    H_list = list(H)
    H_pos = {idx: pos for pos, idx in enumerate(H_list)}
    for _ in range(min(rank, len(groups))):
        best_idx: Optional[int] = None
        best_value = -float("inf")
        for idx in H_list:
            gid = group_of[idx]
            if gid in selected_groups:
                continue
            j = H_pos[idx]
            if not selected_indices:
                value = float(Theta[j, j])
            else:
                S = [H_pos[x] for x in selected_indices]
                block = Theta[np.ix_(S, S)]
                cross = Theta[j, S]
                value = float(Theta[j, j] - cross @ np.linalg.pinv(block, rcond=1e-8) @ cross.T)
            if value > best_value:
                best_value = value
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        selected_groups.append(group_of[best_idx])
        values.append(best_value)
    return selected_groups, selected_indices, values


def reconstruct_correlation_from_groups(
    correlation: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    s = correlation.shape[0]
    if not groups:
        return np.eye(s)
    Theta, H, _ = denoised_theta_on_parallel_set(correlation, scores, groups)
    Hc = [i for i in range(s) if i not in set(H)]
    out = np.zeros_like(correlation)
    out[np.ix_(H, H)] = Theta
    if Hc:
        cross = correlation[np.ix_(Hc, H)]
        out[np.ix_(Hc, H)] = cross
        out[np.ix_(H, Hc)] = cross.T
        out[np.ix_(Hc, Hc)] = cross @ np.linalg.pinv(Theta, rcond=1e-8) @ cross.T
    np.fill_diagonal(out, 1.0)
    return nearest_correlation(out)


def offdiag_frobenius(A: np.ndarray, B: np.ndarray) -> float:
    mask = ~np.eye(A.shape[0], dtype=bool)
    diff = A[mask] - B[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def group_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def match_group_collections(
    left: Sequence[Sequence[int]],
    right: Sequence[Sequence[int]],
) -> List[Tuple[int, int, float]]:
    if not left or not right:
        return []
    score = np.asarray([[group_jaccard(a, b) for b in right] for a in left], dtype=float)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-score)
        return [(int(i), int(j), float(score[i, j])) for i, j in zip(rows, cols)]
    out: List[Tuple[int, int, float]] = []
    used_l, used_r = set(), set()
    for value, i, j in sorted(
        ((float(score[i, j]), i, j) for i in range(score.shape[0]) for j in range(score.shape[1])),
        reverse=True,
    ):
        if i not in used_l and j not in used_r:
            out.append((int(i), int(j), value))
            used_l.add(i)
            used_r.add(j)
    return out


def filter_parallel_groups(
    correlation: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    cfg: StructureConfig,
    *,
    signal_multiplier: float = 1.0,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """Remove weak candidates and select a disjoint high-quality partition.

    Neighborhood grouping can return overlapping local cliques.  PVS requires a
    partition of the replicate set, so candidate cliques are ranked by common
    signal relative to their within-group score and accepted greedily subject to
    disjointness.  This step is inactive for exact population groups.
    """
    records: List[Dict[str, Any]] = []
    s = correlation.shape[0]
    viable: List[Tuple[float, int, List[int]]] = []
    for gid, raw_group in enumerate(groups):
        group = sorted(set(int(x) for x in raw_group))
        diag_values = [estimate_common_diagonal(correlation, scores, group, i) for i in group]
        median_diag = float(np.median(diag_values)) if diag_values else 0.0
        max_diag = float(np.max(diag_values)) if diag_values else 0.0
        fraction = len(group) / max(s, 1)
        within = [float(scores[i, j]) for a, i in enumerate(group) for j in group[a + 1 :]]
        median_within = float(np.median(within)) if within else float("inf")
        max_within = max(within) if within else float("inf")
        reasons: List[str] = []
        if len(group) < cfg.min_parallel_group_size:
            reasons.append("too_small")
        if fraction > cfg.max_group_fraction:
            reasons.append("too_large")
        if median_diag < cfg.min_group_common_diagonal * float(signal_multiplier):
            reasons.append("weak_common_signal")
        quality = median_diag * math.sqrt(max(len(group), 1)) / max(median_within, 1e-4)
        eligible = not reasons
        if eligible:
            viable.append((quality, gid, group))
        records.append({
            "input_group_id": int(gid),
            "group_size": len(group),
            "group_fraction": fraction,
            "median_common_diagonal": median_diag,
            "max_common_diagonal": max_diag,
            "median_within_score": median_within,
            "max_within_score": max_within,
            "quality": quality,
            "eligible": eligible,
            "selected_disjoint": False,
            "reasons": reasons,
        })

    # Prefer high-signal tight groups.  Ties favor larger groups and then a
    # deterministic lexicographic order.
    viable.sort(key=lambda x: (-x[0], -len(x[2]), tuple(x[2])))
    selected: List[List[int]] = []
    used: set[int] = set()
    for _, gid, group in viable:
        if any(i in used for i in group):
            records[gid]["reasons"] = list(records[gid]["reasons"]) + ["overlaps_better_group"]
            continue
        selected.append(group)
        used.update(group)
        records[gid]["selected_disjoint"] = True
    selected.sort(key=lambda g: (g[0], len(g)))
    return selected, records


def retain_split_stable_groups(
    full_groups: Sequence[Sequence[int]],
    first_groups: Sequence[Sequence[int]],
    second_groups: Sequence[Sequence[int]],
    min_jaccard: float,
) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    kept: List[List[int]] = []
    records: List[Dict[str, Any]] = []
    for gid, group in enumerate(full_groups):
        best1 = max((group_jaccard(group, other) for other in first_groups), default=0.0)
        best2 = max((group_jaccard(group, other) for other in second_groups), default=0.0)
        stable = min(best1, best2) >= float(min_jaccard)
        if stable:
            kept.append(list(group))
        records.append({
            "full_group_id": int(gid),
            "group_size": len(group),
            "best_jaccard_first": float(best1),
            "best_jaccard_second": float(best2),
            "stable": bool(stable),
        })
    return kept, records


def select_delta_split_stability(
    Rfull: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    base_delta: float,
    cfg: StructureConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Choose the tolerance yielding the largest reproducible full-sample family."""
    scores_full = pvs_s2_scores(Rfull)
    scores1 = pvs_s2_scores(R1)
    scores2 = pvs_s2_scores(R2)
    records: List[Dict[str, Any]] = []
    for multiplier in cfg.delta_multipliers:
        delta = float(base_delta * multiplier)
        full_threshold = 2.0 * delta
        split_threshold = full_threshold * cfg.split_threshold_multiplier
        raw_full = parallel_groups_from_scores(
            scores_full,
            full_threshold,
            cfg.min_parallel_group_size,
            method=cfg.parallel_grouping,
            max_group_size=cfg.max_parallel_group_size,
            clique_multiplier=cfg.neighborhood_clique_multiplier,
        )
        raw1 = parallel_groups_from_scores(
            scores1,
            split_threshold,
            cfg.min_parallel_group_size,
            method=cfg.parallel_grouping,
            max_group_size=cfg.max_parallel_group_size,
            clique_multiplier=cfg.neighborhood_clique_multiplier,
        )
        raw2 = parallel_groups_from_scores(
            scores2,
            split_threshold,
            cfg.min_parallel_group_size,
            method=cfg.parallel_grouping,
            max_group_size=cfg.max_parallel_group_size,
            clique_multiplier=cfg.neighborhood_clique_multiplier,
        )
        groups_full, _ = filter_parallel_groups(Rfull, scores_full, raw_full, cfg)
        groups1, _ = filter_parallel_groups(R1, scores1, raw1, cfg, signal_multiplier=0.65)
        groups2, _ = filter_parallel_groups(R2, scores2, raw2, cfg, signal_multiplier=0.65)
        stable_full, stable_records = retain_split_stable_groups(
            groups_full, groups1, groups2, cfg.split_group_jaccard
        )
        stable_jaccards = [
            min(float(x["best_jaccard_first"]), float(x["best_jaccard_second"]))
            for x in stable_records
            if bool(x["stable"])
        ]
        mean_jaccard = float(np.mean(stable_jaccards)) if stable_jaccards else 0.0
        stable_mass = float(sum(len(g) for g in stable_full))
        largest_fraction = max(
            [len(g) / max(Rfull.shape[0], 1) for g in groups_full], default=0.0
        )
        records.append({
            "multiplier": float(multiplier),
            "delta": delta,
            "full_threshold": full_threshold,
            "split_threshold": split_threshold,
            "n_groups_full": len(groups_full),
            "n_groups_first": len(groups1),
            "n_groups_second": len(groups2),
            "n_stable_groups": len(stable_full),
            "mean_stable_jaccard": mean_jaccard,
            "stable_mass": stable_mass,
            "largest_group_fraction": largest_fraction,
        })
    viable = [r for r in records if int(r["n_stable_groups"]) > 0]
    if not viable:
        return float(base_delta), records
    # Recover as many reproducible directions as possible; among ties prefer
    # agreement, a larger stable replicate mass, and a smaller tolerance.
    best = max(
        viable,
        key=lambda r: (
            int(r["n_stable_groups"]),
            float(r["mean_stable_jaccard"]),
            float(r["stable_mass"]),
            -float(r["largest_group_fraction"]),
            -float(r["delta"]),
        ),
    )
    return float(best["delta"]), records


def select_delta_split_reconstruction(
    R1: np.ndarray,
    R2: np.ndarray,
    base_delta: float,
    cfg: StructureConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    scores1 = pvs_s2_scores(R1)
    records: List[Dict[str, Any]] = []
    for multiplier in cfg.delta_multipliers:
        delta = float(base_delta * multiplier)
        groups = parallel_groups_from_scores(
            scores1, 2.0 * delta, cfg.min_parallel_group_size, method=cfg.parallel_grouping, max_group_size=cfg.max_parallel_group_size,
            clique_multiplier=cfg.neighborhood_clique_multiplier
        )
        if not groups:
            loss = float("inf")
        else:
            reconstructed = reconstruct_correlation_from_groups(R1, scores1, groups)
            loss = offdiag_frobenius(R2, reconstructed)
        records.append({
            "multiplier": float(multiplier),
            "delta": delta,
            "n_groups": len(groups),
            "n_parallel_features": int(sum(len(g) for g in groups)),
            "loss": loss,
        })
    finite = [r for r in records if math.isfinite(float(r["loss"]))]
    if not finite:
        return float(base_delta), records
    best = min(finite, key=lambda r: (float(r["loss"]), float(r["delta"])))
    return float(best["delta"]), records


def choose_core_and_halo(
    correlation: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    selected_group_ids: Sequence[int],
    candidate_global_indices: np.ndarray,
    cfg: StructureConfig,
    delta: float,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], Dict[str, Any]]:
    core_groups: List[List[int]] = []
    anchor_signs: List[List[int]] = []
    halo_groups: List[List[int]] = []
    diagnostics: Dict[str, Any] = {"group_details": []}

    for gid in selected_group_ids:
        group = list(groups[gid])
        common_diag = {i: estimate_common_diagonal(correlation, scores, group, i) for i in group}
        representative = max(group, key=lambda i: (common_diag[i], np.sum(correlation[i] ** 2)))
        ordered = sorted(group, key=lambda i: (common_diag[i], -scores[i, representative] if i != representative else 0.0), reverse=True)
        if representative in ordered:
            ordered.remove(representative)
        ordered.insert(0, representative)
        n_core = min(cfg.core_anchors_per_factor, len(ordered))
        if cfg.require_all_core_anchors and n_core < cfg.core_anchors_per_factor:
            raise RuntimeError(
                f"Selected PVS group {gid} has only {len(ordered)} features; "
                f"{cfg.core_anchors_per_factor} core anchors are required."
            )
        core_local = ordered[:n_core]
        signs = [1]
        for i in core_local[1:]:
            signs.append(1 if correlation[representative, i] >= 0 else -1)
        halo_local = [i for i in group if i not in set(core_local)]
        # Add near-parallel candidates as halo only; they are never hard-constrained.
        for i in range(correlation.shape[0]):
            if i in group:
                continue
            if scores[i, representative] <= cfg.halo_score_multiplier * 2.0 * delta:
                halo_local.append(i)
        halo_local = sorted(set(halo_local))

        core_groups.append([int(candidate_global_indices[i]) for i in core_local])
        anchor_signs.append([int(x) for x in signs])
        halo_groups.append([int(candidate_global_indices[i]) for i in halo_local])
        diagnostics["group_details"].append({
            "group_id": int(gid),
            "representative_local": int(representative),
            "representative_global": int(candidate_global_indices[representative]),
            "parallel_group_size": len(group),
            "core_local": core_local,
            "core_global": core_groups[-1],
            "core_signs": signs,
            "halo_global": halo_groups[-1],
            "common_diagonal": {str(int(candidate_global_indices[i])): float(common_diag[i]) for i in group},
        })
    return core_groups, anchor_signs, halo_groups, diagnostics


def discover_structure_once(
    residualized: ResidualizedData,
    candidate_indices: np.ndarray,
    *,
    screening_rank: int,
    cfg: StructureConfig,
    seed: int,
) -> ParallelStructure:
    all_subjects = np.unique(residualized.subject_ids)
    rng = np.random.default_rng(seed)
    shuffled = all_subjects.copy()
    rng.shuffle(shuffled)
    half = max(1, len(shuffled) // 2)
    first, second = shuffled[:half], shuffled[half:]
    if second.size == 0:
        second = first

    covariance, correlation, cov_info = subject_balanced_covariance(
        residualized, candidate_indices, psd_floor=cfg.covariance_psd_floor
    )
    cov1, R1, _ = subject_balanced_covariance(
        residualized, candidate_indices, subject_subset=first, psd_floor=cfg.covariance_psd_floor
    )
    cov2, R2, _ = subject_balanced_covariance(
        residualized, candidate_indices, subject_subset=second, psd_floor=cfg.covariance_psd_floor
    )

    initial_candidate_indices = np.asarray(candidate_indices, dtype=int)
    keep_signal, pure_noise_diag = select_pure_noise_prescreen(correlation, R1, R2, cfg)
    if int(np.sum(keep_signal)) < max(cfg.min_parallel_group_size, cfg.core_anchors_per_factor):
        raise RuntimeError("The PVS pure-noise pre-screen removed too many candidate features.")
    candidate_indices = initial_candidate_indices[keep_signal]
    covariance = covariance[np.ix_(keep_signal, keep_signal)]
    correlation = correlation[np.ix_(keep_signal, keep_signal)]
    cov1 = cov1[np.ix_(keep_signal, keep_signal)]
    cov2 = cov2[np.ix_(keep_signal, keep_signal)]
    R1 = R1[np.ix_(keep_signal, keep_signal)]
    R2 = R2[np.ix_(keep_signal, keep_signal)]

    base_delta = cfg.delta_base_constant * math.sqrt(
        math.log(max(len(candidate_indices), len(all_subjects), 2)) / max(len(all_subjects), 1)
    )
    if cfg.delta_selection == "split_stability":
        delta, delta_path = select_delta_split_stability(correlation, R1, R2, base_delta, cfg)
    elif cfg.delta_selection == "split_reconstruction":
        delta, delta_path = select_delta_split_reconstruction(R1, R2, base_delta, cfg)
    elif cfg.delta_selection == "theory":
        delta = base_delta
        delta_path = [{"delta": delta, "multiplier": 1.0, "loss": None}]
    else:
        raise ValueError("delta_selection must be split_stability, split_reconstruction, or theory")

    scores = pvs_s2_scores(correlation)
    raw_groups = parallel_groups_from_scores(
        scores, 2.0 * delta, cfg.min_parallel_group_size, method=cfg.parallel_grouping, max_group_size=cfg.max_parallel_group_size,
        clique_multiplier=cfg.neighborhood_clique_multiplier
    )
    groups, group_filter_diag = filter_parallel_groups(correlation, scores, raw_groups, cfg)

    # A group used to define a dynamic coordinate must be reproducible across
    # independent subject halves.  This is stricter than static PVS and is
    # needed because finite-sample near-parallel chains can otherwise create an
    # apparently interpretable but unstable drift coordinate.
    scores1 = pvs_s2_scores(R1)
    scores2 = pvs_s2_scores(R2)
    split_threshold = 2.0 * delta * cfg.split_threshold_multiplier
    raw_groups1 = parallel_groups_from_scores(
        scores1, split_threshold, cfg.min_parallel_group_size, method=cfg.parallel_grouping, max_group_size=cfg.max_parallel_group_size,
        clique_multiplier=cfg.neighborhood_clique_multiplier
    )
    raw_groups2 = parallel_groups_from_scores(
        scores2, split_threshold, cfg.min_parallel_group_size, method=cfg.parallel_grouping, max_group_size=cfg.max_parallel_group_size,
        clique_multiplier=cfg.neighborhood_clique_multiplier
    )
    groups1, group_filter1 = filter_parallel_groups(R1, scores1, raw_groups1, cfg, signal_multiplier=0.65)
    groups2, group_filter2 = filter_parallel_groups(R2, scores2, raw_groups2, cfg, signal_multiplier=0.65)
    groups, split_stability_diag = retain_split_stable_groups(
        groups, groups1, groups2, cfg.split_group_jaccard
    )
    if not groups:
        raise RuntimeError(
            "PVS found no signal-bearing replicate groups that were stable across subject halves."
        )
    reps = group_representatives(correlation, groups)
    M = representative_common_matrix(correlation, scores, groups, reps)

    # Rank prediction uses the same discovered groups/reps in two independent subject halves.
    M1 = representative_common_matrix(R1, scores1, groups, reps)
    M2 = representative_common_matrix(R2, scores2, groups, reps)
    if cfg.rank_method == "split_prediction":
        rank12, losses12 = estimate_rank_split_prediction(M1, M2, penalty=cfg.rank_penalty, n_subjects=len(all_subjects))
        rank21, losses21 = estimate_rank_split_prediction(M2, M1, penalty=cfg.rank_penalty, n_subjects=len(all_subjects))
        rank_hat = min(len(groups), max(1, int(round((rank12 + rank21) / 2))))
        rank_diag = {"rank12": rank12, "rank21": rank21, "losses12": losses12, "losses21": losses21}
    elif cfg.rank_method == "threshold":
        rank_hat, eigvals = estimate_rank_threshold(M, delta, cfg.rank_threshold_multiplier)
        rank_diag = {"eigenvalues": eigvals}
    else:
        raise ValueError("rank_method must be split_prediction or threshold")

    if len(groups) > rank_hat:
        Theta, H, diag_map = denoised_theta_on_parallel_set(correlation, scores, groups)
        selected_group_ids, selected_indices, schur_values = schur_complement_group_pruning(Theta, H, groups, rank_hat)
        pruning_diag = {
            "used": True,
            "selected_indices_local": selected_indices,
            "schur_values": schur_values,
            "common_diagonal": {str(k): float(v) for k, v in diag_map.items()},
        }
    else:
        selected_group_ids = list(range(len(groups)))
        pruning_diag = {"used": False}

    core_groups, anchor_signs, halo_groups, core_diag = choose_core_and_halo(
        correlation, scores, groups, selected_group_ids, candidate_indices, cfg, delta
    )
    if len(core_groups) != rank_hat:
        raise RuntimeError("The number of selected core groups does not equal the estimated rank.")

    return ParallelStructure(
        delta=delta,
        candidate_indices=np.asarray(candidate_indices, dtype=int),
        correlation=correlation,
        covariance=covariance,
        scores=scores,
        parallel_groups_local=[list(map(int, g)) for g in groups],
        parallel_groups_global=[[int(candidate_indices[i]) for i in g] for g in groups],
        representatives_local=[int(x) for x in reps],
        representative_matrix=M,
        estimated_rank=int(rank_hat),
        selected_group_ids=[int(x) for x in selected_group_ids],
        core_groups=core_groups,
        anchor_signs=anchor_signs,
        halo_groups=halo_groups,
        screening_rank=int(screening_rank),
        screening_indices=initial_candidate_indices,
        diagnostics={
            "covariance": cov_info,
            "pure_noise_prescreen": pure_noise_diag,
            "initial_screen_size": int(len(initial_candidate_indices)),
            "post_noise_screen_size": int(len(candidate_indices)),
            "base_delta": base_delta,
            "delta_path": delta_path,
            "parallel_grouping": cfg.parallel_grouping,
            "raw_group_count": len(raw_groups),
            "group_filter": group_filter_diag,
            "split_group_filter_first": group_filter1,
            "split_group_filter_second": group_filter2,
            "split_group_stability": split_stability_diag,
            "rank": rank_diag,
            "pruning": pruning_diag,
            "core": core_diag,
        },
    )


def match_groups_jaccard(reference: Sequence[Sequence[int]], candidate: Sequence[Sequence[int]]) -> List[Tuple[int, int, float]]:
    if not reference or not candidate:
        return []
    score = np.zeros((len(candidate), len(reference)), dtype=float)
    for i, cg in enumerate(candidate):
        c = set(cg)
        for j, rg in enumerate(reference):
            r = set(rg)
            score[i, j] = len(c & r) / max(len(c | r), 1)
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-score)
        return [(int(c), int(r), float(score[c, r])) for c, r in zip(rows, cols)]
    pairs: List[Tuple[int, int, float]] = []
    used_c, used_r = set(), set()
    for val, c, r in sorted(((score[c, r], c, r) for c in range(score.shape[0]) for r in range(score.shape[1])), reverse=True):
        if c not in used_c and r not in used_r:
            pairs.append((int(c), int(r), float(val)))
            used_c.add(c)
            used_r.add(r)
    return pairs


def bootstrap_structure_stability(
    residualized: ResidualizedData,
    reference: ParallelStructure,
    *,
    cfg: StructureConfig,
    seed: int,
) -> Tuple[np.ndarray, int, List[Dict[str, Any]]]:
    B = max(0, int(cfg.bootstrap_replicates))
    p = residualized.residuals.shape[1]
    K = reference.estimated_rank
    stability = np.zeros((p, K), dtype=float)
    if B == 0:
        return stability, 0, []
    rng = np.random.default_rng(seed)
    subjects = np.unique(residualized.subject_ids)
    records: List[Dict[str, Any]] = []
    successful = 0

    for b in range(B):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        row_blocks = []
        new_subject_ids = []
        new_weights = []
        for new_id, original in enumerate(sampled):
            rows = np.flatnonzero(residualized.subject_ids == original)
            row_blocks.extend(rows.tolist())
            new_subject_ids.extend([new_id] * len(rows))
            new_weights.extend([1.0 / max(len(rows), 1)] * len(rows))
        rows_arr = np.asarray(row_blocks, dtype=int)
        boot = ResidualizedData(
            residuals=residualized.residuals[rows_arr],
            observed=residualized.observed[rows_arr],
            design=residualized.design[rows_arr],
            subject_ids=np.asarray(new_subject_ids, dtype=int),
            row_weights=np.asarray(new_weights, dtype=float),
            row_subject_position=[residualized.row_subject_position[r] for r in rows_arr],
            feature_coverage=residualized.feature_coverage,
            feature_scale=residualized.feature_scale,
            metadata={**residualized.metadata, "n_subjects": len(subjects)},
        )
        try:
            if cfg.bootstrap_full_screen:
                raise NotImplementedError("Full-screen bootstrap is intentionally disabled in the default implementation.")
            candidate_indices = reference.candidate_indices
            boot_cfg = dataclasses.replace(cfg, bootstrap_replicates=0, delta_selection="theory")
            structure = discover_structure_once(
                boot,
                candidate_indices,
                screening_rank=reference.screening_rank,
                cfg=boot_cfg,
                seed=seed + 1009 * (b + 1),
            )
            matches = match_groups_jaccard(reference.core_groups, structure.core_groups)
            matched = 0
            for cand_idx, ref_idx, jac in matches:
                if jac <= 0:
                    continue
                matched += 1
                for feature in structure.core_groups[cand_idx]:
                    stability[feature, ref_idx] += 1.0
            successful += 1
            records.append({"bootstrap": b, "status": "success", "rank": structure.estimated_rank, "matched_groups": matched})
        except Exception as exc:
            records.append({"bootstrap": b, "status": "error", "error": str(exc)})
    if successful:
        stability /= float(successful)
    return stability, successful, records


def discover_screened_pvs_structure(
    core: Any,
    subjects: Sequence[Mapping[str, torch.Tensor]],
    *,
    cfg: StructureConfig,
    seed: int,
    include_level: bool = False,
) -> Tuple[ParallelStructure, ResidualizedData]:
    residualized = crossfit_residualize(
        subjects,
        n_folds=cfg.crossfit_folds,
        ridge=cfg.mean_ridge,
        chunk_size=cfg.feature_chunk_size,
        seed=seed,
        include_level=include_level,
    )
    embedding, strength, screen_rank, spectral_diag = truncated_embedding(core, residualized, cfg)
    candidates, screen_diag = diversified_directional_screen(
        embedding, strength, residualized.feature_coverage, cfg
    )
    structure = discover_structure_once(
        residualized, candidates, screening_rank=screen_rank, cfg=cfg, seed=seed + 17
    )
    stability, successful, bootstrap_records = bootstrap_structure_stability(
        residualized, structure, cfg=cfg, seed=cfg.bootstrap_seed + seed
    )
    structure.bootstrap_stability = stability
    structure.bootstrap_successful = successful
    selected_stabilities = [
        float(stability[idx, r]) if successful else float("nan")
        for r, group in enumerate(structure.core_groups)
        for idx in group
    ]
    structure.diagnostics.update({
        "spectral_screen": spectral_diag,
        "directional_screen": screen_diag,
        "bootstrap_records": bootstrap_records,
        "mean_selected_anchor_stability": float(np.nanmean(selected_stabilities)) if successful else float("nan"),
        "selected_anchor_stability": selected_stabilities,
    })
    return structure, residualized


# -----------------------------------------------------------------------------
# Signed-anchor ALOHA subclass and fitting
# -----------------------------------------------------------------------------


def signed_anchor_clouds_class(core: Any):
    class SignedAnchorCLOUDS(core.CLOUDS):
        """CLOUDS extension allowing known relative signs within each anchor group."""

        def __init__(self, *args: Any, anchor_signs: Optional[Sequence[Sequence[int]]] = None, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if anchor_signs is None:
                anchor_signs = [[1] * len(group) for group in self.anchor_groups]
            if len(anchor_signs) != self.K:
                raise ValueError("anchor_signs must contain one list per latent factor")
            flat: List[int] = []
            normalized: List[List[int]] = []
            for group, signs in zip(self.anchor_groups, anchor_signs):
                if len(group) != len(signs):
                    raise ValueError("Each anchor sign list must match its anchor group")
                s = [1 if int(x) >= 0 else -1 for x in signs]
                if s and s[0] < 0:
                    s = [-x for x in s]
                normalized.append(s)
                flat.extend(s)
            self.anchor_signs = normalized
            self.register_buffer("anchor_sign_flat", torch.tensor(flat, dtype=self.Lambda_raw.dtype))

        @property
        def Lambda(self) -> torch.Tensor:
            result = self.Lambda_raw * self.struct_mask
            anchor_raw = self.Lambda_raw[self.anchor_idx, self.anchor_factor_idx]
            magnitudes = F.softplus(anchor_raw) + self.min_anchor_loading
            signed_values = magnitudes * self.anchor_sign_flat.to(magnitudes)
            return result.index_put((self.anchor_idx, self.anchor_factor_idx), signed_values)

        @torch.no_grad()
        def _assign_lambda_from_matrix(self, Lambda_target: torch.Tensor) -> None:
            target = Lambda_target.to(self.Lambda_raw).clone()
            for r, (group, signs) in enumerate(zip(self.anchor_groups, self.anchor_signs)):
                representative = int(group[0])
                if float(target[representative, r].item()) * float(signs[0]) < 0.0:
                    target[:, r].mul_(-1.0)
                idx = torch.tensor(group, dtype=torch.long, device=target.device)
                sign_t = torch.tensor(signs, dtype=target.dtype, device=target.device)
                target[idx, r] = sign_t * torch.abs(target[idx, r]).clamp_min(self.min_anchor_loading * 2.0)
            self.Lambda_raw.copy_(target)
            self.Lambda_raw[self.anchor_idx, :] = 0.0
            values = torch.abs(target[self.anchor_idx, self.anchor_factor_idx])
            transformed = (values - self.min_anchor_loading).clamp_min(torch.finfo(target.dtype).eps)
            self.Lambda_raw[self.anchor_idx, self.anchor_factor_idx] = core.inverse_softplus(transformed)

    SignedAnchorCLOUDS.__name__ = "SignedAnchorCLOUDS"
    return SignedAnchorCLOUDS


def fit_signed_aloha_candidate(
    core: Any,
    train_subjects: Sequence[Mapping[str, torch.Tensor]],
    structure: ParallelStructure,
    *,
    covar_dim: int,
    mode: str,
    fit_profile: str,
    include_level: bool,
    inverse_ns_threshold: int = 256,
) -> Any:
    K = structure.estimated_rank
    D = int(train_subjects[0]["x"].shape[1])
    Signed = signed_anchor_clouds_class(core)
    scenario = {
        "fit_profile": fit_profile,
        "inverse_ns_threshold": inverse_ns_threshold,
        "include_latent_level": include_level,
        "learn_observation_intercept": True,
    }
    recipe = core.adaptive_fit_recipe(K, scenario)
    kwargs = dict(
        obs_dim=D,
        latent_dim=K,
        covar_dim=covar_dim,
        anchor_groups=structure.core_groups,
        anchor_signs=structure.anchor_signs,
        inverse_ns_threshold=inverse_ns_threshold,
        omega_correlation=True,
        diagonal_fix_omega=True,
        include_latent_level=include_level,
        learn_observation_intercept=True,
        **core.regularization_defaults_for_k(K, scenario),
    )
    if mode == "exact":
        diagonal = Signed(theta_mode="diagonal", **kwargs)
        diagonal.fit_em_multistart(
            train_subjects,
            num_em_epochs=recipe.diag_pre_epochs,
            warmup_epochs=recipe.diag_pre_warmup,
            m_step_iters=recipe.diag_pre_mstep,
            lr=recipe.diag_lr,
            n_starts=recipe.diag_pre_starts,
            burn_in_epochs=recipe.diag_pre_burn,
        )
        model = Signed(theta_mode="exact", **kwargs)
        model.initialize_exact_from_diagonal_model(diagonal)
        model.fit_exact_continuation(
            train_subjects,
            num_em_epochs=recipe.exact_epochs,
            m_step_iters=recipe.exact_mstep,
            lr=recipe.exact_lr,
            lbfgs_max_iter=recipe.exact_lbfgs,
        )
        return model
    if mode != "diagonal":
        raise ValueError("mode must be exact or diagonal")
    model = Signed(theta_mode="diagonal", **kwargs)
    model.fit_em_multistart(
        train_subjects,
        num_em_epochs=recipe.diag_epochs,
        warmup_epochs=recipe.diag_warmup,
        m_step_iters=recipe.diag_mstep,
        lr=recipe.diag_lr,
        n_starts=recipe.diag_starts,
        burn_in_epochs=recipe.diag_burn,
    )
    return model


# -----------------------------------------------------------------------------
# Evaluation metrics
# -----------------------------------------------------------------------------


def best_group_matching(true_groups: Sequence[Sequence[int]], estimated_groups: Sequence[Sequence[int]]) -> List[Tuple[int, int, float]]:
    return match_groups_jaccard(true_groups, estimated_groups)


def partition_pairwise_f1(true_groups: Sequence[Sequence[int]], est_groups: Sequence[Sequence[int]]) -> float:
    universe = sorted(set(i for g in true_groups for i in g) | set(i for g in est_groups for i in g))
    if len(universe) < 2:
        return float("nan")
    true_pair = set()
    est_pair = set()
    for g in true_groups:
        gs = sorted(set(g))
        true_pair.update((gs[a], gs[b]) for a in range(len(gs)) for b in range(a + 1, len(gs)))
    for g in est_groups:
        gs = sorted(set(g))
        est_pair.update((gs[a], gs[b]) for a in range(len(gs)) for b in range(a + 1, len(gs)))
    tp = len(true_pair & est_pair)
    fp = len(est_pair - true_pair)
    fn = len(true_pair - est_pair)
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def set_precision_recall_f1(truth: Iterable[int], estimate: Iterable[int]) -> Tuple[float, float, float]:
    t, e = set(truth), set(estimate)
    tp = len(t & e)
    precision = tp / max(len(e), 1)
    recall = tp / max(len(t), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def structure_metrics(structure: ParallelStructure, truth: Mapping[str, Any]) -> Dict[str, float]:
    true_core = truth["core_groups"]
    est_core = structure.core_groups
    true_set = set(i for g in true_core for i in g)
    est_set = set(i for g in est_core for i in g)
    precision, recall, f1 = set_precision_recall_f1(true_set, est_set)
    screen_recall = len(true_set & set(structure.candidate_indices.tolist())) / max(len(true_set), 1)
    dense_set = set(truth["all_dense_replicate_indices"])
    dense_false_core = len(est_set & dense_set) / max(len(est_set), 1)
    matches = best_group_matching(true_core, est_core)
    coverage = sum(1 for _, _, j in matches if j > 0) / max(len(true_core), 1)
    mean_jaccard = float(np.mean([j for _, _, j in matches])) if matches else 0.0
    return {
        "rank_hat": float(structure.estimated_rank),
        "rank_true": float(len(true_core)),
        "rank_correct": float(structure.estimated_rank == len(true_core)),
        "screen_core_recall": float(screen_recall),
        "core_precision": float(precision),
        "core_recall": float(recall),
        "core_f1": float(f1),
        "partition_pairwise_f1": float(partition_pairwise_f1(true_core, est_core)),
        "factor_coverage": float(coverage),
        "mean_group_jaccard": float(mean_jaccard),
        "dense_replicate_false_core_rate": float(dense_false_core),
        "mean_anchor_stability": float(structure.diagnostics.get("mean_selected_anchor_stability", float("nan"))),
        "screen_size": float(len(structure.candidate_indices)),
        "n_parallel_groups": float(len(structure.parallel_groups_global)),
        "pvs_pruning_used": float(bool(structure.diagnostics.get("pruning", {}).get("used", False))),
    }


def align_estimates_by_anchor_groups(
    estimated_groups: Sequence[Sequence[int]],
    true_groups: Sequence[Sequence[int]],
    estimated_signs: Sequence[Sequence[int]],
    truth_anchor_signs: Sequence[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    K_est, K_true = len(estimated_groups), len(true_groups)
    if K_est != K_true:
        raise ValueError("Alignment requires equal estimated and true dimensions")
    matches = match_groups_jaccard(true_groups, estimated_groups)
    # match_groups returns candidate-index, reference-index; candidate=estimated, reference=true.
    perm = np.full(K_true, -1, dtype=int)
    signs = np.ones(K_true, dtype=float)
    for est_idx, true_idx, _ in matches:
        perm[true_idx] = est_idx
        est_rep = estimated_groups[est_idx][0]
        true_group = list(true_groups[true_idx])
        if est_rep in true_group:
            tpos = true_group.index(est_rep)
            true_sign = truth_anchor_signs[true_idx][tpos]
            est_sign = estimated_signs[est_idx][0]
            signs[true_idx] = float(true_sign * est_sign)
    if np.any(perm < 0):
        raise RuntimeError("Could not align all estimated factors to truth")
    return perm, signs


def dynamic_metrics(model: Any, structure: ParallelStructure, truth: Mapping[str, Any], core: Any) -> Dict[str, float]:
    if structure.estimated_rank != int(truth["Lambda"].shape[1]):
        return {
            "lambda_rel_rmse": float("nan"),
            "gamma_rel_rmse": float("nan"),
            "gamma_offdiag_sign_accuracy": float("nan"),
            "transition_rel_rmse": float("nan"),
        }
    ident = model.get_identifiable_parameters()
    perm, signs = align_estimates_by_anchor_groups(
        structure.core_groups,
        truth["core_groups"],
        structure.anchor_signs,
        truth["anchor_signs"],
    )
    P = np.zeros((len(perm), len(perm)), dtype=float)
    for true_idx, est_idx in enumerate(perm):
        P[est_idx, true_idx] = signs[true_idx]
    Pt = torch.tensor(P, dtype=ident["Lambda"].dtype)
    Lambda_est = ident["Lambda"].detach().cpu() @ Pt
    Gamma_est = Pt.T @ ident["Gamma"].detach().cpu() @ Pt
    Lambda_true = truth["Lambda"].detach().cpu()
    Gamma_true = truth["Gamma"].detach().cpu()
    lambda_rel = float(torch.linalg.norm(Lambda_est - Lambda_true) / torch.linalg.norm(Lambda_true).clamp_min(1e-12))
    gamma_rel = float(torch.linalg.norm(Gamma_est - Gamma_true) / torch.linalg.norm(Gamma_true).clamp_min(1e-12))
    mask = ~torch.eye(Gamma_true.shape[0], dtype=torch.bool)
    true_off = Gamma_true[mask]
    est_off = Gamma_est[mask]
    informative = torch.abs(true_off) > 1e-6
    sign_acc = float(torch.mean((torch.sign(true_off[informative]) == torch.sign(est_off[informative])).double())) if torch.any(informative) else float("nan")
    dt = torch.linspace(0.1, 0.5, 5, dtype=Gamma_true.dtype)
    _, transition_rmse = core.transition_matrix_summary(Gamma_true, Gamma_est, dt)
    return {
        "lambda_rel_rmse": lambda_rel,
        "gamma_rel_rmse": gamma_rel,
        "gamma_offdiag_sign_accuracy": sign_acc,
        "transition_rel_rmse": float(transition_rmse),
    }


# -----------------------------------------------------------------------------
# Simulation scenarios and experiment runners
# -----------------------------------------------------------------------------


def scenario_config(name: str, seed: int, *, n_subjects: Optional[int] = None, obs_dim: Optional[int] = None) -> SimulationConfig:
    base = SimulationConfig(name=name, seed=seed)
    if name == "baseline":
        cfg = base
    elif name == "strong_correlation":
        cfg = dataclasses.replace(base, omega_rho=0.75)
    elif name == "heteroskedastic":
        cfg = dataclasses.replace(base, noise_log_sd=0.90)
    elif name == "dense_replicates":
        cfg = dataclasses.replace(base, dense_replicate_groups=6, dense_replicates_per_group=4)
    elif name == "weak_anchors":
        cfg = dataclasses.replace(base, noise_scale=1.20, anchor_log_scale_sd=0.20, dense_replicate_scale=0.50)
    elif name == "quasi_pure":
        cfg = dataclasses.replace(base, halo_cross_loading=0.25, halos_per_factor=4)
    elif name == "signed_anchors":
        cfg = dataclasses.replace(base, anchors_per_factor=5, positive_anchors_per_factor=3)
    elif name == "missingness":
        cfg = dataclasses.replace(base, item_missing_rate=0.25, visit_missing_rate=0.15)
    elif name == "residual_correlation":
        cfg = dataclasses.replace(base, residual_block_rho=0.25)
    elif name == "large_p":
        cfg = dataclasses.replace(base, obs_dim=30000, n_subjects=120, item_missing_rate=0.10)
    elif name == "dynamic_burden":
        cfg = dataclasses.replace(base, latent_dim=8, obs_dim=4000, n_subjects=80, visit_min=3, visit_max=5)
    else:
        raise ValueError(f"Unknown scenario: {name}")
    if n_subjects is not None:
        cfg = dataclasses.replace(cfg, n_subjects=int(n_subjects))
    if obs_dim is not None:
        cfg = dataclasses.replace(cfg, obs_dim=int(obs_dim))
    return cfg


def run_current_varimax_oracle_k(core: Any, subjects: Sequence[Mapping[str, torch.Tensor]], true_k: int, anchors_per_factor: int, bootstrap: int, seed: int) -> Dict[str, Any]:
    started = time.time()
    try:
        groups, diagnostics = core.discover_anchor_groups(
            subjects,
            true_k,
            anchors_per_factor=anchors_per_factor,
            n_bootstrap=bootstrap,
            seed=seed,
            min_purity_ratio=2.0,
            allow_relaxed=False,
        )
        return {
            "status": "success",
            "groups": groups,
            "mean_anchor_stability": diagnostics.get("mean_chosen_stability"),
            "mean_anchor_purity": diagnostics.get("mean_anchor_purity"),
            "runtime": time.time() - started,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "runtime": time.time() - started}


def run_structure_replication(
    core: Any,
    sim_cfg: SimulationConfig,
    struct_cfg: StructureConfig,
    *,
    compare_varimax: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    started = time.time()
    subjects, truth = simulate_structured_aloha_cohort(core, sim_cfg)
    result: Dict[str, Any] = {
        "scenario": sim_cfg.name,
        "seed": sim_cfg.seed,
        "n_subjects": sim_cfg.n_subjects,
        "obs_dim": sim_cfg.obs_dim,
        "latent_dim": sim_cfg.latent_dim,
        "method": "screened_pvs",
    }
    detail: Dict[str, Any] = {"simulation_config": sim_cfg, "structure_config": struct_cfg}
    try:
        structure, _ = discover_screened_pvs_structure(
            core,
            subjects,
            cfg=struct_cfg,
            seed=sim_cfg.seed + 101,
            include_level=sim_cfg.include_latent_level,
        )
        result.update({"status": "success", **structure_metrics(structure, truth)})
        result["runtime"] = time.time() - started
        detail.update({"truth_structure": {k: truth[k] for k in ["core_groups", "anchor_signs", "halo_groups", "dense_replicate_groups", "pure_noise_indices"]}, "estimated_structure": structure})
    except Exception as exc:
        result.update({"status": "error", "error": str(exc), "runtime": time.time() - started})
        detail["traceback"] = traceback.format_exc()

    if compare_varimax:
        varimax = run_current_varimax_oracle_k(
            core,
            subjects,
            sim_cfg.latent_dim,
            anchors_per_factor=min(struct_cfg.core_anchors_per_factor, sim_cfg.positive_anchors_per_factor),
            bootstrap=min(10, struct_cfg.bootstrap_replicates),
            seed=sim_cfg.seed + 303,
        )
        if varimax["status"] == "success":
            p, r, f = set_precision_recall_f1(
                [i for g in truth["core_groups"] for i in g if truth["anchor_signs"][truth["core_groups"].index(g)][g.index(i)] > 0],
                [i for g in varimax["groups"] for i in g],
            )
            varimax.update({
                "core_precision": p,
                "core_recall": r,
                "core_f1": f,
                "partition_pairwise_f1": partition_pairwise_f1(truth["core_groups"], varimax["groups"]),
            })
        detail["current_varimax_oracle_k"] = varimax
    return result, detail


def run_end_to_end_replication(
    core: Any,
    sim_cfg: SimulationConfig,
    struct_cfg: StructureConfig,
    *,
    fit_profile: str,
    exact_max_k: int,
    min_trans_per_gamma_param: float,
    validation_fraction: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    subjects, truth = simulate_structured_aloha_cohort(core, sim_cfg)
    train, holdouts = core.make_visit_holdout_split(
        subjects, validation_fraction=validation_fraction, seed=sim_cfg.seed + 41, holdout="last"
    )
    started = time.time()
    result: Dict[str, Any] = {
        "scenario": sim_cfg.name,
        "seed": sim_cfg.seed,
        "n_subjects": sim_cfg.n_subjects,
        "obs_dim": sim_cfg.obs_dim,
        "latent_dim": sim_cfg.latent_dim,
        "method": "screened_pvs_aloha",
    }
    detail: Dict[str, Any] = {"simulation_config": sim_cfg, "structure_config": struct_cfg}
    try:
        structure, _ = discover_screened_pvs_structure(
            core,
            train,
            cfg=struct_cfg,
            seed=sim_cfg.seed + 101,
            include_level=sim_cfg.include_latent_level,
        )
        result.update(structure_metrics(structure, truth))
        n_trans = core.count_transitions(train)
        full_supported = (
            structure.estimated_rank <= exact_max_k
            and n_trans >= math.ceil(min_trans_per_gamma_param * structure.estimated_rank ** 2)
        )
        result["n_transitions"] = n_trans
        result["full_drift_supported"] = float(full_supported)
        mode = "exact" if full_supported else "diagonal"
        model = fit_signed_aloha_candidate(
            core,
            train,
            structure,
            covar_dim=sim_cfg.covar_dim,
            mode=mode,
            fit_profile=fit_profile,
            include_level=sim_cfg.include_latent_level,
        )
        _, cells, predictive = core.heldout_visit_predictive_loglik(model, train, holdouts)
        result.update(dynamic_metrics(model, structure, truth, core))
        result.update({
            "status": "success" if full_supported else "full_drift_unsupported_diagonal_diagnostic",
            "fitted_mode": mode,
            "heldout_cells": cells,
            "heldout_loglik_per_cell": predictive,
            "runtime": time.time() - started,
        })
        detail.update({"truth_structure": {k: truth[k] for k in ["core_groups", "anchor_signs", "halo_groups", "dense_replicate_groups"]}, "estimated_structure": structure, "fit_history": model.fit_history})
    except Exception as exc:
        result.update({"status": "error", "error": str(exc), "runtime": time.time() - started})
        detail["traceback"] = traceback.format_exc()
    return result, detail


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("scenario")), str(row.get("method"))), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for (scenario, method), block in sorted(groups.items()):
        ok = [r for r in block if str(r.get("status", "")).startswith("success") or "diagnostic" in str(r.get("status", ""))]
        summary: Dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "n_total": len(block),
            "n_success": len(ok),
        }
        numeric_keys = sorted({k for r in ok for k, v in r.items() if isinstance(v, (int, float, np.number)) and k not in {"seed"}})
        for key in numeric_keys:
            vals = np.asarray([float(r[key]) for r in ok if r.get(key) is not None and math.isfinite(float(r[key]))], dtype=float)
            if vals.size:
                summary[f"{key}_mean"] = float(np.mean(vals))
                summary[f"{key}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        summaries.append(summary)
    return summaries


def run_experiment(args: argparse.Namespace) -> int:
    core = load_aloha_core(args.aloha_core)
    core.configure_torch_runtime(args.threads, args.dtype)
    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "smoke":
        # Fixed seed and a moderate structure-only problem make the smoke test
        # deterministic while exercising screening, pure-noise removal, signed
        # anchors, generic replicate groups, rank recovery, and PVS pruning.
        sim_cfg = dataclasses.replace(
            scenario_config("dense_replicates", 1234, n_subjects=120, obs_dim=500),
            latent_dim=4,
            covar_dim=2,
            visit_min=5,
            visit_max=7,
            anchors_per_factor=4,
            positive_anchors_per_factor=3,
            halos_per_factor=2,
            dense_replicate_groups=3,
            dense_replicates_per_group=3,
            item_missing_rate=0.05,
            visit_missing_rate=0.02,
        )
        struct_cfg = StructureConfig(
            max_screen_rank=12,
            screen_size=240,
            direction_cells=24,
            top_per_cell=8,
            global_top_features=40,
            core_anchors_per_factor=3,
            bootstrap_replicates=0,
        )
        result, detail = run_structure_replication(core, sim_cfg, struct_cfg, compare_varimax=True)
        write_json(out_dir / "smoke_detail.json", detail)
        write_csv(out_dir / "smoke_result.csv", [result])
        print(json.dumps(json_safe(result), indent=2))
        if result.get("status") != "success":
            return 2
        if float(result.get("screen_core_recall", 0.0)) < 0.9:
            raise AssertionError("Smoke test did not retain the true core anchors in the structural screen.")
        if float(result.get("rank_correct", 0.0)) < 1.0:
            raise AssertionError("Smoke test did not recover the structural latent dimension.")
        if float(result.get("factor_coverage", 0.0)) < 1.0:
            raise AssertionError("Smoke test did not recover at least one core group for every factor.")
        if float(result.get("pvs_pruning_used", 0.0)) < 1.0:
            raise AssertionError("Smoke test did not exercise the generic-replicate PVS pruning path.")
        return 0

    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    detail_dir = out_dir / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name in scenarios:
        for rep in range(args.replicates):
            seed = args.seed + 1009 * rep + 100_003 * scenarios.index(scenario_name)
            sim_cfg = scenario_config(
                scenario_name,
                seed,
                n_subjects=args.n_subjects,
                obs_dim=args.obs_dim,
            )
            struct_cfg = StructureConfig(
                crossfit_folds=args.crossfit_folds,
                max_screen_rank=args.max_screen_rank,
                screen_size=args.screen_size,
                direction_cells=args.direction_cells,
                top_per_cell=args.top_per_cell,
                global_top_features=args.global_top_features,
                core_anchors_per_factor=args.core_anchors,
                bootstrap_replicates=args.bootstrap,
                delta_base_constant=args.delta_base_constant,
                delta_selection=args.delta_selection,
                parallel_grouping=args.parallel_grouping,
                max_parallel_group_size=args.max_parallel_group_size,
                max_group_fraction=args.max_group_fraction,
                min_group_common_diagonal=args.min_group_common_diagonal,
                split_group_jaccard=args.split_group_jaccard,
                split_threshold_multiplier=args.split_threshold_multiplier,
                pure_noise_max_quantile=args.pure_noise_max_quantile,
                pure_noise_grid_size=args.pure_noise_grid_size,
                rank_penalty=args.rank_penalty,
            )
            if args.experiment == "structure":
                row, detail = run_structure_replication(core, sim_cfg, struct_cfg, compare_varimax=args.compare_varimax)
            elif args.experiment == "end-to-end":
                row, detail = run_end_to_end_replication(
                    core,
                    sim_cfg,
                    struct_cfg,
                    fit_profile=args.fit_profile,
                    exact_max_k=args.exact_max_k,
                    min_trans_per_gamma_param=args.min_trans_per_gamma_param,
                    validation_fraction=args.validation_fraction,
                )
            else:
                raise ValueError(f"Unsupported experiment: {args.experiment}")
            rows.append(row)
            write_json(detail_dir / f"{scenario_name}_rep{rep:04d}.json", detail)
            write_csv(out_dir / "results.csv", rows)
            print(f"[{scenario_name} rep={rep}] status={row.get('status')} rank_hat={row.get('rank_hat')} runtime={row.get('runtime'):.2f}s")

    summary = summarize_rows(rows)
    write_csv(out_dir / "summary.csv", summary)
    write_json(out_dir / "run_manifest.json", {
        "arguments": vars(args),
        "results_file": str(out_dir / "results.csv"),
        "summary_file": str(out_dir / "summary.csv"),
        "n_rows": len(rows),
    })
    return 0 if all(r.get("status") != "error" for r in rows) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Screened PVS/LOVE initialization and ALOHA simulation study")
    parser.add_argument("--aloha-core", required=True, help="Path to the supplied original ALOHA/CLOUDS Python file")
    parser.add_argument("--experiment", choices=["smoke", "structure", "end-to-end"], default="smoke")
    parser.add_argument("--scenarios", default="baseline,strong_correlation,heteroskedastic,dense_replicates,weak_anchors,quasi_pure,signed_anchors,missingness")
    parser.add_argument("--scenario", dest="scenarios", help=argparse.SUPPRESS)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--output-dir", default="screened_pvs_results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--max-screen-rank", type=int, default=16)
    parser.add_argument("--screen-size", type=int, default=320)
    parser.add_argument("--direction-cells", type=int, default=24)
    parser.add_argument("--top-per-cell", type=int, default=10)
    parser.add_argument("--global-top-features", type=int, default=40)
    parser.add_argument("--core-anchors", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=30)
    parser.add_argument("--delta-base-constant", type=float, default=0.15)
    parser.add_argument(
        "--delta-selection",
        choices=["split_stability", "split_reconstruction", "theory"],
        default="split_stability",
    )
    parser.add_argument(
        "--parallel-grouping",
        choices=["neighborhood", "complete", "graph"],
        default="neighborhood",
    )
    parser.add_argument("--max-parallel-group-size", type=int, default=8)
    parser.add_argument("--max-group-fraction", type=float, default=0.20)
    parser.add_argument("--min-group-common-diagonal", type=float, default=0.06)
    parser.add_argument("--split-group-jaccard", type=float, default=0.35)
    parser.add_argument("--split-threshold-multiplier", type=float, default=1.35)
    parser.add_argument("--pure-noise-max-quantile", type=float, default=0.60)
    parser.add_argument("--pure-noise-grid-size", type=int, default=25)
    parser.add_argument("--rank-penalty", type=float, default=0.01)
    parser.add_argument("--compare-varimax", action="store_true")

    parser.add_argument("--fit-profile", choices=["fast", "standard", "thorough"], default="fast")
    parser.add_argument("--exact-max-k", type=int, default=8)
    parser.add_argument("--min-trans-per-gamma-param", type=float, default=5.0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
