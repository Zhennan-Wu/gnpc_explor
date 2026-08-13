#!/usr/bin/env python3
"""Reproduce the ALOHA preliminary Figure 1 and Table 1 used in the exam materials.

The preliminary evidence in the current writing/slides is explicitly a *two-stage
proof of concept*: training-only PCA/varimax and adaptive anchor assignment are
followed by projected latent scores and exact-gap OU fitting.  It is not the final
joint generalized-EM estimator.  This script therefore reproduces that exact
prototype rather than silently replacing it with a different analysis.

The supplied ``clouds_anchor_simu_multi_core_chat8.py`` implementation is loaded
and checked for the model components used by the project.  The simulation below is
a transparent special case of its equations:

    xi(t+dt) | xi(t) ~ N(exp(-Gamma dt) xi(t),
                         I - exp(-Gamma dt) exp(-Gamma.T dt))
    y(t) = intercept + Lambda xi(t) + epsilon(t)

with zero moving mean and Omega=I.  The prototype fitting code is kept
self-contained so the numerical values reproduce the preliminary results already
reported in the documents.

Default outputs (names match the LaTeX projects):

    aloha_poc_figure1.png
    aloha_poc_figure1_top.png
    aloha_poc_figure1_bottom.png
    aloha_poc_table1.csv
    aloha_poc_table1_writing.tex
    aloha_poc_table1_slides.tex
    aloha_poc_summary_long.csv
    aloha_poc_replicates_long.csv
    aloha_poc_structure.json
    aloha_poc_dimension_curve.csv
    aloha_poc_representative.json
    aloha_poc_macros.tex
    aloha_poc_run_metadata.json

Example
-------
python reproduce_aloha_preliminary_results.py \
    --implementation clouds_anchor_simu_multi_core_chat8.py \
    --outdir aloha_preliminary_results
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

# Configure BLAS before NumPy/scikit-learn are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize, minimize_scalar
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=RuntimeWarning)

METHOD_ORDER = [
    "ALOHA PoC: full exact-gap OU",
    "Restricted diagonal OU",
    "Equal-gap VAR(1)",
    "Static carry-forward",
    "Oracle-state exact-gap OU",
]

METRIC_ORDER = [
    "latent_prediction_rmse",
    "feature_prediction_rmse",
    "transition_relative_error",
    "drift_relative_error",
    "offdiag_sign_accuracy",
]


@dataclass
class SimData:
    y: NDArray[np.float64]
    x: NDArray[np.float64]
    subject: NDArray[np.int64]
    visit: NDArray[np.int64]
    time: NDArray[np.float64]
    is_holdout: NDArray[np.bool_]
    lambda_true: NDArray[np.float64]
    intercept_true: NDArray[np.float64]
    psi_true: NDArray[np.float64]
    gamma_true: NDArray[np.float64]
    anchor_groups: List[List[int]]


@dataclass
class FactorFit:
    mean: NDArray[np.float64]
    sd: NDArray[np.float64]
    loadings_std: NDArray[np.float64]
    scores_all: NDArray[np.float64]
    selected_anchors: List[List[int]]
    residual_var_std: NDArray[np.float64]
    pca: PCA
    rotation: NDArray[np.float64]
    score_scale: NDArray[np.float64]


def load_and_check_implementation(path: Path, threads: int = 1) -> Tuple[Any, str]:
    """Import the supplied implementation and verify the expected API."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Implementation not found: {path}")

    value = str(max(1, int(threads)))
    os.environ["CLOUDS_CPU_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value

    name = f"aloha_implementation_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    required = [
        "CLOUDS",
        "simulate_ad_cohort_stress",
        "discover_anchor_groups",
        "fit_clouds_candidate",
        "transition_matrix_summary",
    ]
    missing = [item for item in required if not hasattr(module, item)]
    if missing:
        raise RuntimeError(f"Implementation is missing required objects: {missing}")

    module.configure_process_environment(int(threads))
    module.configure_torch_runtime(int(threads), "float64")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return module, digest


def varimax(
    phi: NDArray[np.float64],
    gamma: float = 1.0,
    q: int = 100,
    tol: float = 1e-7,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    p, k = phi.shape
    rotation = np.eye(k)
    objective_old = 0.0
    for _ in range(q):
        rotated = phi @ rotation
        u, singular, vh = np.linalg.svd(
            phi.T
            @ (
                rotated**3
                - (gamma / p)
                * rotated
                @ np.diag(np.diag(rotated.T @ rotated))
            ),
            full_matrices=False,
        )
        rotation = u @ vh
        objective = float(np.sum(singular))
        if objective_old > 0 and objective / objective_old < 1 + tol:
            break
        objective_old = objective
    return phi @ rotation, rotation


def make_gamma(r: int) -> NDArray[np.float64]:
    if r != 3:
        raise ValueError("The reported proof of concept is configured for R=3.")
    symmetric = np.array(
        [
            [0.62, 0.08, -0.03],
            [0.08, 0.52, 0.07],
            [-0.03, 0.07, 0.46],
        ]
    )
    skew = np.array(
        [
            [0.0, 0.24, -0.12],
            [-0.24, 0.0, 0.17],
            [0.12, -0.17, 0.0],
        ]
    )
    return symmetric + skew


def simulate(seed: int, n: int = 120, k: int = 240, r: int = 3) -> SimData:
    """Generate the exact proof-of-concept design used in the documents."""
    rng = np.random.default_rng(seed)
    gamma = make_gamma(r)
    omega = np.eye(r)

    loadings = np.zeros((k, r))
    anchor_groups: List[List[int]] = []
    cursor = 0
    for factor in range(r):
        anchors = list(range(cursor, cursor + 3))
        anchor_groups.append(anchors)
        loadings[anchors, factor] = rng.uniform(1.65, 2.05, size=3)
        cursor += 3

    for feature in range(cursor, k):
        primary = int(rng.integers(0, r))
        loadings[feature, primary] = rng.normal(0.0, 0.78)
        if abs(loadings[feature, primary]) < 0.28:
            loadings[feature, primary] += (
                np.sign(loadings[feature, primary] + 1e-8) * 0.35
            )

        secondary = int(rng.choice([value for value in range(r) if value != primary]))
        value = rng.normal(0.0, 0.18)
        if abs(value) < 0.07:
            value = (1.0 if value >= 0 else -1.0) * 0.07
        loadings[feature, secondary] = value
        if rng.random() < 0.10:
            remaining = [value for value in range(r) if value not in (primary, secondary)]
            loadings[feature, remaining] += rng.normal(0.0, 0.08, size=len(remaining))

    intercept = rng.normal(0.0, 0.25, size=k)
    psi = rng.uniform(0.45**2, 0.75**2, size=k)
    for anchors in anchor_groups:
        psi[anchors] = rng.uniform(0.20**2, 0.30**2, size=len(anchors))

    gap_grid = np.array([0.35, 0.50, 0.75, 1.00, 1.30])
    observations: List[NDArray[np.float64]] = []
    states: List[NDArray[np.float64]] = []
    subjects: List[int] = []
    visits: List[int] = []
    times: List[float] = []
    holdouts: List[bool] = []

    for subject in range(n):
        n_visits = int(rng.integers(4, 8))
        gaps = rng.choice(gap_grid, size=n_visits - 1, replace=True)
        subject_times = np.concatenate([[0.0], np.cumsum(gaps)])
        state = rng.multivariate_normal(np.zeros(r), omega)
        for visit, current_time in enumerate(subject_times):
            if visit > 0:
                dt = subject_times[visit] - subject_times[visit - 1]
                transition = expm(-gamma * dt)
                innovation = omega - transition @ omega @ transition.T
                innovation = (innovation + innovation.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(innovation)
                innovation = (
                    eigenvectors
                    @ np.diag(np.maximum(eigenvalues, 1e-10))
                    @ eigenvectors.T
                )
                state = transition @ state + rng.multivariate_normal(np.zeros(r), innovation)

            observed = intercept + loadings @ state + rng.normal(0.0, np.sqrt(psi), size=k)
            observations.append(observed)
            states.append(state.copy())
            subjects.append(subject)
            visits.append(visit)
            times.append(float(current_time))
            holdouts.append(visit == n_visits - 1)

    return SimData(
        y=np.vstack(observations),
        x=np.vstack(states),
        subject=np.asarray(subjects, dtype=int),
        visit=np.asarray(visits, dtype=int),
        time=np.asarray(times, dtype=float),
        is_holdout=np.asarray(holdouts, dtype=bool),
        lambda_true=loadings,
        intercept_true=intercept,
        psi_true=psi,
        gamma_true=gamma,
        anchor_groups=anchor_groups,
    )


def select_anchors(loadings: NDArray[np.float64], anchors_per_factor: int = 3) -> List[List[int]]:
    """Global distinct-anchor assignment used in the reported prototype."""
    absolute = np.abs(loadings)
    strength = absolute / (np.max(absolute, axis=0, keepdims=True) + 1e-12)
    purity = absolute / (np.sum(absolute, axis=1, keepdims=True) + 1e-12)
    communality = np.sum(loadings**2, axis=1, keepdims=True)
    communality /= np.max(communality) + 1e-12
    score = strength * purity * communality

    r = loadings.shape[1]
    slot_factor = np.repeat(np.arange(r), anchors_per_factor)
    cost = -score[:, slot_factor].T
    rows, columns = linear_sum_assignment(cost)
    groups = [[] for _ in range(r)]
    for slot, feature in zip(rows, columns):
        groups[int(slot_factor[slot])].append(int(feature))
    for group in groups:
        group.sort()
    return groups


def fit_factor_model(data: SimData, r: int) -> FactorFit:
    train = ~data.is_holdout
    y_train = data.y[train]
    mean = y_train.mean(axis=0)
    sd = y_train.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-7, 1.0, sd)
    z_train = (y_train - mean) / sd
    z_all = (data.y - mean) / sd

    pca = PCA(n_components=r, svd_solver="randomized", random_state=1729)
    scores_train = pca.fit_transform(z_train)
    base_loadings = pca.components_.T
    rotated_loadings, rotation = varimax(base_loadings)
    rotated_scores_train = scores_train @ rotation
    rotated_scores_all = pca.transform(z_all) @ rotation

    score_scale = rotated_scores_train.std(axis=0, ddof=1)
    score_scale = np.where(score_scale < 1e-8, 1.0, score_scale)
    scores_all = rotated_scores_all / score_scale
    loadings_std = rotated_loadings * score_scale[None, :]

    selected = select_anchors(loadings_std, anchors_per_factor=3)
    for factor, anchors in enumerate(selected):
        if np.mean(loadings_std[anchors, factor]) < 0:
            loadings_std[:, factor] *= -1
            scores_all[:, factor] *= -1
            rotation[:, factor] *= -1

    reconstructed_train = scores_all[train] @ loadings_std.T
    residual_var = np.var(z_train - reconstructed_train, axis=0, ddof=1)
    residual_var = np.maximum(residual_var, 0.05)

    return FactorFit(
        mean=mean,
        sd=sd,
        loadings_std=loadings_std,
        scores_all=scores_all,
        selected_anchors=selected,
        residual_var_std=residual_var,
        pca=pca,
        rotation=rotation,
        score_scale=score_scale,
    )


def transition_arrays(
    data: SimData,
    scores: NDArray[np.float64],
    train_only: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    previous: List[NDArray[np.float64]] = []
    following: List[NDArray[np.float64]] = []
    gaps: List[float] = []
    for subject in np.unique(data.subject):
        indices = np.where(data.subject == subject)[0]
        indices = indices[np.argsort(data.visit[indices])]
        for first, second in zip(indices[:-1], indices[1:]):
            if train_only and data.is_holdout[second]:
                continue
            previous.append(scores[first])
            following.append(scores[second])
            gaps.append(data.time[second] - data.time[first])
    return np.vstack(previous), np.vstack(following), np.asarray(gaps)


def fit_diag_ou(
    previous: NDArray[np.float64],
    following: NDArray[np.float64],
    dt: NDArray[np.float64],
    extra_noise: float = 0.08,
) -> NDArray[np.float64]:
    r = previous.shape[1]
    rates = np.zeros(r)
    for factor in range(r):

        def objective(log_rate: float) -> float:
            rate = math.exp(float(log_rate))
            decay = np.exp(-rate * dt)
            variance = np.maximum(1.0 - decay * decay + extra_noise, 1e-8)
            residual = following[:, factor] - decay * previous[:, factor]
            return float(0.5 * np.sum(np.log(variance) + residual * residual / variance))

        result = minimize_scalar(
            objective,
            bounds=(-4.0, 2.0),
            method="bounded",
            options={"xatol": 1e-5},
        )
        rates[factor] = math.exp(float(result.x))
    return np.diag(rates)


def unpack_full(theta: NDArray[np.float64], r: int) -> NDArray[np.float64]:
    lower = np.zeros((r, r))
    position = 0
    for row in range(r):
        for column in range(row + 1):
            if row == column:
                lower[row, column] = math.exp(float(theta[position]))
            else:
                lower[row, column] = float(theta[position])
            position += 1
    symmetric = lower @ lower.T + 1e-5 * np.eye(r)

    skew = np.zeros((r, r))
    for row in range(r):
        for column in range(row + 1, r):
            value = float(theta[position])
            skew[row, column] = value
            skew[column, row] = -value
            position += 1
    return symmetric + skew


def pack_start(gamma_diag: NDArray[np.float64]) -> NDArray[np.float64]:
    r = gamma_diag.shape[0]
    values: List[float] = []
    for row in range(r):
        for column in range(row + 1):
            if row == column:
                values.append(0.5 * math.log(max(gamma_diag[row, row], 1e-4)))
            else:
                values.append(0.0)
    values.extend([0.0] * (r * (r - 1) // 2))
    return np.asarray(values, dtype=float)


def fit_full_ou(
    previous: NDArray[np.float64],
    following: NDArray[np.float64],
    dt: NDArray[np.float64],
    extra_noise: float = 0.08,
    maxiter: int = 180,
) -> NDArray[np.float64]:
    r = previous.shape[1]
    diagonal = fit_diag_ou(previous, following, dt, extra_noise=extra_noise)
    start = pack_start(diagonal)
    rounded = np.round(dt, 8)
    groups: Dict[float, NDArray[np.int64]] = {
        float(gap): np.where(rounded == gap)[0] for gap in np.unique(rounded)
    }

    def objective(theta: NDArray[np.float64]) -> float:
        gamma = unpack_full(theta, r)
        total = 0.0
        try:
            for gap, indices in groups.items():
                transition = expm(-gamma * gap)
                covariance = (
                    np.eye(r)
                    - transition @ transition.T
                    + extra_noise * np.eye(r)
                )
                covariance = (covariance + covariance.T) / 2
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0 or not np.isfinite(logdet):
                    return 1e10
                residual = following[indices] - previous[indices] @ transition.T
                solution = np.linalg.solve(covariance, residual.T).T
                total += 0.5 * (
                    len(indices) * logdet + np.sum(residual * solution)
                )
            total += 1e-3 * float(np.sum(theta * theta))
            return float(total)
        except (np.linalg.LinAlgError, ValueError, OverflowError):
            return 1e10

    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-5, "maxls": 30},
    )
    return unpack_full(result.x, r)


def fit_var(
    previous: NDArray[np.float64],
    following: NDArray[np.float64],
) -> NDArray[np.float64]:
    # Row-vector regression following = previous @ B; column-state transition is B.T.
    coefficient, *_ = np.linalg.lstsq(previous, following, rcond=None)
    return coefficient.T


def factor_alignment(
    estimated: NDArray[np.float64],
    truth: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r = truth.shape[1]
    correlation = np.corrcoef(estimated[mask].T, truth[mask].T)[:r, r:]
    estimated_rows, truth_columns = linear_sum_assignment(-np.abs(correlation))
    permutation = np.zeros(r, dtype=int)
    signs = np.ones(r)
    for estimated_factor, truth_factor in zip(estimated_rows, truth_columns):
        permutation[truth_factor] = estimated_factor
        signs[truth_factor] = 1.0 if correlation[estimated_factor, truth_factor] >= 0 else -1.0

    transform = np.zeros((r, r))
    for truth_factor in range(r):
        transform[truth_factor, permutation[truth_factor]] = signs[truth_factor]
    aligned = estimated @ transform.T
    return aligned, transform, correlation


def anchor_metrics(
    selected: List[List[int]],
    truth: List[List[int]],
) -> Tuple[float, float, NDArray[np.int64]]:
    r = len(truth)
    overlap = np.zeros((r, r), dtype=int)
    for estimated_factor in range(r):
        for true_factor in range(r):
            overlap[estimated_factor, true_factor] = len(
                set(selected[estimated_factor]) & set(truth[true_factor])
            )
    estimated_rows, truth_columns = linear_sum_assignment(-overlap)
    hits = int(
        sum(overlap[estimated_factor, true_factor] for estimated_factor, true_factor in zip(estimated_rows, truth_columns))
    )
    return (
        hits / sum(len(group) for group in selected),
        hits / sum(len(group) for group in truth),
        overlap,
    )


def evaluate_rep(
    seed: int,
    n: int = 120,
    k: int = 240,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, object]]:
    data = simulate(seed=seed, n=n, k=k)
    fit = fit_factor_model(data, r=3)
    aligned_scores, transform, _ = factor_alignment(
        fit.scores_all, data.x, ~data.is_holdout
    )
    state_corr = float(
        np.mean(
            [
                np.corrcoef(
                    aligned_scores[~data.is_holdout, factor],
                    data.x[~data.is_holdout, factor],
                )[0, 1]
                for factor in range(3)
            ]
        )
    )
    state_rmse = float(
        np.sqrt(
            np.mean(
                (aligned_scores[~data.is_holdout] - data.x[~data.is_holdout]) ** 2
            )
        )
    )
    anchor_precision, anchor_recall, overlap = anchor_metrics(
        fit.selected_anchors, data.anchor_groups
    )

    previous, following, dt = transition_arrays(data, fit.scores_all, train_only=True)
    full = fit_full_ou(previous, following, dt, extra_noise=0.08)
    diagonal = fit_diag_ou(previous, following, dt, extra_noise=0.08)
    var_transition = fit_var(previous, following)

    previous_oracle, following_oracle, dt_oracle = transition_arrays(
        data, data.x, train_only=True
    )
    oracle = fit_full_ou(
        previous_oracle,
        following_oracle,
        dt_oracle,
        extra_noise=0.0,
        maxiter=220,
    )

    transform_inverse = transform.T
    full_aligned = transform @ full @ transform_inverse
    diagonal_aligned = transform @ diagonal @ transform_inverse

    methods = {
        "ALOHA PoC: full exact-gap OU": ("gamma", full, full_aligned),
        "Restricted diagonal OU": ("gamma", diagonal, diagonal_aligned),
        "Equal-gap VAR(1)": ("fixed", var_transition, None),
        "Static carry-forward": ("fixed", np.eye(3), None),
        "Oracle-state exact-gap OU": ("oracle", oracle, oracle),
    }

    holdout_indices = np.where(data.is_holdout)[0]
    previous_indices: List[int] = []
    holdout_gaps: List[float] = []
    for holdout in holdout_indices:
        candidate = np.where(
            (data.subject == data.subject[holdout])
            & (data.visit == data.visit[holdout] - 1)
        )[0]
        previous_indices.append(int(candidate[0]))
        holdout_gaps.append(float(data.time[holdout] - data.time[candidate[0]]))
    previous_indices_arr = np.asarray(previous_indices, dtype=int)
    holdout_gaps_arr = np.asarray(holdout_gaps)

    metrics: Dict[str, Dict[str, float]] = {}
    for name, (kind, parameter, aligned_gamma) in methods.items():
        predicted_estimated = np.zeros((len(holdout_indices), 3))
        if kind == "oracle":
            for row, (previous_index, gap) in enumerate(
                zip(previous_indices_arr, holdout_gaps_arr)
            ):
                predicted_estimated[row] = expm(-parameter * gap) @ data.x[previous_index]
            predicted_true_coordinates = predicted_estimated
            predicted_features = (
                data.intercept_true[None, :]
                + predicted_estimated @ data.lambda_true.T
            )
        else:
            for row, (previous_index, gap) in enumerate(
                zip(previous_indices_arr, holdout_gaps_arr)
            ):
                transition = expm(-parameter * gap) if kind == "gamma" else parameter
                predicted_estimated[row] = transition @ fit.scores_all[previous_index]
            predicted_true_coordinates = predicted_estimated @ transform.T
            standardized_prediction = predicted_estimated @ fit.loadings_std.T
            predicted_features = (
                fit.mean[None, :] + standardized_prediction * fit.sd[None, :]
            )

        latent_rmse = float(
            np.sqrt(
                np.mean(
                    (predicted_true_coordinates - data.x[holdout_indices]) ** 2
                )
            )
        )
        feature_rmse = float(
            np.sqrt(
                np.mean((predicted_features - data.y[holdout_indices]) ** 2)
            )
        )

        transition_errors: List[float] = []
        for gap in np.array([0.35, 0.75, 1.30]):
            true_transition = expm(-data.gamma_true * gap)
            if kind == "oracle":
                estimated_transition = expm(-parameter * gap)
            elif kind == "gamma":
                estimated_transition = (
                    transform @ expm(-parameter * gap) @ transform_inverse
                )
            else:
                estimated_transition = transform @ parameter @ transform_inverse
            transition_errors.append(
                np.linalg.norm(estimated_transition - true_transition, ord="fro")
                / np.linalg.norm(true_transition, ord="fro")
            )

        drift_error = float("nan")
        sign_accuracy = float("nan")
        if aligned_gamma is not None:
            drift_error = float(
                np.linalg.norm(aligned_gamma - data.gamma_true, ord="fro")
                / np.linalg.norm(data.gamma_true, ord="fro")
            )
            off_diagonal = ~np.eye(3, dtype=bool)
            sign_accuracy = float(
                np.mean(
                    np.sign(aligned_gamma[off_diagonal])
                    == np.sign(data.gamma_true[off_diagonal])
                )
            )

        metrics[name] = {
            "latent_prediction_rmse": latent_rmse,
            "feature_prediction_rmse": feature_rmse,
            "transition_relative_error": float(np.mean(transition_errors)),
            "drift_relative_error": drift_error,
            "offdiag_sign_accuracy": sign_accuracy,
        }

    extra: Dict[str, object] = {
        "data": data,
        "fit": fit,
        "aligned_scores": aligned_scores,
        "alignment": transform,
        "full_gamma_aligned": full_aligned,
        "anchor_precision": anchor_precision,
        "anchor_recall": anchor_recall,
        "anchor_overlap": overlap,
        "state_corr": state_corr,
        "state_rmse": state_rmse,
    }
    return metrics, extra


def dimension_curve(
    data: SimData,
    candidates: Sequence[int] = tuple(range(1, 7)),
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for r in candidates:
        fit = fit_factor_model(data, r=r)
        previous, following, dt = transition_arrays(data, fit.scores_all, train_only=True)
        diagonal = fit_diag_ou(previous, following, dt, extra_noise=0.08)
        subject_rmse: List[float] = []
        for holdout in np.where(data.is_holdout)[0]:
            previous_index = np.where(
                (data.subject == data.subject[holdout])
                & (data.visit == data.visit[holdout] - 1)
            )[0][0]
            gap = data.time[holdout] - data.time[previous_index]
            latent_prediction = expm(-diagonal * gap) @ fit.scores_all[previous_index]
            standardized_prediction = fit.loadings_std @ latent_prediction
            feature_prediction = fit.mean + fit.sd * standardized_prediction
            subject_rmse.append(
                float(np.sqrt(np.mean((feature_prediction - data.y[holdout]) ** 2)))
            )
        rows.append(
            {
                "R": int(r),
                "rmse": float(np.mean(subject_rmse)),
                "se": float(
                    np.std(subject_rmse, ddof=1) / np.sqrt(len(subject_rmse))
                ),
            }
        )

    frame = pd.DataFrame(rows)
    best_index = int(frame["rmse"].idxmin())
    threshold = float(frame.loc[best_index, "rmse"] + frame.loc[best_index, "se"])
    selected = int(frame.loc[frame["rmse"] <= threshold, "R"].min())
    frame["one_se_threshold"] = threshold
    frame["selected"] = frame["R"] == selected
    return frame


def _panel_a(ax: Any, extra: Mapping[str, object], figure: Any) -> None:
    data: SimData = extra["data"]  # type: ignore[assignment]
    fit: FactorFit = extra["fit"]  # type: ignore[assignment]
    _, transform, _ = factor_alignment(fit.scores_all, data.x, ~data.is_holdout)
    loading_aligned = fit.loadings_std @ transform.T

    shown: List[int] = []
    for factor in range(3):
        shown.extend(data.anchor_groups[factor])
        candidates = [
            feature
            for feature in np.argsort(-np.abs(data.lambda_true[:, factor]))
            if feature not in shown
        ]
        shown.extend(candidates[:12])
    shown = list(dict.fromkeys(shown))[:45]

    image = ax.imshow(
        loading_aligned[shown].T,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("A. Adaptive measurement structure")
    ax.set_ylabel("Latent domain")
    ax.set_xlabel("Anchor and high-loading features")
    ax.set_yticks([0, 1, 2], labels=["1", "2", "3"])
    ax.set_xticks([])
    figure.colorbar(image, ax=ax, shrink=0.78, label="Estimated loading")


def _panel_b(ax: Any, extra: Mapping[str, object]) -> None:
    data: SimData = extra["data"]  # type: ignore[assignment]
    aligned_scores: NDArray[np.float64] = extra["aligned_scores"]  # type: ignore[assignment]
    gamma_hat: NDArray[np.float64] = extra["full_gamma_aligned"]  # type: ignore[assignment]

    chosen = None
    best_variance = -np.inf
    for subject in np.unique(data.subject):
        indices = np.where(data.subject == subject)[0]
        if len(indices) >= 6:
            variance = float(np.var(data.x[indices]))
            if variance > best_variance:
                best_variance = variance
                chosen = int(subject)
    if chosen is None:
        chosen = 0

    indices = np.where(data.subject == chosen)[0]
    indices = indices[np.argsort(data.visit[indices])]
    time_values = data.time[indices]
    for factor in range(3):
        line = ax.plot(
            time_values,
            data.x[indices, factor],
            "-",
            linewidth=1.8,
            label=f"True domain {factor + 1}",
        )[0]
        ax.scatter(
            time_values[:-1],
            aligned_scores[indices[:-1], factor],
            marker="o",
            s=36,
            color=line.get_color(),
        )

    holdout = indices[-1]
    previous = indices[-2]
    gap = data.time[holdout] - data.time[previous]
    prediction = expm(-gamma_hat * gap) @ aligned_scores[previous]
    for factor in range(3):
        ax.scatter(
            [time_values[-1]],
            [prediction[factor]],
            marker="*",
            s=130,
            zorder=5,
        )
    ax.axvline(time_values[-1], linestyle="--", linewidth=1)
    ax.set_title("B. Irregular latent trajectory reconstruction")
    ax.set_xlabel("Continuous time")
    ax.set_ylabel("Latent state")
    ax.legend(fontsize=8, ncol=2, loc="best")


def _panel_c(ax: Any, extra: Mapping[str, object]) -> None:
    data: SimData = extra["data"]  # type: ignore[assignment]
    gamma_hat: NDArray[np.float64] = extra["full_gamma_aligned"]  # type: ignore[assignment]
    true_values = data.gamma_true.ravel()
    estimated_values = gamma_hat.ravel()
    diagonal = np.eye(3, dtype=bool).ravel()

    ax.scatter(
        true_values[~diagonal],
        estimated_values[~diagonal],
        marker="o",
        s=65,
        label="Off-diagonal",
    )
    ax.scatter(
        true_values[diagonal],
        estimated_values[diagonal],
        marker="s",
        s=70,
        label="Diagonal",
    )
    lower = min(true_values.min(), estimated_values.min()) - 0.08
    upper = max(true_values.max(), estimated_values.max()) + 0.08
    ax.plot([lower, upper], [lower, upper], "--", linewidth=1.2)
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title("C. Full asymmetric drift recovery")
    ax.set_xlabel("True drift entry")
    ax.set_ylabel("Estimated drift entry")
    ax.legend(fontsize=8)


def _panel_d(ax: Any, curve: pd.DataFrame) -> None:
    ax.errorbar(
        curve["R"],
        curve["rmse"],
        yerr=curve["se"],
        marker="o",
        capsize=4,
    )
    threshold = float(curve["one_se_threshold"].iloc[0])
    ax.axhline(threshold, linestyle="--", linewidth=1, label="One-SE threshold")
    selected = curve.loc[curve["selected"]].iloc[0]
    ax.scatter(
        [selected["R"]],
        [selected["rmse"]],
        marker="*",
        s=160,
        zorder=5,
        label=f"Selected R={int(selected['R'])}",
    )
    ax.axvline(3, linestyle=":", linewidth=1, label="True R=3")
    ax.set_xticks(curve["R"])
    ax.set_title("D. Held-out dimension selection")
    ax.set_xlabel("Candidate latent dimension R")
    ax.set_ylabel("Held-out feature RMSE")
    ax.legend(fontsize=8)


def make_figures(extra: Mapping[str, object], curve: pd.DataFrame, outdir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    _panel_a(axes[0, 0], extra, figure)
    _panel_b(axes[0, 1], extra)
    _panel_c(axes[1, 0], extra)
    _panel_d(axes[1, 1], curve)
    figure.suptitle(
        "Figure 1. Two-stage proof of concept for ALOHA (representative replicate)",
        fontsize=15,
    )
    figure.savefig(outdir / "aloha_poc_figure1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    top, top_axes = plt.subplots(1, 2, figsize=(12.4, 4.45), constrained_layout=True)
    _panel_a(top_axes[0], extra, top)
    _panel_b(top_axes[1], extra)
    top.savefig(outdir / "aloha_poc_figure1_top.png", dpi=220, bbox_inches="tight")
    plt.close(top)

    bottom, bottom_axes = plt.subplots(1, 2, figsize=(12.4, 4.35), constrained_layout=True)
    _panel_c(bottom_axes[0], extra)
    _panel_d(bottom_axes[1], curve)
    bottom.savefig(outdir / "aloha_poc_figure1_bottom.png", dpi=220, bbox_inches="tight")
    plt.close(bottom)


def aggregate_results(
    replicate_results: List[Dict[str, Dict[str, float]]],
    extras: List[Dict[str, object]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    records: List[Dict[str, object]] = []
    replicate_records: List[Dict[str, object]] = []
    for replicate, result in enumerate(replicate_results):
        for method in METHOD_ORDER:
            for metric in METRIC_ORDER:
                value = float(result[method][metric])
                records.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
                replicate_records.append(
                    {
                        "replicate": replicate,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )

    long = pd.DataFrame(records)
    summary_rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        for metric in METRIC_ORDER:
            values = long.loc[
                (long["method"] == method) & (long["metric"] == metric),
                "value",
            ].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(finite)) if len(finite) else np.nan,
                    "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "se": (
                        float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                        if len(finite) > 1
                        else np.nan
                    ),
                    "n": int(len(finite)),
                }
            )
    summary = pd.DataFrame(summary_rows)

    structure = {
        "anchor_precision_mean": float(np.mean([extra["anchor_precision"] for extra in extras])),
        "anchor_precision_sd": float(np.std([extra["anchor_precision"] for extra in extras], ddof=1)),
        "anchor_recall_mean": float(np.mean([extra["anchor_recall"] for extra in extras])),
        "anchor_recall_sd": float(np.std([extra["anchor_recall"] for extra in extras], ddof=1)),
        "state_corr_mean": float(np.mean([extra["state_corr"] for extra in extras])),
        "state_corr_sd": float(np.std([extra["state_corr"] for extra in extras], ddof=1)),
        "state_rmse_mean": float(np.mean([extra["state_rmse"] for extra in extras])),
        "state_rmse_sd": float(np.std([extra["state_rmse"] for extra in extras], ddof=1)),
    }
    return summary, pd.DataFrame(replicate_records), structure


def table_wide(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        row: Dict[str, object] = {"Method": method}
        for metric in METRIC_ORDER:
            subset = summary[
                (summary["method"] == method) & (summary["metric"] == metric)
            ].iloc[0]
            if np.isfinite(subset["mean"]):
                row[metric] = f"{subset['mean']:.3f} ({subset['sd']:.3f})"
            else:
                row[metric] = "--"
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex_outputs(
    outdir: Path,
    summary: pd.DataFrame,
    table: pd.DataFrame,
    structure: Mapping[str, float],
) -> None:
    method_tex = {
        "ALOHA PoC: full exact-gap OU": r"\ALOHA{} PoC: full exact-gap OU",
        "Restricted diagonal OU": "Restricted diagonal OU",
        "Equal-gap VAR(1)": "Equal-gap VAR(1)",
        "Static carry-forward": "Static carry-forward",
        "Oracle-state exact-gap OU": "Oracle-state exact-gap OU",
    }

    writing = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Qualification Table 1: proof-of-concept performance over 24 independent replicates, mean (SD). Lower is better except sign accuracy. ``Oracle-state'' uses true latent states to isolate dynamic-estimation error.}",
        r"\label{tab:poc}",
        r"\small",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Latent prediction RMSE & Feature prediction RMSE & Transition rel. error & Drift rel. error & Off-diagonal sign accuracy \\",
        r"\midrule",
    ]
    slides = [
        r"\begin{tabularx}{\textwidth}{p{2.65cm}ccccc}",
        r"\toprule",
        r"Method & Latent RMSE & Feature RMSE & Transition error & Drift error & Sign accuracy \\",
        r"\midrule",
    ]

    for method in METHOD_ORDER:
        table_row = table.loc[table["Method"] == method].iloc[0]
        formatted = [str(table_row[metric]) for metric in METRIC_ORDER]
        writing.append(
            method_tex[method]
            + " & "
            + " & ".join(f"${value}$" if value != "--" else "--" for value in formatted)
            + r" \\"
        )

        means: List[str] = []
        for metric in METRIC_ORDER:
            value = float(
                summary[
                    (summary["method"] == method) & (summary["metric"] == metric)
                ].iloc[0]["mean"]
            )
            means.append(f"{value:.3f}" if np.isfinite(value) else "--")
        slides.append(
            method_tex[method]
            + " & "
            + " & ".join(f"${value}$" if value != "--" else "--" for value in means)
            + r" \\"
        )

    writing.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\end{table}",
            "",
        ]
    )
    slides.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (outdir / "aloha_poc_table1_writing.tex").write_text(
        "\n".join(writing), encoding="utf-8"
    )
    (outdir / "aloha_poc_table1_slides.tex").write_text(
        "\n".join(slides), encoding="utf-8"
    )

    macros = [
        r"% Generated by reproduce_aloha_preliminary_results.py",
        rf"\newcommand{{\ALOHAAnchorPrecisionMean}}{{{structure['anchor_precision_mean']:.3f}}}",
        rf"\newcommand{{\ALOHAAnchorPrecisionSD}}{{{structure['anchor_precision_sd']:.3f}}}",
        rf"\newcommand{{\ALOHAAnchorRecallMean}}{{{structure['anchor_recall_mean']:.3f}}}",
        rf"\newcommand{{\ALOHAAnchorRecallSD}}{{{structure['anchor_recall_sd']:.3f}}}",
        rf"\newcommand{{\ALOHAStateCorrelationMean}}{{{structure['state_corr_mean']:.3f}}}",
        rf"\newcommand{{\ALOHAStateCorrelationSD}}{{{structure['state_corr_sd']:.4f}}}",
        rf"\newcommand{{\ALOHAStateRMSEMean}}{{{structure['state_rmse_mean']:.3f}}}",
        rf"\newcommand{{\ALOHAStateRMSESD}}{{{structure['state_rmse_sd']:.3f}}}",
        "",
    ]
    (outdir / "aloha_poc_macros.tex").write_text(
        "\n".join(macros), encoding="utf-8"
    )


def write_metadata(
    outdir: Path,
    args: argparse.Namespace,
    implementation_hash: str,
) -> None:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "implementation": str(args.implementation.expanduser().resolve()),
        "implementation_sha256": implementation_hash,
        "analysis_type": "two-stage preliminary proof of concept",
        "n_replicates": int(args.n_reps),
        "n_subjects": int(args.n_subjects),
        "n_features": int(args.n_features),
        "latent_dimension": 3,
        "seed_base": int(args.seed_base),
        "seed_step": int(args.seed_step),
    }
    (outdir / "aloha_poc_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reproduce the ALOHA preliminary Figure 1 and Table 1.",
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path("clouds_anchor_simu_multi_core_chat8.py"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("aloha_preliminary_results"))
    parser.add_argument("--n-reps", type=int, default=24)
    parser.add_argument("--n-subjects", type=int, default=120)
    parser.add_argument("--n-features", type=int, default=240)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--seed-step", type=int, default=37)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--skip-implementation-check", action="store_true")
    args = parser.parse_args()

    if args.n_reps < 1:
        raise ValueError("n_reps must be positive.")
    if args.n_subjects < 1:
        raise ValueError("n_subjects must be positive.")
    if args.n_features < 9:
        raise ValueError("n_features must be at least 9 for three anchors per domain.")

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    implementation_hash = "not checked"
    if not args.skip_implementation_check:
        _, implementation_hash = load_and_check_implementation(
            args.implementation, threads=args.threads
        )
    elif args.implementation.exists():
        implementation_hash = hashlib.sha256(
            args.implementation.expanduser().resolve().read_bytes()
        ).hexdigest()
    write_metadata(outdir, args, implementation_hash)

    seeds = [args.seed_base + index * args.seed_step for index in range(args.n_reps)]
    replicate_results: List[Dict[str, Dict[str, float]]] = []
    extras: List[Dict[str, object]] = []
    for index, seed in enumerate(seeds, start=1):
        print(f"Replicate {index}/{len(seeds)} seed={seed}", flush=True)
        metrics, extra = evaluate_rep(
            seed,
            n=args.n_subjects,
            k=args.n_features,
        )
        replicate_results.append(metrics)
        extras.append(extra)

    summary, replicate_long, structure = aggregate_results(replicate_results, extras)
    table = table_wide(summary)

    full_errors = np.asarray(
        [
            result["ALOHA PoC: full exact-gap OU"]["transition_relative_error"]
            for result in replicate_results
        ]
    )
    representative_index = int(np.argsort(full_errors)[len(full_errors) // 2])
    curve = dimension_curve(extras[representative_index]["data"])  # type: ignore[arg-type]
    make_figures(extras[representative_index], curve, outdir)

    summary.to_csv(outdir / "aloha_poc_summary_long.csv", index=False)
    replicate_long.to_csv(outdir / "aloha_poc_replicates_long.csv", index=False)
    table.to_csv(outdir / "aloha_poc_table1.csv", index=False)
    curve.to_csv(outdir / "aloha_poc_dimension_curve.csv", index=False)
    (outdir / "aloha_poc_structure.json").write_text(
        json.dumps(structure, indent=2), encoding="utf-8"
    )

    representative_data: SimData = extras[representative_index]["data"]  # type: ignore[assignment]
    representative = {
        "replicate_index": representative_index,
        "seed": seeds[representative_index],
        "anchor_precision": extras[representative_index]["anchor_precision"],
        "anchor_recall": extras[representative_index]["anchor_recall"],
        "state_corr": extras[representative_index]["state_corr"],
        "state_rmse": extras[representative_index]["state_rmse"],
        "gamma_true": representative_data.gamma_true.tolist(),
        "gamma_hat_aligned": extras[representative_index]["full_gamma_aligned"].tolist(),  # type: ignore[union-attr]
        "dimension_curve": curve.to_dict(orient="records"),
    }
    (outdir / "aloha_poc_representative.json").write_text(
        json.dumps(representative, indent=2), encoding="utf-8"
    )
    write_latex_outputs(outdir, summary, table, structure)

    narrative = (
        f"Across {args.n_reps} independently simulated cohorts, mean anchor precision "
        f"and recall were both {structure['anchor_precision_mean']:.3f} "
        f"(SD {structure['anchor_precision_sd']:.3f}); mean coordinatewise latent-state "
        f"correlation was {structure['state_corr_mean']:.3f} "
        f"(SD {structure['state_corr_sd']:.4f}); and training-state RMSE was "
        f"{structure['state_rmse_mean']:.3f} (SD {structure['state_rmse_sd']:.3f})."
    )
    (outdir / "aloha_poc_narrative.txt").write_text(narrative + "\n", encoding="utf-8")

    print("\nStructure summary:")
    print(json.dumps(structure, indent=2))
    print("\nQualification Table 1:")
    print(table.to_string(index=False))
    print(f"\nRepresentative replicate: {representative_index + 1}, seed={seeds[representative_index]}")
    print(f"Selected dimension: R={int(curve.loc[curve['selected'], 'R'].iloc[0])}")
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
