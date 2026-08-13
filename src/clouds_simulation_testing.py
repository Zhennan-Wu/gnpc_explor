"""
Corrected and hardened CLOUDS simulation/fitting framework.

The latent process uses

    xi(t+dt) | xi(t) ~ N(mu(t+dt) + exp(-Gamma dt)(xi(t)-mu(t)),
                         Omega - exp(-Gamma dt) Omega exp(-Gamma.T dt)),

with mu_i(t)=level_i+slope_i*t (the covariate-level block is optional) and

    d xi_i(t) = [slope_i - Gamma(xi_i(t)-mu_i(t))] dt + G dW_i(t).

The observation model is

    x_ij = c + Lambda xi_ij + epsilon_ij,

where the item intercept c is learned by default. To avoid exact confounding,
the constant latent-level term is free only when item intercepts are disabled.

This edition keeps the equation-consistent Kalman/RTS and generalized-EM core,
and additionally fixes PCA loading scale, safe anchor transforms, diagonal-scale
identification, diffusion consistency, and post-LBFGS profiling. It centralizes
SPD stabilization, packs exact-dynamics parameters without null directions,
ranks starts by observed-data likelihood, guards EM updates, batches transition
matrix exponentials, uses subject-level bootstrap anchors with strict purity/sign
checks, compares models on matched cohorts, supports controlled K-scaling, and
writes structured results plus a versioned, refitted selected-model bundle.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# BLAS/OpenMP controls must be set before NumPy/PyTorch are imported. Importing
# this module as a library does not mutate process environment, hide CUDA,
# install logging handlers, or alter torch runtime state.
CPU_THREADS = int(os.environ.get("CLOUDS_CPU_THREADS", "8"))


def configure_process_environment(num_threads: int = CPU_THREADS) -> None:
    """Configure CPU backends explicitly for CLI or spawned-worker execution."""
    value = str(max(1, int(num_threads)))
    os.environ["CLOUDS_CPU_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")


def _early_cli_thread_count(default: int) -> int:
    """Read only --threads before importing numerical backends."""
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--threads", type=int, default=default)
    parsed, _ = early.parse_known_args()
    return max(1, int(parsed.threads))


if __name__ == "__main__":
    CPU_THREADS = _early_cli_thread_count(CPU_THREADS)
    configure_process_environment(CPU_THREADS)
elif __name__ == "__mp_main__" or os.environ.get("CLOUDS_CONFIGURE_ENV_ON_IMPORT") == "1":
    configure_process_environment(CPU_THREADS)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - a deterministic greedy fallback is used.
    linear_sum_assignment = None

LOG_FILENAME = "clouds_anchor_simulation_chat7_improved.log"


def configure_torch_runtime(num_threads: int = CPU_THREADS, dtype: str = "float64") -> None:
    """Configure torch explicitly; safe to call in the parent and spawned workers."""
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    dtype_l = str(dtype).lower()
    if dtype_l == "float64":
        torch.set_default_dtype(torch.float64)
    elif dtype_l == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError("dtype must be 'float32' or 'float64'.")


def configure_logging(
    log_filename: Optional[str] = LOG_FILENAME,
    *,
    worker: bool = False,
    level: int = logging.INFO,
) -> None:
    """Install deterministic handlers; workers never write the shared log file."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if not worker and log_filename:
        log_path = Path(log_filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


# ---------------------------------------------------------------------
# 1. Linear algebra helpers
# ---------------------------------------------------------------------
def _eye_like(A: torch.Tensor) -> torch.Tensor:
    """Identity matrix broadcastable to A[..., K, K]."""
    K = A.shape[-1]
    eye = torch.eye(K, dtype=A.dtype, device=A.device)
    return eye.expand(A.shape[:-2] + (K, K))


def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


def normalize_spd_to_correlation(A: torch.Tensor, jitter: float = 1e-10) -> torch.Tensor:
    """Normalize an SPD covariance to a correlation matrix with exact unit diagonal."""
    A_sym = symmetrize(A)
    diag = torch.diagonal(A_sym, dim1=-2, dim2=-1).clamp_min(jitter)
    std = torch.sqrt(diag)
    corr = A_sym / (std.unsqueeze(-1) * std.unsqueeze(-2))
    corr = symmetrize(corr)
    # Avoid adding jitter after normalization, which would destroy the unit diagonal.
    idx = torch.arange(corr.shape[-1], device=corr.device)
    corr = corr.clone()
    corr[..., idx, idx] = 1.0
    return corr


_SPD_STABILIZATION_COUNTS: Dict[str, int] = {}
_SPD_MAX_RELATIVE_JITTER: Dict[str, float] = {}
_SPD_WARNED_CONTEXTS: set[str] = set()


def get_spd_stabilization_diagnostics() -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "count": float(_SPD_STABILIZATION_COUNTS.get(key, 0)),
            "max_relative_jitter": float(_SPD_MAX_RELATIVE_JITTER.get(key, 0.0)),
        }
        for key in sorted(_SPD_STABILIZATION_COUNTS)
    }


def reset_spd_stabilization_diagnostics() -> None:
    _SPD_STABILIZATION_COUNTS.clear()
    _SPD_MAX_RELATIVE_JITTER.clear()
    _SPD_WARNED_CONTEXTS.clear()


def safe_inverse_variance(log_variance: torch.Tensor, bound: float = 30.0) -> torch.Tensor:
    """Exponentiate a log inverse variance without overflow."""
    return torch.exp(torch.clamp(-log_variance, min=-bound, max=bound))


@dataclass(frozen=True)
class SPDCholesky:
    """A single stabilized SPD factorization reused for solves and log-determinants."""

    matrix: torch.Tensor
    chol: torch.Tensor
    jitter: float
    relative_jitter: float


def factor_spd(
    A: torch.Tensor,
    *,
    base_relative_jitter: float = 1e-10,
    max_relative_jitter: float = 1e-2,
    max_tries: int = 10,
    context: str = "SPD matrix",
) -> SPDCholesky:
    """Factor an SPD matrix once with bounded per-batch stabilization.

    Every batch member is tried without modification first. Only failed members
    receive scale-relative diagonal jitter, so a small-gap transition covariance
    is not stabilized using the scale of an unrelated large-gap covariance.
    """
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"{context} must be square; got {tuple(A.shape)}.")
    A_sym = symmetrize(A)
    eye = _eye_like(A_sym)
    diag = torch.diagonal(A_sym, dim1=-2, dim2=-1)
    scale = torch.mean(torch.abs(diag), dim=-1).detach().clamp_min(
        torch.finfo(A.dtype).eps
    )
    relative = torch.zeros_like(scale)
    stable = A_sym
    last_info: Optional[torch.Tensor] = None

    for attempt in range(max_tries):
        absolute = relative * scale
        stable = A_sym + absolute[..., None, None] * eye
        chol, info = torch.linalg.cholesky_ex(stable)
        last_info = info
        if not torch.any(info > 0):
            max_relative = float(torch.max(relative).detach().cpu()) if relative.numel() else 0.0
            max_absolute = float(torch.max(absolute).detach().cpu()) if absolute.numel() else 0.0
            stabilized_count = int(torch.sum(relative > 0).item())
            if stabilized_count:
                _SPD_STABILIZATION_COUNTS[context] = (
                    _SPD_STABILIZATION_COUNTS.get(context, 0) + stabilized_count
                )
                _SPD_MAX_RELATIVE_JITTER[context] = max(
                    _SPD_MAX_RELATIVE_JITTER.get(context, 0.0), max_relative
                )
                if max_relative >= 1e-6 and context not in _SPD_WARNED_CONTEXTS:
                    logging.warning(
                        "%s required relative diagonal jitter up to %.3g; further occurrences are counted silently.",
                        context, max_relative,
                    )
                    _SPD_WARNED_CONTEXTS.add(context)
            return SPDCholesky(stable, chol, max_absolute, max_relative)

        next_relative = float(base_relative_jitter) * (10.0 ** attempt)
        if next_relative > max_relative_jitter:
            break
        failing = info > 0
        relative = torch.where(
            failing,
            torch.full_like(relative, next_relative),
            relative,
        )

    failed = int(torch.sum(last_info > 0).item()) if last_info is not None else -1
    max_used = float(torch.max(relative).detach().cpu()) if relative.numel() else 0.0
    raise RuntimeError(
        f"Unable to factor {context}; {failed} batch member(s) remained non-SPD "
        f"after relative jitter up to {max_used:.3g}."
    )

def safe_cholesky(
    A: torch.Tensor,
    jitter: float = 1e-10,
    max_tries: int = 10,
) -> torch.Tensor:
    """Compatibility wrapper around the bounded, scale-aware factorization."""
    return factor_spd(
        A,
        base_relative_jitter=jitter,
        max_tries=max_tries,
        context="safe_cholesky input",
    ).chol


def spd_inverse_logdet(
    A: torch.Tensor,
    *,
    jitter: float = 1e-10,
    context: str = "SPD matrix",
) -> Tuple[torch.Tensor, torch.Tensor, SPDCholesky]:
    """Return inverse and log determinant from one common Cholesky factor."""
    fac = factor_spd(A, base_relative_jitter=jitter, context=context)
    inv_A = symmetrize(torch.cholesky_inverse(fac.chol))
    logdet = 2.0 * torch.sum(
        torch.log(torch.diagonal(fac.chol, dim1=-2, dim2=-1)), dim=-1
    )
    return inv_A, logdet, fac


def cholesky_spd_inverse(A: torch.Tensor, jitter: float = 1e-10) -> torch.Tensor:
    return spd_inverse_logdet(A, jitter=jitter, context="inverse input")[0]


def cholesky_right_solve_spd(
    M: torch.Tensor,
    A: torch.Tensor,
    jitter: float = 1e-10,
) -> torch.Tensor:
    """Compute M A^{-1} using one Cholesky solve."""
    fac = factor_spd(A, base_relative_jitter=jitter, context="right-solve matrix")
    X_t = torch.cholesky_solve(M.transpose(-1, -2), fac.chol)
    return X_t.transpose(-1, -2)


def spd_logdet(A: torch.Tensor, jitter: float = 1e-10) -> torch.Tensor:
    return spd_inverse_logdet(A, jitter=jitter, context="logdet input")[1]


def newton_schulz_inverse(
    A: torch.Tensor,
    num_iters: int = 10,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Batched Newton-Schulz approximate inverse for very large K."""
    I = _eye_like(A)
    frob_norm_sq = torch.sum(A * A, dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = A.transpose(-1, -2) / frob_norm_sq
    for _ in range(num_iters):
        X = torch.matmul(X, 2.0 * I - torch.matmul(A, X))
    return X


def adaptive_spd_inverse(
    A: torch.Tensor,
    *,
    ns_threshold: int = 256,
    ns_iters: int = 10,
    ns_tol: float = 1e-4,
    jitter: float = 1e-10,
    force_method: str = "auto",
) -> torch.Tensor:
    """Use Cholesky for small K and validated Newton-Schulz for large K.

    The Newton-Schulz path does not perform a Cholesky factorization first;
    otherwise it would pay the O(K^3) cost it is intended to avoid. A residual
    check triggers a safe Cholesky fallback when the approximation is inadequate.
    """
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("adaptive_spd_inverse expects square matrices.")
    method = force_method.lower()
    if method not in {"auto", "torch", "newton_schulz"}:
        raise ValueError("force_method must be auto, torch, or newton_schulz.")
    use_ns = method == "newton_schulz" or (method == "auto" and A.shape[-1] >= ns_threshold)
    if not use_ns:
        return cholesky_spd_inverse(A, jitter=jitter)

    A_sym = symmetrize(A)
    scale = torch.mean(
        torch.abs(torch.diagonal(A_sym, dim1=-2, dim2=-1)), dim=-1
    ).detach().clamp_min(torch.finfo(A.dtype).eps)
    A_stable = A_sym + float(jitter) * scale[..., None, None] * _eye_like(A_sym)
    X_ns = symmetrize(newton_schulz_inverse(A_stable, num_iters=ns_iters))
    with torch.no_grad():
        I = _eye_like(A_stable)
        residual = torch.linalg.norm(A_stable @ X_ns - I, dim=(-2, -1))
        denom = torch.linalg.norm(I, dim=(-2, -1)).clamp_min(1e-12)
        rel_residual = residual / denom
        ok = (
            bool(torch.isfinite(X_ns).all())
            and bool(torch.isfinite(rel_residual).all())
            and float(torch.max(rel_residual).cpu()) <= ns_tol
        )
    if ok:
        return X_ns
    return cholesky_spd_inverse(A, jitter=jitter)

def adaptive_right_solve_spd(
    M: torch.Tensor,
    A: torch.Tensor,
    *,
    ns_threshold: int = 256,
    ns_iters: int = 10,
    ns_tol: float = 1e-4,
    jitter: float = 1e-10,
    force_method: str = "auto",
) -> torch.Tensor:
    method = force_method.lower()
    use_ns = method == "newton_schulz" or (method == "auto" and A.shape[-1] >= ns_threshold)
    if not use_ns:
        return cholesky_right_solve_spd(M, A, jitter=jitter)
    return M @ adaptive_spd_inverse(
        A,
        ns_threshold=ns_threshold,
        ns_iters=ns_iters,
        ns_tol=ns_tol,
        jitter=jitter,
        force_method="newton_schulz",
    )


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns NaN rather than crashing for degenerate inputs."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive y."""
    y = y.clamp_min(torch.finfo(y.dtype).eps)
    return y + torch.log(-torch.expm1(-y))


def lowrank_gaussian_loglik(
    x: torch.Tensor,
    mean: torch.Tensor,
    P: torch.Tensor,
    Lambda: torch.Tensor,
    log_psi: torch.Tensor,
    *,
    obs_intercept: Optional[torch.Tensor] = None,
    jitter: float = 1e-10,
) -> Tuple[torch.Tensor, int]:
    """Log N(x; c+Lambda mean, diag(psi)+Lambda P Lambda') using K-dimensional algebra."""
    obs = torch.isfinite(x)
    m = int(obs.sum().item())
    if m == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype), 0
    H = Lambda[obs]
    centered = x if obs_intercept is None else x - obs_intercept
    y = centered[obs] - H @ mean
    logpsi = log_psi[obs]
    invpsi = safe_inverse_variance(logpsi)
    Ht_invR = H.T * invpsi.unsqueeze(0)
    obs_info = Ht_invR @ H
    q = Ht_invR @ y

    P_inv, logdet_P, _ = spd_inverse_logdet(P, jitter=jitter, context="predictive latent covariance")
    M_inv, logdet_M, _ = spd_inverse_logdet(
        symmetrize(P_inv + obs_info), jitter=jitter, context="predictive Woodbury precision"
    )
    quad = torch.sum(y.square() * invpsi) - q @ M_inv @ q
    logdet = torch.sum(logpsi) + logdet_P + logdet_M
    ll = -0.5 * (m * math.log(2.0 * math.pi) + logdet + quad)
    return ll, m

# ---------------------------------------------------------------------
# 1b. Anchor helpers
# ---------------------------------------------------------------------
def normalize_anchor_groups(
    K: int,
    D: int,
    *,
    anchor_items: Optional[Sequence[int]] = None,
    anchor_groups: Optional[Sequence[Sequence[int]]] = None,
) -> List[List[int]]:
    """Return validated anchor groups as a list of K nonempty row-index lists.

    Backward-compatible behavior:
      - anchor_groups=[[...], ...] gives multiple anchors per factor.
      - anchor_items=[a_0,...,a_{K-1}] gives one anchor per factor.
      - neither supplied defaults to the first K rows.

    An item may anchor at most one factor, because anchor rows are forced pure.
    """
    if anchor_groups is not None and anchor_items is not None:
        raise ValueError("Provide either anchor_groups or anchor_items, not both.")

    if anchor_groups is None:
        if anchor_items is None:
            anchor_items = list(range(K))
        if len(anchor_items) != K:
            raise ValueError(f"anchor_items must contain exactly K={K} entries.")
        groups = [[int(a)] for a in anchor_items]
    else:
        if len(anchor_groups) != K:
            raise ValueError(f"anchor_groups must contain exactly K={K} groups.")
        groups = [[int(a) for a in g] for g in anchor_groups]

    used = set()
    for r, group in enumerate(groups):
        if len(group) == 0:
            raise ValueError(f"Anchor group {r} is empty.")
        for a in group:
            if a < 0 or a >= D:
                raise ValueError(f"Anchor row index {a} is outside [0, {D}).")
            if a in used:
                raise ValueError(f"Anchor row index {a} appears in more than one anchor group.")
            used.add(a)
    return groups


def flatten_anchor_groups(anchor_groups: Sequence[Sequence[int]]) -> Tuple[List[int], List[int], List[int]]:
    """Return flat anchor rows, their assigned factors, and representative anchors."""
    flat_rows: List[int] = []
    flat_factors: List[int] = []
    representative_rows: List[int] = []
    for r, group in enumerate(anchor_groups):
        representative_rows.append(int(group[0]))
        for a in group:
            flat_rows.append(int(a))
            flat_factors.append(int(r))
    return flat_rows, flat_factors, representative_rows

# ---------------------------------------------------------------------
# 2. CLOUDS model
# ---------------------------------------------------------------------
class CLOUDS(nn.Module):
    """Continuous-time latent-factor state-space model with anchored loadings.

    The implementation uses a generalized EM algorithm.  The E-step is exact
    Gaussian smoothing; the M-step profiles linear trend/level coefficients and
    uses guarded gradient updates for the remaining parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        covar_dim: int,
        *,
        delta: float = 1e-6,
        theta_mode: str = "exact",
        anchor_items: Optional[Sequence[int]] = None,
        anchor_groups: Optional[Sequence[Sequence[int]]] = None,
        inverse_ns_threshold: int = 256,
        inverse_ns_iters: int = 10,
        inverse_ns_tol: float = 1e-4,
        inverse_force_method: str = "auto",
        jitter: float = 1e-10,
        omega_correlation: bool = True,
        diagonal_fix_omega: bool = True,
        include_latent_level: bool = False,
        learn_observation_intercept: bool = True,
        observation_intercept_prior: float = 0.01,
        min_anchor_loading: float = 1e-4,
        pca_randomized_threshold: int = 5_000_000,
        lambda_skew: float = 0.75,
        lambda_offdiag_G: float = 0.75,
        lambda_gamma_offdiag: float = 0.10,
        lambda_rate: float = 0.25,
        target_rate: float = 1.0,
        profile_ridge: float = 1.0,
    ) -> None:
        super().__init__()
        if theta_mode not in {"exact", "diagonal"}:
            raise ValueError("theta_mode must be 'exact' or 'diagonal'.")
        if obs_dim < latent_dim:
            raise ValueError("obs_dim must be at least latent_dim.")
        if covar_dim < 0:
            raise ValueError("covar_dim must be nonnegative.")

        self.D = int(obs_dim)
        self.K = int(latent_dim)
        self.C_dim = int(covar_dim)
        self.delta = float(delta)
        self.theta_mode = theta_mode
        self.inverse_ns_threshold = int(inverse_ns_threshold)
        self.inverse_ns_iters = int(inverse_ns_iters)
        self.inverse_ns_tol = float(inverse_ns_tol)
        self.inverse_force_method = str(inverse_force_method)
        self.jitter = float(jitter)
        self.omega_correlation = bool(omega_correlation)
        if self.theta_mode == "exact" and not self.omega_correlation:
            raise ValueError(
                "Exact mode requires omega_correlation=True so latent scale is identified. "
                "Use diagonal_fix_omega=False only for an explicitly scale-free diagonal study."
            )
        self.diagonal_fix_omega = bool(diagonal_fix_omega)
        self.include_latent_level = bool(include_latent_level)
        self.learn_observation_intercept = bool(learn_observation_intercept)
        self.observation_intercept_prior = float(observation_intercept_prior)
        if self.observation_intercept_prior < 0.0:
            raise ValueError("observation_intercept_prior must be nonnegative.")
        # A free item intercept and a free constant latent level are exactly
        # confounded through Lambda.  Keep the covariate-dependent latent level,
        # but free its constant term only when item intercepts are disabled.
        self.level_bias_free = self.include_latent_level and not self.learn_observation_intercept
        self.min_anchor_loading = float(min_anchor_loading)
        self.pca_randomized_threshold = int(pca_randomized_threshold)
        self.lambda_skew = float(lambda_skew)
        self.lambda_offdiag_G = float(lambda_offdiag_G)
        self.lambda_gamma_offdiag = float(lambda_gamma_offdiag)
        self.lambda_rate = float(lambda_rate)
        self.target_rate = float(target_rate)
        self.profile_ridge = float(profile_ridge)
        self.fit_history: List[Dict[str, float]] = []

        self.anchor_groups = normalize_anchor_groups(
            self.K,
            self.D,
            anchor_items=anchor_items,
            anchor_groups=anchor_groups,
        )
        flat_anchor_rows, flat_anchor_factors, representative_rows = flatten_anchor_groups(
            self.anchor_groups
        )

        if self.theta_mode == "exact":
            tril = torch.tril_indices(self.K, self.K)
            strict = torch.tril_indices(self.K, self.K, offset=-1)
            self.register_buffer("tril_rows", tril[0])
            self.register_buffer("tril_cols", tril[1])
            self.register_buffer("strict_rows", strict[0])
            self.register_buffer("strict_cols", strict[1])
            self.register_buffer("tril_is_diag", tril[0] == tril[1])

            Lg0 = torch.tril(torch.eye(self.K) + 0.05 * torch.randn(self.K, self.K))
            self.L_G_packed = nn.Parameter(Lg0[tril[0], tril[1]])
            # Omega is normalized to a correlation matrix.  Fixing the
            # unconstrained Cholesky diagonal to one removes K redundant row-scale
            # directions that normalization would otherwise erase.
            self.L_Omega_packed = nn.Parameter(0.05 * torch.randn(strict.shape[1]))
            self.gamma_skew_packed = nn.Parameter(0.02 * torch.randn(strict.shape[1]))
        else:
            self.log_rho = nn.Parameter(0.1 * torch.randn(self.K) - 1.0)
            if self.diagonal_fix_omega:
                self.register_buffer("log_omega", torch.zeros(self.K))
            else:
                self.log_omega = nn.Parameter(torch.zeros(self.K))

        self.Phi_int = nn.Parameter(0.05 * torch.randn(self.K, self.C_dim))
        self.alpha_bias = nn.Parameter(0.05 * torch.randn(self.K))
        if self.include_latent_level:
            self.Phi_level = nn.Parameter(0.05 * torch.randn(self.K, self.C_dim))
            if self.level_bias_free:
                self.level_bias = nn.Parameter(0.05 * torch.randn(self.K))
            else:
                self.register_buffer("level_bias", torch.zeros(self.K))
        else:
            self.register_buffer("Phi_level", torch.zeros(self.K, self.C_dim))
            self.register_buffer("level_bias", torch.zeros(self.K))

        self.Lambda_raw = nn.Parameter(0.1 * torch.randn(self.D, self.K))
        self.log_psi = nn.Parameter(torch.zeros(self.D))
        if self.learn_observation_intercept:
            self.obs_intercept = nn.Parameter(torch.zeros(self.D))
        else:
            self.register_buffer("obs_intercept", torch.zeros(self.D))

        self.register_buffer("anchor_idx", torch.tensor(flat_anchor_rows, dtype=torch.long))
        self.register_buffer(
            "anchor_factor_idx", torch.tensor(flat_anchor_factors, dtype=torch.long)
        )
        self.register_buffer("repr_anchor_idx", torch.tensor(representative_rows, dtype=torch.long))
        self.register_buffer("anchor_cols", torch.arange(self.K, dtype=torch.long))

        struct_mask = torch.ones(self.D, self.K)
        struct_mask[self.anchor_idx, :] = 0.0
        struct_mask[self.anchor_idx, self.anchor_factor_idx] = 1.0
        self.register_buffer("struct_mask", struct_mask)

        positivity_mask = torch.zeros(self.D, self.K, dtype=torch.bool)
        positivity_mask[self.anchor_idx, self.anchor_factor_idx] = True
        self.register_buffer("positivity_mask", positivity_mask)

    def _unpack_lower(self, packed: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.K, self.K, dtype=packed.dtype, device=packed.device)
        return out.index_put((self.tril_rows, self.tril_cols), packed)

    @torch.no_grad()
    def _copy_packed_lower(self, parameter: nn.Parameter, matrix: torch.Tensor) -> None:
        parameter.copy_(matrix[self.tril_rows, self.tril_cols])

    @torch.no_grad()
    def _copy_packed_strict_lower(self, parameter: nn.Parameter, matrix: torch.Tensor) -> None:
        parameter.copy_(matrix[self.strict_rows, self.strict_cols])

    @property
    def L_G(self) -> torch.Tensor:
        if self.theta_mode != "exact":
            raise AttributeError("L_G is available only in exact mode.")
        return self._unpack_lower(self.L_G_packed)

    @property
    def L_Omega_unc(self) -> torch.Tensor:
        if self.theta_mode != "exact":
            raise AttributeError("L_Omega_unc is available only in exact mode.")
        out = torch.eye(
            self.K, dtype=self.L_Omega_packed.dtype, device=self.L_Omega_packed.device
        )
        return out.index_put((self.strict_rows, self.strict_cols), self.L_Omega_packed)

    @property
    def gamma_skew(self) -> torch.Tensor:
        if self.theta_mode != "exact":
            raise AttributeError("gamma_skew is available only in exact mode.")
        out = torch.zeros(
            self.K,
            self.K,
            dtype=self.gamma_skew_packed.dtype,
            device=self.gamma_skew_packed.device,
        )
        return out.index_put((self.strict_rows, self.strict_cols), self.gamma_skew_packed)

    @property
    def Lambda(self) -> torch.Tensor:
        """Apply structural zeros and stable positive transforms only at anchor cells."""
        result = self.Lambda_raw * self.struct_mask
        anchor_raw = self.Lambda_raw[self.anchor_idx, self.anchor_factor_idx]
        anchor_values = F.softplus(anchor_raw) + self.min_anchor_loading
        return result.index_put((self.anchor_idx, self.anchor_factor_idx), anchor_values)

    def _spd_inverse(self, A: torch.Tensor) -> torch.Tensor:
        return adaptive_spd_inverse(
            A,
            ns_threshold=self.inverse_ns_threshold,
            ns_iters=self.inverse_ns_iters,
            ns_tol=self.inverse_ns_tol,
            jitter=self.jitter,
            force_method=self.inverse_force_method,
        )

    def _right_solve_spd(self, M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return adaptive_right_solve_spd(
            M,
            A,
            ns_threshold=self.inverse_ns_threshold,
            ns_iters=self.inverse_ns_iters,
            ns_tol=self.inverse_ns_tol,
            jitter=self.jitter,
            force_method=self.inverse_force_method,
        )

    def get_dynamics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.Lambda_raw.device
        dtype = self.Lambda_raw.dtype
        eye = torch.eye(self.K, device=device, dtype=dtype)

        if self.theta_mode == "exact":
            L_omega = self.L_Omega_unc
            Omega_raw = symmetrize(L_omega @ L_omega.T) + self.delta * eye
            Omega = (
                normalize_spd_to_correlation(Omega_raw, jitter=self.delta)
                if self.omega_correlation
                else Omega_raw
            )

            L_g = self.L_G
            S = symmetrize(0.5 * (L_g @ L_g.T)) + self.delta * eye
            lower_skew = self.gamma_skew
            R = lower_skew - lower_skew.T

            # Omega is SPD by construction, so the unmodified matrix should factor.
            # Using a direct Cholesky solve preserves Gamma Omega + Omega Gamma' = 2S.
            Gamma = torch.linalg.solve(Omega.T, (S + R).T).T
            G_consistent = torch.linalg.cholesky(symmetrize(2.0 * S))
            return Gamma, Omega, G_consistent

        rho = F.softplus(self.log_rho) + self.delta
        if self.diagonal_fix_omega:
            omega = torch.ones(self.K, dtype=dtype, device=device)
        else:
            omega = F.softplus(self.log_omega) + self.delta
        Gamma = torch.diag(rho)
        Omega = torch.diag(omega)
        G = torch.diag(torch.sqrt(2.0 * rho * omega))
        return Gamma, Omega, G

    @torch.no_grad()
    def get_identifiable_parameters(self) -> Dict[str, torch.Tensor]:
        Gamma_est, Omega_est, _ = self.get_dynamics()
        stds = torch.sqrt(torch.diag(Omega_est)).clamp_min(1e-12)
        D_scale = torch.diag(stds)
        D_inv = torch.diag(1.0 / stds)
        out = {
            "Omega_corr": symmetrize(D_inv @ Omega_est @ D_inv),
            "Gamma": D_inv @ Gamma_est @ D_scale,
            "Lambda": self.Lambda @ D_scale,
            "Phi": D_inv @ self.Phi_int,
            "alpha": D_inv @ self.alpha_bias,
            "latent_scale": stds,
        }
        if self.include_latent_level:
            out["Phi_level"] = D_inv @ self.Phi_level
            out["level_bias"] = D_inv @ self.level_bias
        out["obs_intercept"] = self.obs_intercept.detach().clone()
        return out

    def _extract_times(self, subj: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if "t_model" in subj:
            t_model = subj["t_model"]
        elif "t" in subj:
            t_model = subj["t"]
        elif "t_dyn" in subj and "t_trend" in subj:
            if not torch.allclose(subj["t_dyn"], subj["t_trend"], rtol=1e-6, atol=1e-8):
                raise ValueError("t_dyn and t_trend must be the same model-time vector.")
            t_model = subj["t_dyn"]
        elif "t_trend" in subj:
            t_model = subj["t_trend"]
        elif "t_dyn" in subj:
            t_model = subj["t_dyn"]
        else:
            raise KeyError("Subject must contain t_model, t, t_trend, or t_dyn.")
        return t_model, t_model

    def _subject_covariate_vector(self, u: torch.Tensor) -> torch.Tensor:
        """Return a time-invariant subject covariate, rejecting silent variation."""
        if u.ndim == 1:
            return u
        if u.ndim == 2:
            if u.shape[0] == 0:
                raise ValueError("Covariate matrix has no rows.")
            if u.shape[0] > 1 and not torch.allclose(
                u, u[0].unsqueeze(0).expand_as(u), rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    "CLOUDS currently assumes time-invariant covariates; rows of u differ. "
                    "Provide one vector per subject or extend the transition mean explicitly."
                )
            return u[0]
        raise ValueError("Latent covariate u must have shape [C] or [T, C].")

    def _mu_path(self, u: torch.Tensor, t_model: torch.Tensor) -> torch.Tensor:
        x_i = self._subject_covariate_vector(u)
        slope = x_i @ self.Phi_int.T + self.alpha_bias
        level = x_i @ self.Phi_level.T + self.level_bias
        return level.unsqueeze(0) + t_model.unsqueeze(1) * slope.unsqueeze(0)

    def _transition_matrices_from_dt(
        self,
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        dt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = int(dt.numel())
        if n == 0:
            empty = torch.empty(0, self.K, self.K, dtype=Gamma.dtype, device=Gamma.device)
            return empty, empty.clone()
        if torch.any(dt <= 0):
            raise ValueError("Subject times must be strictly increasing.")

        if self.theta_mode == "diagonal":
            rho = torch.diag(Gamma)
            omega = torch.diag(Omega)
            a = torch.exp(-dt.unsqueeze(1) * rho.unsqueeze(0))
            A = torch.diag_embed(a)
            q_diag = omega.unsqueeze(0) * (-torch.expm1(-2.0 * dt.unsqueeze(1) * rho.unsqueeze(0)))
            q_diag = q_diag.clamp_min(torch.finfo(q_diag.dtype).tiny)
            Q = torch.diag_embed(q_diag)
        else:
            A = torch.linalg.matrix_exp(-Gamma.unsqueeze(0) * dt.view(-1, 1, 1))
            Omega_b = Omega.unsqueeze(0).expand(n, self.K, self.K)
            Q = symmetrize(Omega_b - A @ Omega_b @ A.transpose(1, 2))
        return A, Q

    def get_subject_matrices(
        self,
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        u: torch.Tensor,
        t_dyn: torch.Tensor,
        t_trend: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t_trend is not None and not torch.allclose(t_dyn, t_trend, rtol=1e-6, atol=1e-8):
            raise ValueError("The dynamics and trend must use one common model time.")
        dt = t_dyn[1:] - t_dyn[:-1]
        A, Q = self._transition_matrices_from_dt(Gamma, Omega, dt)
        mu = self._mu_path(u, t_dyn)
        if A.shape[0] == 0:
            b = torch.empty(0, self.K, dtype=mu.dtype, device=mu.device)
        else:
            b = mu[1:] - (A @ mu[:-1].unsqueeze(-1)).squeeze(-1)
        return A, b, dt, self.Lambda, Q

    def get_all_subject_matrices(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch all matrix exponentials in one call and split by subject."""
        times: List[torch.Tensor] = []
        lengths: List[int] = []
        for subj in subjects_data:
            t, t2 = self._extract_times(subj)
            if not torch.allclose(t, t2, rtol=1e-6, atol=1e-8):
                raise ValueError("Inconsistent model times.")
            dt = t[1:] - t[:-1]
            if torch.any(dt <= 0):
                raise ValueError("Subject times must be strictly increasing.")
            times.append(t)
            lengths.append(int(dt.numel()))

        if sum(lengths) > 0:
            all_dt = torch.cat([t[1:] - t[:-1] for t in times])
            all_A, all_Q = self._transition_matrices_from_dt(Gamma, Omega, all_dt)
        else:
            all_A = torch.empty(0, self.K, self.K, device=Gamma.device, dtype=Gamma.dtype)
            all_Q = all_A.clone()

        out = []
        offset = 0
        Lambda = self.Lambda
        for subj, t, n in zip(subjects_data, times, lengths):
            A = all_A[offset : offset + n]
            Q = all_Q[offset : offset + n]
            dt = t[1:] - t[:-1]
            mu = self._mu_path(subj.get("x_xi", subj["u"]), t)
            b = mu[1:] - (A @ mu[:-1].unsqueeze(-1)).squeeze(-1) if n else torch.empty(
                0, self.K, dtype=t.dtype, device=t.device
            )
            out.append((A, b, dt, Lambda, Q))
            offset += n
        return out

    def _measurement_update(
        self,
        mean_pred: torch.Tensor,
        cov_pred: torch.Tensor,
        x_t: torch.Tensor,
        Lambda: torch.Tensor,
        full_obs_info: torch.Tensor,
        full_Ht_invR: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = torch.isfinite(x_t)
        if not torch.any(valid):
            return mean_pred, symmetrize(cov_pred)

        pred_info = self._spd_inverse(cov_pred)
        if bool(torch.all(valid)):
            obs_info = full_obs_info
            Ht_invR = full_Ht_invR
            x_valid = x_t - self.obs_intercept
        else:
            H = Lambda[valid]
            inv_psi = safe_inverse_variance(self.log_psi[valid])
            Ht_invR = H.T * inv_psi.unsqueeze(0)
            obs_info = Ht_invR @ H
            x_valid = x_t[valid] - self.obs_intercept[valid]

        filt_cov = self._spd_inverse(symmetrize(pred_info + obs_info))
        natural_mean = pred_info @ mean_pred + Ht_invR @ x_valid
        return filt_cov @ natural_mean, symmetrize(filt_cov)

    def kalman_smoother(
        self,
        x_obs: torch.Tensor,
        A_trans: torch.Tensor,
        b_shift: torch.Tensor,
        dt: torch.Tensor,
        Lambda: torch.Tensor,
        Q: torch.Tensor,
        init_mean: Optional[torch.Tensor] = None,
        init_cov: Optional[torch.Tensor] = None,
        full_obs_info: Optional[torch.Tensor] = None,
        full_Ht_invR: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del dt
        T = int(x_obs.shape[0])
        if T < 1:
            raise ValueError("Each subject must have at least one visit.")
        device, dtype = x_obs.device, x_obs.dtype
        eye = torch.eye(self.K, device=device, dtype=dtype)

        f_pred = torch.zeros(T, self.K, device=device, dtype=dtype)
        P_pred = torch.zeros(T, self.K, self.K, device=device, dtype=dtype)
        f_filt = torch.zeros_like(f_pred)
        P_filt = torch.zeros_like(P_pred)

        f_pred[0] = torch.zeros(self.K, device=device, dtype=dtype) if init_mean is None else init_mean
        P_pred[0] = eye if init_cov is None else symmetrize(init_cov)

        if full_Ht_invR is None or full_obs_info is None:
            inv_psi = safe_inverse_variance(self.log_psi)
            full_Ht_invR = Lambda.T * inv_psi.unsqueeze(0)
            full_obs_info = full_Ht_invR @ Lambda

        f_filt[0], P_filt[0] = self._measurement_update(
            f_pred[0], P_pred[0], x_obs[0], Lambda, full_obs_info, full_Ht_invR
        )
        for j in range(1, T):
            a = A_trans[j - 1]
            f_pred[j] = a @ f_filt[j - 1] + b_shift[j - 1]
            P_pred[j] = symmetrize(a @ P_filt[j - 1] @ a.T + Q[j - 1])
            f_filt[j], P_filt[j] = self._measurement_update(
                f_pred[j], P_pred[j], x_obs[j], Lambda, full_obs_info, full_Ht_invR
            )

        f_s = torch.zeros_like(f_filt)
        P_s = torch.zeros_like(P_filt)
        P_cross = torch.zeros_like(P_filt)
        f_s[-1], P_s[-1] = f_filt[-1], P_filt[-1]
        for j in range(T - 2, -1, -1):
            J = P_filt[j] @ A_trans[j].T @ self._spd_inverse(P_pred[j + 1])
            f_s[j] = f_filt[j] + J @ (f_s[j + 1] - f_pred[j + 1])
            P_s[j] = symmetrize(P_filt[j] + J @ (P_s[j + 1] - P_pred[j + 1]) @ J.T)
            P_cross[j + 1] = P_s[j + 1] @ J.T
        return f_s, P_s, P_cross

    def smooth_all_subjects(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        Gamma: Optional[torch.Tensor] = None,
        Omega: Optional[torch.Tensor] = None,
        Lambda: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if Gamma is None or Omega is None:
            Gamma, Omega, _ = self.get_dynamics()
        if Lambda is None:
            Lambda = self.Lambda
        matrices = self.get_all_subject_matrices(subjects_data, Gamma, Omega)
        inv_psi = safe_inverse_variance(self.log_psi)
        full_Ht_invR = Lambda.T * inv_psi.unsqueeze(0)
        full_obs_info = full_Ht_invR @ Lambda

        stats = []
        for subj, (A, b, dt, _, Q) in zip(subjects_data, matrices):
            t, _ = self._extract_times(subj)
            u = subj.get("x_xi", subj["u"])
            mu = self._mu_path(u, t)
            stats.append(
                self.kalman_smoother(
                    subj["x"], A, b, dt, Lambda, Q,
                    init_mean=mu[0], init_cov=Omega,
                    full_obs_info=full_obs_info, full_Ht_invR=full_Ht_invR,
                )
            )
        return stats

    @torch.no_grad()
    def _apply_exact_stage_constraints(self, stage: str = "full") -> None:
        """Apply numerical bounds and, in exact mode, continuation constraints."""
        self.log_psi.clamp_(-20.0, 20.0)
        self.Lambda_raw.clamp_(-50.0, 50.0)
        self.Phi_int.clamp_(-50.0, 50.0)
        self.alpha_bias.clamp_(-50.0, 50.0)
        self.obs_intercept.clamp_(-1e4, 1e4)
        if self.include_latent_level:
            self.Phi_level.clamp_(-50.0, 50.0)
            self.level_bias.clamp_(-50.0, 50.0)

        if self.theta_mode == "diagonal":
            self.log_rho.clamp_(-30.0, 30.0)
            if not self.diagonal_fix_omega:
                self.log_omega.clamp_(-30.0, 30.0)
            return

        self.L_G_packed.clamp_(-50.0, 50.0)
        self.L_Omega_packed.clamp_(-50.0, 50.0)
        self.gamma_skew_packed.clamp_(-50.0, 50.0)
        stage_l = stage.lower()
        if stage_l in {"diag", "diagonal", "diag_dynamics"}:
            self.gamma_skew_packed.zero_()
            self.L_G_packed[~self.tril_is_diag] = 0.0
            self.L_Omega_packed.zero_()
        elif stage_l in {"reversible", "symmetric", "no_skew"}:
            self.gamma_skew_packed.zero_()
        elif stage_l not in {"full", "exact"}:
            raise ValueError("exact_stage must be diagonal, reversible, or full.")

    @staticmethod
    def _design_matrix_for_beta(C_mat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        K = C_mat.shape[0]
        P = z.numel()
        return torch.einsum("ar,p->arp", C_mat, z).reshape(K, K * P)

    def _trend_parameters(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = [self.Phi_int, self.alpha_bias]
        if self.include_latent_level:
            out.append(self.Phi_level)
            if self.level_bias_free:
                out.append(self.level_bias)
        return out

    @torch.no_grad()
    def profile_linear_trend_parameters(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        *,
        ridge: Optional[float] = None,
    ) -> None:
        """Profile slope and identifiable latent-level coefficients by ridge GLS.

        A constant latent level and a free item intercept are observationally
        confounded.  With item intercepts enabled, this update profiles only the
        covariate-dependent level block; disabling item intercepts also frees the
        constant latent-level term.
        """
        ridge = self.profile_ridge if ridge is None else float(ridge)
        device, dtype = self.Lambda_raw.device, self.Lambda_raw.dtype
        slope_p = self.C_dim + 1
        level_p = self.C_dim + (1 if self.level_bias_free else 0)
        slope_width = self.K * slope_p
        level_width = self.K * level_p if self.include_latent_level and level_p > 0 else 0
        n_params = slope_width + level_width
        eye = torch.eye(self.K, device=device, dtype=dtype)
        normal = ridge * torch.eye(n_params, device=device, dtype=dtype)
        rhs = torch.zeros(n_params, device=device, dtype=dtype)
        Omega_inv = self._spd_inverse(Omega)
        matrices = self.get_all_subject_matrices(subjects_data, Gamma, Omega)

        def make_H(
            C_level: torch.Tensor,
            C_slope: torch.Tensor,
            z_level: Optional[torch.Tensor],
            z_slope: torch.Tensor,
        ) -> torch.Tensor:
            Hs = self._design_matrix_for_beta(C_slope, z_slope)
            if level_width == 0 or z_level is None:
                return Hs
            Hl = self._design_matrix_for_beta(C_level, z_level)
            return torch.cat([Hl, Hs], dim=1)

        for subj, stats, mats in zip(subjects_data, smoothed_stats, matrices):
            f_s, _, _ = stats
            A, _, _, _, Q = mats
            t, _ = self._extract_times(subj)
            x_i = self._subject_covariate_vector(subj.get("x_xi", subj["u"]))
            x_i = x_i.to(device=device, dtype=dtype)
            z_slope = torch.cat([x_i, torch.ones(1, device=device, dtype=dtype)])
            if level_width:
                z_level = (
                    torch.cat([x_i, torch.ones(1, device=device, dtype=dtype)])
                    if self.level_bias_free else x_i
                )
            else:
                z_level = None

            H0 = make_H(eye, t[0] * eye, z_level, z_slope)
            normal = normal + H0.T @ Omega_inv @ H0
            rhs = rhs + H0.T @ Omega_inv @ f_s[0]

            if A.shape[0]:
                Q_inv = self._spd_inverse(Q)
                for j in range(1, f_s.shape[0]):
                    a = A[j - 1]
                    H = make_H(
                        eye - a,
                        t[j] * eye - t[j - 1] * a,
                        z_level,
                        z_slope,
                    )
                    y = f_s[j] - a @ f_s[j - 1]
                    W = Q_inv[j - 1]
                    normal = normal + H.T @ W @ H
                    rhs = rhs + H.T @ W @ y

        fac = factor_spd(
            normal, base_relative_jitter=self.jitter, context="profile normal equations"
        )
        theta = torch.cholesky_solve(rhs.unsqueeze(1), fac.chol).squeeze(1)
        offset = 0
        if level_width:
            B0 = theta[:level_width].reshape(self.K, level_p)
            self.Phi_level.copy_(B0[:, : self.C_dim])
            if self.level_bias_free:
                self.level_bias.copy_(B0[:, self.C_dim])
            offset = level_width
        elif self.include_latent_level:
            self.Phi_level.zero_()
            self.level_bias.zero_()
        B1 = theta[offset : offset + slope_width].reshape(self.K, slope_p)
        self.Phi_int.copy_(B1[:, : self.C_dim])
        self.alpha_bias.copy_(B1[:, self.C_dim])

    def profile_linear_trend_update(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
    ) -> None:
        self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)

    @torch.no_grad()
    def _assign_lambda_from_matrix(self, Lambda_target: torch.Tensor) -> None:
        target = Lambda_target.to(self.Lambda_raw).clone()
        # One factor-wide sign flip is legitimate.  Enforce positive hard anchors
        # only after orienting by the aggregate sign of each anchor group.
        for r, group in enumerate(self.anchor_groups):
            idx = torch.tensor(group, dtype=torch.long, device=target.device)
            if float(torch.sum(target[idx, r]).item()) < 0.0:
                target[:, r].mul_(-1.0)
            target[idx, r] = torch.abs(target[idx, r]).clamp_min(
                self.min_anchor_loading * 2.0
            )

        self.Lambda_raw.copy_(target)
        self.Lambda_raw[self.anchor_idx, :] = 0.0
        anchor_values = target[self.anchor_idx, self.anchor_factor_idx]
        transformed = (anchor_values - self.min_anchor_loading).clamp_min(
            torch.finfo(target.dtype).eps
        )
        self.Lambda_raw[self.anchor_idx, self.anchor_factor_idx] = inverse_softplus(transformed)

    @torch.no_grad()
    def initialize_exact_from_diagonal_model(self, diagonal_model: "CLOUDS") -> None:
        if self.theta_mode != "exact" or diagonal_model.theta_mode != "diagonal":
            raise ValueError("Requires an exact target and diagonal source model.")
        if (self.D, self.K, self.C_dim) != (
            diagonal_model.D,
            diagonal_model.K,
            diagonal_model.C_dim,
        ):
            raise ValueError("Exact and diagonal models must have matching dimensions.")
        ident = diagonal_model.get_identifiable_parameters()
        self._assign_lambda_from_matrix(ident["Lambda"])
        self.log_psi.copy_(diagonal_model.log_psi.to(self.log_psi))
        self.obs_intercept.copy_(diagonal_model.obs_intercept.to(self.obs_intercept))
        self.Phi_int.copy_(ident["Phi"].to(self.Phi_int))
        self.alpha_bias.copy_(ident["alpha"].to(self.alpha_bias))
        if self.include_latent_level and diagonal_model.include_latent_level:
            self.Phi_level.copy_(ident["Phi_level"].to(self.Phi_level))
            if self.level_bias_free:
                self.level_bias.copy_(ident["level_bias"].to(self.level_bias))

        rho = torch.diag(ident["Gamma"].to(self.Lambda_raw)).clamp_min(self.delta)
        self.L_Omega_packed.zero_()
        self.gamma_skew_packed.zero_()
        Lg = torch.diag(torch.sqrt((2.0 * rho - 2.0 * self.delta).clamp_min(1e-8)))
        self._copy_packed_lower(self.L_G_packed, Lg)
        self._apply_exact_stage_constraints("diagonal")

    def _log_prior_components(
        self,
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        Lambda: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = Lambda.device, Lambda.dtype
        temporal = torch.zeros((), device=device, dtype=dtype)
        spatial = torch.zeros((), device=device, dtype=dtype)

        if self.theta_mode == "exact":
            Omega_corr = normalize_spd_to_correlation(Omega, jitter=self.delta)
            temporal = temporal + 0.5 * spd_logdet(Omega_corr, jitter=self.jitter)
            R = self.gamma_skew - self.gamma_skew.T
            temporal = temporal - self.lambda_skew * 0.5 * torch.sum(torch.abs(R))
            temporal = temporal - self.lambda_offdiag_G * torch.sum(
                torch.abs(self.L_G[self.strict_rows, self.strict_cols])
            )
            offdiag = Gamma - torch.diag(torch.diag(Gamma))
            temporal = temporal - self.lambda_gamma_offdiag * torch.sum(torch.abs(offdiag))
            mean_rate = torch.trace(Gamma) / float(self.K)
            temporal = temporal - self.lambda_rate * (mean_rate - self.target_rate).square()
        else:
            temporal = temporal - 0.5 * torch.sum(self.log_rho.square())
            if not self.diagonal_fix_omega:
                temporal = temporal - 0.5 * torch.sum(self.log_omega.square())

        trend_norm = torch.sum(self.Phi_int.square()) + torch.sum(self.alpha_bias.square())
        if self.include_latent_level:
            trend_norm = trend_norm + torch.sum(self.Phi_level.square())
            if self.level_bias_free:
                trend_norm = trend_norm + torch.sum(self.level_bias.square())
        temporal = temporal - 0.5 * self.profile_ridge * trend_norm

        active_lambda = Lambda[self.struct_mask == 1]
        spatial = spatial - 0.5 * torch.sum(active_lambda.square())
        spatial = spatial - 0.5 * torch.sum(self.log_psi.square())
        if self.learn_observation_intercept and self.observation_intercept_prior > 0.0:
            spatial = spatial - 0.5 * self.observation_intercept_prior * torch.sum(
                self.obs_intercept.square()
            )
        return temporal, spatial

    def expected_complete_log_posterior_vectorized(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        Lambda: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = Lambda.device, Lambda.dtype
        ll_obs = torch.zeros((), device=device, dtype=dtype)
        ll_lat = torch.zeros((), device=device, dtype=dtype)

        inv_psi_full = safe_inverse_variance(self.log_psi)
        L_Psi_L_full = Lambda.T @ (inv_psi_full.unsqueeze(1) * Lambda)
        sum_log_psi_full = torch.sum(self.log_psi)
        Omega_inv, logdet_Omega, _ = spd_inverse_logdet(
            Omega, jitter=self.jitter, context="stationary covariance"
        )
        matrices = self.get_all_subject_matrices(subjects_data, Gamma, Omega)

        for subj, stats, mats in zip(subjects_data, smoothed_stats, matrices):
            x_obs = subj["x"]
            x_centered = x_obs - self.obs_intercept.unsqueeze(0)
            f_s, P_s, P_cross = stats
            A, b_shift, _, _, Q = mats
            t, _ = self._extract_times(subj)
            mu0 = self._mu_path(subj.get("x_xi", subj["u"]), t)[0]
            diff0 = f_s[0] - mu0
            M0 = P_s[0] + torch.outer(diff0, diff0)
            ll_lat = ll_lat - 0.5 * logdet_Omega - 0.5 * torch.sum(Omega_inv.T * M0)

            finite = torch.isfinite(x_obs)
            full_rows = torch.all(finite, dim=1)
            partial_rows = torch.any(finite, dim=1) & (~full_rows)
            if torch.any(full_rows):
                x_v, f_v, P_v = x_centered[full_rows], f_s[full_rows], P_s[full_rows]
                trace_E = torch.sum(P_v * L_Psi_L_full.unsqueeze(0), dim=(1, 2))
                trace_E = trace_E + torch.sum(f_v * (f_v @ L_Psi_L_full), dim=1)
                fitted = f_v @ Lambda.T
                term = torch.sum(x_v.square() * inv_psi_full, dim=1)
                term = term - 2.0 * torch.sum(x_v * fitted * inv_psi_full, dim=1)
                term = term + trace_E
                ll_obs = ll_obs + torch.sum(-0.5 * term - 0.5 * sum_log_psi_full)

            for j in torch.where(partial_rows)[0]:
                valid = finite[j]
                x_j = x_centered[j, valid]
                H = Lambda[valid]
                inv_psi = safe_inverse_variance(self.log_psi[valid])
                info = H.T @ (inv_psi.unsqueeze(1) * H)
                m = f_s[j]
                second = P_s[j] + torch.outer(m, m)
                term = torch.sum(x_j.square() * inv_psi)
                term = term - 2.0 * torch.sum(x_j * (H @ m) * inv_psi)
                term = term + torch.sum(second * info)
                ll_obs = ll_obs - 0.5 * term - 0.5 * torch.sum(self.log_psi[valid])

            if A.shape[0] == 0:
                continue
            Q_inv, logdet_Q, _ = spd_inverse_logdet(
                Q, jitter=self.jitter, context="transition covariance"
            )
            f_j, f_prev = f_s[1:], f_s[:-1]
            E_jj = P_s[1:] + f_j.unsqueeze(-1) @ f_j.unsqueeze(1)
            E_jprev = P_cross[1:] + f_j.unsqueeze(-1) @ f_prev.unsqueeze(1)
            E_prev = P_s[:-1] + f_prev.unsqueeze(-1) @ f_prev.unsqueeze(1)
            AT = A.transpose(1, 2)
            M = E_jj - E_jprev @ AT - A @ E_jprev.transpose(1, 2) + A @ E_prev @ AT

            bcol, brow = b_shift.unsqueeze(-1), b_shift.unsqueeze(1)
            M = M - f_j.unsqueeze(-1) @ brow - bcol @ f_j.unsqueeze(1)
            M = M + A @ (f_prev.unsqueeze(-1) @ brow)
            M = M + bcol @ (f_prev.unsqueeze(1) @ AT)
            M = symmetrize(M + bcol @ brow)
            trace = torch.sum(Q_inv.transpose(1, 2) * M, dim=(1, 2))
            ll_lat = ll_lat + torch.sum(-0.5 * logdet_Q - 0.5 * trace)

        temporal_prior, spatial_prior = self._log_prior_components(Gamma, Omega, Lambda)
        return ll_obs + ll_lat + temporal_prior + spatial_prior

    @torch.no_grad()
    def observed_data_log_posterior(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, int]:
        """Exact observed-data log posterior from Kalman prediction innovations."""
        Gamma, Omega, _ = self.get_dynamics()
        Lambda = self.Lambda
        matrices = self.get_all_subject_matrices(subjects_data, Gamma, Omega)
        inv_psi = safe_inverse_variance(self.log_psi)
        full_Ht_invR = Lambda.T * inv_psi.unsqueeze(0)
        full_obs_info = full_Ht_invR @ Lambda
        total = torch.zeros((), dtype=Lambda.dtype, device=Lambda.device)
        cells = 0

        for subj, mats in zip(subjects_data, matrices):
            A, b, _, _, Q = mats
            t, _ = self._extract_times(subj)
            mu = self._mu_path(subj.get("x_xi", subj["u"]), t)
            f_pred, P_pred = mu[0], Omega
            for j in range(subj["x"].shape[0]):
                ll, n = lowrank_gaussian_loglik(
                    subj["x"][j], f_pred, P_pred, Lambda, self.log_psi,
                    obs_intercept=self.obs_intercept, jitter=self.jitter,
                )
                total = total + ll
                cells += n
                f_filt, P_filt = self._measurement_update(
                    f_pred, P_pred, subj["x"][j], Lambda, full_obs_info, full_Ht_invR
                )
                if j + 1 < subj["x"].shape[0]:
                    f_pred = A[j] @ f_filt + b[j]
                    P_pred = symmetrize(A[j] @ P_filt @ A[j].T + Q[j])

        temporal_prior, spatial_prior = self._log_prior_components(Gamma, Omega, Lambda)
        return total + temporal_prior + spatial_prior, cells

    def _truncated_svd(
        self, X: torch.Tensor, rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return truncated_svd_matrix(
            X, rank, randomized_threshold=self.pca_randomized_threshold
        )

    @torch.no_grad()
    def pca_warm_start(self, subjects_data: Sequence[Dict[str, torch.Tensor]]) -> None:
        x_all = torch.cat([s["x"] for s in subjects_data], dim=0)
        x_valid = x_all[torch.any(torch.isfinite(x_all), dim=1)]
        if x_valid.shape[0] <= self.K:
            raise ValueError("Not enough observed rows for PCA warm start.")
        means = torch.nanmean(x_valid, dim=0)
        means = torch.where(torch.isfinite(means), means, torch.zeros_like(means))
        X = torch.where(torch.isfinite(x_valid), x_valid, means.unsqueeze(0)) - means.unsqueeze(0)
        U, S, Vh = self._truncated_svd(X, self.K)
        # S contains singular values; loading norms are S/sqrt(n-1), not sqrt(S/(n-1)).
        Lambda_pca = Vh.T * (S / math.sqrt(max(X.shape[0] - 1, 1)))
        A_pca = Lambda_pca[self.repr_anchor_idx]
        target_scales = torch.diag(torch.norm(A_pca, dim=1).clamp_min(1e-4))
        W = torch.linalg.pinv(A_pca) @ target_scales
        self._assign_lambda_from_matrix(Lambda_pca @ W)

        approx = (U * S.unsqueeze(0)) @ Vh
        psi = torch.mean((X - approx).square(), dim=0).clamp_min(1e-6)
        self.log_psi.copy_(torch.log(psi))
        if self.learn_observation_intercept:
            self.obs_intercept.copy_(means)
        else:
            self.obs_intercept.zero_()
        self.Phi_int.zero_()
        self.alpha_bias.zero_()
        if self.include_latent_level:
            self.Phi_level.zero_()
            self.level_bias.zero_()

    @torch.no_grad()
    def _randomize_temporal_parameters(self) -> None:
        if self.theta_mode == "exact":
            Lg = torch.tril(0.05 * torch.randn(self.K, self.K, device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype) + torch.eye(self.K, device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype))
            self._copy_packed_lower(self.L_G_packed, Lg)
            self.L_Omega_packed.normal_(0.0, 0.05)
            self.gamma_skew_packed.normal_(0.0, 0.02)
        else:
            self.log_rho.normal_(0.0, 0.15)
            if not self.diagonal_fix_omega:
                self.log_omega.zero_()
        self.Phi_int.normal_(0.0, 0.05)
        self.alpha_bias.normal_(0.0, 0.05)
        if self.include_latent_level:
            self.Phi_level.normal_(0.0, 0.05)
            if self.level_bias_free:
                self.level_bias.normal_(0.0, 0.05)

    def _parameter_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        spatial_names = {"Lambda_raw", "log_psi", "obs_intercept"}
        temporal, spatial = [], []
        for name, param in self.named_parameters():
            (spatial if name in spatial_names else temporal).append(param)
        return temporal, spatial

    @staticmethod
    def _set_requires_grad(params: Iterable[nn.Parameter], value: bool) -> None:
        for p in params:
            p.requires_grad_(value)

    def _objective_units(self, subjects_data: Sequence[Dict[str, torch.Tensor]]) -> Tuple[float, float]:
        obs = sum(int(torch.isfinite(s["x"]).sum().item()) for s in subjects_data)
        latent = sum(max(1, int(s["x"].shape[0])) * self.K for s in subjects_data)
        return float(max(obs, 1)), float(max(latent, 1))

    @staticmethod
    def _snapshot(params: Sequence[nn.Parameter]) -> List[torch.Tensor]:
        return [p.detach().clone() for p in params]

    @staticmethod
    @torch.no_grad()
    def _restore(params: Sequence[nn.Parameter], snapshot: Sequence[torch.Tensor]) -> None:
        for p, value in zip(params, snapshot):
            p.copy_(value)

    @staticmethod
    def _unique_params(params: Sequence[nn.Parameter]) -> List[nn.Parameter]:
        seen = set()
        out = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
        return out

    @staticmethod
    def _scale_and_clip_gradients(
        temporal_params: Sequence[nn.Parameter],
        spatial_params: Sequence[nn.Parameter],
        *,
        latent_normalizer: float,
        observed_normalizer: float,
        grad_clip: float,
    ) -> Tuple[float, float]:
        for p in temporal_params:
            if p.grad is not None:
                p.grad.div_(latent_normalizer)
        for p in spatial_params:
            if p.grad is not None:
                p.grad.div_(observed_normalizer)
        t_norm = float(torch.nn.utils.clip_grad_norm_(temporal_params, grad_clip)) if temporal_params else 0.0
        s_norm = float(torch.nn.utils.clip_grad_norm_(spatial_params, grad_clip)) if spatial_params else 0.0
        return t_norm, s_norm

    def _q_value(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        dynamics_only: bool,
    ) -> torch.Tensor:
        Gamma, Omega, _ = self.get_dynamics()
        Lambda = self.Lambda.detach() if dynamics_only else self.Lambda
        return self.expected_complete_log_posterior_vectorized(
            subjects_data, smoothed_stats, Gamma, Omega, Lambda
        )

    def refine_temporal_lbfgs(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        *,
        max_iter: int = 20,
        lr: float = 0.5,
        profile_linear_mstep: bool = True,
        exact_stage: str = "full",
    ) -> None:
        temporal, spatial = self._parameter_groups()
        trend = self._trend_parameters()
        trend_ids = {id(p) for p in trend}
        fit_params = [p for p in temporal if id(p) not in trend_ids] if profile_linear_mstep else temporal
        _, latent_norm = self._objective_units(subjects_data)
        self._set_requires_grad(spatial, False)
        self._set_requires_grad(temporal, True)
        if profile_linear_mstep:
            self._set_requires_grad(trend, False)

        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
            if profile_linear_mstep:
                self.profile_linear_trend_parameters(subjects_data, stats, Gamma, Omega)

        opt = optim.LBFGS(
            fit_params, lr=lr, max_iter=max_iter, history_size=25,
            line_search_fn="strong_wolfe", tolerance_grad=1e-8, tolerance_change=1e-10,
        )

        def closure() -> torch.Tensor:
            self.zero_grad(set_to_none=True)
            q = self._q_value(subjects_data, stats, dynamics_only=True)
            loss = -q / latent_norm
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite LBFGS loss.")
            loss.backward()
            return loss

        opt.step(closure)
        self._apply_exact_stage_constraints(exact_stage)

        # Dynamics changed during LBFGS, so re-run the E-step and reprofile the
        # linear block before any final smoothing/evaluation.
        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
            if profile_linear_mstep:
                self.profile_linear_trend_parameters(subjects_data, stats, Gamma, Omega)
        self._set_requires_grad(spatial, True)
        self._set_requires_grad(temporal, True)

    def fit_em_multistart(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        *,
        num_em_epochs: int = 40,
        warmup_epochs: int = 15,
        m_step_iters: int = 20,
        lr: float = 0.01,
        n_starts: int = 5,
        burn_in_epochs: int = 10,
        grad_clip: float = 2.0,
        use_pca_warm_start: bool = True,
        randomize_temporal_starts: bool = True,
        exact_stage: str = "full",
        profile_linear_mstep: bool = True,
        lbfgs_refine: bool = False,
        lbfgs_max_iter: int = 20,
        enforce_monotone_q: bool = True,
        monotone_tolerance: float = 1e-8,
        convergence_tol: float = 1e-4,
        convergence_patience: int = 3,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generalized EM with observed-likelihood start ranking and guarded updates."""
        if not subjects_data:
            raise ValueError("subjects_data is empty.")
        temporal, spatial = self._parameter_groups()
        trend = self._trend_parameters()
        trend_ids = {id(p) for p in trend}
        temporal_fit = [p for p in temporal if id(p) not in trend_ids] if profile_linear_mstep else temporal
        obs_norm, latent_norm = self._objective_units(subjects_data)
        n_starts = max(1, int(n_starts))

        if use_pca_warm_start:
            self.pca_warm_start(subjects_data)
        lambda_init = self.Lambda_raw.detach().clone()
        psi_init = self.log_psi.detach().clone()

        best_score = -float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        local_history: List[Dict[str, float]] = []

        for start in range(n_starts):
            if randomize_temporal_starts or start > 0:
                self._randomize_temporal_parameters()
            with torch.no_grad():
                self.Lambda_raw.copy_(lambda_init)
                self.log_psi.copy_(psi_init)
                if randomize_temporal_starts:
                    self.Phi_int.zero_()
                    self.alpha_bias.zero_()
                    if self.include_latent_level:
                        self.Phi_level.zero_()
                        self.level_bias.zero_()
                self._apply_exact_stage_constraints(exact_stage)

            self._set_requires_grad(spatial, False)
            self._set_requires_grad(temporal, True)
            if profile_linear_mstep:
                self._set_requires_grad(trend, False)
            opt = optim.Adam(temporal_fit, lr=lr)

            for burn_epoch in range(max(1, burn_in_epochs)):
                with torch.no_grad():
                    Gamma, Omega, _ = self.get_dynamics()
                    stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
                    if profile_linear_mstep:
                        self.profile_linear_trend_parameters(subjects_data, stats, Gamma, Omega)
                        self._apply_exact_stage_constraints(exact_stage)
                    q_before = float(self._q_value(subjects_data, stats, dynamics_only=True).cpu())

                snap_params = self._unique_params(temporal)
                snap = self._snapshot(snap_params)
                for _ in range(max(1, m_step_iters)):
                    self.zero_grad(set_to_none=True)
                    opt.zero_grad(set_to_none=True)
                    q = self._q_value(subjects_data, stats, dynamics_only=True)
                    if not torch.isfinite(q):
                        raise FloatingPointError("Non-finite burn-in Q function.")
                    (-q).backward()
                    self._scale_and_clip_gradients(
                        temporal_fit, [], latent_normalizer=latent_norm,
                        observed_normalizer=obs_norm, grad_clip=grad_clip,
                    )
                    opt.step()
                    with torch.no_grad():
                        self._apply_exact_stage_constraints(exact_stage)

                with torch.no_grad():
                    q_after_t = self._q_value(subjects_data, stats, dynamics_only=True)
                    q_after = float(q_after_t.cpu()) if torch.isfinite(q_after_t) else -float("inf")
                tolerance = monotone_tolerance * (1.0 + abs(q_before))
                if enforce_monotone_q and q_after + tolerance < q_before:
                    self._restore(snap_params, snap)
                    opt.state.clear()
                    for group in opt.param_groups:
                        group["lr"] *= 0.5
                    q_after = q_before
                local_history.append({
                    "stage": float(start), "epoch": float(burn_epoch),
                    "q_before": q_before, "q_after": q_after, "burn_in": 1.0,
                })

            with torch.no_grad():
                Gamma, Omega, _ = self.get_dynamics()
                stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
                if profile_linear_mstep:
                    self.profile_linear_trend_parameters(subjects_data, stats, Gamma, Omega)
                score_t, score_cells = self.observed_data_log_posterior(subjects_data)
                score = float(score_t.cpu()) / float(max(score_cells, 1))
            if math.isfinite(score) and score > best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        if best_state is None:
            raise RuntimeError("No valid temporal start completed.")
        self.load_state_dict(best_state)
        self._set_requires_grad(spatial, True)
        self._set_requires_grad(temporal, True)

        opt_temporal = optim.Adam(temporal_fit, lr=lr)
        opt_joint = optim.Adam([
            {"params": temporal_fit, "lr": lr},
            {"params": spatial, "lr": lr * 0.05},
        ])

        previous_obs: Optional[float] = None
        stable_epochs = 0
        remaining = max(int(num_em_epochs) - max(1, int(burn_in_epochs)), 0)
        for epoch in range(remaining):
            dynamics_only = epoch < int(warmup_epochs)
            self._set_requires_grad(spatial, not dynamics_only)
            self._set_requires_grad(temporal, True)
            if profile_linear_mstep:
                self._set_requires_grad(trend, False)
            active_opt = opt_temporal if dynamics_only else opt_joint
            active_spatial = [] if dynamics_only else spatial
            snapshot_params = self._unique_params(temporal + active_spatial)

            with torch.no_grad():
                Gamma, Omega, _ = self.get_dynamics()
                stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
                if profile_linear_mstep:
                    self.profile_linear_trend_parameters(subjects_data, stats, Gamma, Omega)
                    self._apply_exact_stage_constraints(exact_stage)
                q_before = float(self._q_value(subjects_data, stats, dynamics_only=dynamics_only).cpu())
            snapshot = self._snapshot(snapshot_params)

            t_grad = s_grad = 0.0
            for _ in range(max(1, m_step_iters)):
                self.zero_grad(set_to_none=True)
                active_opt.zero_grad(set_to_none=True)
                q = self._q_value(subjects_data, stats, dynamics_only=dynamics_only)
                if not torch.isfinite(q):
                    raise FloatingPointError("Non-finite EM Q function.")
                (-q).backward()
                t_grad, s_grad = self._scale_and_clip_gradients(
                    temporal_fit, active_spatial,
                    latent_normalizer=latent_norm, observed_normalizer=obs_norm,
                    grad_clip=grad_clip,
                )
                active_opt.step()
                with torch.no_grad():
                    self._apply_exact_stage_constraints(exact_stage)

            with torch.no_grad():
                q_after_t = self._q_value(subjects_data, stats, dynamics_only=dynamics_only)
                q_after = float(q_after_t.cpu()) if torch.isfinite(q_after_t) else -float("inf")
            tolerance = monotone_tolerance * (1.0 + abs(q_before))
            rolled_back = False
            if enforce_monotone_q and q_after + tolerance < q_before:
                self._restore(snapshot_params, snapshot)
                active_opt.state.clear()
                for group in active_opt.param_groups:
                    group["lr"] *= 0.5
                q_after = q_before
                rolled_back = True

            with torch.no_grad():
                obs_t, obs_cells = self.observed_data_log_posterior(subjects_data)
                obs_per_cell = float(obs_t.cpu()) / float(max(obs_cells, 1))
            local_history.append({
                "stage": 1000.0, "epoch": float(epoch), "q_before": q_before,
                "q_after": q_after, "observed_logpost_per_cell": obs_per_cell,
                "temporal_grad_norm": t_grad, "spatial_grad_norm": s_grad,
                "rolled_back": float(rolled_back), "burn_in": 0.0,
            })

            if previous_obs is not None and math.isfinite(obs_per_cell):
                relative = abs(obs_per_cell - previous_obs) / max(1.0, abs(previous_obs))
                stable_epochs = stable_epochs + 1 if relative < convergence_tol else 0
                if stable_epochs >= max(1, convergence_patience):
                    break
            previous_obs = obs_per_cell

        if lbfgs_refine and self.theta_mode == "exact" and lbfgs_max_iter > 0:
            self.refine_temporal_lbfgs(
                subjects_data, max_iter=lbfgs_max_iter,
                profile_linear_mstep=profile_linear_mstep, exact_stage=exact_stage,
            )

        self._set_requires_grad(spatial, True)
        self._set_requires_grad(temporal, True)
        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            final_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda)
            if profile_linear_mstep:
                self.profile_linear_trend_parameters(subjects_data, final_stats, Gamma, Omega)
                Gamma, Omega, _ = self.get_dynamics()
                final_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda)
        self.fit_history.extend(local_history)
        return final_stats

    def fit_exact_continuation(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        *,
        num_em_epochs: int = 35,
        m_step_iters: int = 15,
        lr: float = 0.005,
        grad_clip: float = 2.0,
        profile_linear_mstep: bool = True,
        lbfgs_max_iter: int = 10,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.theta_mode != "exact":
            return self.fit_em_multistart(
                subjects_data, num_em_epochs=num_em_epochs,
                warmup_epochs=max(1, num_em_epochs // 2), m_step_iters=m_step_iters,
                lr=lr, n_starts=1, burn_in_epochs=max(1, min(3, num_em_epochs // 4)),
                grad_clip=grad_clip, profile_linear_mstep=profile_linear_mstep,
            )

        diag_epochs = max(3, int(round(0.20 * num_em_epochs)))
        rev_epochs = max(4, int(round(0.30 * num_em_epochs)))
        full_epochs = max(5, num_em_epochs - diag_epochs - rev_epochs)
        self.fit_em_multistart(
            subjects_data, num_em_epochs=diag_epochs, warmup_epochs=diag_epochs,
            m_step_iters=m_step_iters, lr=lr, n_starts=1, burn_in_epochs=1,
            grad_clip=grad_clip, use_pca_warm_start=False,
            randomize_temporal_starts=False, exact_stage="diagonal",
            profile_linear_mstep=profile_linear_mstep,
        )
        self.fit_em_multistart(
            subjects_data, num_em_epochs=rev_epochs, warmup_epochs=rev_epochs,
            m_step_iters=m_step_iters, lr=lr, n_starts=1, burn_in_epochs=1,
            grad_clip=grad_clip, use_pca_warm_start=False,
            randomize_temporal_starts=False, exact_stage="reversible",
            profile_linear_mstep=profile_linear_mstep,
        )
        return self.fit_em_multistart(
            subjects_data, num_em_epochs=full_epochs, warmup_epochs=max(1, full_epochs // 2),
            m_step_iters=m_step_iters, lr=lr, n_starts=1, burn_in_epochs=1,
            grad_clip=grad_clip, use_pca_warm_start=False,
            randomize_temporal_starts=False, exact_stage="full",
            profile_linear_mstep=profile_linear_mstep,
            lbfgs_refine=lbfgs_max_iter > 0, lbfgs_max_iter=lbfgs_max_iter,
        )

# ---------------------------------------------------------------------
# 3. Data simulation
# ---------------------------------------------------------------------
def simulate_ad_cohort_stress(
    N: int,
    D: int,
    K: int,
    C_dim: int,
    *,
    theta_mode: str = "exact",
    seed: int = 42,
    missing_visit_rate: float = 0.0,
    item_missing_rate: float = 0.0,
    noise_scale: float = 1.0,
    anchor_items: Optional[Sequence[int]] = None,
    anchor_groups: Optional[Sequence[Sequence[int]]] = None,
    exact_target_rate: float = 1.0,
    exact_skew_strength: float = 0.05,
    visit_min: int = 3,
    visit_max: int = 5,
    gap_year_min: float = 1.5,
    gap_year_max: float = 5.0,
    include_latent_level: bool = False,
    observation_intercept_scale: float = 0.25,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Generate a cohort from the same continuous/discrete equations used in fitting."""
    if theta_mode not in {"exact", "diagonal"}:
        raise ValueError("theta_mode must be exact or diagonal.")
    if N < 1 or D < K or K < 1:
        raise ValueError("Require N>=1, D>=K, and K>=1.")
    torch.manual_seed(int(seed))
    anchors = normalize_anchor_groups(K, D, anchor_items=anchor_items, anchor_groups=anchor_groups)
    visit_min, visit_max = int(visit_min), int(visit_max)
    if visit_min < 2 or visit_max < visit_min:
        raise ValueError("visit_min must be >=2 and visit_max >= visit_min.")
    if not (0.0 < gap_year_min < gap_year_max):
        raise ValueError("Require 0 < gap_year_min < gap_year_max.")

    dtype = torch.get_default_dtype()
    eye = torch.eye(K, dtype=dtype)
    if theta_mode == "diagonal":
        rho = torch.linspace(0.2, 1.5, K, dtype=dtype)
        Omega_true = eye.clone()
        Gamma_true = torch.diag(rho)
        G_true = torch.diag(torch.sqrt(2.0 * rho))
    else:
        L_omega = torch.tril(0.25 * torch.randn(K, K, dtype=dtype) + eye)
        Omega_true = normalize_spd_to_correlation(
            symmetrize(L_omega @ L_omega.T) + 1e-5 * eye
        )
        L_g = torch.tril(0.20 * torch.randn(K, K, dtype=dtype) + 0.6 * eye)
        S = symmetrize(0.5 * (L_g @ L_g.T)) + 1e-5 * eye
        raw = exact_skew_strength * torch.randn(K, K, dtype=dtype)
        R = raw - raw.T
        Gamma_base = torch.linalg.solve(Omega_true.T, (S + R).T).T
        rates = torch.real(torch.linalg.eigvals(Gamma_base)).clamp_min(1e-6)
        scale = float(exact_target_rate) / float(torch.median(rates).item())
        Gamma_true = Gamma_base * scale
        # Rescaling Gamma rescales the Lyapunov diffusion covariance as well.
        G_true = torch.linalg.cholesky(symmetrize(2.0 * S * scale))

    Phi_true = 0.5 * torch.randn(K, C_dim, dtype=dtype)
    alpha_true = 0.5 * torch.randn(K, dtype=dtype)
    if include_latent_level:
        Phi_level_true = 0.35 * torch.randn(K, C_dim, dtype=dtype)
        # Avoid exact confounding with the item intercept in the DGP.  A constant
        # latent level can still be studied by setting observation_intercept_scale=0
        # and fitting CLOUDS(..., learn_observation_intercept=False).
        level_bias_true = torch.zeros(K, dtype=dtype)
    else:
        Phi_level_true = torch.zeros(K, C_dim, dtype=dtype)
        level_bias_true = torch.zeros(K, dtype=dtype)

    observation_intercept_true = float(observation_intercept_scale) * torch.randn(D, dtype=dtype)
    Lambda_true = 0.5 * torch.randn(D, K, dtype=dtype)
    for r, group in enumerate(anchors):
        for idx in group:
            Lambda_true[idx, :] = 0.0
            Lambda_true[idx, r] = torch.exp(0.35 * torch.randn((), dtype=dtype))

    subjects: List[Dict[str, torch.Tensor]] = []
    L_Omega = safe_cholesky(Omega_true, jitter=1e-12)
    for _ in range(N):
        J = int(torch.randint(visit_min, visit_max + 1, (1,)).item())
        baseline = torch.rand((), dtype=dtype) * 20.0 + 55.0
        gaps = torch.rand(J - 1, dtype=dtype) * (gap_year_max - gap_year_min) + gap_year_min
        t_age = torch.cat([baseline.view(1), baseline + torch.cumsum(gaps, dim=0)])
        t_model = (t_age - 70.0) / 10.0

        x_xi = torch.randn(C_dim, dtype=dtype)
        u = x_xi.unsqueeze(0).expand(J, C_dim).clone()
        slope = Phi_true @ x_xi + alpha_true
        level = Phi_level_true @ x_xi + level_bias_true
        mu = level.unsqueeze(0) + t_model.unsqueeze(1) * slope.unsqueeze(0)

        F_true = torch.zeros(J, K, dtype=dtype)
        F_true[0] = mu[0] + L_Omega @ torch.randn(K, dtype=dtype)
        for j in range(1, J):
            dt = t_model[j] - t_model[j - 1]
            if theta_mode == "diagonal":
                rates = torch.diag(Gamma_true)
                a = torch.exp(-rates * dt)
                A = torch.diag(a)
                q = -torch.expm1(-2.0 * rates * dt)
                Q = torch.diag(q.clamp_min(torch.finfo(dtype).tiny))
            else:
                A = torch.linalg.matrix_exp(-Gamma_true * dt)
                Q = symmetrize(Omega_true - A @ Omega_true @ A.T)
            L_Q = safe_cholesky(Q, jitter=1e-12)
            F_true[j] = mu[j] + A @ (F_true[j - 1] - mu[j - 1]) + L_Q @ torch.randn(K, dtype=dtype)

        X = observation_intercept_true.unsqueeze(0) + F_true @ Lambda_true.T \
            + noise_scale * torch.randn(J, D, dtype=dtype)
        if missing_visit_rate > 0.0:
            for j in range(1, J):
                if torch.rand(()).item() < missing_visit_rate:
                    X[j] = float("nan")
        if item_missing_rate > 0.0:
            X[torch.rand_like(X) < item_missing_rate] = float("nan")

        subjects.append({
            "x": X,
            "u": u,
            "x_xi": x_xi,
            "t_model": t_model,
            "t_dyn": t_model,
            "t_trend": t_model,
            "t_age": t_age,
            "t": t_model,
            "F_true": F_true,
        })

    truth = {
        "Lambda": Lambda_true,
        "Gamma": Gamma_true,
        "Omega": Omega_true,
        "G": G_true,
        "Phi": Phi_true,
        "alpha": alpha_true,
        "Phi_level": Phi_level_true,
        "level_bias": level_bias_true,
        "obs_intercept": observation_intercept_true,
    }
    return subjects, truth

# ---------------------------------------------------------------------
# 4. Evaluation wrappers
# ---------------------------------------------------------------------
def scale_true_identifiable(true_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    Omega_true = true_params["Omega"]
    stds = torch.sqrt(torch.diag(Omega_true)).clamp_min(1e-12)
    D_scale = torch.diag(stds)
    D_inv = torch.diag(1.0 / stds)
    out = {
        "Omega_corr": symmetrize(D_inv @ Omega_true @ D_inv),
        "Gamma": D_inv @ true_params["Gamma"] @ D_scale,
        "Lambda": true_params["Lambda"] @ D_scale,
        "Phi": D_inv @ true_params["Phi"],
        "alpha": D_inv @ true_params["alpha"],
        "latent_scale": stds,
    }
    if "Phi_level" in true_params:
        out["Phi_level"] = D_inv @ true_params["Phi_level"]
        out["level_bias"] = D_inv @ true_params["level_bias"]
    if "obs_intercept" in true_params:
        out["obs_intercept"] = true_params["obs_intercept"]
    return out




def transition_matrix_summary(
    Gamma_true: torch.Tensor,
    Gamma_est: torch.Tensor,
    dt_values: torch.Tensor,
    *,
    compare_diagonal_only: bool = False,
    max_dt_values: int = 12,
) -> Tuple[float, float]:
    """Correlation and relative RMSE between true/estimated transition matrices.

    For diagonal models, compare only the diagonal of A(dt). Including K^2-K
    shared off-diagonal zeros can artificially inflate A correlation at large K.
    """
    dt_values = torch.sort(dt_values.detach().cpu().reshape(-1)).values
    if dt_values.numel() > max_dt_values:
        idx = torch.linspace(0, dt_values.numel() - 1, max_dt_values).round().long()
        dt_values = dt_values[idx]

    Gamma_true_cpu = Gamma_true.detach().cpu()
    Gamma_est_cpu = Gamma_est.detach().cpu()

    if compare_diagonal_only:
        rates_true = torch.diag(Gamma_true_cpu)
        rates_est = torch.diag(Gamma_est_cpu)
        A_true = torch.exp(-dt_values.unsqueeze(1) * rates_true.unsqueeze(0)).reshape(-1).numpy()
        A_est = torch.exp(-dt_values.unsqueeze(1) * rates_est.unsqueeze(0)).reshape(-1).numpy()
    else:
        A_true_list = []
        A_est_list = []
        for dt in dt_values:
            A_true_list.append(torch.linalg.matrix_exp(-Gamma_true_cpu * dt))
            A_est_list.append(torch.linalg.matrix_exp(-Gamma_est_cpu * dt))
        A_true = torch.stack(A_true_list).reshape(-1).numpy()
        A_est = torch.stack(A_est_list).reshape(-1).numpy()

    corr = finite_corr(A_true, A_est)
    denom = np.sqrt(np.mean(A_true ** 2)) + 1e-12
    rel_rmse = float(np.sqrt(np.mean((A_true - A_est) ** 2)) / denom)
    return corr, rel_rmse


def offdiag_flat(A: torch.Tensor) -> torch.Tensor:
    K = A.shape[-1]
    mask = ~torch.eye(K, dtype=torch.bool, device=A.device)
    return A[mask]


def slope_metric(true_values: np.ndarray, est_values: np.ndarray) -> float:
    """OLS slope in est ~= intercept + slope * true."""
    a = np.asarray(true_values, dtype=float).reshape(-1)
    b = np.asarray(est_values, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    var = float(np.var(a))
    if var < 1e-12:
        return float("nan")
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / var)


def choose_A_metric_dt_count(K: int, mode: str) -> int:
    if mode == "diagonal":
        return 12
    if K <= 16:
        return 12
    if K <= 32:
        return 8
    if K <= 64:
        return 5
    return 3


@dataclass(frozen=True)
class FitRecipe:
    diag_pre_epochs: int
    diag_pre_warmup: int
    diag_pre_mstep: int
    diag_pre_starts: int
    diag_pre_burn: int
    diag_epochs: int
    diag_warmup: int
    diag_mstep: int
    diag_starts: int
    diag_burn: int
    exact_epochs: int
    exact_mstep: int
    exact_lbfgs: int
    exact_lr: float
    diag_lr: float


def adaptive_fit_recipe(K: int, scenario: Dict[str, object]) -> FitRecipe:
    """Adaptive EM settings for latent-dimension scaling experiments.

    Full exact Gamma scales much more steeply than diagonal Gamma. The recipe
    intentionally reduces multistarts and M-step iterations as K grows. Use
    --latent-profile thorough for stronger recovery settings.
    """
    profile = str(scenario.get("fit_profile", "standard")).lower()

    if K <= 4:
        recipe = FitRecipe(18, 8, 10, 3, 4, 30, 10, 15, 4, 5, 36, 12, 8, 0.005, 0.010)
    elif K <= 8:
        recipe = FitRecipe(14, 6, 8, 2, 3, 24, 8, 10, 3, 4, 28, 9, 6, 0.004, 0.008)
    elif K <= 16:
        recipe = FitRecipe(12, 5, 7, 2, 3, 20, 7, 8, 2, 3, 24, 7, 4, 0.0035, 0.007)
    elif K <= 32:
        recipe = FitRecipe(10, 4, 6, 1, 2, 16, 6, 7, 2, 2, 18, 5, 2, 0.0030, 0.006)
    elif K <= 64:
        recipe = FitRecipe(8, 3, 4, 1, 2, 12, 4, 5, 1, 2, 12, 3, 0, 0.0020, 0.004)
    else:
        recipe = FitRecipe(6, 2, 3, 1, 1, 8, 3, 4, 1, 1, 8, 2, 0, 0.0015, 0.003)

    if profile == "fast":
        scale = 0.35
        recipe = FitRecipe(
            max(2, int(round(recipe.diag_pre_epochs * scale))),
            max(1, int(round(recipe.diag_pre_warmup * scale))),
            max(2, int(round(recipe.diag_pre_mstep * scale))),
            1,
            max(1, int(round(recipe.diag_pre_burn * scale))),
            max(3, int(round(recipe.diag_epochs * scale))),
            max(1, int(round(recipe.diag_warmup * scale))),
            max(2, int(round(recipe.diag_mstep * scale))),
            1,
            max(1, int(round(recipe.diag_burn * scale))),
            max(4, int(round(recipe.exact_epochs * scale))),
            max(2, int(round(recipe.exact_mstep * scale))),
            0,
            recipe.exact_lr,
            recipe.diag_lr,
        )
    elif profile == "thorough":
        scale = 1.35 if K <= 32 else 1.15
        recipe = FitRecipe(
            max(3, int(round(recipe.diag_pre_epochs * scale))),
            max(1, int(round(recipe.diag_pre_warmup * scale))),
            max(3, int(round(recipe.diag_pre_mstep * scale))),
            max(recipe.diag_pre_starts, 2 if K <= 32 else 1),
            max(1, int(round(recipe.diag_pre_burn * scale))),
            max(4, int(round(recipe.diag_epochs * scale))),
            max(1, int(round(recipe.diag_warmup * scale))),
            max(3, int(round(recipe.diag_mstep * scale))),
            max(recipe.diag_starts, 2 if K <= 32 else 1),
            max(1, int(round(recipe.diag_burn * scale))),
            max(6, int(round(recipe.exact_epochs * scale))),
            max(3, int(round(recipe.exact_mstep * scale))),
            recipe.exact_lbfgs if K <= 32 else 0,
            recipe.exact_lr,
            recipe.diag_lr,
        )

    # Scenario-level overrides are useful for one-off heavy runs.
    return FitRecipe(
        int(scenario.get("diag_pre_epochs", recipe.diag_pre_epochs)),
        int(scenario.get("diag_pre_warmup", recipe.diag_pre_warmup)),
        int(scenario.get("diag_pre_mstep", recipe.diag_pre_mstep)),
        int(scenario.get("diag_pre_starts", recipe.diag_pre_starts)),
        int(scenario.get("diag_pre_burn", recipe.diag_pre_burn)),
        int(scenario.get("diag_epochs", recipe.diag_epochs)),
        int(scenario.get("diag_warmup", recipe.diag_warmup)),
        int(scenario.get("diag_mstep", recipe.diag_mstep)),
        int(scenario.get("diag_starts", recipe.diag_starts)),
        int(scenario.get("diag_burn", recipe.diag_burn)),
        int(scenario.get("exact_epochs", recipe.exact_epochs)),
        int(scenario.get("exact_mstep", recipe.exact_mstep)),
        int(scenario.get("exact_lbfgs", recipe.exact_lbfgs)),
        float(scenario.get("exact_lr", recipe.exact_lr)),
        float(scenario.get("diag_lr", recipe.diag_lr)),
    )


def regularization_defaults_for_k(K: int, scenario: Dict[str, object]) -> Dict[str, float]:
    # Off-diagonal parameter count grows as K(K-1), so shrinkage should not stay
    # fixed when K moves from 4 to 64/128.
    root_scale = math.sqrt(max(K, 1) / 4.0)
    return {
        "lambda_skew": float(scenario.get("lambda_skew", 0.75 * root_scale)),
        "lambda_offdiag_G": float(scenario.get("lambda_offdiag_G", 0.75 * root_scale)),
        "lambda_gamma_offdiag": float(scenario.get("lambda_gamma_offdiag", 0.10 * root_scale)),
        "lambda_rate": float(scenario.get("lambda_rate", 0.25)),
        "target_rate": float(scenario.get("target_rate", 1.0)),
        "profile_ridge": float(scenario.get("profile_ridge", 1.0 * root_scale)),
    }


def run_smoke_test() -> bool:
    """Focused numerical and integration checks for the corrected implementation."""
    logging.info("--- STARTING EXPANDED SMOKE TEST ---")
    try:
        started = time.time()
        tol = 5e-7 if torch.get_default_dtype() == torch.float64 else 2e-4

        # Correlation normalization must retain an exact unit diagonal.
        A = torch.tensor([[2.0, 0.4], [0.4, 0.7]])
        corr = normalize_spd_to_correlation(A)
        if not torch.allclose(torch.diag(corr), torch.ones(2, dtype=corr.dtype), atol=tol, rtol=0):
            raise AssertionError("Correlation normalization changed the diagonal.")

        # Anchor transformation must not evaluate exp() on non-anchor cells.
        grad_model = CLOUDS(6, 2, 1, theta_mode="diagonal", anchor_items=[0, 1])
        with torch.no_grad():
            grad_model.Lambda_raw[5, 0] = 1000.0
        grad_model.Lambda.sum().backward()
        if not bool(torch.isfinite(grad_model.Lambda_raw.grad).all()):
            raise AssertionError("Non-anchor loading produced a non-finite gradient.")

        # Returned exact diffusion must satisfy the Lyapunov identity.
        exact_model = CLOUDS(8, 2, 1, theta_mode="exact", anchor_items=[0, 1])
        Gamma, Omega, G = exact_model.get_dynamics()
        if exact_model.L_Omega_packed.numel() != exact_model.K * (exact_model.K - 1) // 2:
            raise AssertionError("Exact Omega still contains redundant Cholesky-scale parameters.")
        lyap_err = torch.linalg.norm(Gamma @ Omega + Omega @ Gamma.T - G @ G.T)
        if float(lyap_err.detach()) > tol:
            raise AssertionError(f"Diffusion/Lyapunov mismatch: {float(lyap_err):.3g}")

        # Verify RTS means, variances, and lag covariance against direct Gaussian conditioning.
        with torch.no_grad():
            H_target = torch.tensor([[0.8, 0.0], [0.0, 0.7], [0.4, -0.3]])
            dense_model = CLOUDS(3, 2, 1, theta_mode="diagonal", anchor_items=[0, 1])
            dense_model._assign_lambda_from_matrix(H_target)
            dense_model.log_psi.copy_(torch.log(torch.tensor([0.3, 0.4, 0.2])))
        H = dense_model.Lambda.detach()
        Atrans = torch.stack([
            torch.tensor([[0.75, 0.10], [0.00, 0.65]]),
            torch.tensor([[0.70, -0.05], [0.08, 0.60]]),
        ])
        Q = torch.stack([torch.tensor([[0.20, 0.03], [0.03, 0.15]]), torch.tensor([[0.18, 0.02], [0.02, 0.16]])])
        b = torch.tensor([[0.1, -0.05], [0.02, 0.04]])
        m0 = torch.tensor([0.2, -0.1])
        P0 = torch.tensor([[0.6, 0.1], [0.1, 0.5]])
        x = torch.tensor([[0.4, float("nan"), -0.2], [float("nan"), float("nan"), float("nan")], [0.1, -0.3, 0.2]])
        fs, Ps, Pcross = dense_model.kalman_smoother(x, Atrans, b, torch.ones(2), H, Q, m0, P0)

        T, K = 3, 2
        mean = torch.zeros(T, K)
        cov = torch.zeros(T, T, K, K)
        mean[0], cov[0, 0] = m0, P0
        for j in range(1, T):
            mean[j] = Atrans[j - 1] @ mean[j - 1] + b[j - 1]
            for s in range(j):
                cov[j, s] = Atrans[j - 1] @ cov[j - 1, s]
                cov[s, j] = cov[j, s].T
            cov[j, j] = Atrans[j - 1] @ cov[j - 1, j - 1] @ Atrans[j - 1].T + Q[j - 1]
        mean_flat = mean.reshape(-1)
        Sigma = cov.permute(0, 2, 1, 3).reshape(T * K, T * K)
        rows, y, noise = [], [], []
        for j in range(T):
            for d in torch.where(torch.isfinite(x[j]))[0].tolist():
                row = torch.zeros(T * K)
                row[j * K : (j + 1) * K] = H[d]
                rows.append(row); y.append(x[j, d]); noise.append(torch.exp(dense_model.log_psi[d]))
        Cmat = torch.stack(rows)
        yv = torch.stack(y)
        R = torch.diag(torch.stack(noise))
        Sy = Cmat @ Sigma @ Cmat.T + R
        gain = Sigma @ Cmat.T @ torch.linalg.inv(Sy)
        cond_mean = mean_flat + gain @ (yv - Cmat @ mean_flat)
        cond_cov = Sigma - gain @ Cmat @ Sigma
        direct_m = cond_mean.reshape(T, K)
        direct_P = torch.stack([cond_cov[j*K:(j+1)*K, j*K:(j+1)*K] for j in range(T)])
        direct_cross = torch.zeros_like(Pcross)
        for j in range(1, T):
            direct_cross[j] = cond_cov[j*K:(j+1)*K, (j-1)*K:j*K]
        if max(float(torch.max(torch.abs(fs-direct_m)).detach()), float(torch.max(torch.abs(Ps-direct_P)).detach()), float(torch.max(torch.abs(Pcross-direct_cross)).detach())) > 20 * tol:
            raise AssertionError("RTS smoother disagrees with dense Gaussian conditioning.")

        # Exercise exact and diagonal fitting, missing data, held-out prediction,
        # automatic anchors, and corrected PCA scaling.
        anchor_groups = [[0, 1], [2, 3]]
        subjects, _ = simulate_ad_cohort_stress(
            10, 20, 2, 1, theta_mode="exact", seed=99,
            anchor_groups=anchor_groups, item_missing_rate=0.08,
            missing_visit_rate=0.1, noise_scale=0.6, visit_min=3, visit_max=4,
        )
        train, holdouts = make_visit_holdout_split(subjects, validation_fraction=0.3, seed=101)
        fitted = CLOUDS(20, 2, 1, theta_mode="exact", anchor_groups=anchor_groups)
        fitted.fit_em_multistart(
            train, num_em_epochs=3, warmup_epochs=1, m_step_iters=2,
            lr=0.005, n_starts=1, burn_in_epochs=1, lbfgs_refine=False,
            convergence_patience=1,
        )
        _, cells, per_cell = heldout_visit_predictive_loglik(fitted, train, holdouts)
        if cells <= 0 or not math.isfinite(per_cell):
            raise AssertionError("Held-out predictive likelihood is not finite.")
        if not bool(torch.isfinite(fitted.obs_intercept).all()):
            raise AssertionError("Observation intercept fit is non-finite.")
        diag = CLOUDS(20, 2, 1, theta_mode="diagonal", anchor_groups=anchor_groups)
        _, Om_diag, _ = diag.get_dynamics()
        if not torch.allclose(Om_diag, torch.eye(2, dtype=Om_diag.dtype), atol=tol, rtol=0):
            raise AssertionError("Diagonal mode did not fix latent scale at Omega=I.")

        discovered, info = discover_anchor_groups(
            subjects, 2, anchors_per_factor=1, n_bootstrap=2,
            seed=123, min_purity_ratio=1.2, allow_relaxed=False,
        )
        if len(discovered) != 2 or not bool(info.get("strict_anchor_solution", False)):
            raise AssertionError("Strict subject-bootstrap anchor discovery failed.")

        # Known singular values: loading column norms must equal S/sqrt(n-1).
        X = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        loads, _ = preliminary_pca_loadings_from_matrix(X, 2, randomized_threshold=10_000)
        expected = torch.tensor([math.sqrt(8.0), math.sqrt(2.0)]) / math.sqrt(4.0)
        got = torch.linalg.norm(loads, dim=0)
        if not torch.allclose(got, expected, atol=20 * tol, rtol=20 * tol):
            raise AssertionError(f"Incorrect PCA loading scale: {got} versus {expected}.")

        logging.info("Expanded smoke test passed in %.2fs.", time.time() - started)
        return True
    except Exception:
        logging.error("Expanded smoke test failed:\n%s", traceback.format_exc())
        return False




def run_single_simulation(
    task_id: int,
    scenario: Dict[str, object],
    mode: str,
    run_idx: int,
    seed: int,
) -> Dict[str, object]:
    configure_torch_runtime(CPU_THREADS, str(scenario.get("dtype", "float64")))
    configure_logging(worker=True)
    reset_spd_stabilization_diagnostics()
    started = time.time()
    try:
        K, D, C = int(scenario["K"]), int(scenario["D"]), int(scenario["C"])
        dgp_mode = str(scenario.get("true_mode", "exact"))
        include_level = bool(scenario.get("include_latent_level", False))
        learn_intercept = bool(scenario.get("learn_observation_intercept", True))
        anchors = list(range(K))
        subjects, truth = simulate_ad_cohort_stress(
            int(scenario["N"]), D, K, C, theta_mode=dgp_mode, seed=seed,
            missing_visit_rate=float(scenario.get("miss", 0.0)),
            item_missing_rate=float(scenario.get("item_miss", 0.0)),
            noise_scale=float(scenario.get("noise", 1.0)), anchor_items=anchors,
            exact_target_rate=float(scenario.get("exact_target_rate", 1.0)),
            exact_skew_strength=float(scenario.get("exact_skew_strength", 0.05 if K <= 32 else 0.025)),
            visit_min=int(scenario.get("visit_min", 3)), visit_max=int(scenario.get("visit_max", 5)),
            gap_year_min=float(scenario.get("gap_year_min", 1.5)),
            gap_year_max=float(scenario.get("gap_year_max", 5.0)),
            include_latent_level=include_level,
            observation_intercept_scale=float(
                scenario.get("observation_intercept_scale", 0.25 if learn_intercept else 0.0)
            ),
        )

        recipe = adaptive_fit_recipe(K, scenario)
        kwargs = dict(
            obs_dim=D, latent_dim=K, covar_dim=C, anchor_items=anchors,
            inverse_ns_threshold=int(scenario.get("inverse_ns_threshold", 256)),
            inverse_ns_iters=int(scenario.get("inverse_ns_iters", 10)),
            inverse_ns_tol=float(scenario.get("inverse_ns_tol", 1e-4)),
            inverse_force_method=str(scenario.get("inverse_force_method", "auto")),
            omega_correlation=True, diagonal_fix_omega=True,
            include_latent_level=include_level,
            learn_observation_intercept=learn_intercept,
            **regularization_defaults_for_k(K, scenario),
        )
        if mode == "exact":
            diagonal = CLOUDS(theta_mode="diagonal", **kwargs)
            diagonal.fit_em_multistart(
                subjects, num_em_epochs=recipe.diag_pre_epochs,
                warmup_epochs=recipe.diag_pre_warmup, m_step_iters=recipe.diag_pre_mstep,
                lr=recipe.diag_lr, n_starts=recipe.diag_pre_starts,
                burn_in_epochs=recipe.diag_pre_burn,
            )
            model = CLOUDS(theta_mode="exact", **kwargs)
            model.initialize_exact_from_diagonal_model(diagonal)
            stats = model.fit_exact_continuation(
                subjects, num_em_epochs=recipe.exact_epochs,
                m_step_iters=recipe.exact_mstep, lr=recipe.exact_lr,
                lbfgs_max_iter=recipe.exact_lbfgs,
            )
            del diagonal
        else:
            model = CLOUDS(theta_mode="diagonal", **kwargs)
            stats = model.fit_em_multistart(
                subjects, num_em_epochs=recipe.diag_epochs,
                warmup_epochs=recipe.diag_warmup, m_step_iters=recipe.diag_mstep,
                lr=recipe.diag_lr, n_starts=recipe.diag_starts,
                burn_in_epochs=recipe.diag_burn,
            )

        with torch.no_grad():
            est = model.get_identifiable_parameters()
            true = scale_true_identifiable(truth)
            mask = model.struct_mask == 1
            l_corr = finite_corr(true["Lambda"][mask].cpu().numpy(), est["Lambda"][mask].cpu().numpy())
            intercept_corr = finite_corr(
                true["obs_intercept"].cpu().numpy(), est["obs_intercept"].cpu().numpy()
            )
            intercept_rmse = float(torch.sqrt(torch.mean(
                (true["obs_intercept"] - est["obs_intercept"]).square()
            )).item())

            f_true = torch.cat([s["F_true"] / true["latent_scale"] for s in subjects])
            f_est = torch.cat([s[0] / est["latent_scale"] for s in stats])
            f_corr = finite_corr(f_true.cpu().numpy(), f_est.cpu().numpy())

            Gamma_true, Gamma_est = true["Gamma"], est["Gamma"]
            diag_true, diag_est = torch.diag(Gamma_true), torch.diag(Gamma_est)
            g_diag_corr = finite_corr(diag_true.cpu().numpy(), diag_est.cpu().numpy())
            g_diag_rmse = float(torch.sqrt(torch.mean((diag_true - diag_est).square())).item())
            if dgp_mode == "diagonal":
                g_true_vec, g_est_vec = diag_true, diag_est
            else:
                g_true_vec, g_est_vec = Gamma_true.reshape(-1), Gamma_est.reshape(-1)
            g_corr = finite_corr(g_true_vec.cpu().numpy(), g_est_vec.cpu().numpy())
            g_rmse = float(torch.sqrt(torch.mean((g_true_vec - g_est_vec).square())).item())
            g_slope = slope_metric(g_true_vec.cpu().numpy(), g_est_vec.cpu().numpy())

            if dgp_mode == "exact" and mode == "exact" and K > 1:
                gt, ge = offdiag_flat(Gamma_true), offdiag_flat(Gamma_est)
                g_off_corr = finite_corr(gt.cpu().numpy(), ge.cpu().numpy())
                g_off_rmse = float(torch.sqrt(torch.mean((gt - ge).square())).item())
                ot, oe = offdiag_flat(true["Omega_corr"]), offdiag_flat(est["Omega_corr"])
                omega_corr = finite_corr(ot.cpu().numpy(), oe.cpu().numpy())
                omega_rmse = float(torch.sqrt(torch.mean((ot - oe).square())).item())
            else:
                g_off_corr = g_off_rmse = omega_corr = omega_rmse = float("nan")

            all_dt = torch.cat([s["t_dyn"][1:] - s["t_dyn"][:-1] for s in subjects])
            A_corr, A_rel_rmse = transition_matrix_summary(
                Gamma_true, Gamma_est, all_dt,
                compare_diagonal_only=(dgp_mode == "diagonal" and mode == "diagonal"),
                max_dt_values=int(scenario.get("max_A_metric_dts", choose_A_metric_dt_count(K, mode))),
            )
            obs_post, obs_cells = model.observed_data_log_posterior(subjects)
            obs_post_per_cell = float(obs_post.item()) / max(obs_cells, 1)
            Gamma_raw, Omega_raw, G_raw = model.get_dynamics()
            lyap_resid = float(torch.linalg.norm(Gamma_raw @ Omega_raw + Omega_raw @ Gamma_raw.T - G_raw @ G_raw.T).item())

        result = {
            "status": "success", "task_id": task_id, "run_idx": int(run_idx), "seed": int(seed),
            "scenario_name": str(scenario["name"]), "mode": mode, "true_mode": dgp_mode,
            "K": K, "D": D, "N": int(scenario["N"]),
            "visit_min": int(scenario.get("visit_min", 3)), "visit_max": int(scenario.get("visit_max", 5)),
            "l_corr": l_corr, "f_corr": f_corr,
            "intercept_corr": intercept_corr, "intercept_rmse": intercept_rmse,
            "g_corr": g_corr, "g_rmse": g_rmse, "g_slope": g_slope,
            "g_diag_corr": g_diag_corr, "g_diag_rmse": g_diag_rmse,
            "g_off_corr": g_off_corr, "g_off_rmse": g_off_rmse,
            "omega_corr": omega_corr, "omega_rmse": omega_rmse,
            "A_corr": A_corr, "A_rel_rmse": A_rel_rmse,
            "observed_logpost_per_cell": obs_post_per_cell,
            "lyapunov_residual": lyap_resid,
            "fit_history_records": len(model.fit_history),
            "spd_stabilization": get_spd_stabilization_diagnostics(),
            "time": time.time() - started,
        }
        del subjects, truth, model, stats
        gc.collect()
        return result
    except Exception as exc:
        return {
            "status": "error", "task_id": task_id, "run_idx": int(run_idx), "seed": int(seed),
            "scenario_name": str(scenario.get("name", "unknown")),
            "mode": mode, "true_mode": str(scenario.get("true_mode", "unknown")),
            "error_msg": str(exc) + "\n" + traceback.format_exc(),
        }




def _format_mean_sd(values: Sequence[float], digits: int = 3) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "nan +/- nan"
    return f"{np.mean(arr):.{digits}f} +/- {np.std(arr):.{digits}f}"


def parse_k_grid(text: str) -> List[int]:
    values: List[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("K values must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("K grid cannot be empty.")
    # Preserve order while removing duplicates.
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def make_baseline_scenarios(
    *, dtype: str = "float64", include_latent_level: bool = False
) -> List[Dict[str, object]]:
    common = {
        "C": 2, "miss": 0.0, "noise": 1.0,
        "modes": ["exact", "diagonal"], "true_mode": "exact",
        "dtype": dtype, "include_latent_level": include_latent_level,
    }
    return [
        {**common, "name": "1. Multi-Omics Base (D=2k)", "N": 100, "D": 2000, "K": 3},
        {**common, "name": "2. Real-World Scale (D=8k)", "N": 100, "D": 8000, "K": 4},
        {**common, "name": "3. Extreme Stress (D=15k)", "N": 100, "D": 15000, "K": 4},
        {**common, "name": "4. Absolute Limit (D=30k)", "N": 100, "D": 30000, "K": 4},
        {**common, "name": "5. Missing Visits (30%)", "N": 150, "D": 2000, "K": 4, "miss": 0.3},
        {**common, "name": "6. High Sensor Noise (3x)", "N": 150, "D": 2000, "K": 4, "noise": 3.0},
    ]


def latent_visits_for_k(K: int, profile: str) -> Tuple[int, int]:
    profile = profile.lower()
    if profile == "fast":
        return (4, 5) if K <= 16 else ((5, 6) if K <= 64 else (5, 7))
    if profile == "thorough":
        return (5, 7) if K <= 16 else ((6, 9) if K <= 64 else (8, 10))
    return (4, 6) if K <= 16 else ((5, 7) if K <= 64 else (6, 8))


def latent_subject_count_for_k(K: int, visit_min: int, visit_max: int, profile: str) -> int:
    avg = 0.5 * (visit_min + visit_max)
    if profile == "fast":
        floor, multiplier = (30 if K <= 32 else 24), 1.35
    elif profile == "thorough":
        floor, multiplier = (90 if K <= 32 else 60), (2.5 if K <= 64 else 2.0)
    else:
        floor, multiplier = (64 if K <= 32 else 40), 1.75
    return int(max(floor, math.ceil(multiplier * K / max(avg, 1.0))))


def make_latent_dimension_scenarios(
    k_grid: Sequence[int],
    *,
    exact_max_k: int = 32,
    include_large_exact: bool = False,
    profile: str = "standard",
    min_d: int = 512,
    d_per_k: int = 20,
    max_d: int = 6000,
    covar_dim: int = 2,
    inverse_ns_threshold: int = 256,
    design: str = "controlled",
    dtype: str = "float64",
    include_latent_level: bool = False,
    true_mode: str = "diagonal",
) -> List[Dict[str, object]]:
    """Create either controlled K-scaling or legacy adaptive stress scenarios.

    ``controlled`` holds D, N, and the visit range fixed across K so changes are
    attributable primarily to latent dimension. ``adaptive`` deliberately changes
    several stress dimensions and is labeled as a composite benchmark.
    """
    values = [int(k) for k in k_grid]
    if not values:
        raise ValueError("k_grid is empty.")
    design = design.lower()
    if design not in {"controlled", "adaptive"}:
        raise ValueError("design must be controlled or adaptive.")
    if true_mode not in {"exact", "diagonal"}:
        raise ValueError("true_mode must be exact or diagonal.")
    max_k = max(values)
    common_j = latent_visits_for_k(max_k, profile)
    common_n = latent_subject_count_for_k(max_k, *common_j, profile)
    common_d = int(min(max_d, max(min_d, d_per_k * max_k, max_k + 16)))
    common_recipe = adaptive_fit_recipe(max_k, {"fit_profile": profile})

    scenarios = []
    for K in values:
        if design == "controlled":
            visit_min, visit_max, N, D = common_j[0], common_j[1], common_n, common_d
        else:
            visit_min, visit_max = latent_visits_for_k(K, profile)
            N = latent_subject_count_for_k(K, visit_min, visit_max, profile)
            D = int(min(max_d, max(min_d, d_per_k * K, K + 16)))
        modes = ["diagonal"]
        if include_large_exact or K <= exact_max_k:
            modes.insert(0, "exact")
        scenario = {
            "name": f"Latent K={K} [{design}] (D={D}, N={N}, J={visit_min}-{visit_max})",
            "N": N, "D": D, "K": K, "C": covar_dim,
            "miss": 0.0, "noise": 1.0, "visit_min": visit_min, "visit_max": visit_max,
            "modes": modes, "true_mode": true_mode, "fit_profile": profile,
            "inverse_ns_threshold": inverse_ns_threshold,
            "exact_skew_strength": 0.05 if K <= 32 else 0.025,
            "exact_target_rate": 1.0,
            "max_A_metric_dts": choose_A_metric_dt_count(K, "exact"),
            "design": design, "dtype": dtype,
            "include_latent_level": include_latent_level,
        }
        if design == "controlled":
            # Hold optimizer budget fixed as well as D/N/visits.
            scenario.update(vars(common_recipe))
        scenarios.append(scenario)
    return scenarios


def run_scenarios_multiprocessing(
    scenarios: Sequence[Dict[str, object]],
    *,
    n_runs: int = 2,
    suite_name: str = "CLOUDS experiments",
    output_dir: str = "clouds_results",
) -> List[Dict[str, object]]:
    tasks, task_id = [], 0
    requested: Dict[Tuple[str, str], int] = {}
    for scenario_index, scenario in enumerate(scenarios):
        for run_idx in range(int(n_runs)):
            # The same seed is deliberately reused across fit modes so both models
            # see the same synthetic cohort.
            seed = 400 + 100_000 * scenario_index + run_idx
            for mode in list(scenario.get("modes", ["exact", "diagonal"])):
                task = {
                    "task_id": task_id, "scenario": dict(scenario), "mode": mode,
                    "run_idx": run_idx, "seed": seed,
                }
                tasks.append(task); task_id += 1
                key = (str(scenario["name"]), str(mode))
                requested[key] = requested.get(key, 0) + 1

    requested_workers = int(os.environ.get("CLOUDS_MAX_WORKERS", "4"))
    cpu_count = os.cpu_count() or requested_workers * CPU_THREADS
    max_workers = max(1, min(requested_workers, max(1, cpu_count // max(CPU_THREADS, 1))))
    logging.info("Launching %s: %s isolated tasks, %s workers.", suite_name, len(tasks), max_workers)

    import multiprocessing as mp
    context = mp.get_context("spawn")
    kwargs = {"max_workers": max_workers, "mp_context": context}
    try:
        executor = concurrent.futures.ProcessPoolExecutor(**kwargs, max_tasks_per_child=1)
    except TypeError:
        executor = concurrent.futures.ProcessPoolExecutor(**kwargs)

    all_results: List[Dict[str, object]] = []
    with executor:
        futures = {
            executor.submit(
                run_single_simulation, task["task_id"], task["scenario"],
                task["mode"], task["run_idx"], task["seed"],
            ): task for task in tasks
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "error", "task_id": task["task_id"],
                    "scenario_name": str(task["scenario"].get("name", "unknown")),
                    "mode": task["mode"],
                    "error_msg": f"Worker future failed: {exc}\n{traceback.format_exc()}",
                }
            all_results.append(result)
            if result.get("status") == "success":
                logging.info(
                    "[%s/%s] %s | %s finished in %.1fs",
                    completed, len(tasks), result["scenario_name"], result["mode"], result["time"],
                )
            else:
                logging.error(
                    "[%s/%s] %s | %s failed: %s",
                    completed, len(tasks), result.get("scenario_name"), result.get("mode"),
                    result.get("error_msg", "unknown error"),
                )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suite_slug = "_".join(suite_name.lower().split())
    run_dir = Path(output_dir) / f"{suite_slug}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(all_results, run_dir / "task_results.csv")
    (run_dir / "task_results.json").write_text(
        json.dumps(_json_safe(all_results), indent=2, allow_nan=False), encoding="utf-8"
    )

    logging.info("Final aggregated results: %s", suite_name)
    for scenario in scenarios:
        name = str(scenario["name"])
        for mode in list(scenario.get("modes", ["exact", "diagonal"])):
            subset = [r for r in all_results if r.get("scenario_name") == name and r.get("mode") == mode]
            ok = [r for r in subset if r.get("status") == "success"]
            n_req = requested.get((name, mode), 0)
            logging.info(
                "%s | %s | success %s/%s | Lambda %s | F %s | intercept RMSE %s | Gamma corr %s | "
                "Gamma RMSE %s | Gamma off RMSE %s | A relRMSE %s | Omega off RMSE %s | time %s",
                name, mode, len(ok), n_req,
                _format_mean_sd([r.get("l_corr", float("nan")) for r in ok]),
                _format_mean_sd([r.get("f_corr", float("nan")) for r in ok]),
                _format_mean_sd([r.get("intercept_rmse", float("nan")) for r in ok]),
                _format_mean_sd([r.get("g_corr", float("nan")) for r in ok]),
                _format_mean_sd([r.get("g_rmse", float("nan")) for r in ok]),
                _format_mean_sd([r.get("g_off_rmse", float("nan")) for r in ok]),
                _format_mean_sd([r.get("A_rel_rmse", float("nan")) for r in ok]),
                _format_mean_sd([r.get("omega_rmse", float("nan")) for r in ok]),
                _format_mean_sd([r.get("time", float("nan")) for r in ok], digits=1),
            )
    logging.info("Structured results saved under %s", run_dir)
    return all_results


def run_stress_test_multiprocessing(
    n_runs: int = 2,
    *,
    output_dir: str = "clouds_results",
    dtype: str = "float64",
    include_latent_level: bool = False,
) -> List[Dict[str, object]]:
    return run_scenarios_multiprocessing(
        make_baseline_scenarios(dtype=dtype, include_latent_level=include_latent_level),
        n_runs=n_runs, suite_name="baseline D-scaling stress test", output_dir=output_dir,
    )


def run_latent_dimension_multiprocessing(
    *,
    n_runs: int,
    k_grid: Sequence[int],
    exact_max_k: int,
    include_large_exact: bool,
    profile: str,
    min_d: int,
    d_per_k: int,
    max_d: int,
    covar_dim: int,
    inverse_ns_threshold: int,
    design: str = "controlled",
    output_dir: str = "clouds_results",
    dtype: str = "float64",
    include_latent_level: bool = False,
    true_mode: str = "diagonal",
) -> List[Dict[str, object]]:
    scenarios = make_latent_dimension_scenarios(
        k_grid, exact_max_k=exact_max_k, include_large_exact=include_large_exact,
        profile=profile, min_d=min_d, d_per_k=d_per_k, max_d=max_d,
        covar_dim=covar_dim, inverse_ns_threshold=inverse_ns_threshold,
        design=design, dtype=dtype, include_latent_level=include_latent_level,
        true_mode=true_mode,
    )
    return run_scenarios_multiprocessing(
        scenarios, n_runs=n_runs, suite_name="latent-dimension K-scaling stress test",
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------
# 8. Automatic anchor discovery and K selection
# ---------------------------------------------------------------------
def clone_subjects_data(subjects_data: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Deep-clone a subject-data list containing tensors and simple metadata."""
    cloned: List[Dict[str, torch.Tensor]] = []
    for subj in subjects_data:
        out: Dict[str, torch.Tensor] = {}
        for key, value in subj.items():
            if torch.is_tensor(value):
                out[key] = value.clone()
            else:
                out[key] = value
        cloned.append(out)
    return cloned


def stack_observed_rows(subjects_data: Sequence[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Stack rows with at least one finite observation."""
    x_all = torch.cat([s["x"] for s in subjects_data], dim=0)
    keep = torch.any(torch.isfinite(x_all), dim=1)
    x = x_all[keep]
    if x.numel() == 0:
        raise ValueError("No observed rows available.")
    return x


def fill_nan_with_column_means(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    col_means = torch.nanmean(x, dim=0)
    col_means = torch.where(torch.isfinite(col_means), col_means, torch.zeros_like(col_means))
    x_filled = torch.where(torch.isfinite(x), x, col_means.unsqueeze(0))
    return x_filled, col_means


def truncated_svd_matrix(
    X: torch.Tensor,
    rank: int,
    *,
    randomized_threshold: int = 5_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact SVD for moderate matrices and randomized low-rank SVD for large ones."""
    min_dim = min(X.shape)
    rank = min(int(rank), min_dim)
    if rank < 1:
        raise ValueError("rank must be positive.")
    if X.numel() <= randomized_threshold or min_dim <= max(64, 4 * rank):
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        return U[:, :rank], S[:rank], Vh[:rank]
    q = min(min_dim, max(rank + 8, 2 * rank))
    devices = [X.device.index] if X.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        U, S, V = torch.pca_lowrank(X, q=q, center=False, niter=4)
    return U[:, :rank], S[:rank], V[:, :rank].T


def preliminary_pca_loadings_from_matrix(
    x: torch.Tensor,
    K: int,
    *,
    randomized_threshold: int = 5_000_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return covariance-consistent PCA loadings and residual variances."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D matrix.")
    if x.shape[0] <= K:
        raise ValueError(f"Need more observed rows than K={K} for preliminary PCA.")
    filled, means = fill_nan_with_column_means(x)
    X = filled - means.unsqueeze(0)
    U, S, Vh = truncated_svd_matrix(X, K, randomized_threshold=randomized_threshold)
    K_eff = S.numel()
    loadings = torch.zeros(X.shape[1], K, dtype=x.dtype, device=x.device)
    # S are singular values; sqrt(covariance eigenvalue) = S/sqrt(n-1).
    loadings[:, :K_eff] = Vh.T * (S / math.sqrt(max(X.shape[0] - 1, 1)))
    approx = (U * S.unsqueeze(0)) @ Vh
    psi = torch.mean((X - approx).square(), dim=0).clamp_min(1e-8)
    return loadings, psi


def preliminary_pca_loadings(
    subjects_data: Sequence[Dict[str, torch.Tensor]], K: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return preliminary_pca_loadings_from_matrix(stack_observed_rows(subjects_data), K)


def varimax_rotation(
    loadings: torch.Tensor,
    *,
    gamma: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    D, K = loadings.shape
    if K <= 1:
        return loadings, torch.eye(K, dtype=loadings.dtype, device=loadings.device)
    R = torch.eye(K, dtype=loadings.dtype, device=loadings.device)
    last = -float("inf")
    for _ in range(max_iter):
        L = loadings @ R
        target = L.pow(3) - (gamma / float(D)) * L * torch.sum(L.square(), dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(loadings.T @ target, full_matrices=False)
        R = U @ Vh
        objective = float(torch.sum(S).item())
        if math.isfinite(last) and objective - last <= tol * max(1.0, abs(last)):
            break
        last = objective
    return loadings @ R, R


def orient_loading_columns(loadings: torch.Tensor) -> torch.Tensor:
    """Choose one factor-wide sign so the strongest loading in each column is positive."""
    out = loadings.clone()
    for r in range(out.shape[1]):
        d = int(torch.argmax(torch.abs(out[:, r])).item())
        if out[d, r] < 0:
            out[:, r].mul_(-1.0)
    return out


def align_loadings_to_reference(loadings: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Globally align columns by maximum absolute correlation and sign."""
    if loadings.shape != reference.shape:
        raise ValueError("loadings and reference must have the same shape.")
    ref = reference - reference.mean(dim=0, keepdim=True)
    cand = loadings - loadings.mean(dim=0, keepdim=True)
    ref = ref / torch.linalg.norm(ref, dim=0, keepdim=True).clamp_min(1e-12)
    cand = cand / torch.linalg.norm(cand, dim=0, keepdim=True).clamp_min(1e-12)
    corr = cand.T @ ref
    K = corr.shape[0]
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment((-torch.abs(corr)).detach().cpu().numpy())
        pairs = list(zip(rows.tolist(), cols.tolist()))
    else:
        pairs, used_c, used_r = [], set(), set()
        candidates = sorted(
            [(float(torch.abs(corr[c, r])), c, r) for c in range(K) for r in range(K)],
            reverse=True,
        )
        for _, c, r in candidates:
            if c not in used_c and r not in used_r:
                pairs.append((c, r)); used_c.add(c); used_r.add(r)
    aligned = torch.zeros_like(reference)
    for c, r in pairs:
        sign = 1.0 if corr[c, r] >= 0 else -1.0
        aligned[:, r] = sign * loadings[:, c]
    return aligned


def anchor_scores_from_loadings(
    loadings: torch.Tensor,
    psi: Optional[torch.Tensor] = None,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    absL = torch.abs(loadings)
    total_sq = torch.sum(absL.square(), dim=1, keepdim=True)
    strength = absL
    purity = strength / (torch.sqrt((total_sq - absL.square()).clamp_min(0.0)) + eps)
    if psi is None:
        communality = total_sq.squeeze(1) / (total_sq.squeeze(1) + 1.0)
    else:
        communality = total_sq.squeeze(1) / (
            total_sq.squeeze(1) + psi.to(loadings).clamp_min(eps)
        )
    score = strength * purity * communality.unsqueeze(1)
    return score, purity, strength, communality


def _solve_anchor_assignment(
    score: torch.Tensor,
    valid: torch.Tensor,
    anchors_per_factor: int,
) -> Optional[List[List[int]]]:
    """Assign distinct rows to factor slots, maximizing total score."""
    D, K = score.shape
    slots = [r for r in range(K) for _ in range(anchors_per_factor)]
    if len(slots) > D:
        return None
    cost = np.full((len(slots), D), 1e12, dtype=float)
    score_np, valid_np = score.detach().cpu().numpy(), valid.detach().cpu().numpy()
    for s, r in enumerate(slots):
        ok = valid_np[:, r]
        cost[s, ok] = -score_np[ok, r]

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        if len(rows) != len(slots) or np.any(cost[rows, cols] >= 1e11):
            return None
        groups = [[] for _ in range(K)]
        for s, d in zip(rows.tolist(), cols.tolist()):
            groups[slots[s]].append(int(d))
        return groups

    # Deterministic fallback when SciPy is unavailable.
    candidates = sorted(
        [(float(score[d, r]), d, r) for d in range(D) for r in range(K) if bool(valid[d, r])],
        reverse=True,
    )
    groups = [[] for _ in range(K)]
    used = set()
    for _, d, r in candidates:
        if d not in used and len(groups[r]) < anchors_per_factor:
            groups[r].append(d); used.add(d)
    return groups if all(len(g) == anchors_per_factor for g in groups) else None


def select_anchor_groups_from_loadings(
    loadings: torch.Tensor,
    *,
    psi: Optional[torch.Tensor] = None,
    anchors_per_factor: int = 1,
    min_purity_ratio: float = 2.0,
    min_strength_quantile: float = 0.25,
    allow_relaxed: bool = False,
) -> Tuple[List[List[int]], Dict[str, torch.Tensor]]:
    """Select sign-consistent, globally distinct hard anchors.

    By default, failure to find a strict simple-structure assignment makes the
    candidate ineligible instead of silently imposing unsupported hard zeros.
    """
    loadings = orient_loading_columns(loadings)
    D, K = loadings.shape
    anchors_per_factor = max(1, int(anchors_per_factor))
    if anchors_per_factor * K > D:
        raise ValueError("anchors_per_factor * K cannot exceed D.")
    score, purity, strength, communality = anchor_scores_from_loadings(loadings, psi)
    dominant = torch.argmax(torch.abs(loadings), dim=1)
    strength_cut = torch.quantile(strength.reshape(-1), float(min_strength_quantile))
    factors = torch.arange(K, device=loadings.device).unsqueeze(0)
    strict_valid = (
        (dominant.unsqueeze(1) == factors)
        & (purity >= min_purity_ratio)
        & (strength >= strength_cut)
        & (loadings > 0.0)
    )
    groups = _solve_anchor_assignment(score, strict_valid, anchors_per_factor)
    strict_solution = groups is not None
    if groups is None and allow_relaxed:
        relaxed_valid = (loadings > 0.0) & (strength >= strength_cut)
        groups = _solve_anchor_assignment(score, relaxed_valid, anchors_per_factor)
    if groups is None:
        raise RuntimeError(
            "No globally distinct strict anchor assignment exists. Reduce K or anchors_per_factor, "
            "collect stronger indicators, or use soft-loading constraints rather than hard anchors."
        )
    info = {
        "score": score.detach().cpu(),
        "purity": purity.detach().cpu(),
        "strength": strength.detach().cpu(),
        "communality": communality.detach().cpu(),
        "oriented_loadings": loadings.detach().cpu(),
        "strict_anchor_solution": torch.tensor(strict_solution),
        "strength_cut": strength_cut.detach().cpu(),
    }
    return groups, info


def discover_anchor_groups(
    subjects_data: Sequence[Dict[str, torch.Tensor]],
    K: int,
    *,
    anchors_per_factor: int = 1,
    n_bootstrap: int = 0,
    seed: int = 0,
    min_purity_ratio: float = 2.0,
    min_strength_quantile: float = 0.25,
    varimax: bool = True,
    allow_relaxed: bool = False,
) -> Tuple[List[List[int]], Dict[str, object]]:
    """Discover anchors with subject-level cluster bootstrapping."""
    base_x = stack_observed_rows(subjects_data)
    base_loadings, base_psi = preliminary_pca_loadings_from_matrix(base_x, K)
    if varimax:
        base_loadings, _ = varimax_rotation(base_loadings)
    base_loadings = orient_loading_columns(base_loadings)
    groups, info = select_anchor_groups_from_loadings(
        base_loadings, psi=base_psi, anchors_per_factor=anchors_per_factor,
        min_purity_ratio=min_purity_ratio,
        min_strength_quantile=min_strength_quantile,
        allow_relaxed=allow_relaxed,
    )

    stability = torch.zeros(base_x.shape[1], K)
    successful_bootstraps = 0
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n_subjects = len(subjects_data)
        for _ in range(int(n_bootstrap)):
            sampled = rng.integers(0, n_subjects, size=n_subjects)
            boot_subjects = [subjects_data[int(i)] for i in sampled]
            try:
                boot_L, boot_psi = preliminary_pca_loadings(boot_subjects, K)
                if varimax:
                    boot_L, _ = varimax_rotation(boot_L)
                boot_L = align_loadings_to_reference(boot_L, base_loadings)
                boot_groups, _ = select_anchor_groups_from_loadings(
                    boot_L, psi=boot_psi, anchors_per_factor=anchors_per_factor,
                    min_purity_ratio=min_purity_ratio,
                    min_strength_quantile=min_strength_quantile,
                    allow_relaxed=allow_relaxed,
                )
            except Exception:
                continue
            successful_bootstraps += 1
            for r, group in enumerate(boot_groups):
                for d in group:
                    stability[d, r] += 1.0
        if successful_bootstraps:
            stability /= float(successful_bootstraps)

    purity, strength, score = info["purity"], info["strength"], info["score"]
    group_purity = [[float(purity[d, r]) for d in g] for r, g in enumerate(groups)]
    group_strength = [[float(strength[d, r]) for d in g] for r, g in enumerate(groups)]
    group_scores = [[float(score[d, r]) for d in g] for r, g in enumerate(groups)]
    chosen_stability = [
        [float(stability[d, r]) if successful_bootstraps else float("nan") for d in g]
        for r, g in enumerate(groups)
    ]
    flat_stability = [v for row in chosen_stability for v in row]
    diagnostics: Dict[str, object] = {
        "anchor_groups": groups,
        "group_purity": group_purity,
        "group_strength": group_strength,
        "group_scores": group_scores,
        "chosen_stability": chosen_stability,
        "mean_chosen_stability": float(np.nanmean(flat_stability)) if successful_bootstraps else float("nan"),
        "mean_anchor_purity": float(np.mean([v for row in group_purity for v in row])),
        "strict_anchor_solution": bool(info["strict_anchor_solution"].item()),
        "bootstrap_requested": int(n_bootstrap),
        "bootstrap_successful": successful_bootstraps,
        "bootstrap_unit": "subject",
        "loadings": base_loadings.detach().cpu(),
        "psi": base_psi.detach().cpu(),
    }
    return groups, diagnostics


def make_visit_holdout_split(
    subjects_data: Sequence[Dict[str, torch.Tensor]],
    *,
    validation_fraction: float = 0.25,
    seed: int = 0,
    holdout: str = "last",
) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, object]]]:
    """Mask one visit for a subset of subjects and return holdout records."""
    rng = np.random.default_rng(seed)
    train = clone_subjects_data(subjects_data)
    n = len(train)
    n_val = max(1, int(round(validation_fraction * n)))
    chosen = rng.choice(n, size=min(n_val, n), replace=False)
    records: List[Dict[str, object]] = []
    for i in chosen.tolist():
        x = train[i]["x"]
        finite_rows = torch.where(torch.any(torch.isfinite(x), dim=1))[0].cpu().tolist()
        finite_rows = [j for j in finite_rows if j > 0]
        if not finite_rows:
            continue
        if holdout == "random":
            j = int(rng.choice(finite_rows))
        else:
            j = int(finite_rows[-1])
        x_true = x[j].clone()
        train[i]["x"][j, :] = float("nan")
        records.append({"subject_index": i, "visit_index": j, "x_true": x_true})
    return train, records


@torch.no_grad()
def heldout_visit_predictive_loglik(
    model: CLOUDS,
    train_subjects: Sequence[Dict[str, torch.Tensor]],
    holdout_records: Sequence[Dict[str, object]],
) -> Tuple[float, int, float]:
    """Evaluate one-step held-out visit log likelihood per observed cell."""
    Gamma, Omega, _ = model.get_dynamics()
    Lambda, log_psi = model.Lambda, model.log_psi
    matrices = model.get_all_subject_matrices(train_subjects, Gamma, Omega)
    by_subject: Dict[int, Dict[int, Dict[str, object]]] = {}
    for record in holdout_records:
        i, j = int(record["subject_index"]), int(record["visit_index"])
        by_subject.setdefault(i, {})[j] = record

    inv_psi = safe_inverse_variance(log_psi)
    full_Ht_invR = Lambda.T * inv_psi.unsqueeze(0)
    full_obs_info = full_Ht_invR @ Lambda
    total_ll, total_cells = 0.0, 0

    for i, visits in by_subject.items():
        subj = train_subjects[i]
        A, b, _, _, Q = matrices[i]
        t, _ = model._extract_times(subj)
        mu = model._mu_path(subj.get("x_xi", subj["u"]), t)
        f_pred, P_pred = mu[0], Omega
        for j in range(subj["x"].shape[0]):
            if j in visits:
                x_true = visits[j]["x_true"].to(device=Lambda.device, dtype=Lambda.dtype)
                ll, cells = lowrank_gaussian_loglik(
                    x_true, f_pred, P_pred, Lambda, log_psi,
                    obs_intercept=model.obs_intercept, jitter=model.jitter,
                )
                total_ll += float(ll.item())
                total_cells += cells
            f_filt, P_filt = model._measurement_update(
                f_pred, P_pred, subj["x"][j], Lambda, full_obs_info, full_Ht_invR
            )
            if j + 1 < subj["x"].shape[0]:
                f_pred = A[j] @ f_filt + b[j]
                P_pred = symmetrize(A[j] @ P_filt @ A[j].T + Q[j])
    return total_ll, total_cells, total_ll / float(max(total_cells, 1))


def candidate_modes_for_k(
    K: int,
    n_transitions: int,
    *,
    requested_mode: str = "auto",
    exact_max_k: int = 8,
    min_trans_per_gamma_param: float = 5.0,
) -> List[str]:
    """In auto mode evaluate diagonal for every K and exact when adequately supported."""
    requested = requested_mode.lower()
    if requested in {"exact", "diagonal"}:
        return [requested]
    if requested != "auto":
        raise ValueError("requested_mode must be auto, exact, or diagonal.")
    modes = ["diagonal"]
    enough = n_transitions >= int(math.ceil(min_trans_per_gamma_param * K * K))
    if K <= exact_max_k and enough:
        modes.append("exact")
    return modes


def choose_mode_for_k(
    K: int,
    n_transitions: int,
    *,
    requested_mode: str = "auto",
    exact_max_k: int = 8,
    min_trans_per_gamma_param: float = 5.0,
) -> str:
    """Compatibility helper; auto returns exact when eligible, otherwise diagonal."""
    modes = candidate_modes_for_k(
        K, n_transitions, requested_mode=requested_mode,
        exact_max_k=exact_max_k,
        min_trans_per_gamma_param=min_trans_per_gamma_param,
    )
    return "exact" if "exact" in modes else modes[0]


def count_transitions(subjects_data: Sequence[Dict[str, torch.Tensor]]) -> int:
    """Conservative count of transitions bracketed by observed visits."""
    total = 0
    for subject in subjects_data:
        observed = torch.any(torch.isfinite(subject["x"]), dim=1)
        if observed.numel() > 1:
            total += int(torch.sum(observed[1:] & observed[:-1]).item())
    return total


def fit_clouds_candidate(
    train_subjects: Sequence[Dict[str, torch.Tensor]],
    *,
    K: int,
    D: int,
    C: int,
    mode: str,
    anchor_groups: Sequence[Sequence[int]],
    scenario: Dict[str, object],
) -> CLOUDS:
    recipe = adaptive_fit_recipe(K, scenario)
    model_kwargs = dict(
        obs_dim=D, latent_dim=K, covar_dim=C, anchor_groups=anchor_groups,
        inverse_ns_threshold=int(scenario.get("inverse_ns_threshold", 256)),
        inverse_ns_iters=int(scenario.get("inverse_ns_iters", 10)),
        inverse_ns_tol=float(scenario.get("inverse_ns_tol", 1e-4)),
        inverse_force_method=str(scenario.get("inverse_force_method", "auto")),
        omega_correlation=True, diagonal_fix_omega=True,
        include_latent_level=bool(scenario.get("include_latent_level", False)),
        learn_observation_intercept=bool(scenario.get("learn_observation_intercept", True)),
        **regularization_defaults_for_k(K, scenario),
    )
    if mode == "exact":
        diagonal = CLOUDS(theta_mode="diagonal", **model_kwargs)
        diagonal.fit_em_multistart(
            train_subjects, num_em_epochs=recipe.diag_pre_epochs,
            warmup_epochs=recipe.diag_pre_warmup, m_step_iters=recipe.diag_pre_mstep,
            lr=recipe.diag_lr, n_starts=recipe.diag_pre_starts,
            burn_in_epochs=recipe.diag_pre_burn,
        )
        model = CLOUDS(theta_mode="exact", **model_kwargs)
        model.initialize_exact_from_diagonal_model(diagonal)
        model.fit_exact_continuation(
            train_subjects, num_em_epochs=recipe.exact_epochs,
            m_step_iters=recipe.exact_mstep, lr=recipe.exact_lr,
            lbfgs_max_iter=recipe.exact_lbfgs,
        )
        del diagonal
        return model
    model = CLOUDS(theta_mode="diagonal", **model_kwargs)
    model.fit_em_multistart(
        train_subjects, num_em_epochs=recipe.diag_epochs,
        warmup_epochs=recipe.diag_warmup, m_step_iters=recipe.diag_mstep,
        lr=recipe.diag_lr, n_starts=recipe.diag_starts,
        burn_in_epochs=recipe.diag_burn,
    )
    return model


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def combined_k_selection_choice(
    results: Sequence[Dict[str, object]],
    *,
    min_anchor_stability: float = 0.5,
    min_anchor_purity: float = 2.0,
    require_anchor_stability: bool = True,
    min_success_runs: int = 3,
    one_se_multiplier: float = 1.0,
    allow_gate_fallback: bool = False,
) -> Dict[str, object]:
    """Apply a one-SE rule with strict-anchor, purity, stability, and run-count gates."""
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        raise RuntimeError("No successful K/mode candidates.")
    best = max(successful, key=lambda r: float(r["val_ll_per_cell_mean"]))
    best_mean = float(best["val_ll_per_cell_mean"])
    best_se = float(best.get("val_ll_per_cell_se", 0.0))
    threshold = best_mean - max(0.0, one_se_multiplier) * best_se

    for row in results:
        if row.get("status") != "success":
            row.update({
                "val_ll_gap_from_best": float("nan"), "within_one_se": False,
                "anchor_ok": False, "purity_ok": False, "strict_anchor_ok": False,
                "success_runs_ok": False, "combined_eligible": False,
            })
            continue
        stability = _finite_float(row.get("mean_anchor_stability"))
        purity = _finite_float(row.get("mean_anchor_purity"))
        strict = bool(row.get("strict_anchor_solution", False))
        within = float(row["val_ll_per_cell_mean"]) >= threshold
        run_ok = int(row.get("n_success", 0)) >= max(1, int(min_success_runs))
        stability_ok = (not require_anchor_stability) or (
            math.isfinite(stability) and stability >= min_anchor_stability
        )
        purity_ok = math.isfinite(purity) and purity >= min_anchor_purity
        eligible = within and run_ok and stability_ok and purity_ok and strict
        row.update({
            "val_ll_gap_from_best": best_mean - float(row["val_ll_per_cell_mean"]),
            "within_one_se": within, "anchor_ok": stability_ok,
            "purity_ok": purity_ok, "strict_anchor_ok": strict,
            "success_runs_ok": run_ok, "combined_eligible": eligible,
        })

    eligible = [r for r in successful if bool(r.get("combined_eligible"))]
    if not eligible:
        if not allow_gate_fallback:
            raise RuntimeError(
                "No candidate passed the combined validation/anchor gates. The selection was not "
                "silently relaxed; inspect the saved diagnostics or explicitly enable fallback."
            )
        eligible = [r for r in successful if bool(r.get("within_one_se")) and bool(r.get("success_runs_ok"))]
        rule = "explicit-fallback-one-SE"
    else:
        rule = "combined-one-SE-strict-anchor"

    min_k = min(int(r["K"]) for r in eligible)
    same_k = [r for r in eligible if int(r["K"]) == min_k]
    # At the selected K choose the best validation mean; prefer diagonal only for an exact tie.
    same_k.sort(key=lambda r: (-float(r["val_ll_per_cell_mean"]), 0 if r["mode"] == "diagonal" else 1))
    chosen = dict(same_k[0])
    chosen.update({
        "selection_rule": rule, "one_se_threshold": threshold,
        "one_se_multiplier": one_se_multiplier,
        "best_K_by_mean": int(best["K"]), "best_mode_by_mean": str(best["mode"]),
        "best_val_ll_per_cell_mean": best_mean, "best_val_ll_per_cell_se": best_se,
        "min_anchor_stability": min_anchor_stability,
        "min_anchor_purity": min_anchor_purity,
        "min_success_runs": int(min_success_runs),
    })
    return chosen


def _json_safe(value: Any) -> Any:
    """Convert tensors, NumPy values, and non-finite floats to strict JSON values."""
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_rows_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys() if k not in {"example_anchors"}})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_safe(row.get(k)) for k in keys})


def save_model_bundle(
    model: CLOUDS,
    path: Path,
    *,
    anchor_groups: Sequence[Sequence[int]],
    metadata: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bundle_format_version": 3,
        "state_dict": model.state_dict(),
        "model_config": {
            "obs_dim": model.D, "latent_dim": model.K, "covar_dim": model.C_dim,
            "theta_mode": model.theta_mode, "anchor_groups": [list(g) for g in anchor_groups],
            "delta": model.delta, "jitter": model.jitter,
            "omega_correlation": model.omega_correlation,
            "diagonal_fix_omega": model.diagonal_fix_omega,
            "include_latent_level": model.include_latent_level,
            "learn_observation_intercept": model.learn_observation_intercept,
            "observation_intercept_prior": model.observation_intercept_prior,
            "min_anchor_loading": model.min_anchor_loading,
            "pca_randomized_threshold": model.pca_randomized_threshold,
            "inverse_ns_threshold": model.inverse_ns_threshold,
            "inverse_ns_iters": model.inverse_ns_iters,
            "inverse_ns_tol": model.inverse_ns_tol,
            "inverse_force_method": model.inverse_force_method,
            "lambda_skew": model.lambda_skew,
            "lambda_offdiag_G": model.lambda_offdiag_G,
            "lambda_gamma_offdiag": model.lambda_gamma_offdiag,
            "lambda_rate": model.lambda_rate,
            "target_rate": model.target_rate,
            "profile_ridge": model.profile_ridge,
            "dtype": str(model.Lambda_raw.dtype),
        },
        "fit_history": model.fit_history,
        "metadata": _json_safe(metadata),
    }, path)


def load_model_bundle(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[CLOUDS, Dict[str, object]]:
    """Reconstruct a saved CLOUDS model bundle for inference or further fitting."""
    try:
        bundle = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        bundle = torch.load(path, map_location=map_location)
    version = int(bundle.get("bundle_format_version", 1))
    if version not in {1, 2, 3}:
        raise ValueError(f"Unsupported CLOUDS model-bundle format version: {version}.")
    config = dict(bundle["model_config"])
    dtype_name = str(config.pop("dtype", "torch.float64"))
    config.setdefault("learn_observation_intercept", False if version < 3 else True)
    config.setdefault("observation_intercept_prior", 0.01)
    model = CLOUDS(**config)
    dtype = torch.float64 if "float64" in dtype_name else torch.float32
    model = model.to(device=map_location, dtype=dtype)
    state = dict(bundle["state_dict"])
    if version < 3:
        state.setdefault("obs_intercept", torch.zeros(model.D, dtype=dtype))
        if model.theta_mode == "exact" and "L_Omega_packed" in state:
            old_packed = state["L_Omega_packed"]
            old_size = model.K * (model.K + 1) // 2
            if int(old_packed.numel()) == old_size:
                old_rows, old_cols = torch.tril_indices(model.K, model.K)
                old_matrix = torch.zeros(model.K, model.K, dtype=old_packed.dtype)
                old_matrix[old_rows, old_cols] = old_packed.cpu()
                old_cov = symmetrize(old_matrix @ old_matrix.T) + model.delta * torch.eye(
                    model.K, dtype=old_matrix.dtype
                )
                old_corr = normalize_spd_to_correlation(old_cov, jitter=model.delta)
                corr_chol = torch.linalg.cholesky(old_corr)
                unit_diag_lower = corr_chol / torch.diag(corr_chol).unsqueeze(1)
                state["L_Omega_packed"] = unit_diag_lower[
                    model.strict_rows.cpu(), model.strict_cols.cpu()
                ].to(old_packed.device)
    model.load_state_dict(state)
    model.fit_history = list(bundle.get("fit_history", []))
    model.eval()
    return model, dict(bundle.get("metadata", {}))


def run_k_selection_experiment(
    *,
    true_K: int,
    k_grid: Sequence[int],
    n_runs: int,
    N: int,
    D: int,
    C: int,
    true_mode: str,
    selection_mode: str,
    anchors_per_factor: int,
    anchor_bootstrap: int,
    validation_fraction: float,
    fit_profile: str,
    exact_max_k: int,
    min_trans_per_gamma_param: float,
    inverse_ns_threshold: int,
    min_anchor_stability: float = 0.5,
    min_anchor_purity: float = 2.0,
    require_anchor_stability: bool = True,
    min_success_runs: int = 3,
    one_se_multiplier: float = 1.0,
    allow_gate_fallback: bool = False,
    refit_selected: bool = True,
    output_dir: str = "clouds_results",
    include_latent_level: bool = False,
    seed_base: int = 9000,
) -> Dict[str, object]:
    """Synthetic unknown-K benchmark with strict anchors and optional final refit."""
    if n_runs < min_success_runs:
        logging.warning(
            "n_runs=%s is below min_success_runs=%s; no candidate can be eligible.",
            n_runs, min_success_runs,
        )
    if 0 < anchor_bootstrap < 20:
        logging.warning(
            "Only %s subject-bootstrap replicates were requested; stability estimates are coarse.",
            anchor_bootstrap,
        )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"k_selection_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    true_anchor_groups = [
        list(range(r * anchors_per_factor, (r + 1) * anchors_per_factor))
        for r in range(true_K)
    ]
    if true_anchor_groups and max(max(g) for g in true_anchor_groups) >= D:
        raise ValueError("D is too small for true_K * anchors_per_factor anchors.")

    per_candidate: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    logging.info("K selection: true K=%s, grid=%s, validation folds=%s", true_K, list(k_grid), n_runs)
    # Generate one cohort and repeat subject-level validation splits.  This estimates
    # selection uncertainty for one dataset rather than mixing independent DGP draws.
    final_subjects, _ = simulate_ad_cohort_stress(
        N, D, true_K, C, theta_mode=true_mode, seed=seed_base,
        anchor_groups=true_anchor_groups, visit_min=5, visit_max=8,
        include_latent_level=include_latent_level,
    )

    for run_idx in range(int(n_runs)):
        seed = seed_base + 101 * run_idx
        train, holdouts = make_visit_holdout_split(
            final_subjects, validation_fraction=validation_fraction, seed=seed + 17, holdout="last"
        )
        n_trans = count_transitions(train)
        for K_value in k_grid:
            K = int(K_value)
            modes = candidate_modes_for_k(
                K, n_trans, requested_mode=selection_mode,
                exact_max_k=exact_max_k,
                min_trans_per_gamma_param=min_trans_per_gamma_param,
            )
            try:
                anchors, anchor_info = discover_anchor_groups(
                    train, K, anchors_per_factor=anchors_per_factor,
                    n_bootstrap=anchor_bootstrap, seed=seed + 1000 + K,
                    min_purity_ratio=min_anchor_purity, allow_relaxed=False,
                )
            except Exception as exc:
                for mode in modes:
                    per_candidate.setdefault((K, mode), []).append({
                        "status": "error", "K": K, "mode": mode,
                        "error": f"anchor discovery: {exc}", "time": 0.0,
                    })
                continue

            scenario = {
                "fit_profile": fit_profile, "inverse_ns_threshold": inverse_ns_threshold,
                "target_rate": 1.0, "include_latent_level": include_latent_level,
            }
            for mode in modes:
                started = time.time()
                try:
                    model = fit_clouds_candidate(
                        train, K=K, D=D, C=C, mode=mode,
                        anchor_groups=anchors, scenario=scenario,
                    )
                    _, cells, val = heldout_visit_predictive_loglik(model, train, holdouts)
                    rec = {
                        "status": "success", "K": K, "mode": mode,
                        "val_ll_per_cell": val, "n_cells": cells,
                        "time": time.time() - started,
                        "mean_anchor_stability": anchor_info["mean_chosen_stability"],
                        "mean_anchor_purity": anchor_info["mean_anchor_purity"],
                        "strict_anchor_solution": anchor_info["strict_anchor_solution"],
                        "bootstrap_successful": anchor_info["bootstrap_successful"],
                        "anchor_groups": anchors,
                    }
                    per_candidate.setdefault((K, mode), []).append(rec)
                    del model
                except Exception as exc:
                    per_candidate.setdefault((K, mode), []).append({
                        "status": "error", "K": K, "mode": mode,
                        "error": str(exc), "time": time.time() - started,
                    })
                gc.collect()

    rows: List[Dict[str, object]] = []
    for (K, mode), runs in sorted(per_candidate.items()):
        ok = [r for r in runs if r.get("status") == "success"]
        if not ok:
            rows.append({
                "status": "error", "K": K, "mode": mode,
                "n_requested": len(runs), "n_success": 0, "n_failed": len(runs),
            })
            continue
        vals = np.asarray([float(r["val_ll_per_cell"]) for r in ok])
        stabs = np.asarray([float(r["mean_anchor_stability"]) for r in ok])
        purities = np.asarray([float(r["mean_anchor_purity"]) for r in ok])
        rows.append({
            "status": "success", "K": K, "mode": mode,
            "n_requested": len(runs), "n_success": len(ok), "n_failed": len(runs) - len(ok),
            "val_ll_per_cell_mean": float(np.mean(vals)),
            "val_ll_per_cell_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "val_ll_per_cell_se": float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "mean_time": float(np.mean([r["time"] for r in ok])),
            "mean_anchor_stability": float(np.nanmean(stabs)) if np.any(np.isfinite(stabs)) else float("nan"),
            "mean_anchor_purity": float(np.mean(purities)),
            "strict_anchor_solution": all(bool(r["strict_anchor_solution"]) for r in ok),
            "mean_bootstrap_successful": float(np.mean([r["bootstrap_successful"] for r in ok])),
            "example_anchors": ok[0]["anchor_groups"],
        })

    try:
        chosen = combined_k_selection_choice(
            rows, min_anchor_stability=min_anchor_stability,
            min_anchor_purity=min_anchor_purity,
            require_anchor_stability=require_anchor_stability,
            min_success_runs=min_success_runs,
            one_se_multiplier=one_se_multiplier,
            allow_gate_fallback=allow_gate_fallback,
        )
    except Exception as exc:
        write_rows_csv(rows, run_dir / "candidate_results.csv")
        summary = {"status": "selection_failed", "error": str(exc), "rows": rows}
        (run_dir / "selection_summary.json").write_text(
            json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
        )
        logging.error("K selection failed without silently relaxing gates: %s", exc)
        return summary

    write_rows_csv(rows, run_dir / "candidate_results.csv")
    summary: Dict[str, object] = {
        "status": "success", "chosen": chosen, "rows": rows,
        "output_directory": str(run_dir),
    }
    if refit_selected and final_subjects is not None:
        final_anchors, final_anchor_info = discover_anchor_groups(
            final_subjects, int(chosen["K"]), anchors_per_factor=anchors_per_factor,
            n_bootstrap=anchor_bootstrap, seed=seed_base + 999_999,
            min_purity_ratio=min_anchor_purity, allow_relaxed=False,
        )
        final_scenario = {
            "fit_profile": fit_profile, "inverse_ns_threshold": inverse_ns_threshold,
            "target_rate": 1.0, "include_latent_level": include_latent_level,
        }
        final_model = fit_clouds_candidate(
            final_subjects, K=int(chosen["K"]), D=D, C=C,
            mode=str(chosen["mode"]), anchor_groups=final_anchors,
            scenario=final_scenario,
        )
        bundle_path = run_dir / "selected_model.pt"
        save_model_bundle(
            final_model, bundle_path, anchor_groups=final_anchors,
            metadata={"selection": chosen, "anchor_diagnostics": final_anchor_info},
        )
        summary["selected_model"] = str(bundle_path)
        summary["final_anchor_groups"] = final_anchors
        del final_model

    (run_dir / "selection_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    logging.info(
        "Selected K=%s, mode=%s by %s; artifacts: %s",
        chosen["K"], chosen["mode"], chosen["selection_rule"], run_dir,
    )
    return summary

if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(
        description="Run corrected CLOUDS simulations, controlled K scaling, and strict unknown-K selection."
    )
    parser.add_argument(
        "--experiment", choices=["baseline", "latent", "select", "all", "both"],
        default=os.environ.get("CLOUDS_EXPERIMENT", "latent"),
        help="Experiment suite. 'both' is retained as a deprecated alias for 'all'.",
    )
    parser.add_argument("--n-runs", type=int, default=int(os.environ.get("CLOUDS_N_RUNS", "2")))
    parser.add_argument(
        "--threads", type=int, default=CPU_THREADS,
        help="CPU threads per process; applied before NumPy/PyTorch imports when run as a script.",
    )
    parser.add_argument(
        "--selection-runs", type=int,
        default=int(os.environ.get("CLOUDS_SELECTION_RUNS", "5")),
        help="Repeated subject-level validation splits for K selection. Default: 5.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default=os.environ.get("CLOUDS_DTYPE", "float64"))
    parser.add_argument("--output-dir", default=os.environ.get("CLOUDS_OUTPUT_DIR", "clouds_results"))
    parser.add_argument(
        "--include-latent-level", action="store_true",
        help=("Fit a covariate-dependent latent level in mu_i(t)=level_i+slope_i*t. "
              "The constant mean is represented by item intercepts to avoid confounding."),
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")

    parser.add_argument("--latent-k-grid", default=os.environ.get("CLOUDS_LATENT_K_GRID", "3,4,8,16,32,64,128"))
    parser.add_argument("--latent-exact-max-k", type=int, default=int(os.environ.get("CLOUDS_LATENT_EXACT_MAX_K", "32")))
    parser.add_argument("--include-large-exact", action="store_true")
    parser.add_argument(
        "--latent-profile", choices=["fast", "standard", "thorough"],
        default=os.environ.get("CLOUDS_LATENT_PROFILE", "fast"),
    )
    parser.add_argument(
        "--latent-design", choices=["controlled", "adaptive"],
        default=os.environ.get("CLOUDS_LATENT_DESIGN", "controlled"),
        help="Controlled holds D, N, and visit counts fixed across K; adaptive is a composite stress design.",
    )
    parser.add_argument(
        "--latent-true-mode", choices=["exact", "diagonal"],
        default=os.environ.get("CLOUDS_LATENT_TRUE_MODE", "diagonal"),
        help="Use one common dynamics family across the K grid. Default: diagonal.",
    )
    parser.add_argument("--latent-min-d", type=int, default=int(os.environ.get("CLOUDS_LATENT_MIN_D", "512")))
    parser.add_argument("--latent-d-per-k", type=int, default=int(os.environ.get("CLOUDS_LATENT_D_PER_K", "20")))
    parser.add_argument("--latent-max-d", type=int, default=int(os.environ.get("CLOUDS_LATENT_MAX_D", "6000")))
    parser.add_argument("--latent-covar-dim", type=int, default=int(os.environ.get("CLOUDS_LATENT_C", "2")))
    parser.add_argument("--inverse-ns-threshold", type=int, default=int(os.environ.get("CLOUDS_INVERSE_NS_THRESHOLD", "256")))

    parser.add_argument("--selection-k-grid", default=os.environ.get("CLOUDS_SELECTION_K_GRID", "2,3,4,6,8,12,16"))
    parser.add_argument("--selection-true-k", type=int, default=int(os.environ.get("CLOUDS_SELECTION_TRUE_K", "4")))
    parser.add_argument("--selection-n", type=int, default=int(os.environ.get("CLOUDS_SELECTION_N", "80")))
    parser.add_argument("--selection-d", type=int, default=int(os.environ.get("CLOUDS_SELECTION_D", "512")))
    parser.add_argument("--selection-covar-dim", type=int, default=int(os.environ.get("CLOUDS_SELECTION_C", "2")))
    parser.add_argument("--selection-true-mode", choices=["exact", "diagonal"], default=os.environ.get("CLOUDS_SELECTION_TRUE_MODE", "exact"))
    parser.add_argument("--selection-mode", choices=["auto", "exact", "diagonal"], default=os.environ.get("CLOUDS_SELECTION_MODE", "auto"))
    parser.add_argument("--anchors-per-factor", type=int, default=int(os.environ.get("CLOUDS_ANCHORS_PER_FACTOR", "3")))
    parser.add_argument(
        "--anchor-bootstrap", type=int,
        default=int(os.environ.get("CLOUDS_ANCHOR_BOOTSTRAP", "50")),
        help="Subject-level bootstrap replicates for anchor stability. Default: 50.",
    )
    parser.add_argument("--validation-fraction", type=float, default=float(os.environ.get("CLOUDS_VALIDATION_FRACTION", "0.25")))
    parser.add_argument("--selection-profile", choices=["fast", "standard", "thorough"], default=os.environ.get("CLOUDS_SELECTION_PROFILE", "fast"))
    parser.add_argument("--selection-exact-max-k", type=int, default=int(os.environ.get("CLOUDS_SELECTION_EXACT_MAX_K", "8")))
    parser.add_argument("--selection-min-trans-per-gamma-param", type=float, default=float(os.environ.get("CLOUDS_SELECTION_MIN_TRANS_PER_GAMMA_PARAM", "5.0")))
    parser.add_argument("--selection-min-anchor-stability", type=float, default=float(os.environ.get("CLOUDS_SELECTION_MIN_ANCHOR_STABILITY", "0.5")))
    parser.add_argument("--selection-min-anchor-purity", type=float, default=float(os.environ.get("CLOUDS_SELECTION_MIN_ANCHOR_PURITY", "2.0")))
    parser.add_argument("--selection-disable-anchor-stability-gate", action="store_true")
    parser.add_argument("--selection-one-se-multiplier", type=float, default=float(os.environ.get("CLOUDS_SELECTION_ONE_SE_MULTIPLIER", "1.0")))
    parser.add_argument("--selection-min-success-runs", type=int, default=int(os.environ.get("CLOUDS_SELECTION_MIN_SUCCESS_RUNS", "3")))
    parser.add_argument(
        "--selection-allow-gate-fallback", action="store_true",
        help="Explicitly permit one-SE selection when strict anchor gates fail. Disabled by default.",
    )
    parser.add_argument("--selection-no-refit", action="store_true", help="Do not refit/save the selected configuration on all visits.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    configure_torch_runtime(args.threads, args.dtype)
    configure_logging(str(Path(args.output_dir) / LOG_FILENAME))
    mp.set_start_method("spawn", force=True)

    experiment = "all" if args.experiment == "both" else args.experiment
    if args.experiment == "both":
        logging.warning("--experiment both is deprecated; it now aliases all three suites.")

    if args.smoke_only:
        sys.exit(0 if run_smoke_test() else 1)
    if not args.skip_smoke and not run_smoke_test():
        logging.error("Aborting because the expanded smoke test failed.")
        sys.exit(1)

    if experiment in {"baseline", "all"}:
        run_stress_test_multiprocessing(
            n_runs=args.n_runs, output_dir=args.output_dir, dtype=args.dtype,
            include_latent_level=args.include_latent_level,
        )
    if experiment in {"latent", "all"}:
        run_latent_dimension_multiprocessing(
            n_runs=args.n_runs, k_grid=parse_k_grid(args.latent_k_grid),
            exact_max_k=args.latent_exact_max_k,
            include_large_exact=args.include_large_exact,
            profile=args.latent_profile, min_d=args.latent_min_d,
            d_per_k=args.latent_d_per_k, max_d=args.latent_max_d,
            covar_dim=args.latent_covar_dim,
            inverse_ns_threshold=args.inverse_ns_threshold,
            design=args.latent_design, output_dir=args.output_dir,
            dtype=args.dtype, include_latent_level=args.include_latent_level,
            true_mode=args.latent_true_mode,
        )
    if experiment in {"select", "all"}:
        summary = run_k_selection_experiment(
            true_K=args.selection_true_k,
            k_grid=parse_k_grid(args.selection_k_grid),
            n_runs=args.selection_runs,
            N=args.selection_n, D=args.selection_d, C=args.selection_covar_dim,
            true_mode=args.selection_true_mode, selection_mode=args.selection_mode,
            anchors_per_factor=args.anchors_per_factor,
            anchor_bootstrap=args.anchor_bootstrap,
            validation_fraction=args.validation_fraction,
            fit_profile=args.selection_profile,
            exact_max_k=args.selection_exact_max_k,
            min_trans_per_gamma_param=args.selection_min_trans_per_gamma_param,
            inverse_ns_threshold=args.inverse_ns_threshold,
            min_anchor_stability=args.selection_min_anchor_stability,
            min_anchor_purity=args.selection_min_anchor_purity,
            require_anchor_stability=not args.selection_disable_anchor_stability_gate,
            min_success_runs=args.selection_min_success_runs,
            one_se_multiplier=args.selection_one_se_multiplier,
            allow_gate_fallback=args.selection_allow_gate_fallback,
            refit_selected=not args.selection_no_refit,
            output_dir=args.output_dir,
            include_latent_level=args.include_latent_level,
        )
        if summary.get("status") != "success":
            sys.exit(2)

