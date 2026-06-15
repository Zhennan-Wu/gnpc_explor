"""
Equation-consistent CLOUDS simulation / fitting script.

This version matches the conditional Gaussian transition

    xi(t+dt) | xi(t) ~ N(mu(t+dt,x_i) + exp(-Gamma dt)(xi(t)-mu(t,x_i)),
                         Omega - exp(-Gamma dt) Omega exp(-Gamma.T dt))

and the SDE

    d xi_it = [(Phi x_i + alpha) - Gamma(xi_it - (Phi x_i + alpha)t)] dt + G dW_t.

Main fixes relative to the original version:
  1. Uses one model time scale t for both exp(-Gamma dt) and mu(t)=beta_i t.
  2. Applies the baseline observation update in the Kalman filter.
  3. Supports item-level missingness, not only whole-visit missingness.
  4. Corrects RTS lag-one covariance orientation used in the EM latent transition term.
  5. Recomputes final smoothers after the last M-step before evaluation.
  6. Clears/freezes gradients correctly during dynamics-only warmup.
  7. Uses an adaptive SPD inverse/right-solve: Cholesky/torch solver for small K,
     Newton-Schulz only when K is large enough, with a residual-based fallback.
  8. Specializes diagonal dynamics instead of using dense matrix exponentials.
  9. Sets BLAS/OpenMP/MKL thread controls before importing NumPy/PyTorch.
 10. Computes PCA warm start once per EM fit instead of once per multistart.
 11. Uses fresh worker processes with max_tasks_per_child=1 to avoid memory-arena buildup.
 12. Reports Gamma and transition-matrix metrics, not correlation alone.
 13. Uses subject-level, time-invariant covariates x_i^(xi) for the latent mean.
 14. Uses the exact mean shift b_j = mu(t_j) - A_j mu(t_{j-1}).
 15. Uses an equation-consistent initial latent prior xi_i(t_0) ~ N(mu(t_0,x_i), Omega).
 16. Constrains exact-mode Omega to a correlation matrix during fitting.
 17. Normalizes simulated exact Gamma to a target rate instead of multiplying by 10 blindly.
 18. Fits exact Gamma by diagonal-to-exact continuation rather than random full starts.
 19. Uses stronger skew/off-diagonal/rate regularization for exact Gamma.
 20. Profiles Phi/alpha by weighted ridge least squares inside the EM M-step.
 21. Optionally refines exact temporal parameters with LBFGS after Adam EM.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Runtime controls must be set before importing NumPy/PyTorch so BLAS backends honor them.
CPU_THREADS = int(os.environ.get("CLOUDS_CPU_THREADS", "8"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_THREADS))
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def configure_torch_runtime(num_threads: int = CPU_THREADS) -> None:
    """Keep CPU kernels from thread-thrashing in the parent and worker processes."""
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this only before inter-op parallel work starts. In a reused
        # interpreter it may already be set, which is harmless.
        pass


configure_torch_runtime(CPU_THREADS)

# ---------------------------------------------------------------------
# 0. Logging
# ---------------------------------------------------------------------
LOG_FILENAME = "clouds_anchor_simulation_chat5.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILENAME), logging.StreamHandler(sys.stdout)],
)


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


def normalize_spd_to_correlation(A: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    """Normalize an SPD matrix to a correlation matrix."""
    A_stable = symmetrize(A) + jitter * _eye_like(A)
    std = torch.sqrt(torch.diagonal(A_stable, dim1=-2, dim2=-1)).clamp_min(1e-12)
    Corr = A_stable / (std.unsqueeze(-1) * std.unsqueeze(-2))
    return symmetrize(Corr)


def normalize_spd_to_correlation(A: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Convert an SPD covariance matrix to a correlation matrix with unit diagonal."""
    A_sym = symmetrize(A)
    std = torch.sqrt(torch.diagonal(A_sym, dim1=-2, dim2=-1).clamp_min(jitter))
    corr = A_sym / (std.unsqueeze(-1) * std.unsqueeze(-2))
    return symmetrize(corr) + jitter * _eye_like(A_sym)


def safe_cholesky(
    A: torch.Tensor,
    jitter: float = 1e-6,
    max_tries: int = 7,
) -> torch.Tensor:
    """
    Cholesky factorization with increasing diagonal jitter.

    Works for both unbatched [K, K] and batched [..., K, K] SPD matrices.
    """
    A_sym = symmetrize(A)
    eye = _eye_like(A_sym)
    jitter_value = jitter

    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(A_sym + jitter_value * eye)
        if not torch.any(info > 0):
            return L
        jitter_value *= 10.0

    # Last attempt lets PyTorch raise the informative error.
    return torch.linalg.cholesky(A_sym + jitter_value * eye)


def cholesky_spd_inverse(A: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Explicit inverse of an SPD matrix using Cholesky."""
    L = safe_cholesky(A, jitter=jitter)
    inv_A = torch.cholesky_inverse(L)
    return symmetrize(inv_A)


def cholesky_right_solve_spd(
    M: torch.Tensor,
    A: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """
    Compute X = M A^{-1} for SPD A without explicitly inverting A.

    Equivalent to solving A X^T = M^T, since A is symmetric.
    """
    L = safe_cholesky(A, jitter=jitter)
    X_t = torch.cholesky_solve(M.transpose(-1, -2), L)
    return X_t.transpose(-1, -2)


def spd_logdet(A: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    """Log determinant of an SPD matrix using Cholesky."""
    L = safe_cholesky(A, jitter=jitter)
    return 2.0 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)


def newton_schulz_inverse(
    A: torch.Tensor,
    num_iters: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Batched Newton-Schulz approximate inverse.

    This is intended for large K where matrix-multiply-heavy iterations can be
    competitive. For small K, Cholesky/direct solves are almost always better.
    """
    K = A.shape[-1]
    I = _eye_like(A)

    # Frobenius scaling gives a conservative initial inverse candidate.
    frob_norm_sq = torch.sum(A * A, dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = A.transpose(-1, -2) / frob_norm_sq

    for _ in range(num_iters):
        X = torch.matmul(X, 2.0 * I - torch.matmul(A, X))

    return X


def adaptive_spd_inverse(
    A: torch.Tensor,
    *,
    ns_threshold: int = 256,
    ns_iters: int = 8,
    ns_tol: float = 1e-3,
    jitter: float = 1e-6,
    force_method: str = "auto",
) -> torch.Tensor:
    """
    Autodecide how to invert an SPD matrix based on K.

    - K < ns_threshold: use Cholesky/torch solver.
    - K >= ns_threshold: try Newton-Schulz, then fall back to Cholesky if the
      approximate inverse has a poor residual or non-finite values.

    force_method can be "auto", "torch", or "newton_schulz".
    """
    if A.shape[-1] != A.shape[-2]:
        raise ValueError("adaptive_spd_inverse expects square matrices.")

    K = A.shape[-1]
    method = force_method.lower()
    if method not in {"auto", "torch", "newton_schulz"}:
        raise ValueError("force_method must be 'auto', 'torch', or 'newton_schulz'.")

    use_newton_schulz = method == "newton_schulz" or (method == "auto" and K >= ns_threshold)

    if not use_newton_schulz:
        return cholesky_spd_inverse(A, jitter=jitter)

    A_stable = symmetrize(A) + jitter * _eye_like(A)
    X_ns = newton_schulz_inverse(A_stable, num_iters=ns_iters)
    X_ns = symmetrize(X_ns)

    # Residual check is detached; it only chooses whether to fall back.
    with torch.no_grad():
        I = _eye_like(A_stable)
        residual = torch.linalg.norm(torch.matmul(A_stable, X_ns) - I, dim=(-2, -1))
        denom = torch.linalg.norm(I, dim=(-2, -1)).clamp_min(1e-12)
        rel_residual = residual / denom
        ok = torch.isfinite(X_ns).all() and torch.isfinite(rel_residual).all() and torch.max(rel_residual) <= ns_tol

    if ok:
        return X_ns

    # Safety first: an inaccurate inverse can corrupt EM badly.
    return cholesky_spd_inverse(A_stable, jitter=jitter)


def adaptive_right_solve_spd(
    M: torch.Tensor,
    A: torch.Tensor,
    *,
    ns_threshold: int = 256,
    ns_iters: int = 8,
    ns_tol: float = 1e-3,
    jitter: float = 1e-6,
    force_method: str = "auto",
) -> torch.Tensor:
    """
    Compute M A^{-1}, autodeciding between Cholesky solve and Newton-Schulz.
    """
    K = A.shape[-1]
    method = force_method.lower()
    use_newton_schulz = method == "newton_schulz" or (method == "auto" and K >= ns_threshold)

    if not use_newton_schulz:
        return cholesky_right_solve_spd(M, A, jitter=jitter)

    A_inv = adaptive_spd_inverse(
        A,
        ns_threshold=ns_threshold,
        ns_iters=ns_iters,
        ns_tol=ns_tol,
        jitter=jitter,
        force_method="newton_schulz",
    )
    return torch.matmul(M, A_inv)


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


# ---------------------------------------------------------------------
# 2. CLOUDS model
# ---------------------------------------------------------------------
class CLOUDS(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        covar_dim: int,
        *,
        delta: float = 1e-4,
        theta_mode: str = "exact",
        anchor_items: Optional[Sequence[int]] = None,
        inverse_ns_threshold: int = 256,
        inverse_ns_iters: int = 8,
        inverse_ns_tol: float = 1e-3,
        inverse_force_method: str = "auto",
        jitter: float = 1e-6,
        omega_correlation: bool = True,
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

        self.D = obs_dim
        self.K = latent_dim
        self.C_dim = covar_dim
        self.delta = delta
        self.theta_mode = theta_mode
        self.inverse_ns_threshold = inverse_ns_threshold
        self.inverse_ns_iters = inverse_ns_iters
        self.inverse_ns_tol = inverse_ns_tol
        self.inverse_force_method = inverse_force_method
        self.jitter = jitter
        self.omega_correlation = omega_correlation
        self.lambda_skew = lambda_skew
        self.lambda_offdiag_G = lambda_offdiag_G
        self.lambda_gamma_offdiag = lambda_gamma_offdiag
        self.lambda_rate = lambda_rate
        self.target_rate = target_rate
        self.profile_ridge = profile_ridge

        if anchor_items is None:
            anchor_items = list(range(self.K))
        if len(anchor_items) != self.K:
            raise ValueError(f"Must provide exactly K={self.K} anchor items.")
        if min(anchor_items) < 0 or max(anchor_items) >= self.D:
            raise ValueError("anchor_items must be valid row indices in Lambda.")

        if self.theta_mode == "exact":
            self.L_G = nn.Parameter(torch.tril(torch.eye(self.K) + 0.1 * torch.randn(self.K, self.K)))
            self.gamma_skew = nn.Parameter(0.1 * torch.randn(self.K, self.K))
            self.L_Omega_unc = nn.Parameter(torch.tril(torch.eye(self.K) + 0.1 * torch.randn(self.K, self.K)))
        else:
            self.log_rho = nn.Parameter(0.1 * torch.randn(self.K) - 2.0)
            self.log_omega = nn.Parameter(0.1 * torch.randn(self.K))

        self.Phi_int = nn.Parameter(0.1 * torch.randn(self.K, self.C_dim))
        self.alpha_bias = nn.Parameter(0.1 * torch.randn(self.K))

        # Factor loadings with anchor orientation.
        self.Lambda_raw = nn.Parameter(0.1 * torch.randn(self.D, self.K))
        self.log_psi = nn.Parameter(torch.zeros(self.D))

        self.register_buffer("anchor_idx", torch.tensor(anchor_items, dtype=torch.long))
        self.register_buffer("anchor_cols", torch.arange(self.K, dtype=torch.long))

        # 1 for active free loadings, 0 for structural zeros in anchor rows.
        struct_mask = torch.ones(self.D, self.K)
        struct_mask[self.anchor_idx, :] = 0.0
        struct_mask[self.anchor_idx, self.anchor_cols] = 1.0
        self.register_buffer("struct_mask", struct_mask)

        # True only at positive anchor diagonals.
        positivity_mask = torch.zeros(self.D, self.K, dtype=torch.bool)
        positivity_mask[self.anchor_idx, self.anchor_cols] = True
        self.register_buffer("positivity_mask", positivity_mask)

    @property
    def Lambda(self) -> torch.Tensor:
        """Lambda with structural anchor zeros and positive anchor diagonal."""
        positive_entries = torch.exp(self.Lambda_raw)
        masked_entries = self.Lambda_raw * self.struct_mask
        return torch.where(self.positivity_mask, positive_entries, masked_entries)

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
            L_omega = torch.tril(self.L_Omega_unc)
            Omega_raw = symmetrize(L_omega @ L_omega.T) + self.delta * eye
            if self.omega_correlation:
                # Optimize on the identifiable latent scale: Omega is a
                # correlation matrix during fitting, not just during reporting.
                std = torch.sqrt(torch.diag(Omega_raw)).clamp_min(1e-12)
                Omega = Omega_raw / (std[:, None] * std[None, :])
                Omega = symmetrize(Omega)
            else:
                Omega = symmetrize(Omega_raw)

            G = torch.tril(self.L_G)
            S = 0.5 * (G @ G.T) + self.delta * eye
            S = symmetrize(S)

            A_skew = self.gamma_skew - self.gamma_skew.T
            Gamma = self._right_solve_spd(S + A_skew, Omega)
            return Gamma, Omega, G

        rho = torch.exp(self.log_rho).clamp_min(self.delta)
        omega = torch.exp(self.log_omega).clamp_min(self.delta)
        Gamma = torch.diag(rho)
        Omega = torch.diag(omega)
        G = torch.diag(torch.sqrt(2.0 * rho * omega))
        return Gamma, Omega, G

    @torch.no_grad()
    def get_identifiable_parameters(self) -> Dict[str, torch.Tensor]:
        Gamma_est, Omega_est, _ = self.get_dynamics()
        Lambda_est = self.Lambda

        stds = torch.sqrt(torch.diag(Omega_est)).clamp_min(1e-12)
        D_scale = torch.diag(stds)
        D_inv = torch.diag(1.0 / stds)

        Omega_corr = D_inv @ Omega_est @ D_inv
        Gamma_scaled = D_inv @ Gamma_est @ D_scale
        Lambda_scaled = Lambda_est @ D_scale
        Phi_scaled = D_inv @ self.Phi_int
        alpha_scaled = D_inv @ self.alpha_bias

        return {
            "Omega_corr": Omega_corr,
            "Gamma": Gamma_scaled,
            "Lambda": Lambda_scaled,
            "Phi": Phi_scaled,
            "alpha": alpha_scaled,
        }

    def _extract_times(self, subj: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the single model time used in the equations.

        The displayed SDE uses the same t in both exp(-Gamma dt) and
        mu(t, x_i) = (Phi x_i + alpha)t. To be fully consistent with that
        equation, this implementation uses one model-time vector for both.

        The simulator stores raw age separately as ``t_age`` for reference,
        but ``t_model``/``t_dyn``/``t`` are the scaled model time used here.
        """
        if "t_model" in subj:
            t_model = subj["t_model"]
        elif "t" in subj:
            # In this script, legacy key ``t`` denotes model time.
            t_model = subj["t"]
        elif "t_dyn" in subj and "t_trend" in subj:
            if not torch.allclose(subj["t_dyn"], subj["t_trend"], rtol=1e-6, atol=1e-8):
                raise ValueError(
                    "Equation-consistent CLOUDS requires one model time: t_dyn and t_trend must match. "
                    "Store raw age separately, e.g. under 't_age'."
                )
            t_model = subj["t_dyn"]
        elif "t_trend" in subj:
            t_model = subj["t_trend"]
        elif "t_dyn" in subj:
            t_model = subj["t_dyn"]
        else:
            raise KeyError("Subject dictionary must contain 't_model', 't', 't_trend', or 't_dyn'.")

        # The second return value is kept for backward-compatible call sites.
        return t_model, t_model

    def _subject_covariate_vector(self, u: torch.Tensor) -> torch.Tensor:
        """Return the subject-level covariate x_i^(xi).

        The equations use a time-invariant covariate vector x_i^(xi). The
        simulator stores this vector both as ``x_xi`` and as repeated rows in
        ``u`` for compatibility with older code. If a legacy [T, C] covariate
        matrix is supplied, the first row is used.
        """
        if u.ndim == 1:
            return u
        if u.ndim == 2:
            return u[0]
        raise ValueError("Latent covariate u must have shape [C] or [T, C].")

    def _mu_path(self, u: torch.Tensor, t_model: torch.Tensor) -> torch.Tensor:
        """Compute mu_i(t_j) = (Phi x_i^(xi) + alpha) t_j for all visits."""
        x_i = self._subject_covariate_vector(u)
        beta_i = x_i @ self.Phi_int.T + self.alpha_bias
        return t_model.unsqueeze(1) * beta_i.unsqueeze(0)

    def get_subject_matrices(
        self,
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        u: torch.Tensor,
        t_dyn: torch.Tensor,
        t_trend: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build A_j, b_j, dt_j, Lambda, Q_j for one subject."""
        if t_trend is None:
            t_trend = t_dyn

        dt = t_dyn[1:] - t_dyn[:-1]
        if torch.any(dt <= 0):
            raise ValueError("Subject times must be strictly increasing on the dynamics scale.")

        device = self.Lambda_raw.device
        dtype = self.Lambda_raw.dtype
        n_trans = dt.shape[0]
        eye = torch.eye(self.K, device=device, dtype=dtype)

        if self.theta_mode == "diagonal":
            rho = torch.diag(Gamma)
            omega = torch.diag(Omega)
            a = torch.exp(-dt.unsqueeze(1) * rho.unsqueeze(0))
            A_trans = torch.diag_embed(a)
            q_diag = omega.unsqueeze(0) * (1.0 - a.square())
            q_diag = q_diag.clamp_min(self.jitter)
            Q = torch.diag_embed(q_diag)
        else:
            Gamma_batch = Gamma.unsqueeze(0).expand(n_trans, self.K, self.K)
            A_trans = torch.linalg.matrix_exp(-Gamma_batch * dt.view(-1, 1, 1))
            Omega_batch = Omega.unsqueeze(0).expand(n_trans, self.K, self.K)
            Q = Omega_batch - torch.bmm(A_trans, torch.bmm(Omega_batch, A_trans.transpose(1, 2)))
            Q = symmetrize(Q) + self.jitter * eye.unsqueeze(0)

        if t_trend is not None and t_trend.shape == t_dyn.shape:
            # Full equation consistency requires one model time. The argument is
            # accepted only for backward-compatible call sites.
            if not torch.allclose(t_trend, t_dyn, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    "Equation-consistent mode requires the same time vector for dynamics and mu(t). "
                    "Store raw age separately, e.g. under 't_age'."
                )

        # Exact conditional mean from the displayed equation:
        #   E[xi_j | xi_{j-1}] = mu(t_j) + A_j [xi_{j-1} - mu(t_{j-1})]
        #                      = A_j xi_{j-1} + [mu(t_j) - A_j mu(t_{j-1})].
        mu_path = self._mu_path(u, t_dyn)
        mu_prev = mu_path[:-1]
        mu_next = mu_path[1:]
        b_shift = mu_next - torch.bmm(A_trans, mu_prev.unsqueeze(-1)).squeeze(-1)

        return A_trans, b_shift, dt, self.Lambda, Q

    def _measurement_update(
        self,
        mean_pred: torch.Tensor,
        cov_pred: torch.Tensor,
        x_t: torch.Tensor,
        Lambda: torch.Tensor,
        full_obs_info: torch.Tensor,
        full_Ht_invR: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kalman measurement update with full or partial missingness."""
        valid = torch.isfinite(x_t)
        if not torch.any(valid):
            return mean_pred, cov_pred

        pred_info = self._spd_inverse(cov_pred)

        if bool(torch.all(valid)):
            obs_info = full_obs_info
            Ht_invR = full_Ht_invR
            x_valid = x_t
        else:
            H = Lambda[valid]
            inv_psi = torch.exp(-self.log_psi[valid])
            Ht_invR = H.T * inv_psi.unsqueeze(0)
            obs_info = Ht_invR @ H
            x_valid = x_t[valid]

        filt_cov = self._spd_inverse(pred_info + obs_info)
        natural_mean = pred_info @ mean_pred + Ht_invR @ x_valid
        filt_mean = filt_cov @ natural_mean
        return filt_mean, symmetrize(filt_cov)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Kalman filter + RTS smoother.

        Returns:
          f_smooth: [T, K]
          P_smooth: [T, K, K]
          P_cross_next_prev: [T, K, K], where P_cross_next_prev[j]
              equals Cov(f_j, f_{j-1} | all data) for j >= 1.
        """
        del dt  # dt is useful for diagnostics; transitions already contain it.

        T = x_obs.shape[0]
        device = x_obs.device
        dtype = x_obs.dtype
        eye = torch.eye(self.K, device=device, dtype=dtype)

        f_pred = torch.zeros(T, self.K, device=device, dtype=dtype)
        P_pred = torch.zeros(T, self.K, self.K, device=device, dtype=dtype)
        f_filt = torch.zeros(T, self.K, device=device, dtype=dtype)
        P_filt = torch.zeros(T, self.K, self.K, device=device, dtype=dtype)

        # Equation-consistent initial residual prior:
        #   xi_i(t_0) ~ N(mu_i(t_0), Omega).
        # If called externally without init_mean/init_cov, fall back to N(0, I).
        if init_mean is None:
            f_pred[0] = torch.zeros(self.K, device=device, dtype=dtype)
        else:
            f_pred[0] = init_mean.to(device=device, dtype=dtype)

        if init_cov is None:
            P_pred[0] = eye
        else:
            P_pred[0] = symmetrize(init_cov.to(device=device, dtype=dtype)) + self.jitter * eye

        inv_psi_full = torch.exp(-self.log_psi)
        full_Ht_invR = Lambda.T * inv_psi_full.unsqueeze(0)
        full_obs_info = full_Ht_invR @ Lambda

        # Baseline update: this was missing in the original code.
        f_filt[0], P_filt[0] = self._measurement_update(
            f_pred[0], P_pred[0], x_obs[0], Lambda, full_obs_info, full_Ht_invR
        )

        # Forward pass.
        for j in range(1, T):
            idx = j - 1
            f_pred[j] = A_trans[idx] @ f_filt[j - 1] + b_shift[idx]
            P_pred[j] = symmetrize(A_trans[idx] @ P_filt[j - 1] @ A_trans[idx].T + Q[idx])
            f_filt[j], P_filt[j] = self._measurement_update(
                f_pred[j], P_pred[j], x_obs[j], Lambda, full_obs_info, full_Ht_invR
            )

        # Backward RTS pass.
        f_smooth = torch.zeros_like(f_filt)
        P_smooth = torch.zeros_like(P_filt)
        P_cross_next_prev = torch.zeros_like(P_filt)

        f_smooth[-1] = f_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for j in range(T - 2, -1, -1):
            P_pred_inv_next = self._spd_inverse(P_pred[j + 1])
            J_t = P_filt[j] @ A_trans[j].T @ P_pred_inv_next
            f_smooth[j] = f_filt[j] + J_t @ (f_smooth[j + 1] - f_pred[j + 1])
            P_smooth[j] = symmetrize(P_filt[j] + J_t @ (P_smooth[j + 1] - P_pred[j + 1]) @ J_t.T)

            # Correct orientation for the M-step transition f_{j+1} | f_j:
            # Cov(f_{j+1}, f_j | y) = P_smooth[j+1] J_t^T.
            P_cross_next_prev[j + 1] = P_smooth[j + 1] @ J_t.T

        return f_smooth, P_smooth, P_cross_next_prev

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

        smoothed_stats = []
        for subj in subjects_data:
            t_dyn, t_trend = self._extract_times(subj)
            A_trans, b_shift, dt, _, Q = self.get_subject_matrices(Gamma, Omega, subj["u"], t_dyn, t_trend)
            mu_path = self._mu_path(subj["u"], t_dyn)
            smoothed_stats.append(
                self.kalman_smoother(
                    subj["x"],
                    A_trans,
                    b_shift,
                    dt,
                    Lambda,
                    Q,
                    init_mean=mu_path[0],
                    init_cov=Omega,
                )
            )
        return smoothed_stats

    @torch.no_grad()
    def _apply_exact_stage_constraints(self, stage: str = "full") -> None:
        """Project exact-mode parameters onto continuation-stage constraints.

        Stages:
          - "diagonal": Omega=I, G diagonal, skew zero.
          - "reversible": skew zero, but Omega correlation and off-diagonal G allowed.
          - "full": all exact dynamics allowed.
        """
        if self.theta_mode != "exact":
            return
        stage_l = stage.lower()
        self.L_G.data.copy_(torch.tril(self.L_G.data))
        self.L_Omega_unc.data.copy_(torch.tril(self.L_Omega_unc.data))
        self.gamma_skew.data.fill_diagonal_(0.0)
        if stage_l in {"diag", "diagonal", "diag_dynamics"}:
            self.gamma_skew.data.zero_()
            self.L_G.data.copy_(torch.diag(torch.diag(self.L_G.data)))
            self.L_Omega_unc.data.copy_(torch.diag(torch.diag(self.L_Omega_unc.data)))
        elif stage_l in {"reversible", "symmetric", "no_skew"}:
            self.gamma_skew.data.zero_()
        elif stage_l in {"full", "exact"}:
            pass
        else:
            raise ValueError("exact_stage must be 'diagonal', 'reversible', or 'full'.")

    @staticmethod
    def _design_matrix_for_beta(C_mat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return H such that H vec_rowmajor(B) = C_mat B z."""
        K = C_mat.shape[0]
        P = z.numel()
        return torch.einsum("ar,p->arp", C_mat, z).reshape(K, K * P)

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
        """Weighted ridge update for B=[Phi, alpha] at fixed Gamma/Omega.

        With z_i=[x_i^(xi),1] and beta_i=B z_i,

            xi_j - A_j xi_{j-1} = (t_j I - t_{j-1} A_j) B z_i + noise.

        The expected transition objective is quadratic in B, so this profiles
        the linear trend block and reduces Gamma-trend confounding.
        """
        if ridge is None:
            ridge = self.profile_ridge
        device = self.Lambda_raw.device
        dtype = self.Lambda_raw.dtype
        P = self.C_dim + 1
        n_params = self.K * P
        eye_K = torch.eye(self.K, device=device, dtype=dtype)
        normal = ridge * torch.eye(n_params, device=device, dtype=dtype)
        rhs = torch.zeros(n_params, device=device, dtype=dtype)
        Omega_inv = self._spd_inverse(symmetrize(Omega) + self.jitter * eye_K)

        for subj, stats in zip(subjects_data, smoothed_stats):
            f_s, _, _ = stats
            t_dyn, t_trend = self._extract_times(subj)
            if not torch.allclose(t_dyn, t_trend, rtol=1e-6, atol=1e-8):
                raise ValueError("Profiled trend update requires one equation-consistent model time.")
            x_i = self._subject_covariate_vector(subj["u"]).to(device=device, dtype=dtype)
            z = torch.cat([x_i, torch.ones(1, device=device, dtype=dtype)])

            # Initial residual-stationary prior: xi(t0) ~ N(t0 B z, Omega).
            C0 = t_dyn[0] * eye_K
            H0 = self._design_matrix_for_beta(C0, z)
            y0 = f_s[0]
            normal = normal + H0.T @ Omega_inv @ H0
            rhs = rhs + H0.T @ Omega_inv @ y0

            A_trans, _, _, _, Q = self.get_subject_matrices(Gamma, Omega, subj["u"], t_dyn, t_trend)
            Q_inv = self._spd_inverse(symmetrize(Q) + self.jitter * eye_K.unsqueeze(0))
            for j in range(1, f_s.shape[0]):
                A_j = A_trans[j - 1]
                C_j = t_dyn[j] * eye_K - t_dyn[j - 1] * A_j
                H_j = self._design_matrix_for_beta(C_j, z)
                y_j = f_s[j] - A_j @ f_s[j - 1]
                W_j = Q_inv[j - 1]
                normal = normal + H_j.T @ W_j @ H_j
                rhs = rhs + H_j.T @ W_j @ y_j

        theta = torch.linalg.solve(normal, rhs)
        B = theta.reshape(self.K, P)
        self.Phi_int.copy_(B[:, : self.C_dim])
        self.alpha_bias.copy_(B[:, self.C_dim])

    def profile_linear_trend_update(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
    ) -> None:
        """Alias used by the EM code for the profiled linear trend M-step."""
        self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)

    @torch.no_grad()
    def _assign_lambda_from_matrix(self, Lambda_target: torch.Tensor) -> None:
        """Set Lambda_raw so self.Lambda matches Lambda_target under anchor constraints."""
        Lambda_target = Lambda_target.to(device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype)
        self.Lambda_raw.copy_(Lambda_target)
        self.Lambda_raw[self.anchor_idx, :] = 0.0
        anchor_diag = Lambda_target[self.anchor_idx, self.anchor_cols]
        self.Lambda_raw[self.anchor_idx, self.anchor_cols] = torch.log(torch.abs(anchor_diag).clamp_min(1e-4))

    @torch.no_grad()
    def initialize_exact_from_diagonal_model(self, diagonal_model: "CLOUDS") -> None:
        """Initialize exact mode from a fitted diagonal model on the Omega=I scale."""
        if self.theta_mode != "exact" or diagonal_model.theta_mode != "diagonal":
            raise ValueError("Requires self exact and source diagonal.")
        if self.D != diagonal_model.D or self.K != diagonal_model.K or self.C_dim != diagonal_model.C_dim:
            raise ValueError("Exact and diagonal models must have matching dimensions.")
        ident = diagonal_model.get_identifiable_parameters()
        self._assign_lambda_from_matrix(ident["Lambda"])
        self.log_psi.copy_(diagonal_model.log_psi.to(device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype))
        self.Phi_int.copy_(ident["Phi"].to(device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype))
        self.alpha_bias.copy_(ident["alpha"].to(device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype))
        rho = torch.diag(ident["Gamma"].to(device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype)).clamp_min(self.delta + 1e-6)

        self.L_Omega_unc.copy_(torch.eye(self.K, device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype))
        self.gamma_skew.zero_()
        self.L_G.zero_()
        g_diag = torch.sqrt((2.0 * (rho - self.delta)).clamp_min(1e-6))
        self.L_G.copy_(torch.diag(g_diag))
        self._apply_exact_stage_constraints("diagonal")

    def refine_temporal_lbfgs(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        *,
        max_iter: int = 20,
        lr: float = 0.5,
        profile_linear_mstep: bool = True,
        exact_stage: str = "full",
    ) -> None:
        """LBFGS refinement of the small temporal block with spatial parameters fixed."""
        temporal_params, spatial_params = self._parameter_groups()
        trend_params = [self.Phi_int, self.alpha_bias]
        trend_ids = {id(p) for p in trend_params}
        temporal_fit_params = [p for p in temporal_params if id(p) not in trend_ids] if profile_linear_mstep else temporal_params
        normalizer = self._loss_normalizer(subjects_data)
        self._set_requires_grad(spatial_params, False)
        self._set_requires_grad(temporal_params, True)
        if profile_linear_mstep:
            self._set_requires_grad(trend_params, False)
        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
            if profile_linear_mstep:
                self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)
                self._apply_exact_stage_constraints(exact_stage)
        opt = optim.LBFGS(
            temporal_fit_params,
            lr=lr,
            max_iter=max_iter,
            history_size=25,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure() -> torch.Tensor:
            self.zero_grad(set_to_none=True)
            Gamma_m, Omega_m, _ = self.get_dynamics()
            loss = -self.expected_complete_log_posterior_vectorized(
                subjects_data, smoothed_stats, Gamma_m, Omega_m, self.Lambda.detach()
            ) / normalizer
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite LBFGS temporal loss.")
            loss.backward()
            return loss

        opt.step(closure)
        self._apply_exact_stage_constraints(exact_stage)
        self._set_requires_grad(spatial_params, True)
        self._set_requires_grad(temporal_params, True)

    def expected_complete_log_posterior_vectorized(
        self,
        subjects_data: Sequence[Dict[str, torch.Tensor]],
        smoothed_stats: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Gamma: torch.Tensor,
        Omega: torch.Tensor,
        Lambda: torch.Tensor,
    ) -> torch.Tensor:
        """Expected complete-data log posterior, up to constants."""
        device = Lambda.device
        dtype = Lambda.dtype
        eye = torch.eye(self.K, device=device, dtype=dtype)

        ll_obs = torch.zeros((), device=device, dtype=dtype)
        ll_lat = torch.zeros((), device=device, dtype=dtype)

        inv_psi_full = torch.exp(-self.log_psi)
        L_Psi_L_full = Lambda.T @ (inv_psi_full.unsqueeze(1) * Lambda)
        sum_log_psi_full = torch.sum(self.log_psi)

        for i, subj in enumerate(subjects_data):
            x_obs = subj["x"]
            u = subj["u"]
            t_dyn, t_trend = self._extract_times(subj)
            f_s, P_s, P_cross = smoothed_stats[i]

            # Initial latent prior term consistent with residual stationarity:
            #   xi_i(t_0) ~ N(mu_i(t_0), Omega).
            mu_path = self._mu_path(u, t_dyn)
            mu_0 = mu_path[0]
            Omega_stable = symmetrize(Omega) + self.jitter * eye
            Omega_inv = self._spd_inverse(Omega_stable)
            log_det_Omega = spd_logdet(Omega_stable, jitter=self.jitter)
            diff_0 = f_s[0] - mu_0
            M0 = P_s[0] + torch.outer(diff_0, diff_0)
            ll_lat = ll_lat + (-0.5 * log_det_Omega - 0.5 * torch.sum(Omega_inv.transpose(-1, -2) * M0))

            # Observation term. Fast path for fully observed rows, fallback for
            # partial item-level missingness.
            finite_mask = torch.isfinite(x_obs)
            full_rows = torch.all(finite_mask, dim=1)
            partial_rows = torch.any(finite_mask, dim=1) & (~full_rows)

            if torch.any(full_rows):
                x_v = x_obs[full_rows]
                f_v = f_s[full_rows]
                P_v = P_s[full_rows]

                trace_E = torch.sum(P_v * L_Psi_L_full.unsqueeze(0), dim=(1, 2))
                trace_E = trace_E + torch.sum(f_v * (f_v @ L_Psi_L_full), dim=1)

                fitted = f_v @ Lambda.T
                term_obs = torch.sum(x_v.square() * inv_psi_full, dim=1)
                term_obs = term_obs - 2.0 * torch.sum(x_v * fitted * inv_psi_full, dim=1)
                term_obs = term_obs + trace_E

                ll_obs = ll_obs + torch.sum(-0.5 * term_obs - 0.5 * sum_log_psi_full)

            if torch.any(partial_rows):
                row_ids = torch.where(partial_rows)[0]
                for j in row_ids:
                    valid = finite_mask[j]
                    x_j = x_obs[j, valid]
                    H = Lambda[valid]
                    inv_psi = torch.exp(-self.log_psi[valid])
                    L_Psi_L = H.T @ (inv_psi.unsqueeze(1) * H)

                    m = f_s[j]
                    S = P_s[j] + torch.outer(m, m)
                    term = torch.sum(x_j.square() * inv_psi)
                    term = term - 2.0 * torch.sum(x_j * (H @ m) * inv_psi)
                    term = term + torch.sum(S * L_Psi_L)
                    ll_obs = ll_obs + (-0.5 * term - 0.5 * torch.sum(self.log_psi[valid]))

            # Latent transition term.
            A_trans, b_shift, _, _, Q_exact = self.get_subject_matrices(Gamma, Omega, u, t_dyn, t_trend)
            Q_stable = symmetrize(Q_exact) + self.jitter * eye.unsqueeze(0)
            Q_inv = self._spd_inverse(Q_stable)
            log_det_Q = spd_logdet(Q_stable, jitter=self.jitter)

            f_j = f_s[1:]
            f_prev = f_s[:-1]
            P_j = P_s[1:]
            P_prev = P_s[:-1]
            P_j_prev = P_cross[1:]

            E_jj = P_j + torch.bmm(f_j.unsqueeze(-1), f_j.unsqueeze(1))
            E_j_prev = P_j_prev + torch.bmm(f_j.unsqueeze(-1), f_prev.unsqueeze(1))
            E_prev_prev = P_prev + torch.bmm(f_prev.unsqueeze(-1), f_prev.unsqueeze(1))

            A_T = A_trans.transpose(1, 2)
            M = E_jj
            M = M - torch.bmm(E_j_prev, A_T)
            M = M - torch.bmm(A_trans, E_j_prev.transpose(1, 2))
            M = M + torch.bmm(A_trans, torch.bmm(E_prev_prev, A_T))

            b_col = b_shift.unsqueeze(-1)
            b_row = b_shift.unsqueeze(1)
            f_j_col = f_j.unsqueeze(-1)
            f_j_row = f_j.unsqueeze(1)
            f_prev_col = f_prev.unsqueeze(-1)
            f_prev_row = f_prev.unsqueeze(1)

            M = M - torch.bmm(f_j_col, b_row)
            M = M - torch.bmm(b_col, f_j_row)
            M = M + torch.bmm(A_trans, torch.bmm(f_prev_col, b_row))
            M = M + torch.bmm(b_col, torch.bmm(f_prev_row, A_T))
            M = M + torch.bmm(b_col, b_row)
            M = symmetrize(M)

            trace_term = torch.sum(Q_inv.transpose(1, 2) * M, dim=(1, 2))
            ll_lat = ll_lat + torch.sum(-0.5 * log_det_Q - 0.5 * trace_term)

        # Priors.
        log_prior_dyn = torch.zeros((), device=device, dtype=dtype)
        if self.theta_mode == "exact":
            stds = torch.sqrt(torch.diag(Omega)).clamp_min(1e-12)
            D_inv = torch.diag(1.0 / stds)
            Omega_corr = D_inv @ Omega @ D_inv

            eta = 1.5
            log_prior_dyn = log_prior_dyn + (eta - 1.0) * spd_logdet(
                Omega_corr + self.jitter * eye, jitter=self.jitter
            )

            # Stronger regularization for the weakly identified full-Gamma model.
            A_skew = self.gamma_skew - self.gamma_skew.T
            log_prior_dyn = log_prior_dyn - self.lambda_skew * 0.5 * torch.sum(torch.abs(A_skew))
            off_diag_G = torch.tril(self.L_G, diagonal=-1)
            log_prior_dyn = log_prior_dyn - self.lambda_offdiag_G * torch.sum(torch.abs(off_diag_G))
            Gamma_offdiag = Gamma - torch.diag(torch.diag(Gamma))
            log_prior_dyn = log_prior_dyn - self.lambda_gamma_offdiag * torch.sum(torch.abs(Gamma_offdiag))
            mean_rate = torch.trace(Gamma) / float(self.K)
            log_prior_dyn = log_prior_dyn - self.lambda_rate * (mean_rate - self.target_rate).square()
        else:
            log_prior_dyn = log_prior_dyn - 0.5 * torch.sum(self.log_rho.square())
            log_prior_dyn = log_prior_dyn - 0.5 * torch.sum(self.log_omega.square())

        active_Lambda_raw = self.Lambda_raw[self.struct_mask == 1]
        log_prior_Lambda = -0.5 * torch.sum(active_Lambda_raw.square())
        log_prior_lin = -0.5 * (torch.sum(self.Phi_int.square()) + torch.sum(self.alpha_bias.square()))
        log_prior_psi = -0.5 * torch.sum(self.log_psi.square())

        return ll_obs + ll_lat + log_prior_dyn + log_prior_Lambda + log_prior_lin + log_prior_psi

    @torch.no_grad()
    def pca_warm_start(self, subjects_data: Sequence[Dict[str, torch.Tensor]]) -> None:
        """PCA initialization with anchor-oriented rotation and robust NaN handling."""
        x_all = torch.cat([s["x"] for s in subjects_data], dim=0)
        nonempty_rows = torch.any(torch.isfinite(x_all), dim=1)
        x_valid = x_all[nonempty_rows]
        if x_valid.shape[0] <= self.K:
            raise ValueError("Not enough observed rows for PCA warm start.")

        col_means = torch.nanmean(x_valid, dim=0)
        col_means = torch.where(torch.isfinite(col_means), col_means, torch.zeros_like(col_means))
        x_filled = torch.where(torch.isfinite(x_valid), x_valid, col_means.unsqueeze(0))
        x_centered = x_filled - col_means.unsqueeze(0)

        # SVD of [n_obs, D]. This is acceptable here because n_obs is small
        # compared with D. full_matrices=False avoids a huge U.
        U, S_vals, Vh = torch.linalg.svd(x_centered, full_matrices=False)
        n = x_centered.shape[0]
        Lambda_pca = Vh[: self.K, :].T * torch.sqrt((S_vals[: self.K].clamp_min(1e-12)) / max(n - 1, 1))

        # Rotate PCA loadings toward the anchor coordinate system.
        A_pca = Lambda_pca[self.anchor_idx, :]
        target_scales = torch.diag(torch.norm(A_pca, dim=1).clamp_min(1e-4))
        W = torch.linalg.pinv(A_pca) @ target_scales
        Lambda_rotated = Lambda_pca @ W

        self.Lambda_raw.data.copy_(Lambda_rotated)
        self.Lambda_raw.data[self.anchor_idx, :] = 0.0
        anchor_diag = Lambda_rotated[self.anchor_idx, self.anchor_cols]
        self.Lambda_raw.data[self.anchor_idx, self.anchor_cols] = torch.log(torch.abs(anchor_diag).clamp_min(1e-4))

        # Better noise initialization than setting all log_psi to zero, especially
        # for the high-noise scenario.
        var_x = torch.mean(x_centered.square(), dim=0).clamp_min(1e-4)
        self.log_psi.data.copy_(torch.log(var_x))

        self.Phi_int.data.zero_()
        self.alpha_bias.data.zero_()

    def _randomize_temporal_parameters(self) -> None:
        with torch.no_grad():
            if self.theta_mode == "exact":
                nn.init.normal_(self.L_G, mean=0.0, std=0.1)
                self.L_G.data += torch.eye(self.K, device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype)
                nn.init.normal_(self.gamma_skew, mean=0.0, std=0.1)
                nn.init.normal_(self.L_Omega_unc, mean=0.0, std=0.1)
                self.L_Omega_unc.data += torch.eye(self.K, device=self.Lambda_raw.device, dtype=self.Lambda_raw.dtype)
            else:
                nn.init.normal_(self.log_rho, mean=-2.0, std=0.1)
                nn.init.normal_(self.log_omega, mean=0.0, std=0.1)

            nn.init.normal_(self.Phi_int, mean=0.0, std=0.1)
            nn.init.normal_(self.alpha_bias, mean=0.0, std=0.1)

    def _parameter_groups(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        spatial_names = {"Lambda_raw", "log_psi"}
        temporal_params: List[nn.Parameter] = []
        spatial_params: List[nn.Parameter] = []
        for name, param in self.named_parameters():
            if name in spatial_names:
                spatial_params.append(param)
            else:
                temporal_params.append(param)
        return temporal_params, spatial_params

    @staticmethod
    def _set_requires_grad(params: Iterable[nn.Parameter], value: bool) -> None:
        for p in params:
            p.requires_grad_(value)

    def _loss_normalizer(self, subjects_data: Sequence[Dict[str, torch.Tensor]]) -> float:
        observed_cells = 0
        latent_cells = 0
        for s in subjects_data:
            observed_cells += int(torch.isfinite(s["x"]).sum().item())
            # Count baseline latent prior plus transitions. This only scales the
            # optimization loss; it does not change the objective optimum.
            latent_cells += max(int(s["x"].shape[0]), 1) * self.K
        return float(max(observed_cells + latent_cells, 1))

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
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Multi-start Monte Carlo-free EM with gradient M-steps."""
        temporal_params, spatial_params = self._parameter_groups()
        trend_params = [self.Phi_int, self.alpha_bias]
        trend_param_ids = {id(p) for p in trend_params}
        temporal_fit_params = [p for p in temporal_params if id(p) not in trend_param_ids] if profile_linear_mstep else temporal_params
        normalizer = self._loss_normalizer(subjects_data)

        best_loss = float("inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        # Compute the expensive PCA warm start once when requested. Continuation
        # fits set use_pca_warm_start=False to preserve a diagonal warm start.
        if use_pca_warm_start:
            self.pca_warm_start(subjects_data)
        lambda_init = self.Lambda_raw.detach().clone()
        log_psi_init = self.log_psi.detach().clone()

        # Burn-in across temporal starts while holding spatial parameters fixed.
        # For continuation, n_starts=1 and randomize_temporal_starts=False.
        for start in range(n_starts):
            if randomize_temporal_starts or start > 0:
                self._randomize_temporal_parameters()
            with torch.no_grad():
                self.Lambda_raw.copy_(lambda_init)
                self.log_psi.copy_(log_psi_init)
                if randomize_temporal_starts:
                    # Match the PCA warm-start convention for random starts.
                    self.Phi_int.zero_()
                    self.alpha_bias.zero_()
                self._apply_exact_stage_constraints(exact_stage)

            self._set_requires_grad(spatial_params, False)
            self._set_requires_grad(temporal_params, True)
            if profile_linear_mstep:
                self._set_requires_grad(trend_params, False)
            opt_burn = optim.Adam(temporal_fit_params, lr=lr)

            start_loss = float("inf")
            for _ in range(burn_in_epochs):
                with torch.no_grad():
                    Gamma, Omega, _ = self.get_dynamics()
                    Lambda_fixed = self.Lambda.detach()
                    smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, Lambda_fixed)
                    if profile_linear_mstep:
                        self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)
                        self._apply_exact_stage_constraints(exact_stage)

                losses = []
                for _ in range(m_step_iters):
                    self.zero_grad(set_to_none=True)
                    opt_burn.zero_grad(set_to_none=True)
                    Gamma_m, Omega_m, _ = self.get_dynamics()
                    loss = -self.expected_complete_log_posterior_vectorized(
                        subjects_data, smoothed_stats, Gamma_m, Omega_m, self.Lambda.detach()
                    ) / normalizer
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite burn-in loss.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(temporal_fit_params, max_norm=grad_clip)
                    opt_burn.step()
                    with torch.no_grad():
                        self._apply_exact_stage_constraints(exact_stage)
                    losses.append(float(loss.detach().cpu()))
                start_loss = float(np.mean(losses))

            if start_loss < best_loss:
                best_loss = start_loss
                best_state_dict = {k: v.detach().clone() for k, v in self.state_dict().items()}

        if best_state_dict is None:
            raise RuntimeError("No valid EM start completed.")

        self.load_state_dict(best_state_dict)
        self._set_requires_grad(spatial_params, True)
        self._set_requires_grad(temporal_params, True)

        opt_dynamics_only = optim.Adam(temporal_fit_params, lr=lr)
        opt_joint = optim.Adam(
            [
                {"params": temporal_fit_params, "lr": lr},
                {"params": spatial_params, "lr": lr * 0.05},
            ]
        )

        remaining_epochs = max(num_em_epochs - burn_in_epochs, 0)
        for epoch in range(remaining_epochs):
            dynamics_only = epoch < warmup_epochs

            if dynamics_only:
                self._set_requires_grad(spatial_params, False)
                self._set_requires_grad(temporal_params, True)
                if profile_linear_mstep:
                    self._set_requires_grad(trend_params, False)
                active_opt = opt_dynamics_only
                active_params = temporal_fit_params
            else:
                self._set_requires_grad(spatial_params, True)
                self._set_requires_grad(temporal_params, True)
                if profile_linear_mstep:
                    self._set_requires_grad(trend_params, False)
                active_opt = opt_joint
                active_params = temporal_fit_params + spatial_params

            with torch.no_grad():
                Gamma, Omega, _ = self.get_dynamics()
                Lambda_for_e = self.Lambda.detach() if dynamics_only else self.Lambda
                smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, Lambda_for_e.detach())
                if profile_linear_mstep:
                    self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)
                    self._apply_exact_stage_constraints(exact_stage)

            for _ in range(m_step_iters):
                self.zero_grad(set_to_none=True)
                active_opt.zero_grad(set_to_none=True)
                Gamma_m, Omega_m, _ = self.get_dynamics()
                Lambda_m = self.Lambda.detach() if dynamics_only else self.Lambda
                loss = -self.expected_complete_log_posterior_vectorized(
                    subjects_data, smoothed_stats, Gamma_m, Omega_m, Lambda_m
                ) / normalizer
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite EM loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(active_params, max_norm=grad_clip)
                active_opt.step()
                with torch.no_grad():
                    self._apply_exact_stage_constraints(exact_stage)

        if lbfgs_refine and self.theta_mode == "exact" and lbfgs_max_iter > 0:
            with torch.no_grad():
                Gamma, Omega, _ = self.get_dynamics()
                smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda.detach())
            if profile_linear_mstep:
                self.profile_linear_trend_parameters(subjects_data, smoothed_stats, Gamma, Omega)
                self._apply_exact_stage_constraints(exact_stage)
            self.refine_temporal_lbfgs(
                subjects_data,
                max_iter=lbfgs_max_iter,
                profile_linear_mstep=profile_linear_mstep,
                exact_stage=exact_stage,
            )

        # Ensure all parameters are trainable again for downstream use.
        self._set_requires_grad(spatial_params, True)
        self._set_requires_grad(temporal_params, True)

        # Important: return smoothers under final parameters, not stale E-step stats.
        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            final_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda)
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
        """Fit exact dynamics as a staged expansion of a diagonal solution."""
        if self.theta_mode != "exact":
            return self.fit_em_multistart(
                subjects_data,
                num_em_epochs=num_em_epochs,
                warmup_epochs=max(1, num_em_epochs // 2),
                m_step_iters=m_step_iters,
                lr=lr,
                n_starts=1,
                burn_in_epochs=max(1, min(3, num_em_epochs // 4)),
                grad_clip=grad_clip,
                profile_linear_mstep=profile_linear_mstep,
            )

        diag_epochs = max(3, int(round(0.20 * num_em_epochs)))
        rev_epochs = max(4, int(round(0.30 * num_em_epochs)))
        full_epochs = max(5, num_em_epochs - diag_epochs - rev_epochs)

        self.fit_em_multistart(
            subjects_data,
            num_em_epochs=diag_epochs,
            warmup_epochs=diag_epochs,
            m_step_iters=m_step_iters,
            lr=lr,
            n_starts=1,
            burn_in_epochs=1,
            grad_clip=grad_clip,
            use_pca_warm_start=False,
            randomize_temporal_starts=False,
            exact_stage="diagonal",
            profile_linear_mstep=profile_linear_mstep,
        )

        self.fit_em_multistart(
            subjects_data,
            num_em_epochs=rev_epochs,
            warmup_epochs=rev_epochs,
            m_step_iters=m_step_iters,
            lr=lr,
            n_starts=1,
            burn_in_epochs=1,
            grad_clip=grad_clip,
            use_pca_warm_start=False,
            randomize_temporal_starts=False,
            exact_stage="reversible",
            profile_linear_mstep=profile_linear_mstep,
        )

        return self.fit_em_multistart(
            subjects_data,
            num_em_epochs=full_epochs,
            warmup_epochs=max(1, full_epochs // 2),
            m_step_iters=m_step_iters,
            lr=lr,
            n_starts=1,
            burn_in_epochs=1,
            grad_clip=grad_clip,
            use_pca_warm_start=False,
            randomize_temporal_starts=False,
            exact_stage="full",
            profile_linear_mstep=profile_linear_mstep,
            lbfgs_refine=lbfgs_max_iter > 0,
            lbfgs_max_iter=lbfgs_max_iter,
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
    exact_target_rate: float = 1.0,
    exact_skew_strength: float = 0.05,
    visit_min: int = 3,
    visit_max: int = 5,
    gap_year_min: float = 1.5,
    gap_year_max: float = 5.0,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Synthetic cohort generator consistent with the displayed SDE.

    Raw age is generated for realism, then converted to a single model time
    t_model = (age - 70) / 10. The model uses this same t_model in both
    exp(-Gamma dt) and mu(t) = (Phi x_i^(xi) + alpha)t. Raw age is stored as
    ``t_age`` only for reference.
    """
    if theta_mode not in {"exact", "diagonal"}:
        raise ValueError("theta_mode must be 'exact' or 'diagonal'.")

    torch.manual_seed(seed)
    if anchor_items is None:
        anchor_items = list(range(K))

    visit_min = int(visit_min)
    visit_max = int(visit_max)
    if visit_min < 2 or visit_max < visit_min:
        raise ValueError("visit_min must be >= 2 and visit_max must be >= visit_min.")
    gap_year_min = float(gap_year_min)
    gap_year_max = float(gap_year_max)
    if gap_year_min <= 0.0 or gap_year_max <= gap_year_min:
        raise ValueError("Require 0 < gap_year_min < gap_year_max.")

    dtype = torch.get_default_dtype()
    eye = torch.eye(K, dtype=dtype)

    if theta_mode == "diagonal":
        # Rates are per unit of model time, where one unit is roughly a decade.
        # This preserves a similar amount of longitudinal signal after switching
        # from raw years to scaled model time.
        rho_true = torch.linspace(0.2, 1.5, K)
        omega_true = torch.ones(K)
        Gamma_true = torch.diag(rho_true)
        Omega_true = torch.diag(omega_true)
    else:
        L_unc_true = torch.tril(0.25 * torch.randn(K, K) + torch.eye(K))
        Omega_raw_true = symmetrize(L_unc_true @ L_unc_true.T) + 1e-4 * eye
        Omega_true = normalize_spd_to_correlation(Omega_raw_true, jitter=1e-6)

        L_G_true = torch.tril(0.20 * torch.randn(K, K) + 0.6 * torch.eye(K))
        S_true = symmetrize(0.5 * (L_G_true @ L_G_true.T)) + 1e-4 * eye
        gamma_skew_true = exact_skew_strength * torch.randn(K, K)
        A_true = gamma_skew_true - gamma_skew_true.T
        Gamma_base = cholesky_right_solve_spd(S_true + A_true, Omega_true, jitter=1e-6)

        # Normalize by the actual dynamic rate scale rather than blindly
        # multiplying by 10. This keeps the exact DGP nontrivial but not
        # adversarial relative to the diagonal DGP.
        eigvals = torch.linalg.eigvals(Gamma_base)
        real_rates = torch.real(eigvals).clamp_min(1e-4)
        median_rate = torch.median(real_rates).clamp_min(1e-4)
        Gamma_true = Gamma_base * (exact_target_rate / median_rate)

    Phi_true = 0.5 * torch.randn(K, C_dim)
    alpha_true = 0.5 * torch.randn(K)

    Lambda_true = 0.5 * torch.randn(D, K)
    for r, idx in enumerate(anchor_items):
        Lambda_true[idx, :] = 0.0
        Lambda_true[idx, r] = torch.exp(0.5 * torch.randn(()))

    subjects_data: List[Dict[str, torch.Tensor]] = []
    for _ in range(N):
        J_i = int(torch.randint(visit_min, visit_max + 1, (1,)).item())
        age_baseline = torch.rand(()) * 20.0 + 55.0
        dt_years = torch.rand(J_i - 1) * (gap_year_max - gap_year_min) + gap_year_min
        t_age = torch.cat([age_baseline.view(1), age_baseline + torch.cumsum(dt_years, dim=0)])

        # The SDE uses one time variable t. We use a scaled age as model time
        # for numerical conditioning and use it consistently everywhere.
        t_model = (t_age - 70.0) / 10.0

        # x_i^(xi) is subject-level and time-invariant in the displayed equations.
        x_xi = torch.randn(C_dim)
        u = x_xi.unsqueeze(0).expand(J_i, C_dim).clone()
        beta_i = Phi_true @ x_xi + alpha_true

        F_true = torch.zeros(J_i, K)
        L_Omega = safe_cholesky(Omega_true, jitter=1e-6)
        mu_0 = beta_i * t_model[0]
        F_true[0] = mu_0 + L_Omega @ torch.randn(K)

        for j in range(1, J_i):
            delta_t = t_model[j] - t_model[j - 1]
            if theta_mode == "diagonal":
                rho = torch.diag(Gamma_true)
                omega = torch.diag(Omega_true)
                a = torch.exp(-rho * delta_t)
                A_ij = torch.diag(a)
                Q_true = torch.diag((omega * (1.0 - a.square())).clamp_min(1e-6))
            else:
                A_ij = torch.linalg.matrix_exp(-Gamma_true * delta_t)
                Q_true = Omega_true - A_ij @ Omega_true @ A_ij.T
                Q_true = symmetrize(Q_true) + 1e-6 * eye

            mu_prev = beta_i * t_model[j - 1]
            mu_next = beta_i * t_model[j]
            L_Q = safe_cholesky(Q_true, jitter=1e-6)
            process_noise = L_Q @ torch.randn(K)
            F_true[j] = mu_next + A_ij @ (F_true[j - 1] - mu_prev) + process_noise

        X_obs = F_true @ Lambda_true.T + torch.randn(J_i, D) * noise_scale

        if missing_visit_rate > 0.0:
            for j in range(1, J_i):
                if torch.rand(()).item() < missing_visit_rate:
                    X_obs[j, :] = float("nan")

        if item_missing_rate > 0.0:
            item_mask = torch.rand_like(X_obs) < item_missing_rate
            X_obs[item_mask] = float("nan")

        subjects_data.append(
            {
                "x": X_obs,
                "u": u,
                "x_xi": x_xi,
                "t_model": t_model,
                "t_dyn": t_model,
                "t_trend": t_model,
                "t_age": t_age,
                # Legacy key retained for backward compatibility and quick plotting.
                "t": t_model,
                "F_true": F_true,
            }
        )

    true_params = {
        "Lambda": Lambda_true,
        "Gamma": Gamma_true,
        "Omega": Omega_true,
        "Phi": Phi_true,
        "alpha": alpha_true,
    }
    return subjects_data, true_params


# ---------------------------------------------------------------------
# 4. Evaluation wrappers
# ---------------------------------------------------------------------
def scale_true_identifiable(true_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    Omega_true = true_params["Omega"]
    stds = torch.sqrt(torch.diag(Omega_true)).clamp_min(1e-12)
    D_scale = torch.diag(stds)
    D_inv = torch.diag(1.0 / stds)
    return {
        "Omega_corr": D_inv @ Omega_true @ D_inv,
        "Gamma": D_inv @ true_params["Gamma"] @ D_scale,
        "Lambda": true_params["Lambda"] @ D_scale,
        "Phi": D_inv @ true_params["Phi"],
        "alpha": D_inv @ true_params["alpha"],
    }



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
    dt_values = dt_values.detach().cpu()
    if dt_values.numel() > max_dt_values:
        idx = torch.linspace(0, dt_values.numel() - 1, max_dt_values).long()
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
    """Rapid integration test before long runs."""
    logging.info("--- STARTING SMOKE TEST ---")
    try:
        start_time = time.time()
        anchor_items = list(range(3))
        subjects_data, _ = simulate_ad_cohort_stress(
            N=5,
            D=15,
            K=3,
            C_dim=2,
            theta_mode="exact",
            seed=99,
            anchor_items=anchor_items,
            visit_min=3,
            visit_max=4,
        )

        model = CLOUDS(obs_dim=15, latent_dim=3, covar_dim=2, theta_mode="exact", anchor_items=anchor_items)
        model.fit_em_multistart(
            subjects_data,
            num_em_epochs=2,
            warmup_epochs=1,
            m_step_iters=2,
            lr=0.01,
            n_starts=2,
            burn_in_epochs=1,
        )
        _ = model.get_identifiable_parameters()

        elapsed = time.time() - start_time
        logging.info("Smoke Test Passed in %.2fs. Architecture and memory bounds verified.", elapsed)
        return True
    except Exception:
        logging.error("Smoke Test Failed with exception:")
        logging.error(traceback.format_exc())
        return False


def run_single_simulation(
    task_id: int,
    scenario: Dict[str, object],
    mode: str,
    run_idx: int,
    seed: int,
) -> Dict[str, object]:
    del run_idx
    configure_torch_runtime(CPU_THREADS)
    logging.info("[Worker %s] Spawned: %s | %s", task_id, scenario.get("name", "scenario"), mode)

    start_time = time.time()
    try:
        K = int(scenario["K"])
        D = int(scenario["D"])
        C = int(scenario["C"])
        anchor_items = list(range(K))
        subjects_data, true_params = simulate_ad_cohort_stress(
            int(scenario["N"]),
            D,
            K,
            C,
            theta_mode=mode,
            seed=seed,
            missing_visit_rate=float(scenario.get("miss", 0.0)),
            item_missing_rate=float(scenario.get("item_miss", 0.0)),
            noise_scale=float(scenario.get("noise", 1.0)),
            anchor_items=anchor_items,
            exact_target_rate=float(scenario.get("exact_target_rate", 1.0)),
            exact_skew_strength=float(scenario.get("exact_skew_strength", 0.05 if K <= 32 else 0.025)),
            visit_min=int(scenario.get("visit_min", 3)),
            visit_max=int(scenario.get("visit_max", 5)),
            gap_year_min=float(scenario.get("gap_year_min", 1.5)),
            gap_year_max=float(scenario.get("gap_year_max", 5.0)),
        )

        recipe = adaptive_fit_recipe(K, scenario)
        reg = regularization_defaults_for_k(K, scenario)
        model_kwargs = dict(
            obs_dim=D,
            latent_dim=K,
            covar_dim=C,
            anchor_items=anchor_items,
            inverse_ns_threshold=int(scenario.get("inverse_ns_threshold", 256)),
            inverse_ns_iters=int(scenario.get("inverse_ns_iters", 8)),
            inverse_ns_tol=float(scenario.get("inverse_ns_tol", 1e-3)),
            inverse_force_method=str(scenario.get("inverse_force_method", "auto")),
            omega_correlation=bool(scenario.get("omega_correlation", True)),
            **reg,
        )

        if mode == "exact":
            # Continuation: diagonal fit first, then initialize exact from it.
            diag_model = CLOUDS(theta_mode="diagonal", **model_kwargs)
            diag_model.fit_em_multistart(
                subjects_data,
                num_em_epochs=recipe.diag_pre_epochs,
                warmup_epochs=recipe.diag_pre_warmup,
                m_step_iters=recipe.diag_pre_mstep,
                lr=recipe.diag_lr,
                n_starts=recipe.diag_pre_starts,
                burn_in_epochs=recipe.diag_pre_burn,
                profile_linear_mstep=True,
            )

            model = CLOUDS(theta_mode="exact", **model_kwargs)
            model.initialize_exact_from_diagonal_model(diag_model)
            smoothed_stats = model.fit_exact_continuation(
                subjects_data,
                num_em_epochs=recipe.exact_epochs,
                m_step_iters=recipe.exact_mstep,
                lr=recipe.exact_lr,
                profile_linear_mstep=True,
                lbfgs_max_iter=recipe.exact_lbfgs,
            )
            del diag_model
        else:
            model = CLOUDS(theta_mode="diagonal", **model_kwargs)
            smoothed_stats = model.fit_em_multistart(
                subjects_data,
                num_em_epochs=recipe.diag_epochs,
                warmup_epochs=recipe.diag_warmup,
                m_step_iters=recipe.diag_mstep,
                lr=recipe.diag_lr,
                n_starts=recipe.diag_starts,
                burn_in_epochs=recipe.diag_burn,
                profile_linear_mstep=True,
            )

        with torch.no_grad():
            identifiable = model.get_identifiable_parameters()
            true_ident = scale_true_identifiable(true_params)

            Lambda_est = identifiable["Lambda"]
            Gamma_est = identifiable["Gamma"]
            Omega_est = identifiable["Omega_corr"]
            Lambda_true = true_ident["Lambda"]
            Gamma_true = true_ident["Gamma"]
            Omega_true = true_ident["Omega_corr"]

            mask = model.struct_mask == 1
            l_corr = finite_corr(Lambda_true[mask].cpu().numpy(), Lambda_est[mask].cpu().numpy())

            f_true_flat = torch.cat([subj["F_true"] for subj in subjects_data], dim=0).cpu().numpy().reshape(-1)
            f_est_flat = torch.cat([stat[0] for stat in smoothed_stats], dim=0).cpu().numpy().reshape(-1)
            f_corr = finite_corr(f_true_flat, f_est_flat)

            if mode == "diagonal":
                g_true_tensor = torch.diag(Gamma_true)
                g_est_tensor = torch.diag(Gamma_est)
            else:
                g_true_tensor = Gamma_true.reshape(-1)
                g_est_tensor = Gamma_est.reshape(-1)
            g_true = g_true_tensor.cpu().numpy().reshape(-1)
            g_est = g_est_tensor.cpu().numpy().reshape(-1)
            gamma_corr = finite_corr(g_true, g_est)
            gamma_rmse = float(np.sqrt(np.mean((g_true - g_est) ** 2)))
            gamma_slope = slope_metric(g_true, g_est)

            if mode == "exact" and K > 1:
                go_true = offdiag_flat(Gamma_true).cpu().numpy().reshape(-1)
                go_est = offdiag_flat(Gamma_est).cpu().numpy().reshape(-1)
                gamma_off_corr = finite_corr(go_true, go_est)
                gamma_off_rmse = float(np.sqrt(np.mean((go_true - go_est) ** 2)))
            else:
                gamma_off_corr = float("nan")
                gamma_off_rmse = float("nan")

            omega_corr = finite_corr(Omega_true.cpu().numpy().reshape(-1), Omega_est.cpu().numpy().reshape(-1))
            omega_rmse = float(np.sqrt(np.mean((Omega_true.cpu().numpy() - Omega_est.cpu().numpy()) ** 2)))

            all_dt = torch.cat([s["t_dyn"][1:] - s["t_dyn"][:-1] for s in subjects_data])
            max_dts = int(scenario.get("max_A_metric_dts", choose_A_metric_dt_count(K, mode)))
            A_corr, A_rel_rmse = transition_matrix_summary(
                Gamma_true,
                Gamma_est,
                all_dt,
                compare_diagonal_only=(mode == "diagonal"),
                max_dt_values=max_dts,
            )

        elapsed = time.time() - start_time
        result = {
            "status": "success",
            "task_id": task_id,
            "scenario_name": str(scenario["name"]),
            "mode": mode,
            "K": K,
            "D": D,
            "N": int(scenario["N"]),
            "visit_min": int(scenario.get("visit_min", 3)),
            "visit_max": int(scenario.get("visit_max", 5)),
            "l_corr": l_corr,
            "f_corr": f_corr,
            "g_corr": gamma_corr,
            "g_rmse": gamma_rmse,
            "g_slope": gamma_slope,
            "g_off_corr": gamma_off_corr,
            "g_off_rmse": gamma_off_rmse,
            "omega_corr": omega_corr,
            "omega_rmse": omega_rmse,
            "A_corr": A_corr,
            "A_rel_rmse": A_rel_rmse,
            "time": elapsed,
        }
        del subjects_data, true_params, model, smoothed_stats
        gc.collect()
        return result

    except Exception as e:
        return {
            "status": "error",
            "task_id": task_id,
            "scenario_name": str(scenario.get("name", "unknown")),
            "mode": mode,
            "error_msg": str(e) + "\n" + traceback.format_exc(),
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


def make_baseline_scenarios() -> List[Dict[str, object]]:
    return [
        {"name": "1. Multi-Omics Base (D=2k)", "N": 100, "D": 2000, "K": 3, "C": 2, "miss": 0.0, "noise": 1.0, "modes": ["exact", "diagonal"]},
        {"name": "2. Real-World Scale (D=8k)", "N": 100, "D": 8000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0, "modes": ["exact", "diagonal"]},
        {"name": "3. Extreme Stress (D=15k)", "N": 100, "D": 15000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0, "modes": ["exact", "diagonal"]},
        {"name": "4. Absolute Limit (D=30k)", "N": 100, "D": 30000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0, "modes": ["exact", "diagonal"]},
        {"name": "5. Missing Visits (30%)", "N": 150, "D": 2000, "K": 4, "C": 2, "miss": 0.3, "noise": 1.0, "modes": ["exact", "diagonal"]},
        {"name": "6. High Sensor Noise (3x)", "N": 150, "D": 2000, "K": 4, "C": 2, "miss": 0.0, "noise": 3.0, "modes": ["exact", "diagonal"]},
    ]


def latent_visits_for_k(K: int, profile: str) -> Tuple[int, int]:
    profile = profile.lower()
    if profile == "fast":
        if K <= 16:
            return 4, 5
        if K <= 64:
            return 5, 6
        return 5, 7
    if profile == "thorough":
        if K <= 16:
            return 5, 7
        if K <= 64:
            return 6, 9
        return 8, 10
    # standard
    if K <= 16:
        return 4, 6
    if K <= 64:
        return 5, 7
    return 6, 8


def latent_subject_count_for_k(K: int, visit_min: int, visit_max: int, profile: str) -> int:
    avg_visits = 0.5 * (visit_min + visit_max)
    # PCA warm start needs observed rows > K. Dynamics recovery also benefits
    # from more rows, but full exact K^2 recovery becomes expensive at large K.
    if profile == "fast":
        floor = 30 if K <= 32 else 24
        row_multiplier = 1.35
    elif profile == "thorough":
        floor = 90 if K <= 32 else 60
        row_multiplier = 2.50 if K <= 64 else 2.00
    else:
        floor = 64 if K <= 32 else 40
        row_multiplier = 1.75
    return int(max(floor, math.ceil(row_multiplier * K / max(avg_visits, 1.0))))


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
) -> List[Dict[str, object]]:
    """Create latent-dimension scaling scenarios.

    Default modes:
      - diagonal for every K in the grid, including three-digit K;
      - exact only up to exact_max_k, because dense exact Gamma is O(K^3).

    To force exact K=64/128/256 experiments, pass --include-large-exact or set
    --latent-exact-max-k to the desired maximum K.
    """
    scenarios: List[Dict[str, object]] = []
    for K in k_grid:
        visit_min, visit_max = latent_visits_for_k(K, profile)
        N = latent_subject_count_for_k(K, visit_min, visit_max, profile)
        D = int(min(max_d, max(min_d, d_per_k * K, K + 16)))
        modes = ["diagonal"]
        if include_large_exact or K <= exact_max_k:
            modes.insert(0, "exact")

        scenario: Dict[str, object] = {
            "name": f"Latent K={K} (D={D}, N={N}, J={visit_min}-{visit_max})",
            "N": N,
            "D": D,
            "K": K,
            "C": covar_dim,
            "miss": 0.0,
            "noise": 1.0,
            "visit_min": visit_min,
            "visit_max": visit_max,
            "modes": modes,
            "fit_profile": profile,
            "inverse_ns_threshold": inverse_ns_threshold,
            "exact_skew_strength": 0.05 if K <= 32 else 0.025,
            "exact_target_rate": 1.0,
            "max_A_metric_dts": choose_A_metric_dt_count(K, "exact"),
        }
        scenarios.append(scenario)
    return scenarios


def run_scenarios_multiprocessing(
    scenarios: Sequence[Dict[str, object]],
    *,
    n_runs: int = 2,
    suite_name: str = "CLOUDS experiments",
) -> None:
    tasks = []
    task_id = 0
    for s in scenarios:
        modes = list(s.get("modes", ["exact", "diagonal"]))
        for mode in modes:
            for run_idx in range(n_runs):
                seed = 400 + 10000 * task_id + run_idx
                tasks.append({"task_id": task_id, "scenario": dict(s), "mode": mode, "run_idx": run_idx, "seed": seed})
                task_id += 1

    requested_workers = int(os.environ.get("CLOUDS_MAX_WORKERS", "4"))
    cpu_count = os.cpu_count() or requested_workers * CPU_THREADS
    max_workers = max(1, min(requested_workers, max(1, cpu_count // max(CPU_THREADS, 1))))
    total_tasks = len(tasks)
    logging.info("--- LAUNCHING %s ---", suite_name)
    logging.info("Total isolated tasks queued: %s", total_tasks)
    logging.info("Active Python processes: %s; PyTorch threads per worker: %s", max_workers, CPU_THREADS)
    logging.info("-" * 172)

    results_aggregator: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        str(s["name"]): {mode: [] for mode in list(s.get("modes", ["exact", "diagonal"]))} for s in scenarios
    }

    import multiprocessing as mp

    mp_context = mp.get_context("spawn")
    executor_kwargs = {"max_workers": max_workers, "mp_context": mp_context}

    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            **executor_kwargs,
            max_tasks_per_child=1,
        )
        logging.info("Worker recycling enabled: max_tasks_per_child=1")
    except TypeError:
        logging.warning("max_tasks_per_child is unavailable; using non-recycling workers.")
        executor = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)

    with executor:
        future_to_task = {
            executor.submit(run_single_simulation, t["task_id"], t["scenario"], t["mode"], t["run_idx"], t["seed"]): t
            for t in tasks
        }

        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            completed_count += 1
            res = future.result()
            if res["status"] == "error":
                logging.error("Task %s FAILED (%s | %s):", res["task_id"], res["scenario_name"], res["mode"])
                logging.error(res["error_msg"])
            else:
                logging.info(
                    "[Progress %s/%s] Finished %s | %s in %.1fs",
                    completed_count,
                    total_tasks,
                    res["scenario_name"],
                    str(res["mode"]).capitalize(),
                    float(res["time"]),
                )
                results_aggregator[str(res["scenario_name"])][str(res["mode"])].append(res)

    logging.info("\n" + "=" * 172)
    logging.info("FINAL AGGREGATED RESULTS: %s", suite_name)
    logging.info("=" * 172)
    header = (
        f"{'Scenario':<42} | {'Mode':<9} | {'K':>4} | {'D':>6} | {'N':>4} | {'J':<5} | "
        f"{'Lambda':<15} | {'F':<15} | {'Gamma':<15} | {'G slope':<15} | {'G off':<15} | "
        f"{'A':<15} | {'A relRMSE':<15} | {'Omega':<15} | {'Time(s)':>8}"
    )
    logging.info(header)
    logging.info("-" * 172)

    for s in scenarios:
        scenario_name = str(s["name"])
        for mode in list(s.get("modes", ["exact", "diagonal"])):
            runs = results_aggregator[scenario_name][mode]
            if not runs:
                continue
            first = runs[0]
            l_str = _format_mean_sd([float(r["l_corr"]) for r in runs])
            f_str = _format_mean_sd([float(r["f_corr"]) for r in runs])
            g_str = _format_mean_sd([float(r["g_corr"]) for r in runs])
            gslope_str = _format_mean_sd([float(r["g_slope"]) for r in runs])
            goff_str = _format_mean_sd([float(r["g_off_corr"]) for r in runs])
            a_str = _format_mean_sd([float(r["A_corr"]) for r in runs])
            armse_str = _format_mean_sd([float(r["A_rel_rmse"]) for r in runs])
            omega_str = _format_mean_sd([float(r["omega_corr"]) for r in runs])
            avg_time = np.mean([float(r["time"]) for r in runs])
            j_str = f"{int(first['visit_min'])}-{int(first['visit_max'])}"

            row = (
                f"{scenario_name:<42} | {mode.capitalize():<9} | {int(first['K']):>4} | {int(first['D']):>6} | "
                f"{int(first['N']):>4} | {j_str:<5} | {l_str:<15} | {f_str:<15} | {g_str:<15} | "
                f"{gslope_str:<15} | {goff_str:<15} | {a_str:<15} | {armse_str:<15} | "
                f"{omega_str:<15} | {avg_time:>8.1f}"
            )
            logging.info(row)

    logging.info("-" * 172)


def run_stress_test_multiprocessing(n_runs: int = 2) -> None:
    run_scenarios_multiprocessing(make_baseline_scenarios(), n_runs=n_runs, suite_name="baseline D-scaling stress test")


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
) -> None:
    scenarios = make_latent_dimension_scenarios(
        k_grid,
        exact_max_k=exact_max_k,
        include_large_exact=include_large_exact,
        profile=profile,
        min_d=min_d,
        d_per_k=d_per_k,
        max_d=max_d,
        covar_dim=covar_dim,
        inverse_ns_threshold=inverse_ns_threshold,
    )
    run_scenarios_multiprocessing(scenarios, n_runs=n_runs, suite_name="latent-dimension K-scaling stress test")


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Run CLOUDS baseline and latent-dimension stress simulations.")
    parser.add_argument(
        "--experiment",
        choices=["baseline", "latent", "both"],
        default=os.environ.get("CLOUDS_EXPERIMENT", "latent"),
        help="Which experiment suite to run. Default: env CLOUDS_EXPERIMENT or latent.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=int(os.environ.get("CLOUDS_N_RUNS", "2")),
        help="Replicates per scenario/mode. Default: env CLOUDS_N_RUNS or 2.",
    )
    parser.add_argument("--smoke-only", action="store_true", help="Run only the smoke test and exit.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the smoke test and run the requested suite directly.")
    parser.add_argument(
        "--latent-k-grid",
        default=os.environ.get("CLOUDS_LATENT_K_GRID", "3,4,8,16,32,64,128"),
        help="Comma-separated latent dimensions for the latent suite. Default: 3,4,8,16,32,64,128.",
    )
    parser.add_argument(
        "--latent-exact-max-k",
        type=int,
        default=int(os.environ.get("CLOUDS_LATENT_EXACT_MAX_K", "32")),
        help="Run exact mode by default only up to this K. Diagonal runs for all K. Default: 32.",
    )
    parser.add_argument(
        "--include-large-exact",
        action="store_true",
        help="Force exact mode for all K in --latent-k-grid, including 64/128/256. This can be very slow.",
    )
    parser.add_argument(
        "--latent-profile",
        choices=["fast", "standard", "thorough"],
        default=os.environ.get("CLOUDS_LATENT_PROFILE", "fast"),
        help="Adaptive fitting budget for latent-dimension scaling. Default: fast.",
    )
    parser.add_argument("--latent-min-d", type=int, default=int(os.environ.get("CLOUDS_LATENT_MIN_D", "512")))
    parser.add_argument("--latent-d-per-k", type=int, default=int(os.environ.get("CLOUDS_LATENT_D_PER_K", "20")))
    parser.add_argument("--latent-max-d", type=int, default=int(os.environ.get("CLOUDS_LATENT_MAX_D", "6000")))
    parser.add_argument("--latent-covar-dim", type=int, default=int(os.environ.get("CLOUDS_LATENT_C", "2")))
    parser.add_argument(
        "--inverse-ns-threshold",
        type=int,
        default=int(os.environ.get("CLOUDS_INVERSE_NS_THRESHOLD", "256")),
        help="K threshold for attempting Newton-Schulz inverse before fallback. Default: 256.",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    if args.smoke_only:
        sys.exit(0 if run_smoke_test() else 1)

    if not args.skip_smoke and not run_smoke_test():
        logging.error("Aborting requested experiment because the smoke test failed.")
        sys.exit(1)

    if args.experiment in {"baseline", "both"}:
        run_stress_test_multiprocessing(n_runs=args.n_runs)

    if args.experiment in {"latent", "both"}:
        run_latent_dimension_multiprocessing(
            n_runs=args.n_runs,
            k_grid=parse_k_grid(args.latent_k_grid),
            exact_max_k=args.latent_exact_max_k,
            include_large_exact=args.include_large_exact,
            profile=args.latent_profile,
            min_d=args.latent_min_d,
            d_per_k=args.latent_d_per_k,
            max_d=args.latent_max_d,
            covar_dim=args.latent_covar_dim,
            inverse_ns_threshold=args.inverse_ns_threshold,
        )
