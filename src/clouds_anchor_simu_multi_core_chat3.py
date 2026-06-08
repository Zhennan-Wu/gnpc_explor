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
LOG_FILENAME = "clouds_anchor_simulation_chat3.log"
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
            Omega = L_omega @ L_omega.T + self.delta * eye
            Omega = symmetrize(Omega)

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

            lambda_laplace = 0.1
            log_prior_dyn = log_prior_dyn - lambda_laplace * torch.sum(torch.abs(self.gamma_skew))
            off_diag_G = torch.tril(self.L_G, diagonal=-1)
            log_prior_dyn = log_prior_dyn - lambda_laplace * torch.sum(torch.abs(off_diag_G))
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
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Multi-start Monte Carlo-free EM with gradient M-steps."""
        temporal_params, spatial_params = self._parameter_groups()
        normalizer = self._loss_normalizer(subjects_data)

        best_loss = float("inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None

        # Compute the expensive PCA warm start once, then reuse its spatial
        # initialization for every temporal start. This avoids repeated SVDs.
        self.pca_warm_start(subjects_data)
        lambda_init = self.Lambda_raw.detach().clone()
        log_psi_init = self.log_psi.detach().clone()

        # Burn-in across several temporal starts while holding spatial parameters fixed.
        for start in range(n_starts):
            self._randomize_temporal_parameters()
            with torch.no_grad():
                self.Lambda_raw.copy_(lambda_init)
                self.log_psi.copy_(log_psi_init)
                # Match the PCA warm-start convention: start drift/intercept from 0.
                self.Phi_int.zero_()
                self.alpha_bias.zero_()

            self._set_requires_grad(spatial_params, False)
            self._set_requires_grad(temporal_params, True)
            opt_burn = optim.Adam(temporal_params, lr=lr)

            start_loss = float("inf")
            for _ in range(burn_in_epochs):
                with torch.no_grad():
                    Gamma, Omega, _ = self.get_dynamics()
                    Lambda_fixed = self.Lambda.detach()
                    smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, Lambda_fixed)

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
                    torch.nn.utils.clip_grad_norm_(temporal_params, max_norm=grad_clip)
                    opt_burn.step()
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

        opt_dynamics_only = optim.Adam(temporal_params, lr=lr)
        opt_joint = optim.Adam(
            [
                {"params": temporal_params, "lr": lr},
                {"params": spatial_params, "lr": lr * 0.05},
            ]
        )

        remaining_epochs = max(num_em_epochs - burn_in_epochs, 0)
        for epoch in range(remaining_epochs):
            dynamics_only = epoch < warmup_epochs

            if dynamics_only:
                self._set_requires_grad(spatial_params, False)
                active_opt = opt_dynamics_only
                active_params = temporal_params
            else:
                self._set_requires_grad(spatial_params, True)
                active_opt = opt_joint
                active_params = temporal_params + spatial_params

            with torch.no_grad():
                Gamma, Omega, _ = self.get_dynamics()
                Lambda_for_e = self.Lambda.detach() if dynamics_only else self.Lambda
                smoothed_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, Lambda_for_e.detach())

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

        # Ensure all parameters are trainable again for downstream use.
        self._set_requires_grad(spatial_params, True)
        self._set_requires_grad(temporal_params, True)

        # Important: return smoothers under final parameters, not stale E-step stats.
        with torch.no_grad():
            Gamma, Omega, _ = self.get_dynamics()
            final_stats = self.smooth_all_subjects(subjects_data, Gamma, Omega, self.Lambda)
        return final_stats


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
        L_unc_true = torch.tril(0.3 * torch.randn(K, K) + torch.eye(K))
        L_corr_true = L_unc_true / torch.norm(L_unc_true, dim=1, keepdim=True).clamp_min(1e-8)
        Omega_true = symmetrize(L_corr_true @ L_corr_true.T) + 1e-4 * eye

        L_G_true = torch.tril(0.3 * torch.randn(K, K) + 0.5 * torch.eye(K))
        S_true = symmetrize(0.5 * (L_G_true @ L_G_true.T)) + 1e-4 * eye
        gamma_skew_true = 0.2 * torch.randn(K, K)
        A_true = gamma_skew_true - gamma_skew_true.T
        # Scale to per-decade model-time rates. Lyapunov validity is preserved
        # because multiplying Gamma by a positive constant also multiplies
        # Gamma Omega + Omega Gamma.T by that constant.
        Gamma_true = 10.0 * cholesky_right_solve_spd(S_true + A_true, Omega_true, jitter=1e-6)

    Phi_true = 0.5 * torch.randn(K, C_dim)
    alpha_true = 0.5 * torch.randn(K)

    Lambda_true = 0.5 * torch.randn(D, K)
    for r, idx in enumerate(anchor_items):
        Lambda_true[idx, :] = 0.0
        Lambda_true[idx, r] = torch.exp(0.5 * torch.randn(()))

    subjects_data: List[Dict[str, torch.Tensor]] = []
    for _ in range(N):
        J_i = int(torch.randint(3, 6, (1,)).item())
        age_baseline = torch.rand(()) * 20.0 + 55.0
        dt_years = torch.rand(J_i - 1) * 3.5 + 1.5
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
) -> Tuple[float, float]:
    """Correlation and relative RMSE between true/estimated transition matrices over observed dts."""
    A_true_list = []
    A_est_list = []
    for dt in dt_values:
        A_true_list.append(torch.linalg.matrix_exp(-Gamma_true * dt))
        A_est_list.append(torch.linalg.matrix_exp(-Gamma_est * dt))
    A_true = torch.stack(A_true_list).reshape(-1).cpu().numpy()
    A_est = torch.stack(A_est_list).reshape(-1).cpu().numpy()
    corr = finite_corr(A_true, A_est)
    denom = np.sqrt(np.mean(A_true ** 2)) + 1e-12
    rel_rmse = float(np.sqrt(np.mean((A_true - A_est) ** 2)) / denom)
    return corr, rel_rmse


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
    logging.info("[Worker %s] Spawned and starting math operations...", task_id)

    start_time = time.time()
    try:
        K = int(scenario["K"])
        anchor_items = list(range(K))
        subjects_data, true_params = simulate_ad_cohort_stress(
            int(scenario["N"]),
            int(scenario["D"]),
            K,
            int(scenario["C"]),
            theta_mode=mode,
            seed=seed,
            missing_visit_rate=float(scenario["miss"]),
            item_missing_rate=float(scenario.get("item_miss", 0.0)),
            noise_scale=float(scenario["noise"]),
            anchor_items=anchor_items,
        )

        model = CLOUDS(
            obs_dim=int(scenario["D"]),
            latent_dim=K,
            covar_dim=int(scenario["C"]),
            theta_mode=mode,
            anchor_items=anchor_items,
            inverse_ns_threshold=256,  # Cholesky for current K=3/4; NS only for large latent K.
            inverse_ns_iters=8,
            inverse_ns_tol=1e-3,
            inverse_force_method="auto",
        )
        smoothed_stats = model.fit_em_multistart(
            subjects_data,
            num_em_epochs=30,
            warmup_epochs=10,
            m_step_iters=15,
            lr=0.01,
            n_starts=4,
            burn_in_epochs=5,
        )

        with torch.no_grad():
            identifiable = model.get_identifiable_parameters()
            true_ident = scale_true_identifiable(true_params)

            Lambda_est = identifiable["Lambda"]
            Gamma_est = identifiable["Gamma"]
            Lambda_true = true_ident["Lambda"]
            Gamma_true = true_ident["Gamma"]

            mask = model.struct_mask == 1
            l_corr = finite_corr(Lambda_true[mask].cpu().numpy(), Lambda_est[mask].cpu().numpy())

            f_true_flat = torch.cat([subj["F_true"] for subj in subjects_data], dim=0).cpu().numpy().reshape(-1)
            f_est_flat = torch.cat([stat[0] for stat in smoothed_stats], dim=0).cpu().numpy().reshape(-1)
            f_corr = finite_corr(f_true_flat, f_est_flat)

            if mode == "diagonal":
                g_true = torch.diag(Gamma_true).cpu().numpy()
                g_est = torch.diag(Gamma_est).cpu().numpy()
            else:
                g_true = Gamma_true.cpu().numpy().reshape(-1)
                g_est = Gamma_est.cpu().numpy().reshape(-1)
            gamma_corr = finite_corr(g_true, g_est)
            gamma_rmse = float(np.sqrt(np.mean((g_true.reshape(-1) - g_est.reshape(-1)) ** 2)))

            all_dt = torch.cat([s["t_dyn"][1:] - s["t_dyn"][:-1] for s in subjects_data])
            sample_dt = all_dt[torch.linspace(0, all_dt.numel() - 1, min(20, all_dt.numel())).long()]
            A_corr, A_rel_rmse = transition_matrix_summary(Gamma_true, Gamma_est, sample_dt)

        elapsed = time.time() - start_time
        result = {
            "status": "success",
            "task_id": task_id,
            "scenario_name": str(scenario["name"]),
            "mode": mode,
            "l_corr": l_corr,
            "f_corr": f_corr,
            "g_corr": gamma_corr,
            "g_rmse": gamma_rmse,
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


def run_stress_test_multiprocessing(n_runs: int = 2) -> None:
    scenarios: List[Dict[str, object]] = [
        {"name": "1. Multi-Omics Base (D=2k)", "N": 100, "D": 2000, "K": 3, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "2. Real-World Scale (D=8k)", "N": 100, "D": 8000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "3. Extreme Stress (D=15k)", "N": 100, "D": 15000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "4. Absolute Limit (D=30k)", "N": 100, "D": 30000, "K": 4, "C": 2, "miss": 0.0, "noise": 1.0},
        {"name": "5. Missing Visits (30%)", "N": 150, "D": 2000, "K": 4, "C": 2, "miss": 0.3, "noise": 1.0},
        {"name": "6. High Sensor Noise (3x)", "N": 150, "D": 2000, "K": 4, "C": 2, "miss": 0.0, "noise": 3.0},
    ]

    tasks = []
    task_id = 0
    for s in scenarios:
        for mode in ["exact", "diagonal"]:
            for run_idx in range(n_runs):
                seed = 400 + run_idx + task_id
                tasks.append({"task_id": task_id, "scenario": s, "mode": mode, "run_idx": run_idx, "seed": seed})
                task_id += 1

    requested_workers = int(os.environ.get("CLOUDS_MAX_WORKERS", "16"))
    cpu_count = os.cpu_count() or requested_workers * CPU_THREADS
    max_workers = max(1, min(requested_workers, max(1, cpu_count // CPU_THREADS)))
    total_tasks = len(tasks)
    logging.info("--- LAUNCHING MULTIPROCESSING ---")
    logging.info("Total isolated tasks queued: %s", total_tasks)
    logging.info("Active Python processes: %s; PyTorch threads per worker: %s", max_workers, CPU_THREADS)
    logging.info("-" * 132)

    results_aggregator: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        str(s["name"]): {"exact": [], "diagonal": []} for s in scenarios
    }

    import multiprocessing as mp

    # Fresh worker per task prevents large D=30k allocations from being retained
    # in a process and slowing later D=2k tasks via allocator bloat or swapping.
    mp_context = mp.get_context("spawn")
    executor_kwargs = {"max_workers": max_workers, "mp_context": mp_context}

    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            **executor_kwargs,
            max_tasks_per_child=1,
        )
        logging.info("Worker recycling enabled: max_tasks_per_child=1")
    except TypeError:
        # Older Python fallback. Correct, but later small tasks may inherit memory
        # arenas from earlier large tasks. Upgrade Python if this warning appears.
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

    logging.info("\n" + "=" * 132)
    logging.info("FINAL AGGREGATED RESULTS")
    logging.info("=" * 132)
    header = (
        f"{'Scenario':<30} | {'Mode':<10} | {'Lambda Corr':<17} | {'F Corr':<17} | "
        f"{'Gamma Corr':<17} | {'Gamma RMSE':<17} | {'A Corr':<17} | {'A relRMSE':<17} | {'Avg Time (s)'}"
    )
    logging.info(header)
    logging.info("-" * 132)

    for s in scenarios:
        scenario_name = str(s["name"])
        for mode in ["exact", "diagonal"]:
            runs = results_aggregator[scenario_name][mode]
            if not runs:
                continue

            l_str = _format_mean_sd([float(r["l_corr"]) for r in runs])
            f_str = _format_mean_sd([float(r["f_corr"]) for r in runs])
            g_str = _format_mean_sd([float(r["g_corr"]) for r in runs])
            grmse_str = _format_mean_sd([float(r["g_rmse"]) for r in runs])
            a_str = _format_mean_sd([float(r["A_corr"]) for r in runs])
            armse_str = _format_mean_sd([float(r["A_rel_rmse"]) for r in runs])
            avg_time = np.mean([float(r["time"]) for r in runs])

            row = (
                f"{scenario_name:<30} | {mode.capitalize():<10} | {l_str:<17} | {f_str:<17} | "
                f"{g_str:<17} | {grmse_str:<17} | {a_str:<17} | {armse_str:<17} | {avg_time:>8.1f}"
            )
            logging.info(row)

    logging.info("-" * 132)


if __name__ == "__main__":
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Run CLOUDS smoke/stress simulations.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=int(os.environ.get("CLOUDS_N_RUNS", "2")),
        help="Replicates per scenario/mode. Default: env CLOUDS_N_RUNS or 2.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the smoke test and exit.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke test and run the full stress test directly.",
    )
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    if args.smoke_only:
        sys.exit(0 if run_smoke_test() else 1)

    if args.skip_smoke or run_smoke_test():
        run_stress_test_multiprocessing(n_runs=args.n_runs)
    else:
        logging.error("Aborting full stress test because the smoke test failed.")
        sys.exit(1)
