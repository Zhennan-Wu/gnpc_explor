import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg


def compute_sigma_diagonal(rho, gamma, delta_t):
    # Ensure GGt is symmetric and finite
    GGt = torch.matmul(gamma, gamma.mT)
    
    # x = (rho_i + rho_j) * dt
    rho_sum = rho[:, None] + rho[None, :]
    x = rho_sum * delta_t
    
    # Numerical stability: Taylor expansion for (1 - exp(-x)) / x
    # This prevents NaN when x is near 0 (Random Walk limit)
    # As x -> 0, the term (1 - exp(-x))/x  -> 1.0
    
    # Use a mask for very small values
    eps = 1e-6
    safe_x = torch.where(x.abs() < eps, torch.ones_like(x), x)
    
    # f(x) = (1 - exp(-x)) / x
    exprel = (1.0 - torch.exp(-x)) / safe_x
    
    # Fill in the Taylor approximation for the small values
    # f(x) ≈ 1 - x/2 + x^2/6
    stable_factor = torch.where(
        x.abs() < eps,
        1.0 - 0.5 * x + (x**2) / 6.0,
        exprel
    )
    
    # Q_ij = (Gamma Gamma^T)_ij * dt * stable_factor
    # The 'dt' here is because x = rho_sum * dt, and the formula is (1-exp(-rho_sum*dt))/rho_sum
    sigma = GGt * delta_t * stable_factor
    return sigma


def make_spd(A, eps=1e-5):
    # Ensure symmetry
    A = (A + A.mT) / 2
    # Handle NaNs/Infs by replacing them with a safe default (identity * eps)
    if not torch.isfinite(A).all():
        return torch.eye(A.shape[-1], device=A.device) * eps
    
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    return V @ torch.diag_embed(L) @ V.mT


class OUDynamicFactorModel(nn.Module):
    """
    Ornstein-Uhlenbeck Dynamic Factor Model (OU-DFM).
    
    Attributes:
        D (int): Number of observed dimensions (features).
        K (int): Number of latent factors.
        M (int): Number of covariates for the drift process.
    """
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M, self.device = D, K, M, device
        
        # --- Parameters ---
        # Lambda: Factor loading matrix (D x K)
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        # log_sigma_obs: Log variance of the observation noise
        self.log_sigma_obs = nn.Parameter(torch.log(torch.tensor(0.5, device=device))) 
        # raw_rho: Mean reversion speed (constrained to be positive)
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        # Gamma: Cholesky factor of the diffusion covariance
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.2)
        
        # Alpha/Phi: Parameters governing the drift of the OU process
        self.alpha = nn.Parameter(torch.zeros(K, device=device))
        self.Phi = nn.Parameter(torch.zeros(K, M, device=device))           
        
        # Regularization constants
        self.tau = torch.tensor(1.0, device=device)
        self.v_dk = torch.ones(D, K, device=device)
        self.c_reg = torch.tensor(1.0, device=device)
        self.eta_lambda = 0.4 # Learning rate/momentum for Lambda updates
        
        self.history = {k: [] for k in ['mse', 'corr_lambda', 'err_theta', 'err_gamma', 'err_phi', 'err_alpha', 'err_sig', 'likelihood']}
        self.to(device)

    # --- Helper Getters ---
    def get_rho(self): 
        """Ensures mean-reversion speed is positive."""
        return F.softplus(self.raw_rho) + 1e-4
    
    def get_gamma(self): 
        """Returns the lower-triangular Cholesky factor for factor correlations."""
        return torch.tril(self.Gamma_raw, -1) + torch.diag(F.softplus(torch.diag(self.Gamma_raw)) + 1e-4)

    def get_Lambda(self):
        """Returns the factor loadings matrix."""
        return self.Lambda
    
    def get_history(self):
        """Returns the training history of metrics."""
        return self.history

    def get_transition_params(self, delta_t, s_i, t_p):
        dt = torch.clamp(delta_t, min=1e-6)
        rho = self.get_rho() # This is already softplus'd
        gamma = self.get_gamma()
        
        # 1. Stable A
        # We use -rho * dt. If rho is positive, A is in (0, 1].
        A = torch.exp(-rho * dt) 
        
        # 2. Stable Q (using the external utility, but let's be safe)
        # If rho is tiny, (1 - exp(-2*rho*dt))/(2*rho) approx = dt
        # print(f"Calculating Q with rho: {rho}, gamma: {gamma}, dt: {dt}")
        Q_diag = compute_sigma_diagonal(rho, gamma, dt)
        # print(f"Computed Q diagonal: {Q_diag}")
        
        # --- FIX 1: Protection against external utility NaNs ---
        if torch.isnan(Q_diag).any():
            # Fallback to a Brownian Motion limit if OU integral fails
            # Q approx = (Gamma @ Gamma.T) * dt
            Sigma = gamma @ gamma.mT
            Q = Sigma * dt
        else:
            Q = 0.5 * (Q_diag + Q_diag.T)

        # 3. Stable b integration
        # The term (1 - A) / rho is the primary source of NaNs.
        # We use a limit: as rho -> 0, (1 - exp(-rho*dt))/rho -> dt
        drift_coeff = torch.where(
            rho > 1e-5, 
            (1 - A) / rho, 
            dt * (1 - 0.5 * rho * dt) # Second-order Taylor expansion
        )
        
        drift_vec = torch.mv(self.Phi, s_i)
        # b calculation restructured for stability
        # b = alpha(1-A) + drift * [ (t_p+dt) - A*t_p - (1-A)/rho ]
        term1 = self.alpha * (1 - A)
        term2 = drift_vec * (t_p + dt - A * t_p - drift_coeff)
        b = term1 + term2
        
        # Final check
        Q = Q + 1e-6 * torch.eye(self.K, device=self.device)
        # print(f"Final Q after SPD adjustment: {Q}")
        
        return A, b, Q
    
    def kalman_filter_smoother(self, x, times, s):
        """
        Performs Forward Filtering and Backward Smoothing to estimate latent factors.
        Returns: smoothed means (f_s), smoothed covariances (P_s), and lag-1 covariances (P_l).
        """
        T = x.size(0)
        assert T == len(times), "Number of time points must match length of times array"
        # Pre-allocate tensors for storage
        f_f = torch.zeros(T, self.K, device=self.device)      # Filtered means
        P_f = torch.zeros(T, self.K, self.K, device=self.device) # Filtered covs
        f_s = torch.zeros(T, self.K, device=self.device)      # Smoothed means
        P_s = torch.zeros(T, self.K, self.K, device=self.device) # Smoothed covs
        P_l = torch.zeros(T, self.K, self.K, device=self.device) # Lag-one covs (for EM)

        f_c, P_c = torch.zeros(self.K, device=self.device), torch.eye(self.K, device=self.device) 
        f_f[0], P_f[0] = f_c, P_c
        sig_sq = torch.exp(self.log_sigma_obs).clamp(min=1e-6)
        inv_sig_sq = 1.0 / (sig_sq + 1e-9)
        
        # Precompute part of the Kalman Gain for efficiency
        LTL = self.Lambda.T @ self.Lambda 

        # --- Forward Pass (Filtering) ---
        for j in range(1, T):
            dt = times[j] - times[j-1]
            A, b, Q = self.get_transition_params(dt, s, times[j-1])
            
            # Predict
            f_p = A * f_c + b
            P_p = A[:, None] * P_c * A[None, :] + Q
            
            # Update (Innovation)
            # Solving (I + P_p * L^T L / sig) * K = P_p * L^T / sig
            M_inv = torch.eye(self.K, device=self.device) + P_p @ LTL * inv_sig_sq
            K_g = torch.linalg.solve(
                M_inv + 1e-7*torch.eye(self.K, device=self.device), 
                P_p @ self.Lambda.T * inv_sig_sq
            )
            f_c = f_p + K_g @ (x[j] - self.Lambda @ f_p)
            
            # Joseph Form update for numerical stability of covariance
            IKL = torch.eye(self.K, device=self.device) - K_g @ self.Lambda
            P_c = 0.5 * (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig_sq + (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig_sq).T)
            
            f_f[j], P_f[j] = f_c, P_c

        # --- Backward Pass (RTS Smoothing) ---
        f_s[-1], P_s[-1] = f_f[-1], P_f[-1]
        for j in range(T-2, -1, -1):
            A, b, Q = self.get_transition_params(times[j+1]-times[j], s, times[j])
            P_pn = A[:, None] * P_f[j] * A[None, :] + Q # Predicted cov at j+1
            
            # Smoothing Gain
            L = torch.linalg.cholesky(P_pn + 1e-7 * torch.eye(self.K, device=self.device))
            J = torch.cholesky_solve(A[:, None] * P_f[j], L).T
            
            f_s[j] = f_f[j] + J @ (f_s[j+1] - (A * f_f[j] + b))
            P_s[j] = 0.5 * (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T + (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T).T)
            P_l[j+1] = J @ P_s[j+1] # Required for M-step transition updates
            
        return f_s, P_s, P_l

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
        """
        M-step: Updates Lambda via Ridge Regression and Dynamics via Gradient Descent.
        """
        # 1. Update Factor Loadings (Lambda) using analytical solution
        with torch.no_grad():
            sum_ffT = torch.zeros(self.K, self.K, device=self.device)
            sum_xf = torch.zeros(self.D, self.K, device=self.device)
            for i in range(len(all_f)): 
                sum_ffT += (torch.einsum('ti,tj->ij', all_f[i], all_f[i]) + all_P[i].sum(0))
                sum_xf += (data[i].T @ all_f[i])
            
            # Horseshoe Prior Logic
            v_reg = (self.c_reg**2 * self.v_dk**2) / (self.c_reg**2 + self.tau**2 * self.v_dk**2 + 1e-9)
            prec = 1.0 / (self.tau**2 * v_reg + 1e-9)
            lhs = sum_ffT.unsqueeze(0) + torch.diag_embed(prec) + 1e-6*torch.eye(self.K, device=self.device)
            
            # Damped Update
            new_L = torch.linalg.solve(lhs, sum_xf.unsqueeze(-1)).squeeze(-1)
            self.Lambda.copy_((1 - self.eta_lambda) * self.Lambda + self.eta_lambda * new_L)

        # 2. Update Dynamics (rho, Gamma, Phi, alpha) using Autograd
        opt = torch.optim.Adam([self.raw_rho, self.Gamma_raw, self.Phi, self.alpha], lr=1e-5)
        total_q = 0
        for _ in range(1): 
            opt.zero_grad()
            loss = 0
            for i in range(len(all_f)):
                f, P, P1 = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach()
                dt_list = torch.clamp(times[i][1:] - times[i][:-1], min=1e-6)
                
                for j in range(1, f.size(0)):
                    A, b, Q = self.get_transition_params(dt_list[j-1], covs[i], times[i][j-1])
                    
                    # --- Numerical Safety ---
                    Q = make_spd(Q, eps=1e-5) 
                    try:
                        LQ = torch.linalg.cholesky(Q)
                    except torch._C._LinAlgError:
                        # Fallback if make_spd fails
                        LQ = torch.linalg.cholesky(torch.eye(self.K, device=self.device) * 1e-3)

                    df = f[j] - (A * f[j-1] + b)
                    Am = torch.diag(A)
                    ecov = P[j] + Am @ P[j-1] @ Am.T - Am @ P1[j].T - P1[j] @ Am.T
                    
                    # Log-det with epsilon for stability
                    log_det_Q = 2.0 * torch.log(LQ.diagonal() + 1e-8).sum()
                    
                    # Solve terms
                    res_solve = torch.cholesky_solve(df.unsqueeze(1), LQ).squeeze()
                    mahalanobis = torch.dot(df, res_solve)
                    trace_term = torch.trace(torch.cholesky_solve(ecov, LQ))

                    step_loss = 0.5 * (log_det_Q + mahalanobis + trace_term)
                    
                    # Ignore individual time steps that have exploded
                    if torch.isfinite(step_loss):
                        loss += step_loss

            # --- The Critical Gradient Latch ---
            if loss > 0:
                loss.backward()
                
                # 1. Check if gradients exist and are finite
                params = [self.raw_rho, self.Gamma_raw, self.Phi, self.alpha]
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                
                if torch.isfinite(grad_norm):
                    opt.step()
                else:
                    print(f"Skipping optimizer step: grad_norm is {grad_norm}")
                    opt.zero_grad()
            total_q = -loss.item()

        # 3. Update Observation Noise (Sigma)
        with torch.no_grad():
            rss, n_tot = 0, 0
            LTL = self.Lambda.T @ self.Lambda
            for i in range(len(data)): 
                n_tot += data[i].numel()
                # Residual Sum of Squares includes uncertainty from the smoother
                rss += (data[i] - all_f[i] @ self.Lambda.T).pow(2).sum() + \
                       torch.diagonal(all_P[i] @ LTL, dim1=-2, dim2=-1).sum()
            self.log_sigma_obs.copy_(torch.log((rss/n_tot).clamp(min=1e-6)))
            
        return total_q

    def fit(self, data, covs, times, gt, epochs=30):
        """Training loop."""
        for epoch in range(epochs):
            all_f, all_P, all_P1 = [], [], []
            # E-Step
            for i in range(len(data)): 
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f); all_P.append(P); all_P1.append(P1)
            
            # M-Step
            q = self.m_step(data, times, covs, all_f, all_P, all_P1)
            
            # Metrics
            metrics = self.evaluate(gt, data, times, covs)
            metrics['likelihood'] = q
            for k in metrics: self.history[k].append(metrics[k])
            
            if epoch % 5 == 0: 
                sig = torch.exp(self.log_sigma_obs).item()
                print(f"Epoch {epoch:02d} | Q: {q/1000:.2f}k | MSE: {metrics['mse']:.4f} | Corr: {metrics['corr_lambda']:.4f} | Sig: {sig:.4f}")

    def evaluate(self, gt, data, times, covs):
        """Calculates error metrics against ground truth (gt)."""
        # Procrustes Alignment: Factors are only identifiable up to a rotation.
        # We find the optimal rotation R to align estimated Lambda with Ground Truth.
        Le, Lt = self.Lambda.detach(), gt['Lambda']
        U, _, Vt = scipy.linalg.svd(Lt.cpu().numpy().T @ Le.cpu().numpy())
        R = torch.tensor(Vt.T @ U.T, device=self.device, dtype=torch.float32)
        La = Le @ R # Aligned Lambda
        
        rss, count = 0, 0
        traj_aligned = []
        for i in range(len(data)):
            f, _, _ = self.kalman_filter_smoother(data[i], times[i], covs[i])
            traj_aligned.append(f @ R)
            # Apply rotation to factors for consistency
            rss += torch.sum((data[i] - (f @ R) @ La.T)**2)
            count += data[i].numel()
            
        return {
            'mse': (rss/count).item(), 
            'corr_lambda': np.corrcoef(Lt.cpu().numpy().flatten(), La.cpu().numpy().flatten())[0,1],
            'err_theta': torch.norm(self.get_rho()-gt['rho']).item(), 
            'err_gamma': torch.norm(self.get_gamma()-gt['Gamma']).item(),
            'err_phi': torch.norm(R.T @ self.Phi.detach() - gt['Phi']).item(), 
            'err_alpha': torch.norm(R.T @ self.alpha.detach() - gt['alpha']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs).sqrt()-gt['sigma_obs']).item()
        }