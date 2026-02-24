import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import compute_sigma_diagonal


class OUDynamicFactorModel(nn.Module):
    """
    Implements a Dynamic Factor Model where latent factors evolve according to 
    an Ornstein-Uhlenbeck process with time-varying drift.
    
    Attributes:
        D (int): Dimensionality of the observed data (number of features).
        K (int): Dimensionality of the latent factors.
        M (int): Dimensionality of the external covariates.
    """
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M, self.device = D, K, M, device
        
        # --- Learnable Parameters ---
        # Factor Loadings Matrix (D x K)
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        
        # Observation noise (Diagonal variance)
        self.log_sigma_obs = nn.Parameter(torch.log(torch.tensor(0.5, device=device))) 
        
        # OU Process Parameters: Reversion rate (rho), Cholesky of Diffusion (Gamma)
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.1)
        
        # Drift parameters: alpha (intercept) and Phi (sensitivity to covariates s)
        self.alpha = nn.Parameter(torch.zeros(K, device=device))
        self.Phi = nn.Parameter(torch.randn(K, M, device=device) * 0.05)           
        
        # Prior/Regularization parameters
        self.log_tau = nn.Parameter(torch.tensor(0.0, device=device))
        self.log_v_dk = nn.Parameter(torch.zeros(D, K, device=device))      
        
        self.history = {
            'mse': [], 'corr_lambda': [], 'err_rho': [], 'err_gamma': [], 
            'err_phi': [], 'err_alpha': [], 'err_sig': [], 'likelihood': []
        }     
        self.to(device)

    def get_rho(self): 
        """Ensures reversion rate is positive using Softplus."""
        return F.softplus(self.raw_rho) + 1e-4
    
    def get_gamma(self): 
        """Ensures the Diffusion matrix is a valid Lower Triangular Cholesky factor."""
        return torch.tril(self.Gamma_raw, -1) + torch.diag(F.softplus(torch.diag(self.Gamma_raw)) + 1e-4)
    
    def get_Lambda(self):
        """Returns the factor loadings matrix."""
        return self.Lambda
    
    def get_history(self):
        """Returns the training history of metrics."""
        return self.history

    def get_transition_params(self, dt, s_i, t):
        """
        Computes the discrete-time transition parameters for the OU process over interval dt.
        Returns:
            A: State transition matrix (Diagonal)
            b: Discretized drift vector
            Q: Process noise covariance matrix
        """
        rho = self.get_rho()
        # State transition: x_t = A*x_{t-1} + b + noise
        A = torch.exp(-rho * dt) 
        Q = compute_sigma_diagonal(rho, self.get_gamma(), dt)
        
        # Time-varying drift logic based on covariates s_i
        drift = torch.mv(self.Phi, s_i)
        b = (self.alpha + drift*(t+dt)) - A*(self.alpha + drift*t) - ((1-A)/rho)*drift
        return A, b, Q

    def kalman_filter_smoother(self, x, times, s):
        """
        Performs Forward Filtering and Backward Smoothing (RTS Smoother).
        Args:
            x: Observed data [T, D]
            times: Timestamps [T]
            s: Covariates [M]
        """
        T = x.size(0)
        # Storage for filtered (f) and smoothed (s) means and covariances (P)
        f_f = torch.zeros(T, self.K, device=self.device)
        P_f = torch.zeros(T, self.K, self.K, device=self.device)
        f_s = torch.zeros(T, self.K, device=self.device)
        P_s = torch.zeros(T, self.K, self.K, device=self.device)
        P_l = torch.zeros(T, self.K, self.K, device=self.device) # Lag-one covariance
        
        # Initial conditions (Prior)
        f_c, P_c = torch.zeros(self.K, device=self.device), torch.eye(self.K, device=self.device)
        f_f[0], P_f[0] = f_c, P_c
        
        sig_sq = torch.exp(self.log_sigma_obs)
        inv_sig_sq = 1.0 / (sig_sq + 1e-9)
        LTL = self.Lambda.T @ self.Lambda 
        # --- Forward Pass (Filtering) ---
        for j in range(1, T):
            dt = times[j] - times[j-1]
            A, b, Q = self.get_transition_params(dt, s, times[j-1])
            
            # Predict step
            f_p = A * f_c + b
            P_p = A[:, None] * P_c * A[None, :] + Q
            
            # Update step (using Woodbury-like Cholesky solve for efficiency/stability)
            M = torch.inverse(P_p + 1e-7*torch.eye(self.K, device=self.device)) + (LTL * inv_sig_sq)
        
            M_chol = torch.linalg.cholesky(M + 1e-7*torch.eye(self.K, device=self.device))
            
            innovation = x[j] - self.Lambda @ f_p
            f_c = f_p + torch.cholesky_solve((self.Lambda.T @ innovation * inv_sig_sq).unsqueeze(1), M_chol).squeeze()
            P_c = torch.cholesky_solve(torch.eye(self.K, device=self.device), M_chol)
            
            f_f[j], P_f[j] = f_c, P_c

        # --- Backward Pass (Rauch-Tung-Striebel Smoothing) ---
        f_s[-1], P_s[-1] = f_f[-1], P_f[-1]
        for j in range(T-2, -1, -1):
            A, _, Q = self.get_transition_params(times[j+1]-times[j], s, times[j])
            P_p = A[:, None] * P_f[j] * A[None, :] + Q # Predicted cov at j+1
            
            # Smoother Gain J
            J = torch.linalg.solve(P_p + 1e-7*torch.eye(self.K, device=self.device), A[:, None] * P_f[j]).T
            
            # Update smoothed estimates
            f_s[j] = f_f[j] + J @ (f_s[j+1] - (A * f_f[j] + b))
            P_s[j] = P_f[j] + J @ (P_s[j+1] - P_p) @ J.T
            P_l[j+1] = J @ P_s[j+1] # Cross-covariance P_{t, t-1}
            
        return f_s, P_s, P_l

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
            optimizer = torch.optim.Adam(self.parameters(), lr=2e-3)
            total_q = 0
            for _ in range(5):
                optimizer.zero_grad()
                loss = 0
                sig_sq = torch.exp(self.log_sigma_obs).detach()
                
                # Pre-calculate LTL to use the trace trick
                # This is K x K (e.g., 20x20), very small!
                LTL = self.Lambda.T @ self.Lambda 
                
                for i in range(len(data)):
                    f, P, P1 = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach()
                    
                    # 1. Observation Term
                    # Part A: ||y - Lambda f||^2 (O(N*D*K) - No large matrices)
                    err = data[i] - f @ self.Lambda.T
                    res_sum_sq = err.pow(2).sum()
                    
                    # Part B: Trace trick (O(N*K^3) - Tiny memory footprint)
                    # Instead of Tr(Lambda @ P @ Lambda.T), we use Tr(LTL @ P)
                    # torch.sum(LTL.unsqueeze(0) * P) performs the trace efficiently
                    tr_obs = torch.sum(LTL.unsqueeze(0) * P)
                    
                    loss += (res_sum_sq + tr_obs) / (2 * sig_sq)
                    
                    # 2. Dynamics Term (K x K operations only)
                    dt = times[i][1:] - times[i][:-1]
                    for j in range(1, f.size(0)):
                        A, b, Q = self.get_transition_params(dt[j-1], covs[i], times[i][j-1])
                        L_Q = torch.linalg.cholesky(Q + 1e-6 * torch.eye(self.K, device=self.device))
                        
                        diff = f[j] - (A * f[j-1] + b)
                        A_m = torch.diag(A)
                        e_cov = P[j] + A_m @ P[j-1] @ A_m.T - A_m @ P1[j].T - P1[j] @ A_m.T
                        
                        quad = torch.dot(diff, torch.cholesky_solve(diff.unsqueeze(1), L_Q).squeeze())
                        tr_dyn = torch.trace(torch.cholesky_solve(e_cov, L_Q))
                        loss += 0.5 * (quad + tr_dyn + 2.0 * L_Q.diagonal().log().sum())

                # 3. Horseshoe Penalty (O(D*K))
                tau, v_dk = torch.exp(self.log_tau), torch.exp(self.log_v_dk)
                loss += 0.5 * (self.Lambda.pow(2) / (tau**2 * v_dk**2 + 1e-9)).sum()
                loss += torch.log(1 + v_dk**2).sum() + torch.log(1 + tau**2).sum()
                
                loss.backward()
                optimizer.step()
                total_q = -loss.item()

            # Analytical sigma update (also using the trace trick)
            with torch.no_grad():
                rss, n_tot = 0, 0
                LTL = self.Lambda.T @ self.Lambda
                for i in range(len(data)):
                    n_tot += data[i].numel()
                    err = data[i] - all_f[i].detach() @ self.Lambda.T
                    tr = torch.sum(LTL.unsqueeze(0) * all_P[i].detach())
                    rss += err.pow(2).sum() + tr
                self.log_sigma_obs.copy_(torch.log(rss/n_tot + 1e-6))
                
            return -total_q

    def fit(self, data, covs, times, gt_params, epochs=30):
        """Optimizes model parameters using the Expected Log-Likelihood."""
        opt = torch.optim.Adam(self.parameters(), lr=2e-3)
        
        for epoch in range(epochs):
            # E-Step: Inference via Kalman Smoothing
            all_f, all_P, all_P1 = [], [], []
            for i in range(len(data)):
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f)
                all_P.append(P)
                all_P1.append(P1)
            
            q = self.m_step(data, times, covs, all_f, all_P, all_P1)

            # Tracking progress
            metrics = self.evaluate(gt_params, data, times, covs)
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
            'err_rho': torch.norm(self.get_rho()-gt['rho']).item(), 
            'err_gamma': torch.norm(self.get_gamma()-gt['Gamma']).item(),
            'err_phi': torch.norm(R.T @ self.Phi.detach() - gt['Phi']).item(), 
            'err_alpha': torch.norm(R.T @ self.alpha.detach() - gt['alpha']).item(),
            'err_sig': torch.abs(torch.exp(self.log_sigma_obs).sqrt()-gt['sigma_obs']).item()
        }