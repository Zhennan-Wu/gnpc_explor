import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from comp_utils import compute_sigma_diagonal


class OUDynamicFactorModel(nn.Module):
    """
    Implements a Dynamic Factor Model where latent factors follow an 
    Ornstein-Uhlenbeck (OU) process with time-varying drift.
    
    Equations:
    Transition: dz_t = (alpha + Phi @ s_t - rho * z_t) dt + Gamma dW_t
    Observation: x_t = Lambda @ z_t + epsilon, epsilon ~ N(0, sigma_obs^2 * I)
    """
    def __init__(self, D, K, M, device='cpu'):
        super().__init__()
        self.D, self.K, self.M, self.device = D, K, M, device
        
        # --- Learnable Parameters ---
        # Factor Loadings (D x K)
        self.Lambda = nn.Parameter(torch.randn(D, K, device=device) * 0.1)
        # Observation noise variance (scalar)
        self.sigma_obs = nn.Parameter(torch.tensor(1.0, device=device)) 
        # Mean reversion speeds (K) - kept positive via softplus
        self.raw_rho = nn.Parameter(torch.ones(K, device=device) * 0.5)
        # Diffusion matrix (Cholesky factor of process noise)
        self.Gamma_raw = nn.Parameter(torch.eye(K, device=device) * 0.1)
        # Drift components: constant (alpha) and exogenous influence (Phi)
        self.alpha = nn.Parameter(torch.zeros(K, device=device))
        self.Phi = nn.Parameter(torch.zeros(K, M, device=device))           
        
        # Priors/Hyperparameters for Lambda regularization (Horsehoe-like)
        self.tau = torch.tensor(1.0, device=device)
        self.v_dk = torch.ones(D, K, device=device)
        self.c_reg = torch.tensor(1.0, device=device)

        self.history = {'mse': [], 'corr_lambda': [], 'err_theta': [], 
                        'err_gamma': [], 'err_phi': [], 'err_alpha': [], 
                        'err_sig': [], 'likelihood': []}
        self.to(device)

    # --- Parameter Accessors (Constraints) ---
    def get_rho(self): 
        """Ensures mean reversion speed is strictly positive."""
        return F.softplus(self.raw_rho) + 1e-4
    
    def get_gamma(self): 
        """Returns lower-triangular Cholesky factor with positive diagonal."""
        return torch.tril(self.Gamma_raw, -1) + torch.diag(F.softplus(torch.diag(self.Gamma_raw)) + 1e-4)

    def get_Lambda(self):
        """Returns factor loadings (no constraints here, but could add if desired)."""
        return self.Lambda
    
    def get_history(self):
        """Returns the training history of metrics."""
        return self.history

    def get_transition_params(self, delta_t, s_i, t_p):
        """
        Calculates discretized OU transition parameters for a time step delta_t.
        z_{t+dt} ~ N(A*z_t + b, Q)
        """
        dt = torch.clamp(delta_t, min=1e-6)
        rho, gamma = self.get_rho(), self.get_gamma()
        
        # State transition matrix (Diagonal assumption for OU)
        A = torch.exp(-rho * dt) 
        
        # Process Noise Covariance (Q) - integrated over dt
        # Note: compute_sigma_diagonal must handle the specific OU integral
        Q_diag = compute_sigma_diagonal(rho, gamma, dt)
        Q = 0.5 * (Q_diag + Q_diag.T) + 1e-8 * torch.eye(self.K, device=self.device)
        
        # Drift: includes external covariates s_i
        drift = torch.mv(self.Phi, s_i)
        # Discretized intercept for the OU process
        b = (self.alpha + drift*(t_p+dt)) - A*(self.alpha + drift*t_p) - ((1-A)/rho)*drift
        return A, b, Q

    def kalman_filter_smoother(self, x, times, s):
        """
        E-Step: Estimates latent factor distributions using forward-filtering
        and backward-smoothing (Rauch-Tung-Striebel).
        """
        T = x.size(0)
        # Storage for means and covariances
        f_f, P_f = torch.zeros(T, self.K, device=self.device), torch.zeros(T, self.K, self.K, device=self.device) # Filtered
        f_s, P_s = torch.zeros(T, self.K, device=self.device), torch.zeros(T, self.K, self.K, device=self.device) # Smoothed
        P_l = torch.zeros(T, self.K, self.K, device=self.device) # Lag-one covariance
        
        # Initial State
        f_c, P_c = torch.zeros(self.K, device=self.device), torch.eye(self.K, device=self.device) 
        f_f[0], P_f[0] = f_c, P_c
        
        sig_sq = self.sigma_obs.clamp(min=1e-6)
        inv_sig_sq = 1.0 / (sig_sq + 1e-9)
        LTL = self.Lambda.T @ self.Lambda  # Precision contribution from observation
        
        # --- Forward Pass (Kalman Filter) ---
        for j in range(1, T):
            A, b, Q = self.get_transition_params(times[j]-times[j-1], s, times[j-1])
            
            # Predict
            f_p = A * f_c + b
            P_p = A[:, None] * P_c * A[None, :] + Q # Diagonal A optimization
            
            # Update (Woodbury-style or Direct Inverse)
            M = torch.eye(self.K, device=self.device) + P_p @ LTL * inv_sig_sq + 1e-7*torch.eye(self.K, device=self.device)
            K_g = torch.linalg.solve(M, P_p @ self.Lambda.T * inv_sig_sq)
            
            f_c = f_p + K_g @ (x[j] - self.Lambda @ f_p)
            IKL = torch.eye(self.K, device=self.device) - K_g @ self.Lambda
            # Joseph Form for numerical stability of covariance update
            P_c = 0.5 * (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig_sq + (IKL @ P_p @ IKL.T + K_g @ K_g.T * sig_sq).T)
            
            f_f[j], P_f[j] = f_c, P_c

        # --- Backward Pass (RTS Smoother) ---
        f_s[-1], P_s[-1] = f_f[-1], P_f[-1]
        for j in range(T-2, -1, -1):
            A, b, Q = self.get_transition_params(times[j+1]-times[j], s, times[j])
            P_pn = A[:, None] * P_f[j] * A[None, :] + Q # Predicted cov at j+1
            
            # Smoothing Gain
            L = torch.linalg.cholesky(P_pn + 1e-7 * torch.eye(self.K, device=self.device))
            J = torch.cholesky_solve(A[:, None] * P_f[j], L).T
            
            f_s[j] = f_f[j] + J @ (f_s[j+1] - (A * f_f[j] + b))
            P_s[j] = 0.5 * (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T + (P_f[j] + J @ (P_s[j+1] - P_pn) @ J.T).T)
            P_l[j+1] = J @ P_s[j+1] # Lag-one covariance for M-step transition updates
            
        return f_s, P_s, P_l

    def m_step(self, data, times, covs, all_f, all_P, all_P1):
        """
        M-Step: Updates parameters to maximize the Expected Log-Likelihood.
        Lambda and Sigma are solved analytically; OU params use Adam.
        """
        # 1. Update Lambda (Factor Loadings) with Ridge-like regularization
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
            self.Lambda.copy_(new_L)

            # Update Horseshoe Hyperpriors
            eta = 1.0 / (1.0 + self.v_dk**2 + 1e-9) 
            self.v_dk.copy_(torch.sqrt((1.0 + (self.Lambda**2/(self.tau**2 + 1e-9))) / (1.0 + eta)))
            xi = 1.0 / (1.0 + self.tau**2 + 1e-9) 
            self.tau.copy_(torch.sqrt((1.0 + (self.Lambda**2/(self.v_dk**2 + 1e-9)).sum()) / (self.D*self.K + 1.1 + xi)))

        # 2. Update OU Parameters (rho, Gamma, Phi, alpha) via Gradient Descent
        opt = torch.optim.Adam([self.raw_rho, self.Gamma_raw, self.Phi, self.alpha], lr=5e-3)
        total_q = 0
        for _ in range(2): # Minor inner loops to stabilize M-step
            opt.zero_grad()
            loss = 0
            for i in range(len(all_f)):
                f, P, P1 = all_f[i].detach(), all_P[i].detach(), all_P1[i].detach()
                dt = torch.clamp(times[i][1:]-times[i][:-1], 1e-6)
                
                for j in range(1, f.size(0)):
                    A, b, Q = self.get_transition_params(dt[j-1], covs[i], times[i][j-1])
                    LQ = torch.linalg.cholesky(Q + 1e-7*torch.eye(self.K, device=self.device))
                    
                    # Residual of the latent state transition
                    df = f[j] - (A*f[j-1] + b)
                    # Expected covariance of the transition residual
                    Am = torch.diag(A)
                    ecov = P[j] + Am @ P[j-1] @ Am.T - Am @ P1[j].T - P1[j] @ Am.T
                    
                    # Log-likelihood of the Gaussian transition
                    # Use a negative sign or define it clearly as NLL
                    # The Cholesky log-determinant part: 2 * sum(log(diag(L))) = log|Q|
                    log_det_Q = 2.0 * LQ.diagonal().log().sum()

                    # The Mahalanobis distance part: (df^T Q^-1 df)
                    mahalanobis = torch.dot(df, torch.cholesky_solve(df.unsqueeze(1), LQ).squeeze())

                    # The Trace term (from the EM-style expected log-likelihood): trace(Q^-1 * ecov)
                    trace_term = torch.trace(torch.cholesky_solve(ecov, LQ))

                    # TOTAL LOSS (NLL): We minimize this to maximize likelihood
                    loss += 0.5*(log_det_Q + mahalanobis + trace_term)
                    loss += 1e-3 * torch.norm(self.Gamma_raw) 
                    loss += 1e-3 * torch.norm(1.0 / (self.get_rho() + 1e-6))

            loss.backward()
            opt.step()
            total_q = -loss.item()

        # 3. Update Observation Noise (sigma_obs)
        with torch.no_grad():
            rss, n_tot = 0, 0
            LTL = self.Lambda.T @ self.Lambda
            for i in range(len(data)): 
                n_tot += data[i].numel()
                # Residual Sum of Squares: Data error + factor uncertainty contribution
                rss += (data[i] - all_f[i] @ self.Lambda.T).pow(2).sum() + \
                       torch.diagonal(all_P[i] @ LTL, dim1=-2, dim2=-1).sum()
            self.sigma_obs.copy_((rss/n_tot).clamp(min=1e-6))
            
        return total_q

    def fit(self, data, covs, times, gt_params, epochs=50):
        """Standard EM Loop."""
        for epoch in range(epochs):
            # E-Step
            all_f, all_P, all_P1 = [], [], []
            for i in range(len(data)): 
                f, P, P1 = self.kalman_filter_smoother(data[i], times[i], covs[i])
                all_f.append(f); all_P.append(P); all_P1.append(P1)
            
            # M-Step
            q = self.m_step(data, times, covs, all_f, all_P, all_P1)
            
            # Evaluation against ground truth (GT)
            metrics = self.evaluate(gt_params, data, times, covs)
            metrics['likelihood'] = q
            for key in metrics: self.history[key].append(metrics[key])
            
            if epoch % 5 == 0: 
                sig = self.sigma_obs.item()
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
            'err_sig': torch.abs(self.sigma_obs.sqrt()-gt['sigma_obs']).item()
        }